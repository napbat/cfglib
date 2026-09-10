//! Leader-based CFG construction from a flat, addressed instruction stream.
//!
//! [`CfgBuilder`](super::CfgBuilder) consumes structured flow markers;
//! machine code and bytecode instead arrive as sized instructions at
//! addresses, with branch targets and exception tables spelled in the same
//! address space. [`build_address_cfg`] owns the classic recovery: leaders
//! at the entry, at every direct target, after every terminator, and at
//! exception-range boundaries; blocks populated between leaders; typed
//! normal edges from each terminator's [`AddressFlow`]; `ExceptionUnwind`
//! edges from protected instructions that retain them; and ordered region
//! metadata with explicitly unknown handler-body extents, enclosing ranges
//! registered before nested ones so innermost-region resolution works.
//!
//! The instruction vocabulary stays consumer-owned through
//! [`AddressInstruction`]; edge payloads are consumer-built from an
//! [`AddressEdgeInfo`] description, so exact source coordinates, case keys,
//! and handler identities survive into the caller's own edge type.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::fmt;
use core::ops::Range;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::region::{Handler, HandlerBody, HandlerKind, HandlerRef, Region, RegionId};

/// A code address: totally ordered, with a numeric distance for range spans.
pub trait AddressSpace: Copy + Ord {
    /// The non-negative distance from `earlier` to `self`.
    ///
    /// Called only with `earlier <= self`; used to order exception ranges by
    /// span and to pick the innermost enclosing region.
    fn distance_from(self, earlier: Self) -> u64;
}

macro_rules! unsigned_address_space {
    ($($ty:ty),*) => {
        $(impl AddressSpace for $ty {
            fn distance_from(self, earlier: Self) -> u64 {
                u64::try_from(self).unwrap_or(u64::MAX)
                    - u64::try_from(earlier).unwrap_or(u64::MAX)
            }
        })*
    };
}
unsigned_address_space!(u8, u16, u32, u64, usize);

/// One instruction's address-shaped control transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AddressFlow<A, K> {
    /// Continues at the instruction's exclusive end address.
    FallThrough,
    /// Leaves the function normally.
    Return,
    /// Leaves the function exceptionally.
    Throw,
    /// Transfers to a target the stream does not name.
    Indirect,
    /// Branches to `target` when taken. Otherwise, continues at the
    /// instruction's exclusive end address.
    Conditional {
        /// Taken-path target address.
        target: A,
    },
    /// Always transfers to `target`.
    Unconditional {
        /// Branch target address.
        target: A,
    },
    /// Calls `target` and continues at the instruction's exclusive end
    /// address.
    Call {
        /// Call target address.
        target: A,
    },
    /// Multi-way dispatch over keyed cases with a default.
    Switch {
        /// Target when no case matches.
        default: A,
        /// Keyed case targets in dispatch order.
        cases: Vec<(K, A)>,
    },
}

impl<A, K> AddressFlow<A, K> {
    /// Whether the instruction ends its basic block.
    ///
    /// Every transfer except a plain fall-through does — including a
    /// [`Call`](Self::Call): its `Call` and continuation edges leave the
    /// block's last instruction, so the return site must lead a block of
    /// its own.
    #[must_use]
    pub const fn ends_basic_block(&self) -> bool {
        !matches!(self, AddressFlow::FallThrough)
    }
}

/// The instruction contract for leader-based construction.
pub trait AddressInstruction {
    /// The consumer's address vocabulary.
    type Address: AddressSpace;
    /// The consumer's switch case key.
    type CaseKey;

    /// The instruction's own address.
    fn address(&self) -> Self::Address;

    /// The exclusive end address, or `None` when it overflows the space.
    fn end_address(&self) -> Option<Self::Address>;

    /// The instruction's control transfer.
    fn flow(&self) -> AddressFlow<Self::Address, Self::CaseKey>;

    /// Whether an exception edge from this protected instruction is kept.
    ///
    /// Return `false` for instructions whose native semantics provably
    /// cannot throw, pruning their unwind edges.
    fn retains_exception_edge(&self) -> bool {
        true
    }
}

/// One exception-table entry in the instruction address space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddressHandler<A> {
    /// Protected half-open address range.
    pub protected: Range<A>,
    /// Handler entry address.
    pub entry: A,
    /// Catch classification stored on the produced region handler.
    ///
    /// [`HandlerKind::Filter`] names a block that does not exist yet at
    /// build time; register such handlers as [`HandlerKind::Catch`] and
    /// patch the kind through the returned [`HandlerRef`] afterwards.
    pub kind: HandlerKind,
}

/// Why one edge exists, as described to the payload builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressEdgeRole<'k, A, K> {
    /// Sequential continuation from an instruction's exclusive end address.
    Sequential,
    /// Taken path of a conditional branch.
    ConditionalTaken,
    /// Fall-through path of a conditional branch.
    ConditionalFallThrough,
    /// Direct unconditional branch.
    Branch,
    /// Switch dispatch when no case matches.
    SwitchDefault,
    /// Switch dispatch for one keyed case.
    SwitchCase {
        /// The consumer's case key.
        key: &'k K,
    },
    /// Direct call transfer.
    Call,
    /// Continuation after a call returns.
    CallContinuation {
        /// Address of the calling instruction.
        call_site: A,
    },
    /// Exceptional transfer from a protected instruction to a handler.
    Unwind {
        /// Index of the handler-table entry selecting this edge.
        handler: usize,
    },
}

/// One edge as described to the consumer's payload builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddressEdgeInfo<'k, A, K> {
    /// Address of the transferring (or protected) instruction.
    pub source: A,
    /// Address of the first instruction of the target block.
    pub target: A,
    /// The structural kind the edge is registered under.
    pub kind: EdgeKind,
    /// Why the edge exists.
    pub role: AddressEdgeRole<'k, A, K>,
}

/// Why leader-based construction rejected the input stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressBuildError<A> {
    /// An instruction occupies no addresses.
    ZeroSizeInstruction {
        /// The empty instruction's address.
        address: A,
    },
    /// Instruction start addresses are not in strictly increasing order.
    UnorderedInstruction {
        /// The preceding instruction's address.
        previous: A,
        /// The next instruction's address.
        address: A,
    },
    /// An instruction's end does not fit the address space.
    AddressOverflow {
        /// The overflowing instruction's address.
        address: A,
    },
    /// A direct branch names an address that is not an instruction start.
    MissingBranchTarget {
        /// The branching instruction.
        source: A,
        /// The invalid target.
        target: A,
    },
    /// A protected range contains no addresses.
    EmptyProtectedRange {
        /// Range start.
        start: A,
        /// Range end.
        end: A,
    },
    /// A protected range starts off an instruction boundary.
    MissingRangeStart {
        /// The invalid start address.
        address: A,
    },
    /// A protected range ends off an instruction boundary before code end.
    MissingRangeEnd {
        /// The invalid end address.
        address: A,
    },
    /// A handler entry is not an instruction start.
    MissingHandlerEntry {
        /// The invalid entry address.
        address: A,
    },
}

impl<A: fmt::Display> fmt::Display for AddressBuildError<A> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSizeInstruction { address } => {
                write!(formatter, "instruction at {address} occupies no addresses")
            }
            Self::UnorderedInstruction { previous, address } => write!(
                formatter,
                "instruction addresses are not increasing: {previous} then {address}"
            ),
            Self::AddressOverflow { address } => write!(
                formatter,
                "instruction at {address} ends outside the address space"
            ),
            Self::MissingBranchTarget { source, target } => write!(
                formatter,
                "branch at {source} targets {target}, which is not an instruction start"
            ),
            Self::EmptyProtectedRange { start, end } => {
                write!(formatter, "protected range {start}..{end} is empty")
            }
            Self::MissingRangeStart { address } => write!(
                formatter,
                "protected range starts at {address}, which is not an instruction start"
            ),
            Self::MissingRangeEnd { address } => write!(
                formatter,
                "protected range ends at {address}, which is not an instruction start"
            ),
            Self::MissingHandlerEntry { address } => write!(
                formatter,
                "handler entry {address} is not an instruction start"
            ),
        }
    }
}

impl<A: fmt::Debug + fmt::Display> core::error::Error for AddressBuildError<A> {}

/// A leader-based construction result.
#[derive(Debug)]
pub struct AddressGraph<I: AddressInstruction, E> {
    /// The constructed graph, instructions distributed into blocks.
    pub cfg: Cfg<I, E>,
    /// The block containing each instruction address.
    pub instruction_blocks: BTreeMap<I::Address, BlockId>,
    /// One region handler per input handler-table entry, in table order.
    pub handler_refs: Vec<HandlerRef>,
}

/// Builds a CFG from a sorted, sized, addressed instruction stream.
///
/// Leaders are introduced at the entry, at every direct branch target,
/// after every block-ending instruction, at each overlapping path boundary,
/// and at exception boundaries —
/// every instruction inside a protected range leads its own block, so
/// unwind edges stay instruction-exact. Handlers sharing one protected
/// range share one region; enclosing regions are registered before nested
/// ones (spans descending, table order preserved between equal spans) and
/// each nested region names its innermost enclosing parent. All handler
/// bodies start [`HandlerBody::Unknown`]; pair with
/// [`promote_handler_extents`](crate::promote_handler_extents) or
/// [`recover_exclusive_extents`](crate::recover_exclusive_extents) to
/// recover extents, and run [`verify`](crate::verify) (or
/// [`verify_with`](crate::verify_with)) afterwards — construction validates
/// the stream, not the finished graph.
///
/// `edge_payload` builds the consumer edge type from each
/// [`AddressEdgeInfo`]; the structural [`EdgeKind`] is chosen here.
///
/// # Errors
///
/// Instructions can overlap when they have different start addresses. The
/// builder uses each instruction's exclusive end address for fall-through
/// edges, so one byte range can belong to more than one control-flow path.
///
/// Returns an [`AddressBuildError`] when instruction starts are not strictly
/// increasing, an instruction has zero size, a direct target is off an
/// instruction boundary, or exception metadata names invalid addresses.
pub fn build_address_cfg<I: AddressInstruction, E>(
    instructions: Vec<I>,
    handlers: &[AddressHandler<I::Address>],
    mut edge_payload: impl FnMut(AddressEdgeInfo<'_, I::Address, I::CaseKey>) -> E,
) -> Result<AddressGraph<I, E>, AddressBuildError<I::Address>> {
    let code_end = validate_instructions(&instructions)?;
    let instruction_addresses: BTreeSet<I::Address> = instructions
        .iter()
        .map(AddressInstruction::address)
        .collect();
    let mut leaders = collect_flow_leaders(&instructions, &instruction_addresses)?;
    validate_handlers(
        &instructions,
        handlers,
        code_end,
        &instruction_addresses,
        &mut leaders,
    )?;

    let (mut cfg, instruction_blocks) = populate_blocks(instructions, &leaders);
    add_normal_edges(&mut cfg, &instruction_blocks, &mut edge_payload);
    add_exception_edges(&mut cfg, handlers, &instruction_blocks, &mut edge_payload);
    let handler_refs = add_regions(&mut cfg, handlers, &instruction_blocks);

    Ok(AddressGraph {
        cfg,
        instruction_blocks,
        handler_refs,
    })
}

fn validate_instructions<I: AddressInstruction>(
    instructions: &[I],
) -> Result<Option<I::Address>, AddressBuildError<I::Address>> {
    let mut previous_address = None;
    let mut code_end = None;
    for instruction in instructions {
        let address = instruction.address();
        let end = instruction
            .end_address()
            .ok_or(AddressBuildError::AddressOverflow { address })?;
        if end <= address {
            return Err(AddressBuildError::ZeroSizeInstruction { address });
        }
        if let Some(previous) = previous_address
            && address <= previous
        {
            return Err(AddressBuildError::UnorderedInstruction { previous, address });
        }
        previous_address = Some(address);
        if code_end.is_none_or(|current| end > current) {
            code_end = Some(end);
        }
    }
    Ok(code_end)
}

fn collect_flow_leaders<I: AddressInstruction>(
    instructions: &[I],
    instruction_addresses: &BTreeSet<I::Address>,
) -> Result<BTreeSet<I::Address>, AddressBuildError<I::Address>> {
    let mut leaders = BTreeSet::new();
    if let Some(first) = instructions.first() {
        leaders.insert(first.address());
    }
    for (position, instruction) in instructions.iter().enumerate() {
        let source = instruction.address();
        let flow = instruction.flow();
        match &flow {
            AddressFlow::FallThrough
            | AddressFlow::Return
            | AddressFlow::Throw
            | AddressFlow::Indirect => {}
            AddressFlow::Conditional { target }
            | AddressFlow::Unconditional { target }
            | AddressFlow::Call { target } => {
                validate_target(source, *target, instruction_addresses)?;
                leaders.insert(*target);
            }
            AddressFlow::Switch { default, cases } => {
                validate_target(source, *default, instruction_addresses)?;
                leaders.insert(*default);
                for (_, target) in cases {
                    validate_target(source, *target, instruction_addresses)?;
                    leaders.insert(*target);
                }
            }
        }
        let next = instructions.get(position + 1);
        let continues_to_next =
            next.is_some_and(|next| instruction.end_address() == Some(next.address()));
        if let Some(next) = next
            && (!matches!(&flow, AddressFlow::FallThrough) || !continues_to_next)
        {
            leaders.insert(next.address());
        }
        let continuation_starts_block = match &flow {
            AddressFlow::FallThrough => !continues_to_next,
            AddressFlow::Conditional { .. } | AddressFlow::Call { .. } => true,
            AddressFlow::Return
            | AddressFlow::Throw
            | AddressFlow::Indirect
            | AddressFlow::Unconditional { .. }
            | AddressFlow::Switch { .. } => false,
        };
        if continuation_starts_block
            && let Some(continuation) = instruction.end_address()
            && instruction_addresses.contains(&continuation)
        {
            leaders.insert(continuation);
        }
    }
    Ok(leaders)
}

fn validate_target<A: AddressSpace>(
    source: A,
    target: A,
    instruction_addresses: &BTreeSet<A>,
) -> Result<(), AddressBuildError<A>> {
    if instruction_addresses.contains(&target) {
        Ok(())
    } else {
        Err(AddressBuildError::MissingBranchTarget { source, target })
    }
}

fn validate_handlers<I: AddressInstruction>(
    instructions: &[I],
    handlers: &[AddressHandler<I::Address>],
    code_end: Option<I::Address>,
    instruction_addresses: &BTreeSet<I::Address>,
    leaders: &mut BTreeSet<I::Address>,
) -> Result<(), AddressBuildError<I::Address>> {
    for handler in handlers {
        let (start, end) = (handler.protected.start, handler.protected.end);
        if end <= start {
            return Err(AddressBuildError::EmptyProtectedRange { start, end });
        }
        if !instruction_addresses.contains(&start) {
            return Err(AddressBuildError::MissingRangeStart { address: start });
        }
        let at_code_end = code_end.is_some_and(|code_end| end == code_end);
        if !at_code_end && !instruction_addresses.contains(&end) {
            return Err(AddressBuildError::MissingRangeEnd { address: end });
        }
        if !instruction_addresses.contains(&handler.entry) {
            return Err(AddressBuildError::MissingHandlerEntry {
                address: handler.entry,
            });
        }
        leaders.insert(start);
        for instruction in instructions {
            if handler.protected.contains(&instruction.address()) {
                leaders.insert(instruction.address());
            }
        }
        if !at_code_end {
            leaders.insert(end);
        }
        leaders.insert(handler.entry);
    }
    Ok(())
}

fn populate_blocks<I: AddressInstruction, E>(
    instructions: Vec<I>,
    leaders: &BTreeSet<I::Address>,
) -> (Cfg<I, E>, BTreeMap<I::Address, BlockId>) {
    let mut cfg = Cfg::with_edge_payload();
    let mut current = cfg.entry();
    let mut instruction_blocks = BTreeMap::new();
    for (position, instruction) in instructions.into_iter().enumerate() {
        let address = instruction.address();
        if position != 0 && leaders.contains(&address) {
            current = cfg.new_block();
        }
        cfg.block_mut(current).push(instruction);
        instruction_blocks.insert(address, current);
    }
    (cfg, instruction_blocks)
}

fn add_normal_edges<I: AddressInstruction, E>(
    cfg: &mut Cfg<I, E>,
    instruction_blocks: &BTreeMap<I::Address, BlockId>,
    edge_payload: &mut impl FnMut(AddressEdgeInfo<'_, I::Address, I::CaseKey>) -> E,
) {
    let blocks: Vec<BlockId> = cfg
        .blocks()
        .iter()
        .filter(|block| !block.is_empty())
        .map(crate::BasicBlock::id)
        .collect();

    for &block in &blocks {
        let Some(terminator) = cfg.block(block).instructions().last() else {
            continue;
        };
        let source = terminator.address();
        let flow = terminator.flow();
        let continuation = terminator
            .end_address()
            .and_then(|address| instruction_blocks.get(&address).copied());
        add_terminator_edges(
            cfg,
            instruction_blocks,
            edge_payload,
            block,
            source,
            flow,
            continuation,
        );
    }
}

fn add_terminator_edges<I: AddressInstruction, E>(
    cfg: &mut Cfg<I, E>,
    instruction_blocks: &BTreeMap<I::Address, BlockId>,
    edge_payload: &mut impl FnMut(AddressEdgeInfo<'_, I::Address, I::CaseKey>) -> E,
    block: BlockId,
    source: I::Address,
    flow: AddressFlow<I::Address, I::CaseKey>,
    continuation: Option<BlockId>,
) {
    let mut add = |cfg: &mut Cfg<I, E>,
                   target: BlockId,
                   kind: EdgeKind,
                   role: AddressEdgeRole<'_, I::Address, I::CaseKey>| {
        let Some(target_address) = cfg
            .block(target)
            .instructions()
            .first()
            .map(AddressInstruction::address)
        else {
            return;
        };
        let payload = edge_payload(AddressEdgeInfo {
            source,
            target: target_address,
            kind,
            role,
        });
        cfg.add_edge_with_payload(block, target, kind, payload);
    };
    match flow {
        AddressFlow::FallThrough => {
            if let Some(next) = continuation {
                add(
                    cfg,
                    next,
                    EdgeKind::Fallthrough,
                    AddressEdgeRole::Sequential,
                );
            }
        }
        AddressFlow::Conditional { target } => {
            add(
                cfg,
                instruction_blocks[&target],
                EdgeKind::ConditionalTrue,
                AddressEdgeRole::ConditionalTaken,
            );
            if let Some(next) = continuation {
                add(
                    cfg,
                    next,
                    EdgeKind::ConditionalFalse,
                    AddressEdgeRole::ConditionalFallThrough,
                );
            }
        }
        AddressFlow::Unconditional { target } => {
            add(
                cfg,
                instruction_blocks[&target],
                EdgeKind::Jump,
                AddressEdgeRole::Branch,
            );
        }
        AddressFlow::Switch { default, cases } => {
            add(
                cfg,
                instruction_blocks[&default],
                EdgeKind::SwitchCase,
                AddressEdgeRole::SwitchDefault,
            );
            for (key, target) in &cases {
                add(
                    cfg,
                    instruction_blocks[target],
                    EdgeKind::SwitchCase,
                    AddressEdgeRole::SwitchCase { key },
                );
            }
        }
        AddressFlow::Call { target } => {
            add(
                cfg,
                instruction_blocks[&target],
                EdgeKind::Call,
                AddressEdgeRole::Call,
            );
            if let Some(next) = continuation {
                add(
                    cfg,
                    next,
                    EdgeKind::CallReturn,
                    AddressEdgeRole::CallContinuation { call_site: source },
                );
            }
        }
        AddressFlow::Return | AddressFlow::Throw | AddressFlow::Indirect => {}
    }
}

fn add_exception_edges<I: AddressInstruction, E>(
    cfg: &mut Cfg<I, E>,
    handlers: &[AddressHandler<I::Address>],
    instruction_blocks: &BTreeMap<I::Address, BlockId>,
    edge_payload: &mut impl FnMut(AddressEdgeInfo<'_, I::Address, I::CaseKey>) -> E,
) {
    let block_starts: Vec<(BlockId, I::Address, bool)> = cfg
        .blocks()
        .iter()
        .filter_map(|block| {
            block.instructions().first().map(|instruction| {
                (
                    block.id(),
                    instruction.address(),
                    instruction.retains_exception_edge(),
                )
            })
        })
        .collect();

    for (index, handler) in handlers.iter().enumerate() {
        let Some(&target) = instruction_blocks.get(&handler.entry) else {
            continue;
        };
        for &(source, address, retains) in &block_starts {
            if handler.protected.contains(&address) && retains {
                let payload = edge_payload(AddressEdgeInfo {
                    source: address,
                    target: handler.entry,
                    kind: EdgeKind::ExceptionUnwind,
                    role: AddressEdgeRole::Unwind { handler: index },
                });
                cfg.add_edge_with_payload(source, target, EdgeKind::ExceptionUnwind, payload);
            }
        }
    }
}

fn add_regions<I: AddressInstruction, E>(
    cfg: &mut Cfg<I, E>,
    handlers: &[AddressHandler<I::Address>],
    instruction_blocks: &BTreeMap<I::Address, BlockId>,
) -> Vec<HandlerRef> {
    struct Draft<A> {
        protected: Range<A>,
        handler_indices: Vec<usize>,
    }

    let mut drafts: Vec<Draft<I::Address>> = Vec::new();
    for (index, handler) in handlers.iter().enumerate() {
        if let Some(draft) = drafts
            .iter_mut()
            .find(|draft| draft.protected == handler.protected)
        {
            draft.handler_indices.push(index);
        } else {
            drafts.push(Draft {
                protected: handler.protected.clone(),
                handler_indices: vec![index],
            });
        }
    }

    // Innermost-region resolution walks reverse insertion order, so
    // enclosing ranges must be registered before nested ranges. Stable
    // sorting preserves table order for peers of equal span.
    let span = |range: &Range<I::Address>| -> u64 { range.end.distance_from(range.start) };
    drafts.sort_by_key(|draft| Reverse(span(&draft.protected)));

    let block_starts: Vec<(BlockId, I::Address)> = cfg
        .blocks()
        .iter()
        .filter_map(|block| {
            block
                .instructions()
                .first()
                .map(|instruction| (block.id(), instruction.address()))
        })
        .collect();

    let mut handler_refs = vec![None; handlers.len()];
    let mut region_ids: Vec<RegionId> = Vec::with_capacity(drafts.len());
    for (index, draft) in drafts.iter().enumerate() {
        let protected_blocks = block_starts
            .iter()
            .filter_map(|(block, address)| draft.protected.contains(address).then_some(*block))
            .collect();
        let region_handlers = draft
            .handler_indices
            .iter()
            .filter_map(|&handler_index| {
                let handler = &handlers[handler_index];
                instruction_blocks
                    .get(&handler.entry)
                    .map(|&entry| Handler {
                        entry,
                        body: HandlerBody::unknown(),
                        kind: handler.kind,
                    })
            })
            .collect();
        let parent = drafts[..index]
            .iter()
            .enumerate()
            .filter(|(_, candidate)| strictly_contains(&candidate.protected, &draft.protected))
            .min_by_key(|(_, candidate)| span(&candidate.protected))
            .and_then(|(parent_index, _)| region_ids.get(parent_index).copied());

        let region = cfg.add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks,
            handlers: region_handlers,
            parent,
        });
        region_ids.push(region);
        for (handler_position, &handler_index) in draft.handler_indices.iter().enumerate() {
            handler_refs[handler_index] = Some(HandlerRef::new(region, handler_position));
        }
    }
    handler_refs.into_iter().flatten().collect()
}

fn strictly_contains<A: AddressSpace>(outer: &Range<A>, inner: &Range<A>) -> bool {
    (outer.start != inner.start || outer.end != inner.end)
        && outer.start <= inner.start
        && inner.end <= outer.end
}

#[cfg(test)]
mod tests;
