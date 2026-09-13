//! Leader-based CFG construction from a flat, addressed instruction stream.
//!
//! [`CfgBuilder`](super::CfgBuilder) consumes structured flow markers;
//! machine code and bytecode instead arrive as sized instructions at
//! addresses, with branch targets and exception tables spelled in the same
//! address space. [`build_address_cfg`] owns the classic recovery: leaders
//! at the entry, at every direct target, after every terminator, at every
//! exceptional instruction, and at exception-range boundaries; blocks
//! populated between leaders; typed normal edges from each terminator's
//! [`Flow`]; `ExceptionUnwind` edges from protected instructions that
//! retain them; explicit unresolved transfers without discarding known
//! successors; call sites flattened or kept as edges by policy; and ordered
//! region metadata with explicitly unknown handler-body extents, enclosing
//! ranges registered before nested ones so innermost-region resolution
//! works.
//!
//! The instruction vocabulary stays consumer-owned through
//! [`AddressInstruction`]; edge payloads are consumer-built from an
//! [`AddressEdgeInfo`] description, so exact source coordinates, case keys,
//! and handler identities survive into the caller's own edge type.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::fmt;
use core::ops::Range;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::flow::{CallSite, EdgeRole, Flow, Transfer, UnresolvedTransfer};
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
    fn flow(&self) -> Flow<Self::Address, Self::CaseKey>;

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

/// One edge as described to the consumer's payload builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddressEdgeInfo<'k, A, K> {
    /// Address of the transferring (or protected) instruction.
    pub source: A,
    /// Address of the first instruction of the target block.
    pub target: A,
    /// The structural kind the edge is registered under.
    ///
    /// Always [`role.kind()`](EdgeRole::kind), carried here so a payload
    /// builder that stores only the kind does not repeat the mapping.
    pub kind: EdgeKind,
    /// Why the edge exists.
    pub role: EdgeRole<'k, A, K>,
}

/// How the built graph treats a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CallPolicy {
    /// A call is a transfer: the graph carries a `Call` edge to a recovered
    /// callee and a `CallReturn` edge to the continuation, and the call ends
    /// its block.
    #[default]
    Edges,
    /// A call is an operation of the caller: the graph carries only the
    /// sequential continuation, the call does not end its block, and every
    /// call site is reported in [`AddressGraph::calls`].
    ///
    /// This is the intraprocedural view a function-local recovery wants: an
    /// edge into the callee would fuse two functions' graphs.
    Flatten,
}

/// What [`build_address_cfg`] is asked to do beyond the stream itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AddressCfgOptions<A> {
    /// The address the function is entered at.
    ///
    /// `None` enters at the first instruction in address order. An explicit
    /// entry separates function identity from storage order: instructions
    /// below it remain available for backward targets.
    pub entry: Option<A>,
    /// How calls appear in the graph.
    pub calls: CallPolicy,
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
    /// The requested function entry is not an instruction start.
    MissingEntry {
        /// The unavailable entry address.
        address: A,
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
            Self::MissingEntry { address } => write!(
                formatter,
                "function entry {address} is not an instruction start"
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
    /// Transfers whose destination is outside the recovered instruction set.
    ///
    /// These are explicit precision results, not build failures. A conditional
    /// or call with an unresolved direct target still keeps its local
    /// continuation, and a switch keeps every destination that did resolve.
    pub unresolved_transfers: Vec<UnresolvedTransfer<I::Address>>,
    /// Every call site, in address order, when calls are flattened.
    ///
    /// Empty under [`CallPolicy::Edges`], where calls are edges instead.
    pub calls: Vec<CallSite<I::Address>>,
}

/// Builds a CFG from a sorted, sized, addressed instruction stream.
///
/// Leaders are introduced at the entry, at every direct branch target,
/// after every block-ending instruction, at each overlapping path boundary,
/// at every instruction with an exceptional transfer, and at exception
/// boundaries — every instruction inside a protected range leads its own
/// block, so unwind edges stay instruction-exact, and an exceptional
/// instruction leads its own block, so its alternate-path edge does too.
/// Handlers sharing one protected range share one region; enclosing regions
/// are registered before nested ones (spans descending, table order
/// preserved between equal spans) and each nested region names its
/// innermost enclosing parent. All handler bodies start
/// [`HandlerBody::Unknown`]; pair with
/// [`promote_handler_extents`](crate::promote_handler_extents) or
/// [`recover_exclusive_extents`](crate::recover_exclusive_extents) to
/// recover extents, and run [`verify`](crate::verify) (or
/// [`verify_with`](crate::verify_with)) afterwards — construction validates
/// the stream, not the finished graph.
///
/// `options` selects the function entry and the call policy. `edge_payload`
/// builds the consumer edge type from each [`AddressEdgeInfo`]; the
/// structural [`EdgeKind`] is chosen here, by [`EdgeRole::kind`].
///
/// # Errors
///
/// Instructions can overlap when they have different start addresses. The
/// builder uses each instruction's exclusive end address for fall-through
/// edges, so one byte range can belong to more than one control-flow path.
///
/// Returns an [`AddressBuildError`] when instruction starts are not strictly
/// increasing, an instruction has zero size, the requested entry is not an
/// instruction start, or exception metadata names invalid addresses. Direct
/// instruction targets outside the recovered stream are returned as
/// [`AddressGraph::unresolved_transfers`] while every known successor is
/// retained.
pub fn build_address_cfg<I: AddressInstruction, E>(
    instructions: Vec<I>,
    handlers: &[AddressHandler<I::Address>],
    options: AddressCfgOptions<I::Address>,
    mut edge_payload: impl FnMut(AddressEdgeInfo<'_, I::Address, I::CaseKey>) -> E,
) -> Result<AddressGraph<I, E>, AddressBuildError<I::Address>> {
    let (addresses, transfers, code_end) = inspect_instructions(&instructions)?;
    let entry_position = options
        .entry
        .map(|address| {
            addresses
                .binary_search(&address)
                .map_err(|_| AddressBuildError::MissingEntry { address })
        })
        .transpose()?;
    let mut leaders = vec![false; instructions.len()];
    if !leaders.is_empty() {
        leaders[0] = true;
    }
    if let Some(position) = entry_position {
        leaders[position] = true;
    }
    collect_flow_leaders(
        &instructions,
        &addresses,
        &transfers,
        options.calls,
        &mut leaders,
    );
    let handler_index = validate_handlers(handlers, code_end, &addresses, &mut leaders)?;
    let calls = match options.calls {
        CallPolicy::Edges => Vec::new(),
        CallPolicy::Flatten => collect_call_sites::<I>(&addresses, &transfers),
    };

    let Populated {
        mut cfg,
        instruction_blocks,
        blocks_by_instruction,
        terminators,
    } = populate_blocks(instructions, &addresses, &leaders);
    if let Some(position) = entry_position {
        cfg.set_entry(blocks_by_instruction[position]);
    }
    let unresolved_transfers = add_normal_edges(
        &mut cfg,
        &addresses,
        &transfers,
        &blocks_by_instruction,
        &terminators,
        options.calls,
        &mut edge_payload,
    );
    add_exception_edges(
        &mut cfg,
        handlers,
        &handler_index,
        &addresses,
        &blocks_by_instruction,
        &mut edge_payload,
    );
    let handler_refs = add_regions(&mut cfg, handlers, &handler_index, &blocks_by_instruction);

    Ok(AddressGraph {
        cfg,
        instruction_blocks,
        handler_refs,
        unresolved_transfers,
        calls,
    })
}

type AddressTransfers<I> = Vec<(
    usize,
    Flow<<I as AddressInstruction>::Address, <I as AddressInstruction>::CaseKey>,
)>;

type Inspected<I> = (
    Vec<<I as AddressInstruction>::Address>,
    AddressTransfers<I>,
    Option<<I as AddressInstruction>::Address>,
);

fn inspect_instructions<I: AddressInstruction>(
    instructions: &[I],
) -> Result<Inspected<I>, AddressBuildError<I::Address>> {
    let mut previous_address = None;
    let mut code_end = None;
    let mut addresses = Vec::with_capacity(instructions.len());
    let mut transfers = Vec::new();
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
        let position = addresses.len();
        let flow = instruction.flow();
        if flow.ends_block() {
            transfers.push((position, flow));
        }
        addresses.push(address);
    }
    Ok((addresses, transfers, code_end))
}

/// Whether `flow` ends its block under `policy`.
///
/// A flattened call is an operation of its block, so only its continuation
/// leaves it and it ends nothing.
fn ends_block<A, K>(flow: &Flow<A, K>, policy: CallPolicy) -> bool {
    flow.ends_block() && !(policy == CallPolicy::Flatten && flow.is_call())
}

fn collect_flow_leaders<I: AddressInstruction>(
    instructions: &[I],
    addresses: &[I::Address],
    transfers: &AddressTransfers<I>,
    policy: CallPolicy,
    leaders: &mut [bool],
) {
    for position in 0..addresses.len() {
        let end = instructions[position]
            .end_address()
            .expect("validated instruction end");
        let next = addresses.get(position + 1);
        let continues_to_next = next.is_some_and(|next| end == *next);
        if !continues_to_next {
            if next.is_some() {
                leaders[position + 1] = true;
            }
            mark_target(addresses, end, leaders);
        }
    }

    for &(position, ref flow) in transfers {
        if !ends_block(flow, policy) {
            // A flattened call is an operation of its block: neither its
            // callee nor its continuation leads anything.
            continue;
        }
        let end = instructions[position]
            .end_address()
            .expect("validated instruction end");
        for transfer in flow.transfers(addresses[position], Some(end)) {
            if let Some(target) = transfer.target {
                mark_target(addresses, target, leaders);
            }
        }
        if matches!(flow, Flow::Exceptional { .. }) {
            // The exceptional edge must be exact to this instruction: an IR
            // built on the graph gives it to one throwing statement.
            leaders[position] = true;
        }
        if position + 1 < addresses.len() {
            leaders[position + 1] = true;
        }
    }
}

/// Every call site in address order, for a graph that flattens calls.
fn collect_call_sites<I: AddressInstruction>(
    addresses: &[I::Address],
    transfers: &AddressTransfers<I>,
) -> Vec<CallSite<I::Address>> {
    transfers
        .iter()
        .filter(|(_, flow)| flow.is_call())
        .map(|&(position, ref flow)| CallSite {
            source: addresses[position],
            target: flow.direct_target(),
        })
        .collect()
}

fn mark_target<A: AddressSpace>(addresses: &[A], target: A, leaders: &mut [bool]) {
    if let Ok(position) = addresses.binary_search(&target) {
        leaders[position] = true;
    }
}

struct HandlerIndex {
    ranges: Vec<Range<usize>>,
    entries: Vec<usize>,
}

fn validate_handlers<A: AddressSpace>(
    handlers: &[AddressHandler<A>],
    code_end: Option<A>,
    addresses: &[A],
    leaders: &mut [bool],
) -> Result<HandlerIndex, AddressBuildError<A>> {
    let mut ranges = Vec::with_capacity(handlers.len());
    let mut entries = Vec::with_capacity(handlers.len());
    let mut range_starts = vec![0usize; addresses.len() + 1];
    let mut range_ends = vec![0usize; addresses.len() + 1];
    for handler in handlers {
        let (start, end) = (handler.protected.start, handler.protected.end);
        if end <= start {
            return Err(AddressBuildError::EmptyProtectedRange { start, end });
        }
        let start_position = addresses
            .binary_search(&start)
            .map_err(|_| AddressBuildError::MissingRangeStart { address: start })?;
        let at_code_end = code_end.is_some_and(|code_end| end == code_end);
        let end_position = if at_code_end {
            addresses.len()
        } else {
            addresses
                .binary_search(&end)
                .map_err(|_| AddressBuildError::MissingRangeEnd { address: end })?
        };
        let entry_position = addresses.binary_search(&handler.entry).map_err(|_| {
            AddressBuildError::MissingHandlerEntry {
                address: handler.entry,
            }
        })?;
        leaders[start_position] = true;
        if end_position < leaders.len() {
            leaders[end_position] = true;
        }
        leaders[entry_position] = true;
        range_starts[start_position] += 1;
        range_ends[end_position] += 1;
        ranges.push(start_position..end_position);
        entries.push(entry_position);
    }

    let mut active = 0usize;
    for position in 0..addresses.len() {
        active -= range_ends[position];
        active += range_starts[position];
        if active != 0 {
            leaders[position] = true;
        }
    }
    Ok(HandlerIndex { ranges, entries })
}

struct Populated<I: AddressInstruction, E> {
    cfg: Cfg<I, E>,
    instruction_blocks: BTreeMap<I::Address, BlockId>,
    blocks_by_instruction: Vec<BlockId>,
    terminators: Vec<(BlockId, usize)>,
}

fn populate_blocks<I: AddressInstruction, E>(
    instructions: Vec<I>,
    addresses: &[I::Address],
    leaders: &[bool],
) -> Populated<I, E> {
    let mut cfg = Cfg::with_edge_payload();
    let mut current = cfg.entry();
    let mut instruction_blocks = BTreeMap::new();
    let mut blocks_by_instruction = Vec::with_capacity(instructions.len());
    let mut terminators = Vec::new();
    for (position, instruction) in instructions.into_iter().enumerate() {
        let address = addresses[position];
        if position != 0 && leaders[position] {
            terminators.push((current, position - 1));
            current = cfg.new_block();
        }
        cfg.block_mut(current).push(instruction);
        instruction_blocks.insert(address, current);
        blocks_by_instruction.push(current);
    }
    if !addresses.is_empty() {
        terminators.push((current, addresses.len() - 1));
    }
    Populated {
        cfg,
        instruction_blocks,
        blocks_by_instruction,
        terminators,
    }
}

fn add_normal_edges<I: AddressInstruction, E, F>(
    cfg: &mut Cfg<I, E>,
    addresses: &[I::Address],
    transfers: &AddressTransfers<I>,
    blocks_by_instruction: &[BlockId],
    terminators: &[(BlockId, usize)],
    policy: CallPolicy,
    edge_payload: &mut F,
) -> Vec<UnresolvedTransfer<I::Address>>
where
    F: for<'k> FnMut(AddressEdgeInfo<'k, I::Address, I::CaseKey>) -> E,
{
    let mut emitter = AddressEdgeEmitter {
        cfg,
        addresses,
        blocks_by_instruction,
        policy,
        edge_payload,
        unresolved: Vec::new(),
    };
    for &(block, position) in terminators {
        let flow = transfers
            .binary_search_by_key(&position, |(position, _)| *position)
            .ok()
            .map(|transfer| &transfers[transfer].1);
        emitter.emit(block, position, flow);
    }
    emitter.unresolved
}

struct AddressEdgeEmitter<'a, I: AddressInstruction, E, F> {
    cfg: &'a mut Cfg<I, E>,
    addresses: &'a [I::Address],
    blocks_by_instruction: &'a [BlockId],
    policy: CallPolicy,
    edge_payload: &'a mut F,
    unresolved: Vec<UnresolvedTransfer<I::Address>>,
}

impl<I: AddressInstruction, E, F> AddressEdgeEmitter<'_, I, E, F>
where
    F: for<'k> FnMut(AddressEdgeInfo<'k, I::Address, I::CaseKey>) -> E,
{
    /// Adds the edges that leave `block`, whose last instruction sits at
    /// `position` and transfers by `flow`.
    ///
    /// A block whose last instruction has no transfer of its own ended at a
    /// leader, so it continues sequentially. The continuation is the
    /// instruction at the block's last end address, which can differ from
    /// the next instruction in the stream when paths overlap.
    fn emit(
        &mut self,
        block: BlockId,
        position: usize,
        flow: Option<&Flow<I::Address, I::CaseKey>>,
    ) {
        let source = self.addresses[position];
        let end = self
            .cfg
            .block(block)
            .instructions()
            .last()
            .and_then(AddressInstruction::end_address)
            .expect("validated block terminator end");
        let Some(flow) = flow.filter(|flow| ends_block(flow, self.policy)) else {
            self.add_transfer(
                block,
                source,
                Transfer {
                    target: Some(end),
                    role: EdgeRole::Sequential,
                },
            );
            return;
        };
        for transfer in flow.transfers(source, Some(end)) {
            self.add_transfer(block, source, transfer);
        }
    }

    fn add_transfer(
        &mut self,
        block: BlockId,
        source: I::Address,
        transfer: Transfer<'_, I::Address, I::CaseKey>,
    ) {
        let Some(target) = transfer.target else {
            if let Some(role) = transfer.unresolved_role() {
                self.unresolved.push(UnresolvedTransfer {
                    source,
                    target: None,
                    role,
                });
            }
            return;
        };
        let Ok(target_position) = self.addresses.binary_search(&target) else {
            if let Some(role) = transfer.unresolved_role() {
                self.unresolved.push(UnresolvedTransfer {
                    source,
                    target: Some(target),
                    role,
                });
            }
            return;
        };
        let kind = transfer.role.kind();
        let payload = (self.edge_payload)(AddressEdgeInfo {
            source,
            target,
            kind,
            role: transfer.role,
        });
        self.cfg.add_edge_with_payload(
            block,
            self.blocks_by_instruction[target_position],
            kind,
            payload,
        );
    }
}

fn add_exception_edges<I: AddressInstruction, E>(
    cfg: &mut Cfg<I, E>,
    handlers: &[AddressHandler<I::Address>],
    handler_index: &HandlerIndex,
    addresses: &[I::Address],
    blocks_by_instruction: &[BlockId],
    edge_payload: &mut impl FnMut(AddressEdgeInfo<'_, I::Address, I::CaseKey>) -> E,
) {
    for (index, handler) in handlers.iter().enumerate() {
        let target = blocks_by_instruction[handler_index.entries[index]];
        for position in handler_index.ranges[index].clone() {
            let source = blocks_by_instruction[position];
            let instruction = cfg
                .block(source)
                .instructions()
                .first()
                .expect("a protected instruction leads its block");
            debug_assert!(instruction.address() == addresses[position]);
            if instruction.retains_exception_edge() {
                let payload = edge_payload(AddressEdgeInfo {
                    source: addresses[position],
                    target: handler.entry,
                    kind: EdgeKind::ExceptionUnwind,
                    role: EdgeRole::Unwind { handler: index },
                });
                cfg.add_edge_with_payload(source, target, EdgeKind::ExceptionUnwind, payload);
            }
        }
    }
}

fn add_regions<I: AddressInstruction, E>(
    cfg: &mut Cfg<I, E>,
    handlers: &[AddressHandler<I::Address>],
    handler_index: &HandlerIndex,
    blocks_by_instruction: &[BlockId],
) -> Vec<HandlerRef> {
    struct Draft<A> {
        protected: Range<A>,
        positions: Range<usize>,
        handler_indices: Vec<usize>,
    }

    let mut drafts: Vec<Draft<I::Address>> = Vec::new();
    let mut draft_index: BTreeMap<(I::Address, I::Address), usize> = BTreeMap::new();
    for (index, handler) in handlers.iter().enumerate() {
        let key = (handler.protected.start, handler.protected.end);
        if let Some(&position) = draft_index.get(&key) {
            drafts[position].handler_indices.push(index);
        } else {
            draft_index.insert(key, drafts.len());
            drafts.push(Draft {
                protected: handler.protected.clone(),
                positions: handler_index.ranges[index].clone(),
                handler_indices: vec![index],
            });
        }
    }
    drop(draft_index);

    // Innermost-region resolution walks reverse insertion order, so
    // enclosing ranges must be registered before nested ranges. Stable
    // sorting preserves table order for peers of equal span.
    let span = |range: &Range<I::Address>| -> u64 { range.end.distance_from(range.start) };
    drafts.sort_by_key(|draft| Reverse(span(&draft.protected)));

    let mut handler_refs = vec![None; handlers.len()];
    let mut region_ids: Vec<RegionId> = Vec::with_capacity(drafts.len());
    for (index, draft) in drafts.iter().enumerate() {
        let protected_blocks = draft
            .positions
            .clone()
            .map(|position| blocks_by_instruction[position])
            .collect();
        let region_handlers = draft
            .handler_indices
            .iter()
            .map(|&handler_position| {
                let handler = &handlers[handler_position];
                Handler {
                    entry: blocks_by_instruction[handler_index.entries[handler_position]],
                    body: HandlerBody::unknown(),
                    kind: handler.kind,
                }
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
