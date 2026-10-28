//! Build harness: assembles a toy program, builds its address CFG with a
//! consumer edge payload, and runs the structural oracles over every result.

use core::fmt::Debug;
use core::ops::Range;

use cfglib::{
    AddressBuildError, AddressCfgOptions, AddressEdgeInfo, AddressGraph, AddressHandler,
    AddressInstruction, BlockId, CallPolicy, Cfg, EdgeKind, EdgeRole, HandlerKind,
    build_address_cfg,
};

use crate::isa::{Isa, Layout, Op};
use crate::oracles;

pub const FALL: EdgeKind = EdgeKind::Fallthrough;
pub const TAKEN: EdgeKind = EdgeKind::ConditionalTrue;
pub const NOT_TAKEN: EdgeKind = EdgeKind::ConditionalFalse;
pub const JUMP: EdgeKind = EdgeKind::Jump;
pub const CASE: EdgeKind = EdgeKind::SwitchCase;
pub const CALL: EdgeKind = EdgeKind::Call;
pub const RETURNS: EdgeKind = EdgeKind::CallReturn;
pub const UNWIND: EdgeKind = EdgeKind::ExceptionUnwind;

/// Asserts that `items` is empty, printing what was found when it is not.
pub fn assert_empty<T: PartialEq + Debug>(items: &[T]) {
    assert_eq!(items, <&[T]>::default());
}

/// The consumer's own spelling of why an edge exists. It carries no lifetime,
/// so it can live in the edge payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Sequential,
    Taken,
    FallThrough,
    Jump,
    Default,
    Case { index: usize, key: i64 },
    Call,
    Continuation { call_site: u64 },
    Exceptional,
    Unwind { handler: usize },
    Route { route: usize },
}

/// The edge payload a dasm-shaped consumer builds from each
/// [`AddressEdgeInfo`], with addresses widened to `u64` so one payload type
/// serves every address space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wire {
    pub source: u64,
    pub target: u64,
    pub kind: EdgeKind,
    pub role: Role,
}

pub fn wire<A: Copy + Into<u64>, K: Copy + Into<i64>>(info: AddressEdgeInfo<'_, A, K>) -> Wire {
    let role = match info.role {
        EdgeRole::Sequential => Role::Sequential,
        EdgeRole::ConditionalTaken => Role::Taken,
        EdgeRole::ConditionalFallThrough => Role::FallThrough,
        EdgeRole::Jump => Role::Jump,
        EdgeRole::SwitchDefault => Role::Default,
        EdgeRole::SwitchCase { index, key } => Role::Case {
            index,
            key: (*key).into(),
        },
        EdgeRole::Call => Role::Call,
        EdgeRole::CallContinuation { call_site } => Role::Continuation {
            call_site: call_site.into(),
        },
        EdgeRole::Exceptional => Role::Exceptional,
        EdgeRole::Unwind { handler } => Role::Unwind { handler },
        EdgeRole::Route { route } => Role::Route { route },
    };
    Wire {
        source: info.source.into(),
        target: info.target.into(),
        kind: info.kind,
        role,
    }
}

/// One exception-table entry, spelled in program positions. `protected.end`
/// may equal the program length, which spells the end of the code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TryRange {
    pub protected: Range<usize>,
    pub entry: usize,
    pub kind: HandlerKind,
}

pub fn catch(protected: Range<usize>, entry: usize) -> TryRange {
    TryRange {
        protected,
        entry,
        kind: HandlerKind::Catch,
    }
}

/// The options of one build, with the entry spelled as a position.
#[derive(Debug, Clone, Copy, Default)]
pub struct Settings {
    pub entry: Option<usize>,
    pub calls: CallPolicy,
}

/// A built program, its layout, and the inputs that produced it.
pub struct Built<I: Isa> {
    pub layout: Layout<I>,
    pub table: Vec<TryRange>,
    pub settings: Settings,
    pub graph: AddressGraph<I::Insn, Wire>,
    /// How many times the edge-payload callback ran.
    pub payload_calls: usize,
}

/// Builds `ops` with default settings and no exception table.
pub fn build<I: Isa>(ops: &[Op]) -> Built<I> {
    build_with(ops, Vec::new(), Settings::default())
}

/// The exception table of `table` in `layout`'s address space.
pub fn handlers_of<I: Isa>(layout: &Layout<I>, table: &[TryRange]) -> Vec<AddressHandler<I::Addr>> {
    table
        .iter()
        .map(|entry| AddressHandler {
            protected: layout.address(entry.protected.start)..layout.address(entry.protected.end),
            entry: layout.address(entry.entry),
            kind: entry.kind,
        })
        .collect()
}

/// Options that enter at the first instruction and keep calls as edges,
/// spelled out because `AddressCfgOptions::default` needs `A: Default`.
pub const fn plain_options<A>() -> AddressCfgOptions<A> {
    AddressCfgOptions {
        entry: None,
        calls: CallPolicy::Edges,
    }
}

/// Builds `ops` and checks the result against every structural oracle.
///
/// # Panics
///
/// Panics when the build fails or an oracle rejects the graph.
pub fn build_with<I: Isa>(ops: &[Op], table: Vec<TryRange>, settings: Settings) -> Built<I> {
    let layout = Layout::<I>::assemble(ops);
    let handlers = handlers_of(&layout, &table);
    let options = AddressCfgOptions {
        entry: settings.entry.map(|position| layout.address(position)),
        calls: settings.calls,
    };
    let mut payload_calls = 0;
    let graph = build_address_cfg(layout.instructions().to_vec(), &handlers, options, |info| {
        payload_calls += 1;
        wire(info)
    })
    .expect("the toy program is a valid stream");
    let built = Built {
        layout,
        table,
        settings,
        graph,
        payload_calls,
    };
    oracles::check(&built);
    built
}

/// Builds raw instructions with the same payload and no oracle, for inputs
/// the position-indexed assembler cannot spell (gaps, overlaps, bad tables).
///
/// # Errors
///
/// Returns the builder's [`AddressBuildError`].
pub fn try_build<T>(
    instructions: Vec<T>,
    handlers: &[AddressHandler<T::Address>],
    options: AddressCfgOptions<T::Address>,
) -> Result<AddressGraph<T, Wire>, AddressBuildError<T::Address>>
where
    T: AddressInstruction,
    T::Address: Into<u64>,
    T::CaseKey: Copy + Into<i64>,
{
    build_address_cfg(instructions, handlers, options, wire)
}

impl<I: Isa> Built<I> {
    pub fn cfg(&self) -> &Cfg<I::Insn, Wire> {
        &self.graph.cfg
    }

    /// The block holding the instruction at `position`.
    pub fn block_of(&self, position: usize) -> BlockId {
        self.graph.instruction_blocks[&self.layout.address(position)]
    }

    /// The positions of every block's instructions, in block order.
    pub fn blocks(&self) -> Vec<Vec<usize>> {
        self.cfg()
            .block_ids()
            .map(|block| {
                self.cfg()
                    .block(block)
                    .instructions()
                    .iter()
                    .map(|instruction| self.position_of(instruction.address()))
                    .collect()
            })
            .collect()
    }

    /// Every block's outgoing edges as `(target block index, kind)`, in the
    /// order the block lists them.
    pub fn successors(&self) -> Vec<Vec<(usize, EdgeKind)>> {
        self.cfg()
            .block_ids()
            .map(|block| {
                self.cfg()
                    .outgoing(block)
                    .map(|edge| {
                        let edge = self.cfg().edge(edge);
                        (edge.target().index(), edge.kind())
                    })
                    .collect()
            })
            .collect()
    }

    /// Every block's outgoing edge roles, in the order the block lists them.
    pub fn roles(&self) -> Vec<Vec<Role>> {
        self.cfg()
            .block_ids()
            .map(|block| {
                self.cfg()
                    .outgoing(block)
                    .map(|edge| self.cfg().edge(edge).payload().role)
                    .collect()
            })
            .collect()
    }

    /// The program position of the instruction that starts at `address`.
    pub fn position_of(&self, address: I::Addr) -> usize {
        self.layout
            .position_of(address)
            .expect("the address starts an instruction")
    }

    /// `address(position)` widened to the payload's `u64`.
    pub fn raw(&self, position: usize) -> u64 {
        self.layout.address(position).into()
    }
}
