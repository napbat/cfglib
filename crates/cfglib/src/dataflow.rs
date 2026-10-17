//! Data flow analysis framework.
//!
//! Provides generic infrastructure for computing data flow properties
//! over any [`CfgView`]:
//!
//! - **Reaching definitions** — which writes can reach a given point
//! - **Liveness** — which variables are live at each point
//! - **Def-use / use-def chains** — linking writers to readers
//!
//! [`CfgView`] is the one read-only trait that the
//! instruction-level machinery reads, and [`Cfg`](crate::Cfg) implements it.
//! The [`node_fixpoint`] module is the graph-shaped counterpart, solving
//! per-node fact problems over any
//! [`GraphView`](crate::GraphView) (taint or reachability
//! over a value-flow graph, closure over an import graph).
//!
//! # Usage
//!
//! Implement [`InstrInfo`] for your instruction type to declare which
//! IR variables each instruction reads from and writes to, then run any
//! of the provided analyses. The [`MemoryEventInfo`](crate::MemoryEventInfo)
//! extension adds ordered memory data flow and an instruction-level
//! read/modify/write summary. Variables and memory locations are supplied by
//! the IR adapter; the framework does not impose a numbering scheme.

pub mod abstract_interpretation;
pub mod bits;
pub mod cfg_view;
pub mod constant_propagation;
pub mod copy_propagation;
pub mod def_use;
pub mod edge_fixpoint;
pub mod fixpoint;
pub mod liveness;
pub mod memory;
pub mod node_fixpoint;
pub mod phi_web;
pub mod reaching;
pub mod sccp;
pub mod ssa;
pub mod ssa_destruction;
mod worklist;

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::edge::{EdgeKind, KindedEdge};
use crate::graph::edge_view::{EdgeRef, FilteredEdges};
use crate::graph::traverse::{TraversalDirection, reachable};

use self::cfg_view::CfgView;

/// An IR-defined identity that participates in data-flow analysis.
///
/// This blanket trait accepts any ordered, cloneable type. An adapter can
/// therefore use a compact integer, an x86 register/flag enum, a shader
/// register-component structure, or another native identity directly.
///
/// The identity must describe the atomic unit tracked by the analyses. For
/// architectures with overlapping resources, such as x86 subregisters, the
/// adapter is responsible for exposing canonical units or all affected units.
pub trait VariableId: Clone + Ord {}

impl<T: Clone + Ord> VariableId for T {}

/// Trait that an instruction type implements to expose its data
/// dependencies (which variables it reads and writes).
///
/// This is the data-flow counterpart of [`FlowControl`](crate::FlowControl),
/// which classifies control-flow effects.
///
/// Returns slices rather than `Vec`s to avoid heap allocations in
/// hot-path fixpoint iteration. The instruction is expected to own (or
/// borrow from its own storage) the def/use sets it reports; adapters over
/// external structures typically materialize them once at construction.
/// Memory accesses are reported separately through the
/// [`MemoryEventInfo`](crate::MemoryEventInfo) extension so adapters without a
/// memory model keep this base contract minimal.
pub trait InstrInfo {
    /// IR variable identity used by this instruction representation.
    type Variable: VariableId;

    /// Variables that this instruction **reads** (uses).
    fn uses(&self) -> &[Self::Variable];

    /// Variables that this instruction **writes** (defines).
    fn defs(&self) -> &[Self::Variable];

    /// Whether execution can unwind before this instruction defines its outputs.
    ///
    /// An [`ExceptionUnwind`](crate::EdgeKind::ExceptionUnwind) successor
    /// observes the values before this instruction, so its own definitions
    /// never reach the handler. Normal successors observe its completed
    /// definitions. Adapters with exceptional control flow must report every
    /// possible unwind point: a block with an unwind successor and no such
    /// instruction is treated as unwinding after its last instruction.
    ///
    /// The analyses read only this method. The
    /// [`FlowEffect::MayThrow`](crate::FlowEffect::MayThrow) classification
    /// the CFG builder sees is separate, so an adapter that classifies a
    /// throwing instruction reports it here as well.
    fn may_unwind(&self) -> bool {
        false
    }
}

/// Opt-in: instructions whose execution is predicated on a condition
/// variable (ARM IT blocks, GPU wave predication, x86 CMOV, guarded
/// statements).
///
/// The predicate is expressed in the consumer's own variable domain.
///
/// # Contract
///
/// The predicate variable is a READ: it **must also appear in
/// [`uses`](InstrInfo::uses)**. The dataflow analyses consult only
/// `uses`/`defs` — an adapter that reports the guard here but omits it
/// from `uses` will see liveness call the predicate's definition dead and
/// dead-code elimination delete it while the guard still names it.
pub trait Predicated: InstrInfo {
    /// The guarding variable and the polarity under which the instruction
    /// executes. `None` for unconditionally executed instructions.
    fn predicate(&self) -> Option<(Self::Variable, bool)>;
}

/// Opt-in: instructions that declare side effects beyond their def/use sets.
///
/// The effect vocabulary is consumer-typed — machine memory and I/O for a
/// binary adapter, GC allocation / panics / channel sends for a source
/// language — the library imposes none. Purity analysis and dead-code
/// elimination consume this trait; requiring it there means a consumer must
/// state an effect vocabulary before code with possible side effects can be
/// classified or deleted.
pub trait EffectInfo: InstrInfo {
    /// Consumer effect vocabulary.
    type Effect: Clone + Ord;

    /// Side effects of this instruction. An empty slice means the
    /// instruction is pure.
    fn effects(&self) -> &[Self::Effect];
}

/// A point in the program identified by a (block, instruction-index) pair.
///
/// Used to refer to both definition sites and use sites in data flow
/// analyses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgramPoint {
    /// The block containing the instruction.
    pub block: BlockId,
    /// Index of the instruction within the block.
    pub inst_idx: usize,
}

impl core::fmt::Display for ProgramPoint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}:{}", self.block, self.inst_idx)
    }
}

/// Alias for [`ProgramPoint`] used in definition contexts.
pub type DefSite = ProgramPoint;
/// Alias for [`ProgramPoint`] used in use-site contexts.
pub type UseSite = ProgramPoint;

/// Whether `edge` is an [`ExceptionUnwind`](crate::EdgeKind::ExceptionUnwind)
/// edge.
///
/// This is the one place the unwind kind is read off an edge. Whether such an
/// edge leaves before a throwing instruction also depends on the source
/// block's instructions; see [`departs_before_throws`].
pub(crate) fn is_unwind_edge<N: Copy, E: Copy, D: KindedEdge + ?Sized>(
    edge: EdgeRef<'_, N, E, D>,
) -> bool {
    edge.data().kind() == EdgeKind::ExceptionUnwind
}

/// The edges out of `block` that are [unwind edges](is_unwind_edge).
pub(crate) fn unwind_edges<V: CfgView>(
    view: &V,
    block: BlockId,
) -> impl Iterator<Item = EdgeRef<'_, BlockId, V::EdgeId, V::EdgeData>> + '_ {
    view.outgoing(block)
        .map(|edge| view.edge(edge))
        .filter(|&edge| is_unwind_edge(edge))
}

/// Whether `edge` leaves its source before a throwing instruction.
///
/// An [`ExceptionUnwind`](crate::EdgeKind::ExceptionUnwind) edge departs
/// immediately before every source instruction that
/// [`may_unwind`](InstrInfo::may_unwind), so it carries the state before that
/// instruction rather than the block's completed state. Every other edge, and
/// an unwind out of a block that reports no throwing instruction, departs
/// after the block's last instruction.
///
/// The rule is the same for memory: [`MemorySSA`](crate::MemorySSA) carries
/// the memory state before the throwing instruction, so the write of a
/// faulting instruction never reaches the handler. An adapter models a write
/// that commits before an unwind as an instruction of its own, ahead of the
/// throwing one.
#[must_use]
pub fn departs_before_throws<V>(
    view: &V,
    edge: EdgeRef<'_, BlockId, V::EdgeId, V::EdgeData>,
) -> bool
where
    V: CfgView,
    V::Inst: InstrInfo,
{
    is_unwind_edge(edge)
        && view
            .instructions(edge.source())
            .iter()
            .any(InstrInfo::may_unwind)
}

/// The first instruction of `block` before which an unwind can leave it.
///
/// `None` when no edge out of `block` [departs before a throwing
/// instruction](departs_before_throws). Every definition at or after the
/// answer may be missing on the unwind path.
pub(crate) fn first_unwind_point<V>(view: &V, block: BlockId) -> Option<usize>
where
    V: CfgView,
    V::Inst: InstrInfo,
{
    unwind_edges(view, block).next()?;
    view.instructions(block)
        .iter()
        .position(InstrInfo::may_unwind)
}

/// Per block, whether an unwind out of `block` reaches it without entering
/// `block` again.
///
/// Such a block can be entered after `block` left before a throwing
/// instruction, so it must not rely on anything `block` does from its
/// [first unwind point](first_unwind_point) on. A path that re-enters
/// `block` and unwinds again returns to the same successors, so excluding
/// `block` loses no such path.
pub(crate) fn unwind_reach<V: CfgView>(view: &V, block: BlockId) -> Vec<bool> {
    let without_block = FilteredEdges::new(view, |edge: V::EdgeId, _: &V::EdgeData| {
        view.edge(edge).target() != block
    });
    let unwind_successors = unwind_edges(view, block)
        .map(|edge| edge.target())
        .filter(|&target| target != block);
    reachable(
        &without_block,
        unwind_successors,
        TraversalDirection::Outgoing,
    )
}

/// Replays `block` from the state `input`, applying `step` to each
/// instruction in program order.
///
/// `before_unwind` sees the state before every instruction that
/// [may unwind](InstrInfo::may_unwind), which is the state an unwind leaving
/// there carries. Returns the state after the last instruction.
pub(crate) fn replay_block<G, S>(
    cfg: &G,
    block: BlockId,
    input: &S,
    mut step: impl FnMut(&mut S, usize, &G::Inst),
    mut before_unwind: impl FnMut(&S),
) -> S
where
    G: CfgView,
    G::Inst: InstrInfo,
    S: Clone,
{
    let mut state = input.clone();
    for (index, instruction) in cfg.instructions(block).iter().enumerate() {
        if instruction.may_unwind() {
            before_unwind(&state);
        }
        step(&mut state, index, instruction);
    }
    state
}

/// The state `edge` carries out of its source when it [departs before
/// throws](departs_before_throws): the `join` of the states `step` has
/// produced before each throwing instruction, replaying the source from
/// `input`. `None` when the edge departs after the block, which carries the
/// block's completed state instead.
pub(crate) fn unwind_departure<G, S>(
    cfg: &G,
    edge: EdgeRef<'_, BlockId, G::EdgeId, G::EdgeData>,
    input: &S,
    step: impl FnMut(&mut S, usize, &G::Inst),
    join: impl Fn(&S, &S) -> S,
) -> Option<S>
where
    G: CfgView,
    G::Inst: InstrInfo,
    S: Clone,
{
    if !departs_before_throws(cfg, edge) {
        return None;
    }
    let mut departing: Option<S> = None;
    replay_block(cfg, edge.source(), input, step, |state| {
        departing = Some(match departing.take() {
            None => state.clone(),
            Some(earlier) => join(&earlier, state),
        });
    });
    departing
}
