//! The unwind departure rule: which edges leave a block before a throwing
//! instruction, and what they carry.

extern crate alloc;
use alloc::vec::Vec;

use super::cfg_view::CfgView;
use super::instruction::InstrInfo;
use crate::block::BlockId;
use crate::edge::{EdgeKind, KindedEdge};
use crate::graph::edge_view::{EdgeRef, FilteredEdges};
use crate::graph::traverse::{TraversalDirection, reachable};

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
/// The rule is the same for memory: [`MemorySsa`](crate::MemorySsa) carries
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
