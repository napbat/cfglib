//! Back edges and reducibility.
//!
//! [`find_back_edges`] lists the edges that close a loop: every edge whose
//! target dominates its source, and every edge a builder tagged
//! [`EdgeKind::Back`] (a structured `loop` or `continue`) that closes a cycle
//! dominance cannot prove, as on an irreducible machine CFG.
//! [`LoopForest::compute`](crate::LoopForest::compute) builds the loops from
//! them. [`is_reducible`] says whether every cycle has a block that dominates
//! the rest of it.
//!
//! The detectors are generic: any [`EdgeView`] whose edge data declares a
//! [`KindedEdge`] kind participates, so they are not a copy written against
//! [`Cfg`](crate::Cfg). The `()` edge data of a plain graph declares no tag,
//! which leaves dominance alone to find its loops.

extern crate alloc;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use super::dominator::DominatorTree;
use super::edge_view::EdgeView;
use super::scc::{SccDecomposition, SccNumbering};
use super::search::{DfsEvent, depth_first_events};
use super::view::{Direction, RootedView};
use crate::block::BlockId;
use crate::edge::{EdgeKind, KindedEdge};

/// A back edge: an edge from `tail` to `header` that closes a loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BackEdge<N = BlockId> {
    /// The node at the tail of the back edge (the source of the jump).
    pub tail: N,
    /// The loop header node (the target of the back edge).
    pub header: N,
}

/// Find the back edges of `graph`, in ascending order of tail, then header.
///
/// A back edge is an edge whose target dominates its source, or an edge
/// tagged [`EdgeKind::Back`] whose ends lie on one cycle. The tags matter on
/// irreducible machine CFGs, where a frontend knows an edge is a loop back
/// edge even though dominance cannot prove it.
///
/// Parallel edges between the same pair appear once. Only code the root
/// reaches has back edges: a block dominates itself whether or not it runs, so
/// unreachable code is skipped rather than reported as loops, and so is a tag
/// on an edge out of it. A tag on an edge that lies on no cycle is ignored
/// too: a header that cannot reach the tail has no loop around it, whatever
/// the edge is called.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, DominatorTree, EdgeKind, find_back_edges};
///
/// let mut cfg = Cfg::<u32>::new();
/// let b0 = cfg.entry();
/// let b1 = cfg.new_block();
/// cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
/// cfg.add_edge(b1, b0, EdgeKind::Back);
///
/// let dom = DominatorTree::compute(&cfg);
/// let backs = find_back_edges(&cfg, &dom);
/// assert_eq!(backs.len(), 1);
/// assert_eq!(backs[0].header, b0);
/// assert_eq!(backs[0].tail, b1);
/// ```
#[must_use]
pub fn find_back_edges<G>(graph: &G, dom: &DominatorTree<G::NodeId>) -> Vec<BackEdge<G::NodeId>>
where
    G: EdgeView,
    G::EdgeData: KindedEdge,
{
    let mut backs = Vec::new();
    for node in graph.node_ids().filter(|&node| dom.is_reachable(node)) {
        for successor in graph.successors(node) {
            if dom.dominates(successor, node) {
                backs.push(BackEdge {
                    tail: node,
                    header: successor,
                });
            }
        }
    }
    backs.sort_unstable();
    backs.dedup();

    let dominance = backs.len();
    let mut components = None;
    for edge in graph.edge_ids() {
        let edge = graph.edge(edge);
        if edge.data().kind() != EdgeKind::Back || !dom.is_reachable(edge.source()) {
            continue;
        }
        let tagged = BackEdge {
            tail: edge.source(),
            header: edge.target(),
        };
        // A dominance back edge closes a cycle by definition. A tag must be
        // on one too, which holds when both ends are in one strongly
        // connected component.
        let on_a_cycle = backs[..dominance].binary_search(&tagged).is_ok() || {
            let components = components
                .get_or_insert_with(|| SccDecomposition::compute(graph, SccNumbering::LeavesFirst));
            components.component_index(tagged.tail) == components.component_index(tagged.header)
        };
        if on_a_cycle {
            backs.push(tagged);
        }
    }
    if backs.len() > dominance {
        backs.sort_unstable();
        backs.dedup();
    }
    backs
}

/// Whether the graph is reducible.
///
/// A graph is **reducible** if and only if every cycle contains a node
/// that dominates all other nodes in that cycle. Equivalently, every
/// retreating edge of a depth-first walk is a back edge (its target dominates
/// its source).
///
/// The walk runs from the view's root; any [`DfsEvent::BackEdge`] whose
/// target does not dominate its source witnesses an irreducible cycle.
/// [`EdgeKind::Back`] tags play no part.
#[must_use]
pub fn is_reducible<G: RootedView>(graph: &G, dom: &DominatorTree<G::NodeId>) -> bool {
    graph.node_bound() == 0
        || depth_first_events(
            graph,
            graph.root(),
            Direction::Forward,
            |event| match event {
                DfsEvent::BackEdge(tail, header) if !dom.dominates(header, tail) => {
                    ControlFlow::Break(())
                }
                _ => ControlFlow::Continue(()),
            },
        )
        .is_none()
}

#[cfg(test)]
mod tests;
