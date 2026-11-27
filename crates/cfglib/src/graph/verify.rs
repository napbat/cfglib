//! Structural verification of a graph view.
//!
//! One definition of the invariants serves every graph: [`verify_view`]
//! checks the node-level contract of any rooted view, [`verify_edge_view`]
//! adds the stable-edge contract on top of it, and
//! [`verify`](crate::verify) is the latter over a [`Cfg`](crate::Cfg). The
//! invariants are an entry that exists, in-bounds identities, mirrored forward
//! and reverse adjacency, and a predecessor for every reachable non-entry
//! node.

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;
use core::convert::Infallible;

use crate::graph::edge_traverse::EdgeAdjacency;
use crate::graph::edge_view::EdgeView;
use crate::graph::traverse::{Incoming, Outgoing, reachable};
use crate::graph::view::Direction;
use crate::graph::view::{DenseId, RootedView};

/// A single verification failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyError {
    /// Human-readable description of the violated invariant.
    pub message: String,
}

/// The message alone: a view of any kind fails verification, not only a CFG.
impl core::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.message)
    }
}

impl core::error::Error for VerifyError {}

/// What a verification found: the structural violations, and the typed errors
/// of a consumer's semantic checks.
///
/// [`verify_view`], [`verify_edge_view`] and [`verify`](crate::verify) check
/// structure only, so their report keeps the default `E = Infallible` and
/// holds no semantic error. [`verify_semantics`](crate::verify_semantics)
/// adds the errors of a [`SemanticValidator`](crate::SemanticValidator).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyReport<E = Infallible> {
    /// Every structural violation, in the order found (empty = valid).
    pub errors: Vec<VerifyError>,
    /// The typed errors the consumer's validator emitted, in hook order.
    pub semantic_errors: Vec<E>,
}

impl<E> VerifyReport<E> {
    /// Whether neither the structural nor the semantic checks found an error.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty() && self.semantic_errors.is_empty()
    }

    /// The number of structural plus semantic errors.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors.len() + self.semantic_errors.len()
    }
}

/// A `(source, target)` adjacency entry as plain indices.
type Pair = (usize, usize);

/// Walks the distinct entries of the sorted slice `left` in order, with each
/// entry's multiplicity in `left` and in the sorted slice `right`.
fn multiplicities<'a>(
    left: &'a [Pair],
    right: &'a [Pair],
) -> impl Iterator<Item = (Pair, usize, usize)> + 'a {
    let mut right_runs = right.chunk_by(|a, b| a == b).peekable();
    left.chunk_by(|a, b| a == b).map(move |run| {
        let pair = run[0];
        while right_runs.next_if(|other| other[0] < pair).is_some() {}
        let mirrored = right_runs
            .next_if(|other| other[0] == pair)
            .map_or(0, <[Pair]>::len);
        (pair, run.len(), mirrored)
    })
}

/// Reports every entry whose multiplicity differs between the sorted
/// `forward` and `reverse` lists: first those present forward, then those
/// present only in reverse.
fn report_adjacency_mismatches(forward: &[Pair], reverse: &[Pair], errors: &mut Vec<VerifyError>) {
    for ((source, target), count, mirrored) in multiplicities(forward, reverse) {
        if mirrored != count {
            errors.push(VerifyError {
                message: alloc::format!(
                    "adjacency mismatch: {count} successor edge(s) {source}->{target} but {mirrored} predecessor entry(ies)"
                ),
            });
        }
    }
    for ((source, target), count, mirrored) in multiplicities(reverse, forward) {
        if mirrored == 0 {
            errors.push(VerifyError {
                message: alloc::format!(
                    "adjacency mismatch: {count} predecessor entry(ies) {source}->{target} with no successor edge"
                ),
            });
        }
    }
}

fn report_out_of_bounds_neighbor(
    errors: &mut Vec<VerifyError>,
    node: usize,
    role: &str,
    neighbor: usize,
    node_bound: usize,
) {
    errors.push(VerifyError {
        message: alloc::format!(
            "node {node} has {role} {neighbor} out of bounds (node_bound={node_bound})"
        ),
    });
}

/// Validate the [`RootedView`] contract on consumer-owned storage.
///
/// Checks performed:
/// 1. The root node's index is within `0..node_bound()`.
/// 2. Every node, successor, and predecessor the view yields is within
///    `0..node_bound()`. An identity beyond the bound is reported where it
///    appears and left out of the checks below, so an adapter that breaks the
///    contract gets a report instead of aborting the validator.
/// 3. Forward and reverse adjacency mirror each other with matching
///    multiplicity (every successor entry has a matching predecessor entry).
/// 4. Every reachable non-root node has at least one predecessor. This walk
///    runs only when check 2 found nothing, because it indexes tables of
///    `node_bound()` entries by those identities.
///
/// This catches adapter bugs when a consumer implements the view traits over
/// its own graph store — the counterpart of [`verify`](crate::verify), which
/// checks the storage invariants of [`Cfg`](crate::Cfg) itself.
#[must_use]
pub fn verify_view<G: RootedView>(graph: &G) -> VerifyReport {
    let mut errors = Vec::new();
    let node_bound = graph.node_bound();

    let root = graph.root();
    if root.index() >= node_bound {
        errors.push(VerifyError {
            message: alloc::format!(
                "root node index {} out of bounds (node_bound={node_bound})",
                root.index()
            ),
        });
        return VerifyReport {
            errors,
            semantic_errors: Vec::new(),
        };
    }

    // Forward and reverse adjacency must agree as multisets of (source, target).
    let mut forward: Vec<Pair> = Vec::with_capacity(node_bound);
    let mut reverse: Vec<Pair> = Vec::with_capacity(node_bound);
    let mut in_bounds = true;
    for node in graph.node_ids() {
        if node.index() >= node_bound {
            errors.push(VerifyError {
                message: alloc::format!(
                    "node index {} out of bounds (node_bound={node_bound})",
                    node.index()
                ),
            });
            in_bounds = false;
            continue;
        }
        for successor in graph.successors(node) {
            if successor.index() < node_bound {
                forward.push((node.index(), successor.index()));
            } else {
                report_out_of_bounds_neighbor(
                    &mut errors,
                    node.index(),
                    "successor",
                    successor.index(),
                    node_bound,
                );
                in_bounds = false;
            }
        }
        for predecessor in graph.predecessors(node) {
            if predecessor.index() < node_bound {
                reverse.push((predecessor.index(), node.index()));
            } else {
                report_out_of_bounds_neighbor(
                    &mut errors,
                    node.index(),
                    "predecessor",
                    predecessor.index(),
                    node_bound,
                );
                in_bounds = false;
            }
        }
    }
    forward.sort_unstable();
    reverse.sort_unstable();
    if forward != reverse {
        report_adjacency_mismatches(&forward, &reverse, &mut errors);
    }

    // Every reachable non-root node has a predecessor.
    if in_bounds {
        let reached = reachable(graph, [root], Direction::Forward);
        for node in graph.node_ids() {
            if reached[node.index()] && node != root && graph.predecessors(node).next().is_none() {
                errors.push(VerifyError {
                    message: alloc::format!("reachable node {} has no predecessors", node.index()),
                });
            }
        }
    }

    VerifyReport {
        errors,
        semantic_errors: Vec::new(),
    }
}

/// What [`verify_edge_view`] has seen of one edge slot.
#[derive(Clone, Copy, Default)]
struct EdgeSlot {
    live: bool,
    outgoing: usize,
    incoming: usize,
}

/// How one adjacency direction is named in a report and counted in an
/// [`EdgeSlot`].
struct Listing {
    /// `"outgoing"` or `"incoming"`.
    adjacency: &'static str,
    /// The end of an edge that a listing node must be: `"source"` or `"target"`.
    end: &'static str,
    count: fn(&mut EdgeSlot) -> &mut usize,
}

const OUTGOING: Listing = Listing {
    adjacency: "outgoing",
    end: "source",
    count: |slot| &mut slot.outgoing,
};

const INCOMING: Listing = Listing {
    adjacency: "incoming",
    end: "target",
    count: |slot| &mut slot.incoming,
};

/// Check the edges `node` lists along `axis`: each must be a live slot, and
/// must name `node` as its [`Listing::end`].
fn check_listings<G: EdgeView, A: EdgeAdjacency>(
    axis: A,
    listing: &Listing,
    graph: &G,
    node: G::NodeId,
    slots: &mut [EdgeSlot],
    errors: &mut Vec<VerifyError>,
) {
    for edge_id in axis.edges(graph, node) {
        let index = edge_id.index();
        let Some(slot) = slots.get_mut(index).filter(|slot| slot.live) else {
            errors.push(VerifyError {
                message: alloc::format!(
                    "node {} has non-live {} edge index {index}",
                    node.index(),
                    listing.adjacency
                ),
            });
            continue;
        };
        *(listing.count)(slot) += 1;
        let stored = axis.previous(graph.edge(edge_id));
        if stored != node {
            errors.push(VerifyError {
                message: alloc::format!(
                    "node {} lists edge index {index}, whose {} is {}",
                    node.index(),
                    listing.end,
                    stored.index()
                ),
            });
        }
    }
}

/// Validate the node and stable-edge contracts of an edge-aware rooted view.
///
/// In addition to [`verify_view`], this checks dense edge-slot bounds, unique
/// live identities, endpoint bounds, exact one-time membership in source and
/// target adjacency, and adjacency endpoint orientation. Parallel edges are
/// valid because their identities remain distinct.
#[must_use]
pub fn verify_edge_view<G>(graph: &G) -> VerifyReport
where
    G: EdgeView + RootedView,
{
    let mut result = verify_view(graph);
    let node_bound = graph.node_bound();
    let edge_bound = graph.edge_bound();
    let mut slots = alloc::vec![EdgeSlot::default(); edge_bound];
    for edge_id in graph.edge_ids() {
        let index = edge_id.index();
        if index >= edge_bound {
            result.errors.push(VerifyError {
                message: alloc::format!(
                    "edge index {index} out of bounds (edge_bound={edge_bound})"
                ),
            });
            continue;
        }
        if core::mem::replace(&mut slots[index].live, true) {
            result.errors.push(VerifyError {
                message: alloc::format!("edge index {index} appears more than once"),
            });
            continue;
        }
        let edge = graph.edge(edge_id);
        if edge.source().index() >= node_bound || edge.target().index() >= node_bound {
            result.errors.push(VerifyError {
                message: alloc::format!(
                    "edge index {index} has endpoint outside node_bound {node_bound}"
                ),
            });
        }
    }

    // A node beyond the bound is already reported by `verify_view`, and asking
    // the view for its edges would only repeat the contract violation.
    for node in graph.node_ids().filter(|node| node.index() < node_bound) {
        check_listings(
            Outgoing,
            &OUTGOING,
            graph,
            node,
            &mut slots,
            &mut result.errors,
        );
        check_listings(
            Incoming,
            &INCOMING,
            graph,
            node,
            &mut slots,
            &mut result.errors,
        );
    }
    for (edge, slot) in slots.iter().enumerate().filter(|(_, slot)| slot.live) {
        let (outgoing, incoming) = (slot.outgoing, slot.incoming);
        if outgoing != 1 || incoming != 1 {
            result.errors.push(VerifyError {
                message: alloc::format!(
                    "edge index {edge} appears {outgoing} time(s) in outgoing and {incoming} time(s) in incoming adjacency"
                ),
            });
        }
    }
    result
}

#[cfg(test)]
mod tests;
