//! Structural verification of a CFG or of a consumer-owned graph view.
//!
//! One definition of the invariants serves both: [`verify_view`] checks the
//! node-level contract of any rooted view, [`verify_edge_view`] adds the
//! stable-edge contract on top of it, and [`verify`] is the latter over a
//! [`Cfg`]. The invariants are an entry that exists, in-bounds
//! identities, mirrored forward and reverse adjacency, and a predecessor for
//! every reachable non-entry node.

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeId;
use crate::graph::edge_traverse::EdgeAdjacency;
use crate::graph::edge_view::EdgeView;
use crate::graph::traverse::{Incoming, Outgoing, TraversalDirection, reachable};
use crate::graph::view::{DenseId, RootedView};

/// A single verification failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyError {
    /// Human-readable description of the violated invariant.
    pub message: String,
}

impl core::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "CFG verify: {}", self.message)
    }
}

/// Result of running [`verify`] on a CFG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyReport {
    /// All detected violations (empty = valid).
    pub errors: Vec<VerifyError>,
}

impl VerifyReport {
    /// True when the CFG passes all invariant checks.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Number of violations found.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
}

/// Consumer-defined semantic checks layered over structural CFG verification.
///
/// Hooks run in block allocation order, then live edge identity order, then
/// once for whole-graph checks. Errors are consumer types, allowing frontends
/// to report branch cardinality, handler order, continuation constraints, or
/// format-specific provenance without putting those semantics in cfglib.
pub trait SemanticValidator<I, E> {
    /// Structured consumer error.
    type Error;

    /// Validate one block and its ordered adjacency.
    fn validate_block(&self, _cfg: &Cfg<I, E>, _block: BlockId, _errors: &mut Vec<Self::Error>) {}

    /// Validate one stable live edge.
    fn validate_edge(&self, _cfg: &Cfg<I, E>, _edge: EdgeId, _errors: &mut Vec<Self::Error>) {}

    /// Validate relationships that span several blocks or edges.
    fn finish(&self, _cfg: &Cfg<I, E>, _errors: &mut Vec<Self::Error>) {}
}

/// Structural and consumer-semantic verification results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticVerifyReport<E> {
    /// Storage and adjacency invariants checked by [`verify`].
    pub structural: VerifyReport,
    /// Typed errors emitted by the consumer validator.
    pub semantic_errors: Vec<E>,
}

impl<E> SemanticVerifyReport<E> {
    /// Whether both structural and semantic validation succeeded.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.structural.is_ok() && self.semantic_errors.is_empty()
    }

    /// Total structural plus semantic error count.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.structural.error_count() + self.semantic_errors.len()
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

/// Validate structural invariants of a CFG.
///
/// A [`Cfg`] is an edge-aware rooted view of itself, so this is
/// [`verify_edge_view`] over the entry block: the CFG store and consumer-owned
/// storage are held to one definition of every invariant, and one set of
/// messages. Checks performed:
/// 1. The entry block index is within bounds.
/// 2. Every live edge references valid source/target block IDs.
/// 3. Adjacency lists (`succs`/`preds`) are consistent with edges, and every
///    live edge is listed exactly once by each of its endpoints.
/// 4. Every reachable non-entry block has at least one predecessor.
/// 5. Every block exception metadata names by a single reference (a handler
///    entry, a filter block, a cleanup resume point, a continuation target)
///    is live: the references [`Cfg::compact`] panics on. The entry cannot
///    be removed, so it is live by construction.
///
/// Returns a [`VerifyReport`] containing all violations found.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, verify};
///
/// let mut cfg = Cfg::<u32>::new();
/// let b1 = cfg.new_block();
/// cfg.add_edge(cfg.entry(), b1, EdgeKind::Fallthrough);
///
/// let result = verify(&cfg);
/// assert!(result.is_ok());
/// ```
#[must_use]
pub fn verify<I, E>(cfg: &Cfg<I, E>) -> VerifyReport {
    let mut report = verify_edge_view(cfg);
    report
        .errors
        .extend(cfg.dangling_anchors().map(|block| VerifyError {
            message: alloc::format!("exception metadata names {block}, which is not a live block"),
        }));
    report
}

/// Validate structural invariants and consumer semantics in stable order.
#[must_use]
pub fn verify_with<I, E, S>(cfg: &Cfg<I, E>, validator: &S) -> SemanticVerifyReport<S::Error>
where
    S: SemanticValidator<I, E>,
{
    let structural = verify(cfg);
    let mut semantic_errors = Vec::new();
    for block_id in cfg.block_ids() {
        validator.validate_block(cfg, block_id, &mut semantic_errors);
    }
    for edge in cfg.edges() {
        validator.validate_edge(cfg, edge.id(), &mut semantic_errors);
    }
    validator.finish(cfg, &mut semantic_errors);
    SemanticVerifyReport {
        structural,
        semantic_errors,
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
/// its own graph store — the counterpart of [`verify`], which checks the
/// storage invariants of [`Cfg`] itself.
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
        return VerifyReport { errors };
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
        let reached = reachable(graph, [root], TraversalDirection::Outgoing);
        for node in graph.node_ids() {
            if reached[node.index()] && node != root && graph.predecessors(node).next().is_none() {
                errors.push(VerifyError {
                    message: alloc::format!("reachable node {} has no predecessors", node.index()),
                });
            }
        }
    }

    VerifyReport { errors }
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
