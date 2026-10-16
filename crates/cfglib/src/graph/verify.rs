//! CFG structural verification.
//!
//! Validates invariants that should always hold for a well-formed CFG:
//! entry block exists, adjacency lists are consistent, no out-of-bounds
//! IDs, and every non-entry reachable block has at least one predecessor.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeId;
use crate::graph::edge_view::EdgeView;
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

fn verify_edge_endpoints<I, E>(cfg: &Cfg<I, E>, block_bound: usize, errors: &mut Vec<VerifyError>) {
    for edge in cfg.edges() {
        if edge.source().index() >= block_bound {
            errors.push(VerifyError {
                message: alloc::format!(
                    "edge {} source {} out of bounds (block_bound={})",
                    edge.id(),
                    edge.source(),
                    block_bound
                ),
            });
        }
        if edge.target().index() >= block_bound {
            errors.push(VerifyError {
                message: alloc::format!(
                    "edge {} target {} out of bounds (block_bound={})",
                    edge.id(),
                    edge.target(),
                    block_bound
                ),
            });
        }
    }
}

/// Name every live edge that its source's successors or its target's
/// predecessors fail to list.
///
/// [`verify_adjacency`] checks the converse, that every listing names an edge
/// that agrees with it, which a dropped listing cannot violate. It counts the
/// agreeing listings as it goes, and only a count that differs from the live
/// edges calls this walk, so a sound CFG never pays for it.
fn report_unlisted_edges<I, E>(cfg: &Cfg<I, E>, errors: &mut Vec<VerifyError>) {
    let bound = cfg.edge_bound();
    let mut successor_listed = alloc::vec![false; bound];
    let mut predecessor_listed = alloc::vec![false; bound];
    for block in cfg.block_ids() {
        for edge in cfg.outgoing(block) {
            if let Some(listed) = successor_listed.get_mut(edge.index()) {
                *listed = true;
            }
        }
        for edge in cfg.incoming(block) {
            if let Some(listed) = predecessor_listed.get_mut(edge.index()) {
                *listed = true;
            }
        }
    }
    for edge in cfg.edges() {
        let index = edge.id().index();
        if index >= bound {
            continue;
        }
        if !successor_listed[index] {
            errors.push(VerifyError {
                message: alloc::format!(
                    "edge {} is not a successor edge of its source {}",
                    edge.id(),
                    edge.source()
                ),
            });
        }
        if !predecessor_listed[index] {
            errors.push(VerifyError {
                message: alloc::format!(
                    "edge {} is not a predecessor edge of its target {}",
                    edge.id(),
                    edge.target()
                ),
            });
        }
    }
}

fn verify_adjacency<I, E>(cfg: &Cfg<I, E>, errors: &mut Vec<VerifyError>) {
    let mut successor_listings = 0;
    let mut predecessor_listings = 0;
    for block_id_ in cfg.block_ids() {
        let block_id = block_id_;
        for edge_id in cfg.outgoing(block_id) {
            if edge_id.index() >= cfg.edge_bound() {
                errors.push(VerifyError {
                    message: alloc::format!(
                        "block {block_id} successor edge {edge_id} out of bounds"
                    ),
                });
            } else if cfg.edge(edge_id).source() != block_id {
                errors.push(VerifyError {
                    message: alloc::format!(
                        "block {} lists edge {} as successor but edge source is {}",
                        block_id,
                        edge_id,
                        cfg.edge(edge_id).source()
                    ),
                });
            } else {
                successor_listings += 1;
            }
        }
        for edge_id in cfg.incoming(block_id) {
            if edge_id.index() >= cfg.edge_bound() {
                errors.push(VerifyError {
                    message: alloc::format!(
                        "block {block_id} predecessor edge {edge_id} out of bounds"
                    ),
                });
            } else if cfg.edge(edge_id).target() != block_id {
                errors.push(VerifyError {
                    message: alloc::format!(
                        "block {} lists edge {} as predecessor but edge target is {}",
                        block_id,
                        edge_id,
                        cfg.edge(edge_id).target()
                    ),
                });
            } else {
                predecessor_listings += 1;
            }
        }
    }
    if successor_listings != cfg.edge_count() || predecessor_listings != cfg.edge_count() {
        report_unlisted_edges(cfg, errors);
    }
}

/// Detects an edge listed twice within one adjacency walk.
///
/// Each walk stamps the edges it meets with its own number, so starting the
/// next walk costs nothing. An identity beyond the edge bound has no stamp
/// slot and is tracked in `stray` instead.
struct WalkStamps {
    stamps: Vec<usize>,
    stray: BTreeSet<EdgeId>,
    walk: usize,
}

impl WalkStamps {
    fn new(edge_bound: usize) -> Self {
        Self {
            stamps: alloc::vec![0; edge_bound],
            stray: BTreeSet::new(),
            walk: 0,
        }
    }

    fn begin_walk(&mut self) {
        self.walk += 1;
        self.stray.clear();
    }

    /// Records `edge` in the current walk; true when it was already recorded.
    fn repeats(&mut self, edge: EdgeId) -> bool {
        match self.stamps.get_mut(edge.index()) {
            Some(stamp) => core::mem::replace(stamp, self.walk) == self.walk,
            None => !self.stray.insert(edge),
        }
    }
}

fn verify_unique_adjacency<I, E>(cfg: &Cfg<I, E>, errors: &mut Vec<VerifyError>) {
    let mut seen = WalkStamps::new(cfg.edge_bound());
    for block_id in cfg.block_ids() {
        seen.begin_walk();
        for edge_id in cfg.outgoing(block_id) {
            if seen.repeats(edge_id) {
                errors.push(VerifyError {
                    message: alloc::format!(
                        "block {block_id} has duplicate successor edge {edge_id}"
                    ),
                });
            }
        }
        seen.begin_walk();
        for edge_id in cfg.incoming(block_id) {
            if seen.repeats(edge_id) {
                errors.push(VerifyError {
                    message: alloc::format!(
                        "block {block_id} has duplicate predecessor edge {edge_id}"
                    ),
                });
            }
        }
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
                    "adjacency mismatch: {count} successor edge(s) {source}->{target} but {mirrored} predecessor entrie(s)"
                ),
            });
        }
    }
    for ((source, target), count, mirrored) in multiplicities(reverse, forward) {
        if mirrored == 0 {
            errors.push(VerifyError {
                message: alloc::format!(
                    "adjacency mismatch: {count} predecessor entrie(s) {source}->{target} with no successor edge"
                ),
            });
        }
    }
}

/// Validate structural invariants of a CFG.
///
/// Checks performed:
/// 1. Entry block index is within bounds.
/// 2. Every edge references valid source/target block IDs.
/// 3. Adjacency lists (`succs`/`preds`) are consistent with edges, and every
///    live edge is listed by both of its endpoints.
/// 4. Every reachable non-entry block has at least one predecessor.
/// 5. No duplicate edges in adjacency lists.
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
    let mut errors = Vec::new();
    // Identities, not quantities: removed blocks keep their slots, so every
    // live index is bounded by `block_bound`, which `block_count` undercuts.
    let n = cfg.block_bound();

    // 1. Entry in bounds.
    if cfg.entry().index() >= n {
        errors.push(VerifyError {
            message: alloc::format!(
                "entry block {} out of bounds (block_bound={})",
                cfg.entry(),
                n
            ),
        });
        // Can't do much more if entry is invalid.
        return VerifyReport { errors };
    }

    verify_edge_endpoints(cfg, n, &mut errors);
    verify_adjacency(cfg, &mut errors);

    // 4. Every reachable non-entry block has a predecessor.
    let reachable = cfg.depth_first_preorder();
    for &bid in &reachable {
        if bid == cfg.entry() {
            continue;
        }
        if cfg.incoming(bid).next().is_none() {
            errors.push(VerifyError {
                message: alloc::format!("reachable block {bid} has no predecessors"),
            });
        }
    }

    verify_unique_adjacency(cfg, &mut errors);

    VerifyReport { errors }
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

/// Validate the [`RootedView`] contract on consumer-owned storage.
///
/// Checks performed:
/// 1. The root node's index is within `0..node_bound()`.
/// 2. Forward and reverse adjacency mirror each other with matching
///    multiplicity (every successor entry has a matching predecessor entry).
/// 3. Every reachable non-root node has at least one predecessor.
///
/// This catches adapter bugs when a consumer implements the view traits over
/// its own graph store — the counterpart of [`verify`], which checks the
/// storage invariants of [`Cfg`] itself.
#[must_use]
pub fn verify_view<G: RootedView>(graph: &G) -> VerifyReport {
    let mut errors = Vec::new();
    let node_count = graph.node_bound();

    let root = graph.root();
    if root.index() >= node_count {
        errors.push(VerifyError {
            message: alloc::format!(
                "root node index {} out of bounds (node_count={node_count})",
                root.index()
            ),
        });
        return VerifyReport { errors };
    }

    // Forward and reverse adjacency must agree as multisets of (source, target).
    let mut forward: Vec<Pair> = Vec::with_capacity(node_count);
    let mut reverse: Vec<Pair> = Vec::with_capacity(node_count);
    for node in graph.node_ids() {
        forward.extend(
            graph
                .successors(node)
                .map(|successor| (node.index(), successor.index())),
        );
        reverse.extend(
            graph
                .predecessors(node)
                .map(|predecessor| (predecessor.index(), node.index())),
        );
    }
    forward.sort_unstable();
    reverse.sort_unstable();
    if forward != reverse {
        report_adjacency_mismatches(&forward, &reverse, &mut errors);
    }

    // Every reachable non-root node has a predecessor.
    let mut visited = alloc::vec![false; node_count];
    let mut stack = alloc::vec![root];
    visited[root.index()] = true;
    while let Some(node) = stack.pop() {
        for successor in graph.successors(node) {
            if !visited[successor.index()] {
                visited[successor.index()] = true;
                stack.push(successor);
            }
        }
    }
    for node in graph.node_ids() {
        if visited[node.index()] && node != root && graph.predecessors(node).next().is_none() {
            errors.push(VerifyError {
                message: alloc::format!("reachable node {} has no predecessors", node.index()),
            });
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
        if edge.source().index() >= graph.node_bound()
            || edge.target().index() >= graph.node_bound()
        {
            result.errors.push(VerifyError {
                message: alloc::format!(
                    "edge index {index} has endpoint outside node_count {}",
                    graph.node_bound()
                ),
            });
        }
    }

    for node in graph.node_ids() {
        for edge_id in graph.outgoing(node) {
            let index = edge_id.index();
            let Some(slot) = slots.get_mut(index).filter(|slot| slot.live) else {
                result.errors.push(VerifyError {
                    message: alloc::format!(
                        "node {} has non-live outgoing edge index {index}",
                        node.index()
                    ),
                });
                continue;
            };
            slot.outgoing += 1;
            if graph.edge(edge_id).source() != node {
                result.errors.push(VerifyError {
                    message: alloc::format!(
                        "node {} lists edge index {index}, whose source is {}",
                        node.index(),
                        graph.edge(edge_id).source().index()
                    ),
                });
            }
        }
        for edge_id in graph.incoming(node) {
            let index = edge_id.index();
            let Some(slot) = slots.get_mut(index).filter(|slot| slot.live) else {
                result.errors.push(VerifyError {
                    message: alloc::format!(
                        "node {} has non-live incoming edge index {index}",
                        node.index()
                    ),
                });
                continue;
            };
            slot.incoming += 1;
            if graph.edge(edge_id).target() != node {
                result.errors.push(VerifyError {
                    message: alloc::format!(
                        "node {} lists edge index {index}, whose target is {}",
                        node.index(),
                        graph.edge(edge_id).target().index()
                    ),
                });
            }
        }
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
