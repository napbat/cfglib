//! CFG structural verification.
//!
//! Validates invariants that should always hold for a well-formed CFG:
//! entry block exists, adjacency lists are consistent, no out-of-bounds
//! IDs, and every non-entry reachable block has at least one predecessor.

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
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

fn verify_adjacency<I, E>(cfg: &Cfg<I, E>, errors: &mut Vec<VerifyError>) {
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
            }
        }
    }
}

fn verify_unique_adjacency<I, E>(cfg: &Cfg<I, E>, errors: &mut Vec<VerifyError>) {
    for block_id_ in cfg.block_ids() {
        let block_id = block_id_;
        let mut seen = alloc::collections::BTreeSet::new();
        for edge_id in cfg.outgoing(block_id) {
            if !seen.insert(edge_id) {
                errors.push(VerifyError {
                    message: alloc::format!(
                        "block {block_id} has duplicate successor edge {edge_id}"
                    ),
                });
            }
        }
        seen.clear();
        for edge_id in cfg.incoming(block_id) {
            if !seen.insert(edge_id) {
                errors.push(VerifyError {
                    message: alloc::format!(
                        "block {block_id} has duplicate predecessor edge {edge_id}"
                    ),
                });
            }
        }
    }
}

/// Validate structural invariants of a CFG.
///
/// Checks performed:
/// 1. Entry block index is within bounds.
/// 2. Every edge references valid source/target block IDs.
/// 3. Adjacency lists (`succs`/`preds`) are consistent with edges.
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
    let mut forward: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut reverse: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for node in graph.node_ids() {
        for successor in graph.successors(node) {
            *forward
                .entry((node.index(), successor.index()))
                .or_insert(0) += 1;
        }
        for predecessor in graph.predecessors(node) {
            *reverse
                .entry((predecessor.index(), node.index()))
                .or_insert(0) += 1;
        }
    }
    for (&(source, target), &count) in &forward {
        let mirrored = reverse.get(&(source, target)).copied().unwrap_or(0);
        if mirrored != count {
            errors.push(VerifyError {
                message: alloc::format!(
                    "adjacency mismatch: {count} successor edge(s) {source}->{target} but {mirrored} predecessor entrie(s)"
                ),
            });
        }
    }
    for (&(source, target), &count) in &reverse {
        if !forward.contains_key(&(source, target)) {
            errors.push(VerifyError {
                message: alloc::format!(
                    "adjacency mismatch: {count} predecessor entrie(s) {source}->{target} with no successor edge"
                ),
            });
        }
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
    let mut live = BTreeSet::new();
    for edge_id in graph.edge_ids() {
        let index = edge_id.index();
        if index >= graph.edge_bound() {
            result.errors.push(VerifyError {
                message: alloc::format!(
                    "edge index {index} out of bounds (edge_bound={})",
                    graph.edge_bound()
                ),
            });
            continue;
        }
        if !live.insert(index) {
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

    let mut outgoing_counts = BTreeMap::new();
    let mut incoming_counts = BTreeMap::new();
    for node in graph.node_ids() {
        for edge_id in graph.outgoing(node) {
            let index = edge_id.index();
            if !live.contains(&index) {
                result.errors.push(VerifyError {
                    message: alloc::format!(
                        "node {} has non-live outgoing edge index {index}",
                        node.index()
                    ),
                });
                continue;
            }
            *outgoing_counts.entry(index).or_insert(0) += 1;
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
            if !live.contains(&index) {
                result.errors.push(VerifyError {
                    message: alloc::format!(
                        "node {} has non-live incoming edge index {index}",
                        node.index()
                    ),
                });
                continue;
            }
            *incoming_counts.entry(index).or_insert(0) += 1;
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
    for edge in live {
        let outgoing = outgoing_counts.get(&edge).copied().unwrap_or(0);
        let incoming = incoming_counts.get(&edge).copied().unwrap_or(0);
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
mod tests {
    use super::*;
    use crate::edge::EdgeKind;
    use crate::test_util::ff;

    #[test]
    fn valid_cfg_passes() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("a"));
        cfg.block_mut(b).instructions_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);

        let result = verify(&cfg);
        assert!(result.is_ok(), "valid CFG should pass: {:?}", result.errors);
    }

    #[test]
    fn single_block_cfg_passes() {
        let mut cfg = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("a"));
        assert!(verify(&cfg).is_ok());
    }

    #[test]
    fn diamond_cfg_passes() {
        let mut cfg = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        let merge = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("e"));
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        cfg.add_edge(a, merge, EdgeKind::Fallthrough);
        cfg.add_edge(b, merge, EdgeKind::Fallthrough);

        assert!(verify(&cfg).is_ok());
    }

    #[test]
    fn merging_a_linear_chain_leaves_one_verifiable_block() {
        let mut cfg = Cfg::new();
        let mut previous = cfg.entry();
        cfg.block_mut(previous).instructions_mut().push(ff("b0"));
        for _ in 1..8 {
            let next = cfg.new_block();
            cfg.block_mut(next).instructions_mut().push(ff("b"));
            cfg.add_edge(previous, next, EdgeKind::Fallthrough);
            previous = next;
        }
        assert_eq!(cfg.block_count(), 8);

        assert_eq!(crate::merge_blocks(&mut cfg), 7);

        let result = verify(&cfg);
        assert!(
            result.is_ok(),
            "a merged chain is still a valid CFG: {:?}",
            result.errors
        );
        assert_eq!(cfg.block_count(), 1, "the chain collapsed into the entry");
        assert_eq!(
            cfg.block_bound(),
            8,
            "the consumed blocks keep their identity slots"
        );
        assert_eq!(cfg.block(cfg.entry()).instructions().len(), 8);
    }

    #[test]
    fn a_diamond_missing_an_arm_survives_every_whole_graph_analysis() {
        use crate::dataflow::liveness::LivenessProblem;
        use crate::test_util::{DfInst, df_def, df_use};

        let mut cfg: Cfg<DfInst> = Cfg::new();
        let left = cfg.new_block();
        let right = cfg.new_block();
        let merge = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_def("def", 0));
        cfg.block_mut(left).push(df_use("left", 0));
        cfg.block_mut(right).push(df_use("right", 0));
        cfg.block_mut(merge).push(df_use("merge", 0));
        cfg.add_edge(cfg.entry(), left, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), right, EdgeKind::ConditionalFalse);
        cfg.add_edge(left, merge, EdgeKind::Fallthrough);
        cfg.add_edge(right, merge, EdgeKind::Fallthrough);

        // Retiring the low-indexed arm leaves a live block whose index sits
        // above the live count — the shape that a count-sized array truncates.
        assert!(cfg.remove_block(left));
        assert_eq!(cfg.block_count(), 3);
        assert_eq!(cfg.block_bound(), 4);

        let report = verify(&cfg);
        assert!(
            report.is_ok(),
            "a retired slot is not a violation: {:?}",
            report.errors
        );

        let dominators = crate::DominatorTree::compute(&cfg);
        assert_eq!(dominators.idom(merge), Some(right));
        assert!(!dominators.is_reachable(left));

        let post_dominators = crate::DominatorTree::compute_post(&cfg);
        assert_eq!(post_dominators.idom(cfg.entry()), Some(right));
        assert_eq!(post_dominators.idom(right), Some(merge));

        let facts = crate::solve_problem(&cfg, &LivenessProblem).expect("liveness solves");
        assert!(facts.fact_in(merge).contains(&0));
        assert!(
            facts.fact_in(left).is_empty(),
            "a retired slot keeps the bottom fact"
        );

        let ssa = crate::SsaForm::compute(&cfg, &dominators);
        assert_eq!(ssa.blocks().len(), cfg.block_bound());
        assert_eq!(ssa.block(merge).instructions.len(), 1);
        assert_eq!(ssa.block(left).instructions.len(), 0);

        let reversed = crate::reverse_cfg(&cfg);
        let reversed_report = verify(&reversed);
        assert!(
            reversed_report.is_ok(),
            "the reverse of a holed CFG verifies: {:?}",
            reversed_report.errors
        );
        assert_eq!(reversed.entry(), merge);
        assert_eq!(reversed.block_count(), cfg.block_count());
        assert_eq!(reversed.block_bound(), cfg.block_bound());

        cfg.compact();
        assert_eq!(cfg.block_bound(), cfg.block_count());
        assert!(verify(&cfg).is_ok());
    }

    #[test]
    fn verify_error_count() {
        let cfg: Cfg<crate::test_util::MockInst> = Cfg::new();
        let result = verify(&cfg);
        assert_eq!(result.error_count(), 0);
    }

    #[test]
    fn verify_view_accepts_consistent_consumer_view() {
        let mut graph = crate::graph::store::Graph::new();
        let root = graph.add_node(());
        let child = graph.add_node(());
        graph.add_edge(root, child, ());
        graph.add_edge(root, child, ());

        let rooted = crate::graph::view::Rooted::new(&graph, root);
        assert!(verify_view(&rooted).is_ok());
    }

    #[test]
    fn verify_view_reports_one_sided_adjacency() {
        struct Broken;
        impl crate::graph::view::GraphView for Broken {
            type NodeId = usize;

            fn node_bound(&self) -> usize {
                2
            }

            fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
                0..2
            }

            fn successors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
                (node == 0).then_some(1).into_iter()
            }

            fn predecessors(&self, _node: usize) -> impl Iterator<Item = usize> + '_ {
                core::iter::empty()
            }
        }
        impl crate::graph::view::RootedView for Broken {
            fn root(&self) -> usize {
                0
            }
        }

        let result = verify_view(&Broken);
        assert!(!result.is_ok());
        assert!(
            result
                .errors
                .iter()
                .any(|error| error.message.contains("adjacency mismatch"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|error| error.message.contains("no predecessors"))
        );
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Route {
        Normal,
        Handler(u8),
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum SemanticError {
        ConditionalCardinality {
            block: BlockId,
            actual: usize,
        },
        HandlerOrder {
            block: BlockId,
            expected: u8,
            actual: u8,
        },
    }

    struct RouteValidator;

    impl SemanticValidator<(), Route> for RouteValidator {
        type Error = SemanticError;

        fn validate_block(
            &self,
            cfg: &Cfg<(), Route>,
            block: BlockId,
            errors: &mut Vec<Self::Error>,
        ) {
            let mut conditional_count = 0;
            let mut expected_handler = 0;
            for edge in cfg.outgoing(block) {
                let edge = cfg.edge(edge);
                if matches!(
                    edge.kind(),
                    EdgeKind::ConditionalTrue | EdgeKind::ConditionalFalse
                ) {
                    conditional_count += 1;
                }
                if let Route::Handler(actual) = *edge.payload() {
                    if actual != expected_handler {
                        errors.push(SemanticError::HandlerOrder {
                            block,
                            expected: expected_handler,
                            actual,
                        });
                    }
                    expected_handler += 1;
                }
            }
            if conditional_count != 0 && conditional_count != 2 {
                errors.push(SemanticError::ConditionalCardinality {
                    block,
                    actual: conditional_count,
                });
            }
        }
    }

    #[test]
    fn semantic_hooks_return_typed_cardinality_and_order_errors() {
        let mut cfg = Cfg::<(), Route>::with_edge_payload();
        let branch = cfg.new_block();
        let handler_a = cfg.new_block();
        let handler_b = cfg.new_block();
        let entry = cfg.entry();
        cfg.add_edge_with_payload(entry, branch, EdgeKind::ConditionalTrue, Route::Normal);
        cfg.add_edge_with_payload(
            entry,
            handler_a,
            EdgeKind::ExceptionHandler,
            Route::Handler(1),
        );
        cfg.add_edge_with_payload(
            entry,
            handler_b,
            EdgeKind::ExceptionHandler,
            Route::Handler(0),
        );

        let result = verify_with(&cfg, &RouteValidator);
        assert!(result.structural.is_ok());
        assert!(!result.is_ok());
        assert_eq!(result.semantic_errors.len(), 3);
        assert!(matches!(
            result.semantic_errors[2],
            SemanticError::ConditionalCardinality { actual: 1, .. }
        ));
    }
}
