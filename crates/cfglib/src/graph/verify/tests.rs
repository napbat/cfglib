//! Tests for structural and semantic CFG verification.

use super::*;
use crate::edge::EdgeKind;
use crate::test_util::ff;
use alloc::vec;

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

    let post_dominators = crate::PostDominatorTree::compute(&cfg);
    assert_eq!(
        post_dominators.ipdom(cfg.entry()),
        Some(crate::PostNode::Node(right))
    );
    assert_eq!(
        post_dominators.ipdom(right),
        Some(crate::PostNode::Node(merge))
    );
    assert_eq!(post_dominators.exits(), [merge]);

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

/// Explicit adjacency tables, so a test can corrupt each side on its own.
struct Table {
    out_edges: Vec<Vec<usize>>,
    in_edges: Vec<Vec<usize>>,
    endpoints: Vec<(usize, usize)>,
    live: Vec<usize>,
}

impl crate::graph::view::GraphView for Table {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        self.out_edges.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.out_edges.len()
    }

    fn successors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        let endpoints = &self.endpoints;
        self.out_edges[node]
            .iter()
            .filter_map(|&edge| endpoints.get(edge))
            .map(|&(_, target)| target)
    }

    fn predecessors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        let endpoints = &self.endpoints;
        self.in_edges[node]
            .iter()
            .filter_map(|&edge| endpoints.get(edge))
            .map(|&(source, _)| source)
    }
}

impl RootedView for Table {
    fn root(&self) -> usize {
        0
    }
}

impl EdgeView for Table {
    type EdgeId = usize;
    type EdgeData = ();

    fn edge_bound(&self) -> usize {
        self.endpoints.len()
    }

    fn edge_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.live.iter().copied()
    }

    fn outgoing(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.out_edges[node].iter().copied()
    }

    fn incoming(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.in_edges[node].iter().copied()
    }

    fn edge(&self, edge: usize) -> crate::graph::edge_view::EdgeRef<'_, usize, usize, ()> {
        let (source, target) = self.endpoints[edge];
        crate::graph::edge_view::EdgeRef::new(edge, source, target, &())
    }
}

fn messages(report: &VerifyReport) -> Vec<&str> {
    report
        .errors
        .iter()
        .map(|error| error.message.as_str())
        .collect()
}

#[test]
fn verify_view_reports_multiplicity_mismatches_in_pair_order() {
    // Edge slots 5 and 6 are listed on the reverse side only.
    let table = Table {
        out_edges: vec![vec![0, 1, 2], vec![3], vec![4], vec![]],
        in_edges: vec![vec![5], vec![0], vec![2], vec![3, 4, 4, 6]],
        endpoints: vec![(0, 1), (0, 1), (0, 2), (1, 3), (2, 3), (3, 0), (0, 3)],
        live: (0..7).collect(),
    };

    let report = verify_view(&table);
    assert_eq!(
        messages(&report),
        [
            "adjacency mismatch: 2 successor edge(s) 0->1 but 1 predecessor entrie(s)",
            "adjacency mismatch: 1 successor edge(s) 2->3 but 2 predecessor entrie(s)",
            "adjacency mismatch: 1 predecessor entrie(s) 0->3 with no successor edge",
            "adjacency mismatch: 1 predecessor entrie(s) 3->0 with no successor edge",
        ]
    );
}

#[test]
fn verify_edge_view_reports_every_edge_level_defect_in_order() {
    // Live list: edge 1 twice, edge 8 beyond the bound, edge 3 pointing at
    // node 9, and edge 4 removed. Adjacency: stray, repeated, misoriented
    // and dangling entries.
    let table = Table {
        out_edges: vec![vec![0, 5, 4], vec![1, 1, 2], vec![2]],
        in_edges: vec![vec![2], vec![0, 7], vec![1, 5, 4, 0]],
        endpoints: vec![(0, 1), (1, 2), (2, 0), (0, 9), (1, 2), (0, 2)],
        live: vec![0, 1, 1, 2, 3, 5, 8],
    };

    let report = verify_edge_view(&table);
    assert_eq!(
        messages(&report),
        [
            "adjacency mismatch: 1 successor edge(s) 1->0 but 0 predecessor entrie(s)",
            "edge index 1 appears more than once",
            "edge index 3 has endpoint outside node_count 3",
            "edge index 8 out of bounds (edge_bound=6)",
            "node 0 has non-live outgoing edge index 4",
            "node 1 lists edge index 2, whose source is 2",
            "node 1 has non-live incoming edge index 7",
            "node 2 has non-live incoming edge index 4",
            "node 2 lists edge index 0, whose target is 1",
            "edge index 0 appears 1 time(s) in outgoing and 2 time(s) in incoming adjacency",
            "edge index 1 appears 2 time(s) in outgoing and 1 time(s) in incoming adjacency",
            "edge index 2 appears 2 time(s) in outgoing and 1 time(s) in incoming adjacency",
            "edge index 3 appears 0 time(s) in outgoing and 0 time(s) in incoming adjacency",
        ]
    );
}

#[test]
fn walk_stamps_report_repeats_within_one_walk_only() {
    let edge = EdgeId::from_index;
    let mut seen = WalkStamps::new(2);
    seen.begin_walk();
    assert!(!seen.repeats(edge(0)));
    assert!(seen.repeats(edge(0)));
    assert!(!seen.repeats(edge(1)));
    // An identity beyond the bound has no stamp slot but still repeats.
    assert!(!seen.repeats(edge(9)));
    assert!(seen.repeats(edge(9)));

    seen.begin_walk();
    assert!(!seen.repeats(edge(0)));
    assert!(!seen.repeats(edge(9)));
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

    fn validate_block(&self, cfg: &Cfg<(), Route>, block: BlockId, errors: &mut Vec<Self::Error>) {
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
