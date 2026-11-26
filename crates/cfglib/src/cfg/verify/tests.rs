//! Tests for structural and semantic CFG verification.

use super::*;
use crate::edge::EdgeKind;
use crate::graph::store::NodeOrder;
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
    use crate::test_util::{DfInst, LiveVariables, df_def, df_use};

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

    let facts = crate::solve_node_problem(&cfg, &LiveVariables, crate::Solve::default())
        .expect("liveness solves");
    assert!(facts.fact_in(merge).contains(&0));
    assert!(
        facts.fact_in(left).is_empty(),
        "a retired slot keeps the bottom fact"
    );

    let frontiers = crate::DominanceFrontiers::compute(&cfg, &dominators);
    assert!(
        frontiers.frontier(right).is_empty(),
        "the surviving arm dominates the merge"
    );
    assert!(
        frontiers.frontier(left).is_empty(),
        "a retired slot has an empty frontier"
    );

    cfg.compact(NodeOrder::Insertion);
    assert_eq!(cfg.block_bound(), cfg.block_count());
    assert!(verify(&cfg).is_ok());
}

#[test]
fn verify_error_count() {
    let cfg: Cfg<crate::test_util::MockInst> = Cfg::new();
    let result = verify(&cfg);
    assert_eq!(result.error_count(), 0);
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
    assert!(result.errors.is_empty(), "{:?}", result.errors);
    assert!(!result.is_ok());
    assert_eq!(result.semantic_errors.len(), 3);
    assert_eq!(result.error_count(), 3);
    assert!(matches!(
        result.semantic_errors[2],
        SemanticError::ConditionalCardinality { actual: 1, .. }
    ));
}
