//! `ControlTree::compute_with_report` into `lift_with_tree`: the tree, its
//! `LiftReport`, and the structured function that reads them.

use cfglib::ir::hlil::{
    LiftError, LiftMetadata, VerificationReport, lift, lift_with_tree, pseudocode, verify,
};
use cfglib::{ControlNode, ControlTree, GotoReason, HandlerKind, LiftReport, LoopKind};

use crate::dialect::{Source, Tiny};
use crate::golden::assert_golden;
use crate::graphs::{self, Fixture};
use crate::machine::{End, run_cfg, run_hlil};
use crate::storage::{Builder, Hlil, StmtNode};

type Lifted = Result<
    cfglib::ir::hlil::LiftedFunction<Hlil, usize, usize, usize>,
    LiftError<VerificationReport>,
>;

fn structure(source: &Source) -> (ControlTree, LiftReport) {
    ControlTree::compute_with_report(&source.cfg)
}

fn lift_tree(source: &Source, tree: &ControlTree, report: &LiftReport) -> Lifted {
    lift_with_tree::<Tiny, _, _>(source, tree, report, Builder::new(), LiftMetadata::Preserve)
}

fn lifted(fixture: &Fixture) -> Hlil {
    let (tree, report) = structure(&fixture.source);
    lift_tree(&fixture.source, &tree, &report)
        .unwrap_or_else(|error| panic!("{}: {error}", fixture.name))
        .function
}

fn every_fixture() -> Vec<Fixture> {
    let mut all = graphs::structured();
    all.push(graphs::guarded_division_with_finally());
    all.push(graphs::irreducible_loop());
    all
}

/// How often each kind of construct occurs.
#[derive(Debug, Default, PartialEq, Eq)]
struct Constructs {
    branches: usize,
    loops: usize,
    switches: usize,
    tries: usize,
    gotos: usize,
    labels: usize,
    breaks: usize,
    continues: usize,
}

fn tree_constructs(tree: &ControlTree) -> Constructs {
    let mut found = Constructs::default();
    for id in tree.preorder() {
        match tree.node(id) {
            ControlNode::IfThenElse { .. } => found.branches += 1,
            ControlNode::Loop { .. } => found.loops += 1,
            ControlNode::Switch { .. } => found.switches += 1,
            ControlNode::TryCatch { .. } => found.tries += 1,
            ControlNode::Goto { .. } => found.gotos += 1,
            ControlNode::Label { .. } => found.labels += 1,
            ControlNode::Break { .. } => found.breaks += 1,
            ControlNode::Continue { .. } => found.continues += 1,
            _ => {}
        }
    }
    found
}

fn function_constructs(function: &Hlil) -> Constructs {
    let mut found = Constructs::default();
    for node in &function.stmts {
        match node {
            StmtNode::If(..) => found.branches += 1,
            StmtNode::While(..) | StmtNode::DoWhile(..) | StmtNode::Loop(_) | StmtNode::For(..) => {
                found.loops += 1;
            }
            StmtNode::Switch(..) => found.switches += 1,
            StmtNode::Try(..) => found.tries += 1,
            StmtNode::Goto(_) => found.gotos += 1,
            StmtNode::Labeled(..) => found.labels += 1,
            StmtNode::Break(_) => found.breaks += 1,
            StmtNode::Continue(_) => found.continues += 1,
            _ => {}
        }
    }
    found
}

#[test]
fn a_structured_graph_has_a_report_with_no_fallback() {
    for fixture in graphs::structured() {
        let (_, report) = structure(&fixture.source);
        assert!(report.is_fully_structured(), "{}: {report:?}", fixture.name);
        assert!(report.gotos.is_empty(), "{}", fixture.name);
        assert!(report.unresolved_labels.is_empty(), "{}", fixture.name);
        assert!(report.unstructured_regions.is_empty(), "{}", fixture.name);
    }
}

#[test]
fn the_tree_has_the_shape_of_each_construct() {
    let (tree, _) = structure(&graphs::counting_loop().source);
    let loops: Vec<_> = tree
        .preorder()
        .filter_map(|id| match *tree.node(id) {
            ControlNode::Loop { kind, .. } => Some(kind),
            _ => None,
        })
        .collect();
    assert!(matches!(
        loops.as_slice(),
        [LoopKind::While {
            exit_on_true: false,
            ..
        }]
    ));

    let (tree, _) = structure(&graphs::dispatching().source);
    let ControlNode::Switch {
        cases,
        default_body,
        ..
    } = tree
        .preorder()
        .map(|id| *tree.node(id))
        .find(|node| matches!(node, ControlNode::Switch { .. }))
        .expect("the graph dispatches")
    else {
        unreachable!("the node was found as a switch");
    };
    assert_eq!(
        tree.cases(cases).len(),
        2,
        "the shared arm and the single one"
    );
    assert!(!default_body.is_empty());

    let (tree, _) = structure(&graphs::guarded_division().source);
    let ControlNode::TryCatch {
        handlers,
        finally_body,
        ..
    } = tree
        .preorder()
        .map(|id| *tree.node(id))
        .find(|node| matches!(node, ControlNode::TryCatch { .. }))
        .expect("the graph has a region")
    else {
        unreachable!("the node was found as a try");
    };
    let arms = tree.handlers(handlers);
    assert_eq!(arms.len(), 1);
    assert_eq!(arms[0].kind, HandlerKind::CatchAll);
    assert!(finally_body.is_empty());
}

#[test]
fn every_lift_verifies() {
    for fixture in every_fixture() {
        let function = lifted(&fixture);
        let report = verify::<Tiny, _>(&function);
        assert!(report.is_ok(), "{}: {report}", fixture.name);
    }
}

#[test]
fn the_lifted_function_observes_what_the_cfg_observes() {
    for fixture in every_fixture() {
        let function = lifted(&fixture);
        let source = &fixture.source;
        for input in &fixture.inputs {
            let expected = run_cfg(&source.cfg, source.names.len(), &source.parameters, input);
            assert!(
                matches!(expected.end, End::Returned(_)),
                "{} {input:?}",
                fixture.name
            );
            assert_eq!(
                run_hlil(&function, input),
                expected,
                "{} on {input:?}",
                fixture.name
            );
        }
    }
}

#[test]
fn a_throwing_statement_reaches_the_handler_in_both_forms() {
    let fixture = graphs::guarded_division();
    let function = lifted(&fixture);
    let source = &fixture.source;
    let caught = run_cfg(&source.cfg, source.names.len(), &source.parameters, &[8, 0]);
    assert_eq!(caught.output, [9], "the handler ran");
    assert_eq!(run_hlil(&function, &[8, 0]), caught);
    let normal = run_cfg(&source.cfg, source.names.len(), &source.parameters, &[8, 2]);
    assert_eq!(normal.output, [4]);
    assert_eq!(run_hlil(&function, &[8, 2]), normal);
}

#[test]
fn the_constructs_of_the_tree_are_the_constructs_of_the_lifted_function() {
    for fixture in every_fixture() {
        // A loop that leaves on true has no negation in this dialect, so the
        // lift states its exit as an extra `if` and `break`; it has a test
        // of its own below.
        if fixture.name == "loop_leaving_on_true" {
            continue;
        }
        let (tree, _) = structure(&fixture.source);
        let function = lifted(&fixture);
        assert_eq!(
            tree_constructs(&tree),
            function_constructs(&function),
            "{}",
            fixture.name
        );
    }
}

#[test]
fn a_loop_that_leaves_on_true_without_a_negation_lifts_to_an_endless_loop_with_a_break() {
    let fixture = graphs::loop_leaving_on_true();
    let (tree, report) = structure(&fixture.source);
    assert!(report.is_fully_structured());
    let kinds: Vec<LoopKind> = tree
        .preorder()
        .filter_map(|id| match *tree.node(id) {
            ControlNode::Loop { kind, .. } => Some(kind),
            _ => None,
        })
        .collect();
    assert!(matches!(
        kinds.as_slice(),
        [LoopKind::While {
            exit_on_true: true,
            ..
        }]
    ));
    let function = lifted(&fixture);
    let constructs = function_constructs(&function);
    assert_eq!(constructs.loops, 1);
    assert_eq!(constructs.breaks, 1);
    assert!(
        !function
            .stmts
            .iter()
            .any(|node| matches!(node, StmtNode::While(..)))
    );
    let leaving = function.stmts.iter().find_map(|node| match node {
        StmtNode::If(_, then_body, _) => Some(then_body),
        _ => None,
    });
    let [only] = leaving.expect("the exit test is an if").as_slice() else {
        panic!("the exit test breaks and does nothing else");
    };
    assert!(matches!(function.stmts[*only], StmtNode::Break(None)));
}

#[test]
fn a_post_tested_loop_lifts_to_a_do_while() {
    let (tree, _) = structure(&graphs::do_while_loop().source);
    assert!(tree.preorder().any(|id| matches!(
        tree.node(id),
        ControlNode::Loop {
            kind: LoopKind::DoWhile { .. },
            ..
        }
    )));
    let function = lifted(&graphs::do_while_loop());
    assert!(
        function
            .stmts
            .iter()
            .any(|node| matches!(node, StmtNode::DoWhile(..)))
    );
}

#[test]
fn a_break_and_a_continue_of_the_tree_are_a_break_and_a_continue_of_the_function() {
    let fixture = graphs::loop_with_break_and_continue();
    let (tree, _) = structure(&fixture.source);
    let expected = tree_constructs(&tree);
    assert!(
        expected.breaks >= 1 && expected.continues >= 1,
        "{expected:?}"
    );
    let found = function_constructs(&lifted(&fixture));
    assert_eq!(
        (found.breaks, found.continues),
        (expected.breaks, expected.continues)
    );
}

#[test]
fn a_finally_that_both_paths_enter_is_lifted_with_a_report_that_matches_its_gotos() {
    // The normal path and the unwind both flow into the cleanup block, which
    // the handler owns. Whatever transfers the structurer emits for that, the
    // report must list each one and every label must have a node.
    let fixture = graphs::guarded_division_with_finally();
    let (tree, report) = structure(&fixture.source);
    assert_eq!(report.unresolved_labels, []);
    let function = lifted(&fixture);
    assert_eq!(function_constructs(&function).gotos, report.gotos.len());
    assert_eq!(tree_constructs(&tree).tries, 1);
    assert_eq!(function_constructs(&function).tries, 1);
    let StmtNode::Try(_, handlers, finally_body) = function
        .stmts
        .iter()
        .find(|node| matches!(node, StmtNode::Try(..)))
        .expect("the region lifts to a try")
    else {
        unreachable!("the node was found as a try");
    };
    assert_eq!(handlers.len(), 1);
    assert!(!finally_body.is_empty(), "the cleanup is the finally body");
}

#[test]
fn lift_structures_the_graph_the_way_compute_with_report_does() {
    for fixture in every_fixture() {
        let direct = lift::<Tiny, _, _>(&fixture.source, Builder::new(), LiftMetadata::Preserve)
            .unwrap_or_else(|error| panic!("{}: {error}", fixture.name));
        let (tree, report) = structure(&fixture.source);
        let through = lift_tree(&fixture.source, &tree, &report)
            .unwrap_or_else(|error| panic!("{}: {error}", fixture.name));
        assert_eq!(direct.function, through.function, "{}", fixture.name);
        assert_eq!(direct.report, through.report, "{}", fixture.name);
        assert_eq!(
            direct.correspondence, through.correspondence,
            "{}",
            fixture.name
        );
        assert_eq!(through.report, report);
    }
}

#[test]
fn an_irreducible_loop_falls_back_to_a_goto_and_keeps_its_behavior() {
    let fixture = graphs::irreducible_loop();
    let (tree, report) = structure(&fixture.source);
    assert!(!report.is_fully_structured());
    assert_eq!(report.unresolved_labels, []);
    assert!(
        report
            .gotos
            .iter()
            .any(|goto| goto.reason == GotoReason::RevisitedTarget),
        "{report:?}"
    );
    let function = lifted(&fixture);
    let constructs = function_constructs(&function);
    assert_eq!(
        constructs.gotos,
        report.gotos.len(),
        "one goto statement per diagnostic"
    );
    assert_eq!(constructs.gotos, tree_constructs(&tree).gotos);
    assert!(constructs.labels >= 1);
    assert!(verify::<Tiny, _>(&function).is_ok());
    let source = &fixture.source;
    for input in &fixture.inputs {
        assert_eq!(
            run_hlil(&function, input),
            run_cfg(&source.cfg, source.names.len(), &source.parameters, input),
            "{input:?}"
        );
    }
}

#[test]
fn a_goto_whose_label_found_no_node_is_reported_and_the_function_it_would_give_is_refused() {
    let fixture = graphs::irreducible_loop_with_an_empty_head();
    let (_, report) = structure(&fixture.source);
    assert!(!report.is_fully_structured());
    assert!(!report.unresolved_labels.is_empty(), "{report:?}");
    let (tree, report) = structure(&fixture.source);
    let error = lift_tree(&fixture.source, &tree, &report)
        .expect_err("a goto to a label that nothing defines");
    let LiftError::Builder(rejected) = &error else {
        panic!("the builder refuses the dangling goto, got {error}");
    };
    assert!(!rejected.is_ok());
}

fn assert_lifted_text(fixture: &Fixture, golden: &str) {
    assert_golden(golden, &pseudocode::<Tiny, _>(&lifted(fixture)));
}

#[test]
fn a_branch_lifts_to_an_if_with_both_arms() {
    assert_lifted_text(
        &graphs::branching(),
        "cfg-to-hlil/branch-lifted-to-if-else.pseudo",
    );
}

#[test]
fn a_pre_tested_loop_lifts_to_a_while() {
    assert_lifted_text(
        &graphs::counting_loop(),
        "cfg-to-hlil/loop-lifted-to-while.pseudo",
    );
}

#[test]
fn a_branch_in_a_loop_lifts_to_an_if_in_the_while() {
    assert_lifted_text(
        &graphs::loop_with_a_branch(),
        "cfg-to-hlil/branch-in-loop-lifted-to-if-in-while.pseudo",
    );
}

#[test]
fn a_dispatch_lifts_to_a_switch_with_a_shared_arm() {
    assert_lifted_text(
        &graphs::dispatching(),
        "cfg-to-hlil/dispatch-lifted-to-switch.pseudo",
    );
}

#[test]
fn a_region_lifts_to_a_try_with_its_catch() {
    assert_lifted_text(
        &graphs::guarded_division(),
        "cfg-to-hlil/region-lifted-to-try-catch.pseudo",
    );
}

#[test]
fn an_irreducible_loop_lifts_to_labels_and_gotos() {
    assert_lifted_text(
        &graphs::irreducible_loop(),
        "cfg-to-hlil/irreducible-loop-lifted-with-goto.pseudo",
    );
}

#[test]
fn a_post_tested_loop_prints_as_do_while() {
    assert_lifted_text(
        &graphs::do_while_loop(),
        "cfg-to-hlil/post-tested-loop-lifted-to-do-while.pseudo",
    );
}

#[test]
fn a_break_and_a_continue_print_in_a_while() {
    assert_lifted_text(
        &graphs::loop_with_break_and_continue(),
        "cfg-to-hlil/break-and-continue-lifted-in-while.pseudo",
    );
}

#[test]
fn a_loop_that_leaves_on_true_prints_its_exit_test_as_a_break() {
    assert_lifted_text(
        &graphs::loop_leaving_on_true(),
        "cfg-to-hlil/leave-on-true-lifted-with-break.pseudo",
    );
}
