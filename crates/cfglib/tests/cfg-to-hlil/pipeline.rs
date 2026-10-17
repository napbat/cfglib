//! The whole chain of a decompiler front end, stage by stage: build a `Cfg`,
//! analyse it (dominators, post-dominators, SSA, SCCP, control dependence,
//! def-use), cut away what the analyses prove dead, structure what is left
//! with `ControlTree`, lift it to the structured IR, verify it, and compare
//! what the programs observe at every stage.
//!
//! The oracle for every stage is the pair of reference interpreters: the
//! program before any stage, the pruned graph, and the lifted function must
//! observe the same output and result on every input. The analyses are also
//! held to hand-derived facts about each fixture, so a pruning that happens
//! to preserve behavior for the wrong reason still fails.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::ir::hlil::{LiftMetadata, lift_with_tree, pseudocode, verify};
use cfglib::{
    BlockId, ConstValue, ConstantFolder, ControlTree, DefUseChains, DominatorTree, EdgeId,
    EdgeKind, HandlerKind, NodeId, PostDominatorTree, SccpAnalysis, SsaForm,
    control_dependence_graph, remove_unreachable, verify as verify_cfg,
};

use crate::dialect::{Form, Link, Prim, Source, Stmt, Tiny, act, konst, ret, set, test};
use crate::graphs::{self, Fixture, Graph};
use crate::machine::{Run, run_cfg, run_hlil};
use crate::storage::{Builder, Hlil};

impl ConstantFolder for Stmt {
    type Const = i64;

    fn fold_constant(&self, known: &BTreeMap<usize, i64>) -> Option<(usize, i64)> {
        let destination = *self.defs.first()?;
        match self.form {
            Form::Const(value) => Some((destination, value)),
            Form::Apply(prim) => {
                let operands: Vec<i64> = self
                    .uses
                    .iter()
                    .map(|variable| known.get(variable).copied())
                    .collect::<Option<_>>()?;
                match prim {
                    Prim::Plus => Some((destination, operands[0].wrapping_add(operands[1]))),
                    Prim::Less => Some((destination, i64::from(operands[0] < operands[1]))),
                    Prim::Quot if operands[1] != 0 => {
                        Some((destination, operands[0].wrapping_div(operands[1])))
                    }
                    Prim::Quot | Prim::Show => None,
                }
            }
            Form::Test | Form::Dispatch | Form::Ret => None,
        }
    }

    fn fold_branch(&self, known: &BTreeMap<usize, i64>) -> Option<bool> {
        if self.form != Form::Test {
            return None;
        }
        known.get(&self.uses[0]).map(|condition| *condition != 0)
    }
}

/// `show(a)` or `show(b)` on a branch whose condition is a constant `1 < 2`,
/// with each arm computing a constant that reaches the join.
fn folded_guard() -> Fixture {
    let mut graph = Graph::new(&["a", "b", "one", "two", "c", "r"], 2);
    let head = graph.entry(vec![
        konst(2, 1),
        konst(3, 2),
        set(4, Prim::Less, &[2, 3]),
        test(4),
    ]);
    let then_block = graph.block(vec![act(Prim::Show, &[0]), set(5, Prim::Plus, &[2, 3])]);
    let else_block = graph.block(vec![act(Prim::Show, &[1]), set(5, Prim::Plus, &[3, 3])]);
    let join = graph.block(vec![ret(5)]);
    graph.edge(head, then_block, Link::True);
    graph.edge(head, else_block, Link::False);
    graph.edge(then_block, join, Link::Next);
    graph.edge(else_block, join, Link::Next);
    Fixture {
        name: "folded_guard",
        source: graph.finish(),
        inputs: vec![vec![3, 4], vec![0, 0], vec![-5, 9]],
    }
}

/// A constant outer guard around a branch that depends on the inputs: the
/// outer test folds away and the inner one must stay.
fn live_branch_in_a_folded_guard() -> Fixture {
    let mut graph = Graph::new(&["a", "b", "t", "c", "r", "nine"], 2);
    let head = graph.entry(vec![konst(2, 1), konst(5, 9), test(2)]);
    let inner = graph.block(vec![set(3, Prim::Less, &[0, 1]), test(3)]);
    let small = graph.block(vec![act(Prim::Show, &[0]), set(4, Prim::Plus, &[0, 0])]);
    let large = graph.block(vec![act(Prim::Show, &[1]), set(4, Prim::Plus, &[1, 1])]);
    let dead = graph.block(vec![act(Prim::Show, &[5]), set(4, Prim::Plus, &[5, 5])]);
    let join = graph.block(vec![ret(4)]);
    graph.edge(head, inner, Link::True);
    graph.edge(head, dead, Link::False);
    graph.edge(inner, small, Link::True);
    graph.edge(inner, large, Link::False);
    graph.edge(small, join, Link::Next);
    graph.edge(large, join, Link::Next);
    graph.edge(dead, join, Link::Next);
    Fixture {
        name: "live_branch_in_a_folded_guard",
        source: graph.finish(),
        inputs: vec![vec![1, 2], vec![2, 1], vec![4, 4]],
    }
}

/// `if (1 < 0) { try { q = a / 0 } catch { show(9) } } else { q = a + 1 }`:
/// the guard is constant false, so the only block that can throw, its
/// handler, and the whole region are dead.
fn dead_try_behind_a_folded_guard() -> Fixture {
    let mut graph = Graph::new(&["a", "zero", "one", "nine", "c", "q"], 1);
    let head = graph.entry(vec![
        konst(1, 0),
        konst(2, 1),
        konst(3, 9),
        set(4, Prim::Less, &[2, 1]),
        test(4),
    ]);
    let guarded = graph.block(vec![set(5, Prim::Quot, &[0, 1])]);
    let handler = graph.block(vec![act(Prim::Show, &[3])]);
    let other = graph.block(vec![set(5, Prim::Plus, &[0, 2])]);
    let join = graph.block(vec![act(Prim::Show, &[5]), ret(5)]);
    graph.edge(head, guarded, Link::True);
    graph.edge(head, other, Link::False);
    graph.edge(guarded, join, Link::Next);
    graph.edge(guarded, handler, Link::Unwind);
    graph.edge(handler, join, Link::Next);
    graph.edge(other, join, Link::Next);
    graph.region(&[guarded], &[(handler, HandlerKind::CatchAll, &[handler])]);
    Fixture {
        name: "dead_try_behind_a_folded_guard",
        source: graph.finish(),
        inputs: vec![vec![5], vec![0], vec![-3]],
    }
}

/// A division that can really throw, next to a folded guard: the dead arm
/// goes, the region and its handler stay.
fn live_try_beside_a_folded_guard() -> Fixture {
    let mut graph = Graph::new(&["a", "b", "zero", "one", "nine", "c", "q"], 2);
    let head = graph.entry(vec![
        konst(2, 0),
        konst(3, 1),
        konst(4, 9),
        set(5, Prim::Less, &[3, 2]),
        test(5),
    ]);
    let dead = graph.block(vec![act(Prim::Show, &[0])]);
    let divide = graph.block(vec![set(6, Prim::Quot, &[0, 1]), act(Prim::Show, &[6])]);
    let handler = graph.block(vec![act(Prim::Show, &[4])]);
    let join = graph.block(vec![ret(6)]);
    graph.edge(head, dead, Link::True);
    graph.edge(head, divide, Link::False);
    graph.edge(dead, join, Link::Next);
    graph.edge(divide, join, Link::Next);
    graph.edge(divide, handler, Link::Unwind);
    graph.edge(handler, join, Link::Next);
    graph.region(&[divide], &[(handler, HandlerKind::CatchAll, &[handler])]);
    Fixture {
        name: "live_try_beside_a_folded_guard",
        source: graph.finish(),
        inputs: vec![vec![8, 2], vec![8, 0], vec![0, 0]],
    }
}

/// What every stage left behind for one fixture.
struct Pipeline {
    original: Source,
    pruned: Source,
    sccp: SccpAnalysis<usize, i64>,
    ssa: SsaForm<usize>,
    /// The blocks, in the original numbering, SCCP proves never execute.
    dead_blocks: BTreeSet<BlockId>,
    /// The blocks whose conditional transfer the constants decide.
    folded_tests: BTreeSet<BlockId>,
}

fn run_pipeline(source: &Source) -> Pipeline {
    let mut cfg = source.cfg.clone();
    let dominators = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dominators);
    let sccp = SccpAnalysis::compute(&cfg, &ssa);

    let dead_blocks: BTreeSet<BlockId> = cfg
        .block_ids()
        .filter(|block| !sccp.reachable_blocks.contains(block))
        .collect();
    let dead_edges: Vec<EdgeId> = cfg
        .edges()
        .filter(|edge| {
            sccp.reachable_blocks.contains(&edge.source())
                && !sccp
                    .executable_edges
                    .contains(&(edge.source(), edge.target()))
        })
        .map(|edge| edge.id())
        .collect();
    let folded_tests: BTreeSet<BlockId> = dead_edges
        .iter()
        .map(|&edge| cfg.edge(edge).source())
        .filter(|&block| {
            cfg.block(block)
                .instructions()
                .last()
                .is_some_and(|last| last.form == Form::Test)
        })
        .collect();

    for edge in dead_edges {
        assert!(cfg.remove_edge(edge));
    }
    for &block in &folded_tests {
        let test = cfg.block_mut(block).instructions_mut().pop();
        assert_eq!(test.map(|stmt| stmt.form), Some(Form::Test));
        let remaining: Vec<EdgeId> = cfg.outgoing(block).collect();
        let taken: Vec<EdgeId> = remaining
            .into_iter()
            .filter(|&edge| {
                matches!(
                    cfg.edge(edge).kind(),
                    EdgeKind::ConditionalTrue | EdgeKind::ConditionalFalse
                )
            })
            .collect();
        assert_eq!(taken.len(), 1, "a decided test keeps exactly one arm");
        let edge = cfg.edge_mut(taken[0]);
        edge.set_kind(EdgeKind::Fallthrough);
        *edge.payload_mut() = Link::Next;
    }
    // A region whose handlers all became unreachable goes first, so no region
    // names a block the sweep is about to remove.
    cfg.remove_unreachable_regions();
    remove_unreachable(&mut cfg);
    cfg.compact();

    Pipeline {
        original: Source {
            cfg: source.cfg.clone(),
            names: source.names.clone(),
            parameters: source.parameters.clone(),
        },
        pruned: Source {
            cfg,
            names: source.names.clone(),
            parameters: source.parameters.clone(),
        },
        sccp,
        ssa,
        dead_blocks,
        folded_tests,
    }
}

fn observe_cfg(source: &Source, input: &[i64]) -> Run {
    run_cfg(&source.cfg, source.names.len(), &source.parameters, input)
}

fn lift(source: &Source) -> (Hlil, bool) {
    let (tree, report) = ControlTree::compute_with_report(&source.cfg);
    let lifted = lift_with_tree::<Tiny, _, _>(
        source,
        &tree,
        &report,
        Builder::new(),
        LiftMetadata::Preserve,
    )
    .unwrap_or_else(|error| panic!("the pruned graph lifts: {error}"));
    (lifted.function, report.is_fully_structured())
}

fn folding_fixtures() -> Vec<(Fixture, usize)> {
    vec![
        (folded_guard(), 1),
        (live_branch_in_a_folded_guard(), 1),
        (dead_try_behind_a_folded_guard(), 2),
        (live_try_beside_a_folded_guard(), 1),
    ]
}

#[test]
fn constants_prove_exactly_the_dead_blocks_of_each_fixture() {
    for (fixture, expected_dead) in folding_fixtures() {
        let pipeline = run_pipeline(&fixture.source);
        assert_eq!(
            pipeline.dead_blocks.len(),
            expected_dead,
            "{}",
            fixture.name
        );
        assert_eq!(
            pipeline.folded_tests.len(),
            1,
            "{}: one decided test",
            fixture.name
        );
    }
}

#[test]
fn no_stage_changes_what_the_program_observes() {
    for (fixture, _) in folding_fixtures() {
        let pipeline = run_pipeline(&fixture.source);
        let (lifted, _) = lift(&pipeline.pruned);
        for input in &fixture.inputs {
            let expected = observe_cfg(&pipeline.original, input);
            assert_eq!(
                observe_cfg(&pipeline.pruned, input),
                expected,
                "{} pruned, {input:?}",
                fixture.name
            );
            assert_eq!(
                run_hlil(&lifted, input),
                expected,
                "{} lifted, {input:?}",
                fixture.name
            );
        }
    }
}

#[test]
fn the_pruned_graph_is_whole_and_fully_structured() {
    for (fixture, _) in folding_fixtures() {
        let pipeline = run_pipeline(&fixture.source);
        let cfg = &pipeline.pruned.cfg;
        assert!(verify_cfg(cfg).is_ok(), "{}", fixture.name);
        assert!(cfg.is_compact(), "{}", fixture.name);
        let dominators = DominatorTree::compute(cfg);
        assert!(
            cfg.block_ids().all(|block| dominators.is_reachable(block)),
            "{}: nothing unreachable is left",
            fixture.name
        );
        assert_eq!(
            cfg.block_count(),
            pipeline.original.cfg.block_count() - pipeline.dead_blocks.len(),
            "{}: exactly the dead blocks went",
            fixture.name
        );
        let (lifted, structured) = lift(&pipeline.pruned);
        assert!(
            structured,
            "{}: a tree of if and try needs no goto",
            fixture.name
        );
        let report = verify::<Tiny, _>(&lifted);
        assert!(report.is_ok(), "{}: {report}", fixture.name);
    }
}

#[test]
fn a_decided_branch_leaves_no_branch_and_an_undecided_one_stays() {
    let folded = run_pipeline(&folded_guard().source);
    let (lifted, _) = lift(&folded.pruned);
    let text = pseudocode::<Tiny, _>(&lifted);
    assert!(
        !text.contains("if ("),
        "the only branch was decided:\n{text}"
    );

    let nested = run_pipeline(&live_branch_in_a_folded_guard().source);
    let (lifted, _) = lift(&nested.pruned);
    let text = pseudocode::<Tiny, _>(&lifted);
    assert_eq!(
        text.matches("if (").count(),
        1,
        "the input-dependent branch stays:\n{text}"
    );
}

#[test]
fn a_region_whose_handlers_are_dead_leaves_no_try_and_a_live_one_stays() {
    let dead = run_pipeline(&dead_try_behind_a_folded_guard().source);
    let (lifted, _) = lift(&dead.pruned);
    let text = pseudocode::<Tiny, _>(&lifted);
    assert!(
        !text.contains("try {"),
        "no block that can throw is left:\n{text}"
    );

    let live = run_pipeline(&live_try_beside_a_folded_guard().source);
    let (lifted, _) = lift(&live.pruned);
    let text = pseudocode::<Tiny, _>(&lifted);
    assert!(
        text.contains("try {"),
        "the division can still throw:\n{text}"
    );
}

#[test]
fn a_value_that_reaches_a_join_only_through_the_live_arm_is_a_constant() {
    let pipeline = run_pipeline(&folded_guard().source);
    let join = BlockId::from_raw(3);
    let phi = &pipeline.ssa.block(join).phis[0];
    assert_eq!(phi.result.variable, 5);
    assert_eq!(
        pipeline.sccp.values.get(&phi.result),
        Some(&ConstValue::Const(3)),
        "the arm that never runs must not spoil the constant"
    );
}

#[test]
fn an_induction_variable_is_never_folded() {
    for fixture in [
        graphs::counting_loop(),
        graphs::branching(),
        graphs::loop_with_a_branch(),
    ] {
        let pipeline = run_pipeline(&fixture.source);
        assert!(
            pipeline.dead_blocks.is_empty(),
            "{}: nothing is provably dead",
            fixture.name
        );
        assert!(pipeline.folded_tests.is_empty(), "{}", fixture.name);
        assert_eq!(
            pipeline.pruned.cfg.block_count(),
            fixture.source.cfg.block_count(),
            "{}",
            fixture.name
        );
        let (lifted, _) = lift(&pipeline.pruned);
        for input in &fixture.inputs {
            assert_eq!(
                run_hlil(&lifted, input),
                observe_cfg(&fixture.source, input),
                "{} on {input:?}",
                fixture.name
            );
        }
    }
}

#[test]
fn every_dead_block_is_control_dependent_on_a_decided_test() {
    for (fixture, _) in folding_fixtures() {
        let pipeline = run_pipeline(&fixture.source);
        let cfg = &pipeline.original.cfg;
        let post_dominators = PostDominatorTree::compute(cfg);
        let dependences = control_dependence_graph(cfg, &post_dominators);

        let mut influenced: BTreeSet<BlockId> = pipeline.folded_tests.clone();
        let mut pending: Vec<BlockId> = influenced.iter().copied().collect();
        while let Some(block) = pending.pop() {
            let node = NodeId::from_index(block.index());
            for dependent in dependences.successors(node) {
                let dependent = dependences[dependent];
                if influenced.insert(dependent) {
                    pending.push(dependent);
                }
            }
        }
        for dead in &pipeline.dead_blocks {
            assert!(
                influenced.contains(dead),
                "{}: {dead} is dead but no decided test controls it",
                fixture.name
            );
        }
    }
}

#[test]
fn what_survives_still_has_its_definitions() {
    for (fixture, _) in folding_fixtures() {
        let pipeline = run_pipeline(&fixture.source);
        let chains = DefUseChains::compute(&pipeline.pruned.cfg);
        for (use_site, definitions) in &chains.use_def {
            let statement =
                &pipeline.pruned.cfg.block(use_site.block).instructions()[use_site.inst_idx];
            let is_parameter_read = statement
                .uses
                .iter()
                .any(|variable| pipeline.pruned.parameters.contains(variable));
            assert!(
                !definitions.is_empty() || is_parameter_read,
                "{}: the read at {use_site:?} has no definition",
                fixture.name
            );
        }
    }
}
