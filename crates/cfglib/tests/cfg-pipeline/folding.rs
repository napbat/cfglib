//! SCCP and constant propagation: soundness against every execution, and
//! precision on programs whose constants are known by inspection.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::{ConstValue, ConstantPropagation, DominatorTree, SccpAnalysis, SsaForm, SsaValue};

use crate::check::{assert_same_behavior, assert_well_formed, block_holding};
use crate::interp::{Inputs, Termination, input_vectors, run, run_ssa};
use crate::oracle::endpoints;
use crate::passes::fold_with_sccp;
use crate::programs::{corpus, program};
use crate::toy::{Op, Var};

#[test]
fn sccp_never_declares_an_executed_block_or_edge_dead() {
    for case in corpus() {
        let dominators = DominatorTree::compute(&case.cfg);
        let ssa = SsaForm::compute(&case.cfg, &dominators);
        let sccp = SccpAnalysis::compute(&case.cfg, &ssa);
        for block in &sccp.reachable_blocks {
            assert!(
                dominators.is_reachable(*block),
                "{}: {block:?} is not even reachable",
                case.name
            );
        }
        let executable = endpoints(&case.cfg, &sccp.executable_edges);
        for (source, _) in &executable {
            assert!(
                sccp.reachable_blocks.contains(source),
                "{}: edge from dead {source:?}",
                case.name
            );
        }
        for inputs in input_vectors(case.inputs) {
            let (_, trace) = run(&case.cfg, &inputs);
            assert!(
                trace.blocks.is_subset(&sccp.reachable_blocks),
                "{}: {inputs:?} runs a block SCCP calls unreachable",
                case.name
            );
            assert!(
                trace.edges.is_subset(&executable),
                "{}: {inputs:?} takes an edge SCCP calls unexecutable",
                case.name
            );
        }
    }
}

#[test]
fn sccp_constants_hold_in_every_execution_and_every_executed_value_was_visited() {
    for case in corpus() {
        let dominators = DominatorTree::compute(&case.cfg);
        let ssa = SsaForm::compute(&case.cfg, &dominators);
        let sccp = SccpAnalysis::compute(&case.cfg, &ssa);
        let mut seen: BTreeMap<SsaValue<Var>, BTreeSet<i64>> = BTreeMap::new();
        for inputs in input_vectors(case.inputs) {
            for (value, held) in run_ssa(&case.cfg, &ssa, &inputs).1 {
                seen.entry(value).or_default().extend(held);
            }
        }
        for (value, held) in &seen {
            match sccp.values.get(value) {
                Some(ConstValue::Const(constant)) => assert_eq!(
                    held,
                    &BTreeSet::from([*constant]),
                    "{}: {value:?} is claimed constant",
                    case.name
                ),
                Some(ConstValue::Bottom) => {}
                other if value.is_live_in() => {
                    assert!(
                        other.is_none(),
                        "{}: live-in {value:?} is {other:?}",
                        case.name
                    );
                }
                other => panic!(
                    "{}: executed {value:?} has lattice value {other:?}",
                    case.name
                ),
            }
        }
    }
}

#[test]
fn constant_propagation_facts_hold_at_every_block_boundary_of_every_execution() {
    for case in corpus() {
        let facts = ConstantPropagation::compute(&case.cfg);
        for inputs in input_vectors(case.inputs) {
            let (_, trace) = run(&case.cfg, &inputs);
            let boundaries = trace
                .entry_states
                .iter()
                .map(|(block, state)| (facts.fact_in(*block), block, state))
                .chain(
                    trace
                        .exit_states
                        .iter()
                        .map(|(block, state)| (facts.fact_out(*block), block, state)),
                );
            for (fact, block, state) in boundaries {
                for (variable, lattice) in fact {
                    if let (ConstValue::Const(constant), Some(actual)) =
                        (lattice, state.get(variable))
                    {
                        assert_eq!(
                            actual, constant,
                            "{}: {variable:?} at {block:?} on {inputs:?}",
                            case.name
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn sccp_folding_preserves_behavior_and_structure() {
    for case in corpus() {
        let dominators = DominatorTree::compute(&case.cfg);
        let ssa = SsaForm::compute(&case.cfg, &dominators);
        let folded = fold_with_sccp(&case.cfg, &ssa, &SccpAnalysis::compute(&case.cfg, &ssa));
        assert_same_behavior(&case, "SCCP folding", &case.cfg, &folded);
        assert_well_formed(case.name, "SCCP folding", &folded);
    }
}

#[test]
fn sccp_proves_the_constant_diamond_and_prunes_its_untaken_arm() {
    let case = program("constant_condition_diamond");
    let dominators = DominatorTree::compute(&case.cfg);
    let ssa = SsaForm::compute(&case.cfg, &dominators);
    let sccp = SccpAnalysis::compute(&case.cfg, &ssa);

    let untaken = block_holding(&case.cfg, Op::Mul);
    let taken = block_holding(&case.cfg, Op::Add);
    assert!(sccp.reachable_blocks.contains(&taken));
    assert!(!sccp.reachable_blocks.contains(&untaken));
    assert!(!endpoints(&case.cfg, &sccp.executable_edges).contains(&(case.cfg.entry(), untaken)));

    let merge = block_holding(&case.cfg, Op::Print);
    let [phi] = ssa.block(merge).phis.as_slice() else {
        panic!("one variable merges at the join");
    };
    assert_eq!(sccp.values[&phi.result], ConstValue::Const(7));
    let condition = ssa.block(case.cfg.entry()).instructions[2].defs()[0].clone();
    assert_eq!(sccp.values[&condition], ConstValue::Const(1));
}

#[test]
fn sccp_keeps_a_value_merged_from_input_dependent_arms_non_constant() {
    let case = program("minimum_diamond");
    let dominators = DominatorTree::compute(&case.cfg);
    let ssa = SsaForm::compute(&case.cfg, &dominators);
    let sccp = SccpAnalysis::compute(&case.cfg, &ssa);
    assert_eq!(sccp.reachable_blocks.len(), case.cfg.block_count());
    let merge = block_holding(&case.cfg, Op::Print);
    let [phi] = ssa.block(merge).phis.as_slice() else {
        panic!("one variable merges at the join");
    };
    assert_eq!(sccp.values[&phi.result], ConstValue::Bottom);
}

#[test]
fn a_throwing_division_by_a_known_zero_is_never_folded() {
    let case = program("constant_zero_divisor");
    let dominators = DominatorTree::compute(&case.cfg);
    let ssa = SsaForm::compute(&case.cfg, &dominators);
    let sccp = SccpAnalysis::compute(&case.cfg, &ssa);
    let quotient = ssa.block(case.cfg.entry()).instructions[1].defs()[0].clone();
    assert_eq!(sccp.values[&quotient], ConstValue::Bottom);
    let folded = fold_with_sccp(&case.cfg, &ssa, &sccp);
    assert_eq!(
        folded.block(case.cfg.entry()).instructions()[1].op,
        Op::Divide
    );
    let (outcome, _) = run(&folded, &Inputs::from([(Var("a"), 7)]));
    assert_eq!(outcome.output, [-1]);
    assert_eq!(outcome.termination, Termination::Returned(-1));
}

#[test]
fn sccp_with_known_live_ins_folds_the_branch_they_decide() {
    let case = program("minimum_diamond");
    let dominators = DominatorTree::compute(&case.cfg);
    let ssa = SsaForm::compute(&case.cfg, &dominators);
    let else_arm = block_holding(&case.cfg, Op::Marker);
    let merge = block_holding(&case.cfg, Op::Print);
    let [phi] = ssa.block(merge).phis.as_slice() else {
        panic!("one variable merges at the join");
    };

    // With nothing known about the inputs both arms are live.
    let unknown = SccpAnalysis::compute_seeded(&case.cfg, &ssa, &BTreeMap::new());
    assert!(unknown.reachable_blocks.contains(&else_arm));
    assert_eq!(unknown.values[&phi.result], ConstValue::Bottom);

    // a = 1 and b = 5 decide `a < b`: only the true arm runs, and the merged
    // value is the constant that arm copies.
    let known = BTreeMap::from([(Var("a"), 1), (Var("b"), 5)]);
    let seeded = SccpAnalysis::compute_seeded(&case.cfg, &ssa, &known);
    assert!(!seeded.reachable_blocks.contains(&else_arm));
    assert_eq!(seeded.values[&phi.result], ConstValue::Const(1));
    assert_eq!(
        seeded.values[&SsaValue::live_in(Var("a"))],
        ConstValue::Const(1)
    );

    // One known input is not enough to decide the comparison.
    let partial = SccpAnalysis::compute_seeded(&case.cfg, &ssa, &BTreeMap::from([(Var("a"), 1)]));
    assert!(partial.reachable_blocks.contains(&else_arm));

    // The seeded verdicts are sound for the executions the seeds describe.
    let inputs = Inputs::from([(Var("a"), 1), (Var("b"), 5)]);
    let (_, trace) = run(&case.cfg, &inputs);
    assert!(trace.blocks.is_subset(&seeded.reachable_blocks));
    assert!(
        trace
            .edges
            .is_subset(&endpoints(&case.cfg, &seeded.executable_edges))
    );
}
