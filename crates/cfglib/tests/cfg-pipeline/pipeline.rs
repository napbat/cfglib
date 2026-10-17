//! The stages composed end to end, SSA lowering, and the rendered result.

use cfglib::{
    Cfg, DominatorTree, EdgeKind, SccpAnalysis, SsaForm, alias_propagation, copy_propagation,
    remove_dead_code, verify_with,
};

use crate::check::{assert_lowering_preserves_behavior, assert_same_behavior, assert_well_formed};
use crate::golden::assert_golden;
use crate::interp::{Inputs, Outcome, Termination, run};
use crate::oracle::{Reference, has_unwind_edges};
use crate::passes::{ToyShape, destruct, fold_with_sccp};
use crate::programs::{corpus, program};
use crate::toy::{Inst, Var};

#[test]
fn lowering_ssa_after_splitting_critical_edges_preserves_behavior() {
    for case in corpus()
        .into_iter()
        .filter(|case| !has_unwind_edges(&case.cfg))
    {
        assert_lowering_preserves_behavior(&case, "construction", &case.cfg);
    }
}

#[test]
fn the_whole_pipeline_composes_and_preserves_behavior_at_every_stage() {
    for case in corpus() {
        let dominators = DominatorTree::compute(&case.cfg);
        let ssa = SsaForm::compute(&case.cfg, &dominators);
        let sccp = SccpAnalysis::compute(&case.cfg, &ssa);
        let mut current = fold_with_sccp(&case.cfg, &ssa, &sccp);
        assert_same_behavior(&case, "folding", &case.cfg, &current);

        copy_propagation(&mut current);
        assert_same_behavior(&case, "copy propagation", &case.cfg, &current);
        alias_propagation(&mut current);
        assert_same_behavior(&case, "alias propagation", &case.cfg, &current);
        remove_dead_code(&mut current);
        assert_same_behavior(&case, "dead code removal", &case.cfg, &current);
        assert_well_formed(case.name, "the pipeline", &current);

        let reference = Reference::new(&current);
        assert_eq!(
            reference.reachable.len(),
            current.block_count(),
            "{}: the cleaned program has no unreachable block",
            case.name
        );
        if !has_unwind_edges(&current) {
            assert_lowering_preserves_behavior(&case, "the pipeline", &current);
        }
    }
}

#[test]
fn the_semantic_validator_rejects_a_branch_with_a_missing_arm() {
    let mut cfg = Cfg::<Inst<Var>>::new();
    let next = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(Inst::branch("c"));
    cfg.add_edge(cfg.entry(), next, EdgeKind::ConditionalTrue);
    cfg.block_mut(next).push(Inst::ret("c"));
    let report = verify_with(&cfg, &ToyShape);
    assert!(report.structural.is_ok());
    assert_eq!(report.semantic_errors.len(), 1);
    assert_eq!(report.error_count(), 1);
}

#[test]
fn the_interpreter_reports_a_trap_when_no_handler_catches_the_division() {
    let mut cfg = Cfg::<Inst<Var>>::new();
    cfg.block_mut(cfg.entry()).push(Inst::print("a"));
    cfg.block_mut(cfg.entry()).push(Inst::divide("q", "a", "b"));
    let inputs = Inputs::from([(Var("a"), 4), (Var("b"), 0)]);
    assert_eq!(
        run(&cfg, &inputs).0,
        Outcome {
            output: vec![4],
            termination: Termination::Trapped
        }
    );
}

#[test]
fn the_constant_diamond_renders_before_and_after_the_pipeline() {
    let case = program("constant_condition_diamond");
    assert_golden(
        "cfg-pipeline/constant-diamond-before.txt",
        &case.cfg.to_text(),
    );

    let dominators = DominatorTree::compute(&case.cfg);
    let ssa = SsaForm::compute(&case.cfg, &dominators);
    let mut current = fold_with_sccp(&case.cfg, &ssa, &SccpAnalysis::compute(&case.cfg, &ssa));
    copy_propagation(&mut current);
    alias_propagation(&mut current);
    remove_dead_code(&mut current);
    assert_golden(
        "cfg-pipeline/constant-diamond-after.txt",
        &current.to_text(),
    );
}

#[test]
fn a_lowered_program_renders_its_phi_copies_over_renamed_values() {
    let case = program("minimum_diamond");
    let dominators = DominatorTree::compute(&case.cfg);
    let ssa = SsaForm::compute(&case.cfg, &dominators);
    assert_golden(
        "cfg-pipeline/minimum-diamond-lowered.txt",
        &destruct(&case.cfg, &ssa).to_text(),
    );
}
