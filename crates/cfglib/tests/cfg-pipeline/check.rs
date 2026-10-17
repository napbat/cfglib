//! Assertions every stage of the pipeline shares: same behavior, still
//! well formed, and lowering to copies preserves behavior.

use cfglib::{BlockId, Cfg, DominatorTree, SsaForm, split_critical_edges, verify, verify_with};

use crate::interp::{Inputs, input_vectors, run};
use crate::oracle::{Reference, ssa_violations};
use crate::passes::{ToyShape, destruct, lowered_inputs};
use crate::programs::Case;
use crate::toy::{Inst, Op, Var};

pub fn assert_same_behavior(
    case: &Case,
    stage: &str,
    before: &Cfg<Inst<Var>>,
    after: &Cfg<Inst<Var>>,
) {
    for inputs in input_vectors(case.inputs) {
        assert_eq!(
            run(after, &inputs).0,
            run(before, &inputs).0,
            "{}: {stage} changed the behavior on {inputs:?}",
            case.name
        );
    }
}

pub fn assert_well_formed(name: &str, stage: &str, cfg: &Cfg<Inst<Var>>) {
    let structural = verify(cfg);
    assert!(
        structural.is_ok(),
        "{name}: {stage} broke the structure: {structural:?}"
    );
    let semantic = verify_with(cfg, &ToyShape);
    assert!(
        semantic.is_ok(),
        "{name}: {stage} broke the toy shape: {:?}",
        semantic.semantic_errors
    );
}

pub fn pipeline_inputs(names: &[&'static str]) -> Vec<Inputs> {
    input_vectors(names)
}

pub fn block_holding(cfg: &Cfg<Inst<Var>>, op: Op) -> BlockId {
    cfg.block_ids()
        .find(|block| {
            cfg.block(*block)
                .instructions()
                .iter()
                .any(|inst| inst.op == op)
        })
        .unwrap_or_else(|| panic!("no block holds {op:?}"))
}

pub fn assert_lowering_preserves_behavior(case: &Case, stage: &str, candidate: &Cfg<Inst<Var>>) {
    let mut split = candidate.clone();
    split_critical_edges(&mut split);
    let dominators = DominatorTree::compute(&split);
    let ssa = SsaForm::compute(&split, &dominators);
    let violations = ssa_violations(&split, &ssa, &Reference::new(&split));
    assert!(
        violations.is_empty(),
        "{}: SSA after {stage}: {violations:#?}",
        case.name
    );
    let lowered = destruct(&split, &ssa);
    assert!(
        verify(&lowered).is_ok(),
        "{}: lowered program after {stage}",
        case.name
    );
    for inputs in pipeline_inputs(case.inputs) {
        assert_eq!(
            run(&lowered, &lowered_inputs(&split, &inputs)).0,
            run(&case.cfg, &inputs).0,
            "{}: lowering after {stage} on {inputs:?}",
            case.name
        );
    }
}
