//! Generic SSA construction: placement, renaming, and the value that
//! reaches a point.

use super::*;
use crate::builder::CfgBuilder;
use crate::edge::EdgeKind;
use crate::test_util::{DfInst, df_def, df_use};
use alloc::vec;

#[test]
fn no_phis_in_linear_cfg() {
    let cfg = CfgBuilder::build(vec![df_def("def r0", 0), df_use("use r0", 0)]).unwrap();
    let dom = DominatorTree::compute(&cfg);
    assert!(PhiPlacements::compute(&cfg, &dom).is_empty());
}

#[test]
fn phi_at_merge_point() {
    let mut cfg = Cfg::<DfInst>::new();
    let then_block = cfg.new_block();
    let else_block = cfg.new_block();
    let merge = cfg.new_block();
    cfg.add_edge(cfg.entry(), then_block, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), else_block, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_block, merge, EdgeKind::Fallthrough);
    cfg.add_edge(else_block, merge, EdgeKind::Fallthrough);
    cfg.block_mut(then_block).push(df_def("then", 0));
    cfg.block_mut(else_block).push(df_def("else", 0));
    cfg.block_mut(merge).push(df_use("use", 0));

    let dom = DominatorTree::compute(&cfg);
    let placements = PhiPlacements::compute(&cfg, &dom);
    assert_eq!(placements.len(), 1);
    assert_eq!(placements.at(merge)[0].variable, 0);
}

#[test]
fn edge_payloads_are_preserved_while_computing_ssa() {
    let mut cfg = Cfg::<DfInst, &'static str>::with_edge_payload();
    let left = cfg.new_block();
    let right = cfg.new_block();
    let merge = cfg.new_block();
    cfg.add_edge_with_payload(cfg.entry(), left, EdgeKind::ConditionalTrue, "left");
    cfg.add_edge_with_payload(cfg.entry(), right, EdgeKind::ConditionalFalse, "right");
    cfg.add_edge_with_payload(left, merge, EdgeKind::Fallthrough, "left-merge");
    cfg.add_edge_with_payload(right, merge, EdgeKind::Fallthrough, "right-merge");
    cfg.block_mut(left).push(df_def("left", 0));
    cfg.block_mut(right).push(df_def("right", 0));
    cfg.block_mut(merge).push(df_use("merged", 0));

    let dom = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dom);

    assert_eq!(ssa.block(merge).phis.len(), 1);
    assert_eq!(
        cfg.edges().map(|edge| *edge.payload()).collect::<Vec<_>>(),
        ["left", "right", "left-merge", "right-merge"]
    );
}

#[test]
fn renaming_uses_latest_definition() {
    let mut cfg = Cfg::<DfInst>::new();
    cfg.block_mut(cfg.entry()).instructions_mut().extend([
        df_def("first", 0),
        df_def("second", 0),
        df_use("use", 0),
    ]);

    let dom = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dom);
    let instructions = &ssa.block(cfg.entry()).instructions;
    assert_eq!(instructions[0].defs[0], SsaValue::new(0, 1));
    assert_eq!(instructions[1].defs[0], SsaValue::new(0, 2));
    assert_eq!(instructions[2].uses[0], SsaValue::new(0, 2));
}

#[test]
fn read_before_definition_is_live_in() {
    let cfg = CfgBuilder::build(vec![df_use("use", 7), df_def("def", 7)]).unwrap();
    let dom = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dom);
    assert_eq!(
        ssa.block(cfg.entry()).instructions[0].uses[0],
        SsaValue::live_in(7)
    );
}

#[test]
fn diamond_phi_has_renamed_result_and_operands() {
    let mut cfg = Cfg::<DfInst>::new();
    let left = cfg.new_block();
    let right = cfg.new_block();
    let merge = cfg.new_block();
    cfg.add_edge(cfg.entry(), left, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), right, EdgeKind::ConditionalFalse);
    cfg.add_edge(left, merge, EdgeKind::Fallthrough);
    cfg.add_edge(right, merge, EdgeKind::Fallthrough);
    cfg.block_mut(left).push(df_def("left", 0));
    cfg.block_mut(right).push(df_def("right", 0));
    cfg.block_mut(merge).push(df_use("merged", 0));

    let dom = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dom);
    let phi = &ssa.block(merge).phis[0];
    assert_eq!(phi.result.variable, 0);
    assert_ne!(phi.result.version, 0);
    assert_eq!(phi.operands.len(), 2);
    assert!(phi.operands.iter().all(|(_, value)| value.version != 0));
    assert_ne!(phi.operands[0].1, phi.operands[1].1);
    assert_eq!(ssa.block(merge).instructions[0].uses[0], phi.result);
}

#[test]
fn unreachable_block_is_still_annotated() {
    let mut cfg = Cfg::<DfInst>::new();
    let unreachable = cfg.new_block();
    cfg.block_mut(unreachable).push(df_def("dead", 3));
    let dom = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dom);
    assert_eq!(ssa.block(unreachable).instructions.len(), 1);
    assert_eq!(ssa.block(unreachable).instructions[0].defs[0].variable, 3);
}

#[test]
fn disconnected_definition_reaches_its_component_successor() {
    let mut cfg = Cfg::<DfInst>::new();
    let landing = cfg.new_block();
    let handler = cfg.new_block();
    cfg.add_edge(landing, handler, EdgeKind::Fallthrough);
    cfg.block_mut(landing).push(df_def("caught", 0));
    cfg.block_mut(handler).push(df_use("handler", 0));

    let dom = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dom);
    assert_eq!(
        ssa.block(handler).instructions[0].uses[0],
        ssa.block(landing).instructions[0].defs[0]
    );
}

#[test]
fn value_at_finds_the_reaching_definition_in_the_same_block() {
    let mut cfg = Cfg::<DfInst>::new();
    cfg.block_mut(cfg.entry()).instructions_mut().extend([
        df_use("read live-in", 0),
        df_def("first", 0),
        df_use("read first", 0),
        df_def("second", 0),
    ]);

    let dom = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dom);
    let at = |index| {
        ssa.value_at(
            ProgramPoint {
                block: cfg.entry(),
                inst_idx: index,
            },
            &0,
        )
    };

    assert_eq!(at(0), SsaValue::live_in(0), "no definition precedes it");
    assert_eq!(at(1), SsaValue::live_in(0), "its own definition is later");
    assert_eq!(at(2), SsaValue::new(0, 1));
    assert_eq!(at(4), SsaValue::new(0, 2), "the block end reads the last");
}

#[test]
fn value_at_reads_the_phi_and_the_dominator() {
    let mut cfg = Cfg::<DfInst>::new();
    let left = cfg.new_block();
    let right = cfg.new_block();
    let merge = cfg.new_block();
    let entry = cfg.entry();
    cfg.add_edge(entry, left, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, right, EdgeKind::ConditionalFalse);
    cfg.add_edge(left, merge, EdgeKind::Fallthrough);
    cfg.add_edge(right, merge, EdgeKind::Fallthrough);
    cfg.block_mut(entry).push(df_def("entry", 1));
    cfg.block_mut(left).push(df_def("left", 0));
    cfg.block_mut(right).push(df_def("right", 0));
    cfg.block_mut(merge).push(df_use("merged", 0));

    let dom = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dom);
    let start_of_merge = ProgramPoint {
        block: merge,
        inst_idx: 0,
    };

    assert_eq!(
        ssa.value_at(start_of_merge, &0),
        ssa.block(merge).phis[0].result,
        "the phi of the block answers before any dominator"
    );
    assert_eq!(
        ssa.value_at(start_of_merge, &1),
        ssa.block(entry).instructions[0].defs[0],
        "the immediate dominator supplies a variable the block merges not"
    );
    assert_eq!(
        ssa.value_at(
            ProgramPoint {
                block: left,
                inst_idx: 0
            },
            &0
        ),
        SsaValue::live_in(0),
        "no dominator of the arm defines it"
    );
}

#[test]
fn disconnected_component_does_not_inherit_entry_definitions() {
    let mut cfg = Cfg::<DfInst>::new();
    let disconnected = cfg.new_block();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(df_def("entry", 0));
    cfg.block_mut(disconnected).push(df_use("dead", 0));

    let dom = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dom);
    assert_eq!(
        ssa.block(disconnected).instructions[0].uses[0],
        SsaValue::live_in(0)
    );
}

/// Give every block definitions and uses over a small recycled variable set,
/// so the merges of the shared shapes actually take phis.
fn filled(mut cfg: Cfg<DfInst>) -> Cfg<DfInst> {
    for (index, block) in cfg.block_ids().collect::<Vec<_>>().into_iter().enumerate() {
        let variable = u16::try_from(index % 4).expect("a small index fits in u16");
        cfg.block_mut(block).push(df_use("use", (variable + 1) % 4));
        cfg.block_mut(block).push(df_def("def", variable));
        cfg.block_mut(block)
            .push(df_def("redef", (variable + 2) % 4));
        cfg.block_mut(block).push(df_use("read back", variable));
    }
    cfg
}

#[test]
fn one_scratch_reused_down_a_sequence_computes_the_allocating_answer() {
    let sequence: Vec<_> = crate::test_util::shapes::scratch_sequence::<DfInst>()
        .into_iter()
        .map(filled)
        .collect();
    let mut scratch = SsaScratch::new();
    // Twice, so the second pass sees a scratch every buffer of which is
    // already at the sequence's high-water mark, and every pooled vector of
    // which came back from an earlier procedure.
    for _ in 0..2 {
        for cfg in &sequence {
            let dom = DominatorTree::compute(cfg);
            assert_eq!(
                SsaForm::compute_in(&mut scratch, cfg, &dom),
                SsaForm::compute(cfg, &dom),
                "{} blocks",
                cfg.block_count()
            );
        }
    }
}

#[test]
fn a_reused_scratch_survives_a_large_procedure_before_a_small_one() {
    let large = filled(crate::test_util::shapes::diamond_chain::<DfInst>(40));
    let small = filled(crate::test_util::shapes::diamond_chain::<DfInst>(1));
    let mut scratch = SsaScratch::new();

    let large_dominators = DominatorTree::compute(&large);
    drop(SsaForm::compute_in(&mut scratch, &large, &large_dominators));
    let small_dominators = DominatorTree::compute(&small);
    assert_eq!(
        SsaForm::compute_in(&mut scratch, &small, &small_dominators),
        SsaForm::compute(&small, &small_dominators)
    );
    assert_eq!(
        SsaForm::compute_in(&mut scratch, &large, &large_dominators),
        SsaForm::compute(&large, &large_dominators)
    );
}

#[test]
fn a_reused_scratch_crosses_procedures_with_disjoint_variables() {
    // Nothing of one procedure's variables may survive into the next, which
    // is what the pooled definition lists and renaming stacks risk.
    let mut first = crate::test_util::shapes::diamond_chain::<DfInst>(2);
    let blocks: Vec<_> = first.block_ids().collect();
    for &block in &blocks {
        first.block_mut(block).push(df_def("def", 7));
    }
    let mut second = crate::test_util::shapes::diamond_chain::<DfInst>(2);
    let blocks: Vec<_> = second.block_ids().collect();
    for &block in &blocks {
        second.block_mut(block).push(df_def("def", 9));
    }

    let mut scratch = SsaScratch::new();
    for (cfg, present, absent) in [(&first, 7, 9), (&second, 9, 7), (&first, 7, 9)] {
        let dom = DominatorTree::compute(cfg);
        let form = SsaForm::compute_in(&mut scratch, cfg, &dom);
        assert_eq!(form, SsaForm::compute(cfg, &dom));
        assert!(form.max_version(&present) > 0);
        assert_eq!(
            form.max_version(&absent),
            0,
            "a variable of another procedure reached this form"
        );
    }
}

#[test]
fn parallel_edges_give_every_phi_operand_position_its_value() {
    // Two edges from one predecessor occupy two operand positions, and the
    // flat operand array has to fill both.
    let mut cfg = Cfg::<DfInst>::new();
    let source = cfg.new_block();
    let merge = cfg.new_block();
    cfg.add_edge(cfg.entry(), source, EdgeKind::Fallthrough);
    cfg.add_edge(source, merge, EdgeKind::ConditionalTrue);
    cfg.add_edge(source, merge, EdgeKind::ConditionalFalse);
    let other = cfg.new_block();
    cfg.add_edge(cfg.entry(), other, EdgeKind::ConditionalTrue);
    cfg.add_edge(other, merge, EdgeKind::Fallthrough);
    cfg.block_mut(source).push(df_def("def", 0));
    cfg.block_mut(other).push(df_def("def", 0));
    cfg.block_mut(merge).push(df_use("use", 0));

    let dom = DominatorTree::compute(&cfg);
    let form = SsaForm::compute(&cfg, &dom);
    let phi = &form.block(merge).phis[0];
    assert_eq!(phi.operands.len(), 3);
    assert!(
        phi.operands.iter().all(|(_, value)| !value.is_live_in()),
        "every operand position names the definition reaching it: {:?}",
        phi.operands
    );
}
