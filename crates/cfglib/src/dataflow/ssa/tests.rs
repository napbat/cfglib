//! Generic SSA construction: placement, renaming, and the value that
//! reaches a point.

use super::*;
use crate::builder::CfgBuilder;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::test_util::{DfInst, df_def, df_ff, df_throwing, df_use, protected_region};
use alloc::vec;

fn departure(block: BlockId, inst_idx: usize) -> ProgramPoint {
    ProgramPoint { block, inst_idx }
}

#[test]
fn a_handler_reads_the_definition_before_the_faulting_instruction() {
    let (mut cfg, [protected, normal, handler]) =
        protected_region([df_throwing(df_def("load", 0))]);
    cfg.block_mut(cfg.entry()).push(df_def("prior", 0));
    cfg.block_mut(normal).push(df_use("success", 0));
    cfg.block_mut(handler).push(df_use("catch", 0));

    let ssa = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
    let prior = ssa.block(cfg.entry()).instructions[0].defs()[0].clone();
    let loaded = &ssa.block(protected).instructions[0].defs()[0];
    assert_eq!(&ssa.block(normal).instructions[0].uses()[0], loaded);
    let [phi] = ssa.block(handler).phis.as_slice() else {
        panic!("the handler merges the value before the load");
    };
    assert_eq!(phi.operands(), [(departure(protected, 0), prior)]);
    assert_eq!(ssa.block(handler).instructions[0].uses()[0], phi.result);

    let live = crate::Liveness::compute(&cfg);
    assert!(live.live_in(protected).contains(&0));
    assert!(live.live_before_instructions(&cfg, protected)[0].contains(&0));
    assert!(live.live_after_instructions(&cfg, cfg.entry())[0].contains(&0));
}

#[test]
fn multiple_unwind_points_keep_their_distinct_completed_definitions() {
    let (mut cfg, [protected, normal, handler]) =
        protected_region(["first load", "second load"].map(|name| df_throwing(df_def(name, 0))));
    cfg.block_mut(cfg.entry()).push(df_def("prior", 0));
    cfg.block_mut(handler).push(df_use("caught", 0));
    cfg.block_mut(normal).push(df_use("completed", 0));

    let ssa = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
    let phi = &ssa.block(handler).phis[0];
    let prior = ssa.block(cfg.entry()).instructions[0].defs()[0].clone();
    let first = ssa.block(protected).instructions[0].defs()[0].clone();
    assert_eq!(
        phi.operands(),
        [
            (departure(protected, 0), prior),
            (departure(protected, 1), first),
        ]
    );
    assert_eq!(ssa.block(handler).instructions[0].uses()[0], phi.result);
    assert_eq!(
        ssa.block(normal).instructions[0].uses()[0],
        ssa.block(protected).instructions[1].defs()[0]
    );
    let live = crate::Liveness::compute(&cfg);
    assert!(
        live.live_before_instructions(&cfg, protected)
            .iter()
            .all(|set| set.contains(&0))
    );
}

#[test]
fn a_definition_before_every_throw_reaches_the_handler_without_a_phi() {
    let (mut cfg, [protected, normal, handler]) =
        protected_region([df_def("completed", 0), df_throwing(df_ff("call"))]);
    cfg.block_mut(normal).push(df_use("success", 0));
    cfg.block_mut(handler).push(df_use("catch", 0));

    let ssa = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
    let completed = &ssa.block(protected).instructions[0].defs()[0];
    assert!(
        ssa.block(handler).phis.is_empty(),
        "every throw already observes the one definition"
    );
    assert_eq!(&ssa.block(handler).instructions[0].uses()[0], completed);
    assert_eq!(&ssa.block(normal).instructions[0].uses()[0], completed);
}

#[test]
fn a_join_of_normal_and_unwind_paths_merges_their_distinct_values() {
    let (mut cfg, [protected, normal, handler]) = protected_region([
        df_def("before call", 0),
        df_throwing(df_ff("call")),
        df_def("after call", 0),
    ]);
    let join = cfg.new_block();
    cfg.add_edge(normal, join, EdgeKind::Fallthrough);
    cfg.add_edge(handler, join, EdgeKind::Fallthrough);
    cfg.block_mut(join).push(df_use("merged", 0));

    let ssa = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
    let before = ssa.block(protected).instructions[0].defs()[0].clone();
    let after = ssa.block(protected).instructions[2].defs()[0].clone();
    let [unwound] = ssa.block(handler).phis.as_slice() else {
        panic!("the definition after the call needs a handler phi");
    };
    assert_eq!(unwound.operands(), [(departure(protected, 1), before)]);
    let [merged] = ssa.block(join).phis.as_slice() else {
        panic!("the join merges the normal and the unwind value");
    };
    assert_eq!(
        merged.operands(),
        [
            (departure(normal, 0), after),
            (departure(handler, 0), unwound.result.clone()),
        ]
    );
    assert_eq!(ssa.block(join).instructions[0].uses()[0], merged.result);
}

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
    assert_eq!(instructions[0].defs()[0], SsaValue::new(0, 1));
    assert_eq!(instructions[1].defs()[0], SsaValue::new(0, 2));
    assert_eq!(instructions[2].uses()[0], SsaValue::new(0, 2));
}

#[test]
fn read_before_definition_is_live_in() {
    let cfg = CfgBuilder::build(vec![df_use("use", 7), df_def("def", 7)]).unwrap();
    let dom = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dom);
    assert_eq!(
        ssa.block(cfg.entry()).instructions[0].uses()[0],
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
    assert_eq!(phi.operands().len(), 2);
    assert!(phi.operands().iter().all(|(_, value)| value.version != 0));
    assert_ne!(phi.operands()[0].1, phi.operands()[1].1);
    assert_eq!(ssa.block(merge).instructions[0].uses()[0], phi.result);
}

#[test]
fn unreachable_block_is_still_annotated() {
    let mut cfg = Cfg::<DfInst>::new();
    let unreachable = cfg.new_block();
    cfg.block_mut(unreachable).push(df_def("dead", 3));
    let dom = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dom);
    assert_eq!(ssa.block(unreachable).instructions.len(), 1);
    assert_eq!(ssa.block(unreachable).instructions[0].defs()[0].variable, 3);
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
        ssa.block(handler).instructions[0].uses()[0],
        ssa.block(landing).instructions[0].defs()[0]
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
        ssa.block(entry).instructions[0].defs()[0],
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
        ssa.block(disconnected).instructions[0].uses()[0],
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
    assert_eq!(phi.operands().len(), 3);
    assert!(
        phi.operands().iter().all(|(_, value)| !value.is_live_in()),
        "every operand position names the definition reaching it: {:?}",
        phi.operands()
    );
}

#[test]
fn an_instruction_wider_than_its_inline_storage_keeps_every_value_in_order() {
    // More uses and definitions than an instruction record holds inline, with
    // a repeated variable on each side.
    let mut cfg = Cfg::<DfInst>::new();
    let entry = cfg.entry();
    for variable in 0..5 {
        cfg.block_mut(entry).push(df_def("seed", variable));
    }
    cfg.block_mut(entry).push(DfInst {
        uses: vec![4, 2, 2, 0, 3, 1],
        defs: vec![1, 0, 1, 7],
        ..df_ff("wide")
    });

    let form = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
    let instructions = &form.block(entry).instructions;
    let wide = &instructions[5];
    let seeded = |variable: u16| instructions[usize::from(variable)].defs()[0].clone();
    assert_eq!(
        wide.uses(),
        [
            seeded(4),
            seeded(2),
            seeded(2),
            seeded(0),
            seeded(3),
            seeded(1)
        ]
    );
    assert_eq!(
        wide.defs(),
        [
            SsaValue::new(1, 2),
            SsaValue::new(0, 2),
            SsaValue::new(1, 3),
            SsaValue::new(7, 1),
        ]
    );
    assert_eq!(form.max_version(&1), 3);
}

/// The entry doubles as a loop header: it reads variable 0, and its latch
/// redefines it and branches back.
fn entry_loop() -> (Cfg<DfInst>, BlockId) {
    let mut cfg = Cfg::<DfInst>::new();
    let latch = cfg.new_block();
    let exit = cfg.new_block();
    let entry = cfg.entry();
    cfg.add_edge(entry, latch, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(latch, entry, EdgeKind::Back);
    cfg.block_mut(entry).push(df_use("read", 0));
    cfg.block_mut(latch).push(df_def("redefine", 0));
    (cfg, latch)
}

#[test]
#[should_panic(expected = "branch target")]
fn ssa_construction_rejects_an_entry_that_is_a_branch_target() {
    let (cfg, _) = entry_loop();
    drop(SsaForm::compute(&cfg, &DominatorTree::compute(&cfg)));
}

#[test]
#[should_panic(expected = "branch target")]
fn phi_placement_rejects_an_entry_that_is_a_branch_target() {
    let (cfg, _) = entry_loop();
    drop(PhiPlacements::compute(&cfg, &DominatorTree::compute(&cfg)));
}

#[test]
fn a_new_entry_block_restores_the_value_entering_the_function() {
    // The remedy the panic names: an empty block that falls through to the old
    // entry. The old entry is then an ordinary loop header, and its phi
    // merges the live-in value from the new entry with the latch's.
    let (mut cfg, latch) = entry_loop();
    let header = cfg.entry();
    let new_entry = cfg.new_block();
    cfg.add_edge(new_entry, header, EdgeKind::Fallthrough);
    cfg.set_entry(new_entry);

    let ssa = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
    let [phi] = ssa.block(header).phis.as_slice() else {
        panic!("the header merges the live-in and the latch definition");
    };
    let defined = ssa.block(latch).instructions[0].defs()[0].clone();
    assert_eq!(
        phi.operands(),
        [
            (departure(latch, 1), defined),
            (departure(new_entry, 0), SsaValue::live_in(0)),
        ],
        "operands follow the CFG's predecessor order"
    );
    assert_eq!(ssa.block(header).instructions[0].uses()[0], phi.result);
}

#[test]
fn an_instruction_answers_the_value_it_reads_and_the_value_it_leaves() {
    // Variable 0 is read, then defined twice; variable 1 is only read, and
    // variable 2 only defined.
    let mut cfg = Cfg::<DfInst>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(df_def("seed", 0));
    cfg.block_mut(entry).push(DfInst {
        uses: vec![0, 1, 0],
        defs: vec![0, 2, 0],
        ..df_ff("wide")
    });

    let form = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
    let wide = form.instruction(departure(entry, 1)).unwrap();

    assert_eq!(wide.use_of(&0), Some(&SsaValue::new(0, 1)));
    assert_eq!(wide.use_of(&1), Some(&SsaValue::live_in(1)));
    assert_eq!(wide.use_of(&2), None, "variable 2 is only defined here");
    assert_eq!(
        wide.def_of(&0),
        Some(&SsaValue::new(0, 3)),
        "the later of two definitions is the one left behind"
    );
    assert_eq!(wide.def_of(&2), Some(&SsaValue::new(2, 1)));
    assert_eq!(wide.def_of(&1), None, "variable 1 is only read here");
    assert_eq!(
        &form.value_at(departure(entry, 2), &0),
        wide.def_of(&0).unwrap(),
        "the next point reads what the instruction left"
    );
}

#[test]
fn a_phi_wider_than_its_inline_storage_lists_every_departure_in_cfg_order() {
    // Four arms merge, one more operand than a phi record holds inline.
    let mut cfg = Cfg::<DfInst>::new();
    let merge = cfg.new_block();
    let arms: Vec<BlockId> = (0..4).map(|_| cfg.new_block()).collect();
    for (index, &arm) in arms.iter().enumerate() {
        let kind = if index == 0 {
            EdgeKind::ConditionalTrue
        } else {
            EdgeKind::ConditionalFalse
        };
        cfg.add_edge(cfg.entry(), arm, kind);
        cfg.add_edge(arm, merge, EdgeKind::Fallthrough);
        cfg.block_mut(arm).push(df_def("arm", 0));
    }
    cfg.block_mut(merge).push(df_use("merged", 0));

    let dom = DominatorTree::compute(&cfg);
    let form = SsaForm::compute(&cfg, &dom);
    let [phi] = form.block(merge).phis.as_slice() else {
        panic!("the merge needs exactly one phi");
    };
    let expected: Vec<ProgramPoint> = cfg
        .predecessors(merge)
        .map(|block| departure(block, cfg.block(block).instructions().len()))
        .collect();
    let departures: Vec<ProgramPoint> = phi.operands().iter().map(|(point, _)| *point).collect();
    assert_eq!(departures, expected);
    for (point, value) in phi.operands() {
        assert_eq!(
            value,
            &form.instruction(departure(point.block, 0)).unwrap().defs()[0]
        );
    }

    let placements = PhiPlacements::compute(&cfg, &dom);
    assert_eq!(placements.at(merge)[0].predecessors(), expected);
}

#[test]
fn a_blocks_instruction_vector_holds_no_spare_capacity() {
    // The vector moves into the form, so room the allocator's growth step
    // would add stays allocated for the life of the form.
    let cfg = CfgBuilder::build(vec![
        df_def("first", 0),
        df_use("second", 0),
        df_def("third", 1),
    ])
    .unwrap();
    let form = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
    for block in form.blocks() {
        assert_eq!(block.instructions.capacity(), block.instructions.len());
    }
}
