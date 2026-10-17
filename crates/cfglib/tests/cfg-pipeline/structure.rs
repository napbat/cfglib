//! Dominance, frontiers, and SSA structure against brute-force definitions.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::{
    BlockId, DeadCode, DominanceFrontiers, DominatorTree, EdgeKind, FrontierScratch, PhiPlacements,
    PostDominatorTree, ProgramPoint, SccpAnalysis, SsaForm, SsaValue,
};

use crate::check::assert_well_formed;
use crate::interp::{input_vectors, run, run_ssa};
use crate::oracle::{Reference, expected_phi_departures, has_unwind_edges, ssa_violations};
use crate::programs::{corpus, program};
use crate::toy::Var;

#[test]
fn every_program_in_the_corpus_is_structurally_valid_before_any_pass() {
    for case in corpus() {
        assert_well_formed(case.name, "construction", &case.cfg);
    }
}

#[test]
fn dominators_match_the_remove_the_node_and_check_reachability_definition() {
    for case in corpus() {
        let dominators = DominatorTree::compute(&case.cfg);
        let reference = Reference::new(&case.cfg);
        for block in case.cfg.block_ids() {
            assert_eq!(
                dominators.is_reachable(block),
                reference.reachable.contains(&block),
                "{}: reachability of {block:?}",
                case.name
            );
        }
        for &node in &reference.reachable {
            assert_eq!(
                dominators.idom(node),
                reference.immediate_dominator(node),
                "{}: idom of {node:?}",
                case.name
            );
            for &candidate in &reference.reachable {
                assert_eq!(
                    dominators.dominates(candidate, node),
                    reference.dominates(candidate, node),
                    "{}: does {candidate:?} dominate {node:?}",
                    case.name
                );
            }
        }
    }
}

#[test]
fn post_dominators_match_the_every_path_to_an_exit_definition() {
    for case in corpus() {
        let reference = Reference::new(&case.cfg);
        let post = PostDominatorTree::compute(&case.cfg);
        let mut exits = post.exits().to_vec();
        exits.sort();
        assert_eq!(exits, reference.exits(), "{}: exits", case.name);
        let finishing: Vec<BlockId> = reference
            .reachable
            .iter()
            .copied()
            .filter(|block| reference.reaches_exit(*block))
            .collect();
        for &block in &finishing {
            assert!(
                post.reaches_exit(block),
                "{}: {block:?} reaches an exit",
                case.name
            );
            for &candidate in &finishing {
                assert_eq!(
                    post.post_dominates(candidate, block),
                    reference.post_dominates(candidate, block),
                    "{}: does {candidate:?} post-dominate {block:?}",
                    case.name
                );
            }
        }
    }
}

#[test]
fn dominance_frontiers_match_the_definition_from_every_entry_point() {
    for case in corpus() {
        let dominators = DominatorTree::compute(&case.cfg);
        let reference = Reference::new(&case.cfg);
        let table = DominanceFrontiers::compute(&case.cfg, &dominators);
        let mut queried = Vec::new();
        for &block in &reference.reachable {
            let expected = reference.frontier(block);
            assert_eq!(
                table
                    .frontier(block)
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>(),
                expected,
                "{}: table frontier of {block:?}",
                case.name
            );
            dominators.frontier(&case.cfg, block, &mut queried);
            assert_eq!(
                queried.iter().copied().collect::<BTreeSet<_>>(),
                expected,
                "{}: queried frontier of {block:?}",
                case.name
            );
        }
    }
}

#[test]
fn ssa_defines_each_value_once_and_dominates_every_use() {
    for case in corpus() {
        let dominators = DominatorTree::compute(&case.cfg);
        let ssa = SsaForm::compute(&case.cfg, &dominators);
        let violations = ssa_violations(&case.cfg, &ssa, &Reference::new(&case.cfg));
        assert!(violations.is_empty(), "{}: {violations:#?}", case.name);
    }
}

#[test]
fn phi_operands_come_one_per_incoming_edge_from_that_edges_source() {
    for case in corpus()
        .into_iter()
        .filter(|case| !has_unwind_edges(&case.cfg))
    {
        let dominators = DominatorTree::compute(&case.cfg);
        let ssa = SsaForm::compute(&case.cfg, &dominators);
        for block in case.cfg.block_ids() {
            let mut expected = expected_phi_departures(&case.cfg, block);
            expected.sort();
            for phi in &ssa.block(block).phis {
                let mut departures: Vec<ProgramPoint> =
                    phi.operands().iter().map(|(point, _)| *point).collect();
                departures.sort();
                assert_eq!(departures, expected, "{}: phi of {block:?}", case.name);
            }
        }
    }
}

#[test]
fn phis_sit_exactly_at_the_iterated_frontier_of_each_variables_definitions() {
    let mut scratch = FrontierScratch::new();
    for case in corpus()
        .into_iter()
        .filter(|case| !has_unwind_edges(&case.cfg))
    {
        let reference = Reference::new(&case.cfg);
        if reference.reachable.len() != case.cfg.block_count() {
            continue;
        }
        let dominators = DominatorTree::compute(&case.cfg);
        let ssa = SsaForm::compute(&case.cfg, &dominators);
        let placements = PhiPlacements::compute(&case.cfg, &dominators);
        let mut defined_in: BTreeMap<Var, BTreeSet<BlockId>> = BTreeMap::new();
        for block in case.cfg.block_ids() {
            for inst in case.cfg.block(block).instructions() {
                for variable in &inst.defs {
                    defined_in.entry(*variable).or_default().insert(block);
                }
            }
        }
        let mut phi_blocks: BTreeMap<Var, BTreeSet<BlockId>> = BTreeMap::new();
        for (block, phi) in ssa.phis() {
            phi_blocks
                .entry(phi.result.variable)
                .or_default()
                .insert(block);
        }
        for (variable, definitions) in &defined_in {
            let expected = reference.iterated_frontier(definitions);
            assert_eq!(
                phi_blocks.get(variable).cloned().unwrap_or_default(),
                expected,
                "{}: phi blocks of {variable:?}",
                case.name
            );
            let mut queried = Vec::new();
            dominators.iterated_frontier(
                &case.cfg,
                &mut scratch,
                definitions.iter().copied(),
                None,
                &mut queried,
            );
            assert_eq!(
                queried.into_iter().collect::<BTreeSet<_>>(),
                expected,
                "{}: iterated frontier query of {variable:?}",
                case.name
            );
        }
        assert_eq!(
            placements.len(),
            ssa.phis().count(),
            "{}: placement count",
            case.name
        );
        for (block, placement) in placements.iter() {
            let phi = ssa
                .block(block)
                .phis
                .iter()
                .find(|phi| phi.result.variable == placement.variable)
                .expect("a placement names a phi of the form");
            let points: Vec<ProgramPoint> =
                phi.operands().iter().map(|(point, _)| *point).collect();
            assert_eq!(
                placement.predecessors(),
                points,
                "{}: placement operands",
                case.name
            );
        }
    }
}

#[test]
fn ssa_execution_matches_plain_execution_on_every_input() {
    for case in corpus() {
        let dominators = DominatorTree::compute(&case.cfg);
        let ssa = SsaForm::compute(&case.cfg, &dominators);
        for inputs in input_vectors(case.inputs) {
            assert_eq!(
                run_ssa(&case.cfg, &ssa, &inputs).0,
                run(&case.cfg, &inputs).0,
                "{}: {inputs:?}",
                case.name
            );
        }
    }
}

#[test]
fn a_handler_phi_takes_one_operand_per_throwing_instruction() {
    let case = program("division_with_handler");
    let entry = case.cfg.entry();
    let handler = case
        .cfg
        .block_ids()
        .find(|block| {
            case.cfg
                .incoming(*block)
                .any(|edge| case.cfg.edge(edge).kind() == EdgeKind::ExceptionUnwind)
        })
        .expect("the program has a handler");
    let dominators = DominatorTree::compute(&case.cfg);
    let ssa = SsaForm::compute(&case.cfg, &dominators);
    let phi = ssa
        .block(handler)
        .phis
        .iter()
        .find(|phi| phi.result.variable == Var("t"))
        .expect("t is redefined after the first throwing instruction");
    // The divisions are instructions 1 and 3; `t` is written by 0 and 2.
    let before_first = ssa.block(entry).instructions[0].defs()[0].clone();
    let before_second = ssa.block(entry).instructions[2].defs()[0].clone();
    assert_eq!(
        phi.operands(),
        [
            (
                ProgramPoint {
                    block: entry,
                    inst_idx: 1
                },
                before_first
            ),
            (
                ProgramPoint {
                    block: entry,
                    inst_idx: 3
                },
                before_second
            ),
        ]
    );
}

#[test]
fn a_removed_block_slot_keeps_every_other_identity_through_the_pipeline() {
    let case = program("removed_block_slot");
    assert_eq!(case.cfg.block_bound(), case.cfg.block_count() + 1);
    let dominators = DominatorTree::compute(&case.cfg);
    let ssa = SsaForm::compute(&case.cfg, &dominators);
    assert_eq!(ssa.blocks().len(), case.cfg.block_bound());
    let hole = case
        .cfg
        .block_ids()
        .map(BlockId::index)
        .collect::<BTreeSet<_>>();
    let removed = (0..case.cfg.block_bound())
        .find(|index| !hole.contains(index))
        .unwrap();
    assert!(ssa.blocks()[removed].phis.is_empty() && ssa.blocks()[removed].instructions.is_empty());
    assert!(!dominators.is_reachable(BlockId::from_index(removed)));
    let sccp = SccpAnalysis::compute(&case.cfg, &ssa);
    assert_eq!(sccp.reachable_blocks.len(), case.cfg.block_count());
}

#[test]
fn unreachable_blocks_are_reported_and_get_their_own_ssa_roots() {
    let case = program("unreachable_loop_component");
    let dead = DeadCode::compute(&case.cfg);
    assert_eq!(dead.unreachable_blocks.len(), 2);
    let dominators = DominatorTree::compute(&case.cfg);
    let ssa = SsaForm::compute(&case.cfg, &dominators);
    let entry_read = &ssa.block(case.cfg.entry()).instructions[0].uses()[0];
    assert_eq!(entry_read, &SsaValue::live_in(Var("a")));
    let head = dead.unreachable_blocks[0];
    let head_read = &ssa.block(head).instructions[0].uses()[0];
    assert_ne!(
        head_read, entry_read,
        "unreachable code does not inherit the entry's live-in"
    );
}
