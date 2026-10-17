//! `ParamPlacement`: the pruned iterated dominance frontier.
//!
//! The oracle computes dominators by removal-reachability, frontiers from the
//! textbook definition, the iterated frontier by a worklist, and keeps the
//! blocks where the root is live on entry.

use std::collections::BTreeSet;

use cfglib::ir::rtl::{BlockSummaries, FunctionView, ParamPlacement, RootLiveness, RootTable};
use cfglib::{DenseId, DominanceFrontiers, DominatorTree};

use crate::fixtures;
use crate::kestrel::{Kestrel, Reg};
use crate::kestrel_store::BlockRef;
use crate::model::{
    Access::{Read, WriteFull},
    Model, Target, block, build_kestrel, build_tessera, inst,
};
use crate::oracle;
use crate::recipe::{Recipe, run_recipe};
use crate::tessera::{Slot, Tessera};

/// Compares the placement of a recipe with the oracle, block by block.
pub fn assert_placement_matches<D: Target, F: FunctionView<D>>(
    recipe: &Recipe<D, F>,
    model: &Model,
    label: &str,
) {
    let named = model.named_roots();
    let live = oracle::liveness(model, &|_| named.clone());
    let expected = oracle::placement(model, &live.live_in);
    let mut total = 0;
    for (index, expected) in expected.iter().enumerate() {
        let actual = recipe.placement.roots_at(F::NodeId::from_index(index));
        assert!(
            actual.windows(2).all(|pair| pair[0] < pair[1]),
            "{label}: block {index} lists its roots in the dialect order, once each"
        );
        let neutral: BTreeSet<u8> = actual.iter().map(|&root| D::id_of(root)).collect();
        assert_eq!(&neutral, expected, "{label}: parameters of block {index}");
        total += actual.len();
    }
    assert_eq!(
        recipe.placement.param_count(),
        total,
        "{label}: param_count"
    );
}

fn placed(
    recipe: &Recipe<Kestrel, crate::kestrel_store::KestrelFunction>,
    block: usize,
) -> Vec<u8> {
    recipe
        .placement
        .roots_at(BlockRef::from_index(block))
        .iter()
        .map(|&root| Kestrel::id_of(root))
        .collect()
}

#[test]
fn a_root_that_two_arms_define_needs_a_parameter_at_the_join() {
    let function = build_kestrel(&fixtures::diamond());
    let recipe = run_recipe::<Kestrel, _>(&function);
    assert_eq!(placed(&recipe, 3), [0]);
    for other in [0, 1, 2] {
        assert_eq!(placed(&recipe, other), []);
    }
    assert_eq!(recipe.placement.param_count(), 1);
}

#[test]
fn a_loop_header_needs_a_parameter_for_a_root_that_its_body_redefines() {
    // Block 1 branches back to itself, so it is in its own frontier, and it
    // reads `R0` first, so `R0` is live on entry. `R1` is defined only before
    // the loop and needs no parameter.
    let function = build_kestrel(&fixtures::self_loop());
    let recipe = run_recipe::<Kestrel, _>(&function);
    assert_eq!(placed(&recipe, 1), [0]);
    assert_eq!(recipe.placement.param_count(), 1);
}

#[test]
fn an_entry_block_with_a_back_edge_is_in_its_own_frontier_and_gets_a_parameter() {
    let function = build_kestrel(&fixtures::entry_loop());
    let recipe = run_recipe::<Kestrel, _>(&function);
    assert_eq!(placed(&recipe, 0), [0]);
    assert_eq!(placed(&recipe, 1), []);
}

#[test]
fn a_parameter_is_itself_a_definition_so_the_frontier_is_iterated() {
    // `R1` is written only in the inner body (block 3). Its frontier is the
    // inner header (block 2), and the parameter there defines `R1` again, so
    // the frontier of block 2 puts a second parameter at the outer header
    // (block 1). `R0` is written in block 4 and reaches only block 1.
    let function = build_kestrel(&fixtures::nested_loops());
    let recipe = run_recipe::<Kestrel, _>(&function);
    assert_eq!(placed(&recipe, 1), [0, 1]);
    assert_eq!(placed(&recipe, 2), [1]);
    assert_eq!(recipe.placement.param_count(), 3);
    for other in [0, 3, 4, 5, 6] {
        assert!(placed(&recipe, other).is_empty(), "block {other}");
    }
}

#[test]
fn the_parameters_of_a_block_follow_the_order_of_each_dialect() {
    let model = fixtures::nested_loops();
    let kestrel = run_recipe::<Kestrel, _>(&build_kestrel(&model));
    assert_eq!(
        kestrel.placement.roots_at(BlockRef::from_index(1)),
        [Reg::R0, Reg::R1]
    );
    let tessera = run_recipe::<Tessera, _>(&build_tessera(&model));
    assert_eq!(
        tessera.placement.roots_at(1_u32),
        [Slot::of(1), Slot::of(0)],
        "slots run against the neutral numbering"
    );
}

#[test]
fn an_unreachable_block_defines_nothing() {
    // Block 8 writes `R2` and flows into block 5, which the reachable loop
    // also reaches. `R2` is live at block 5 (every root is live after an
    // exit), but no reachable block defines it, so it needs no parameter.
    let function = build_kestrel(&fixtures::loop_with_exits());
    let recipe = run_recipe::<Kestrel, _>(&function);
    let r2 = recipe.roots.index_of(Reg::R2).expect("named");
    assert!(recipe.liveness.live_in(BlockRef::from_index(5)).get(r2));
    assert!(!placed(&recipe, 5).contains(&2));
    assert_eq!(placed(&recipe, 8), []);
}

/// A diamond whose join reads only `R1`, while both arms and the entry write
/// `R0`.
fn diamond_with_a_dead_join() -> Model {
    Model::new(vec![
        block(vec![inst(vec![WriteFull(0), WriteFull(1)])], &[1, 2]),
        block(vec![inst(vec![WriteFull(0)])], &[3]),
        block(vec![inst(vec![WriteFull(0)])], &[3]),
        block(vec![inst(vec![Read(1)])], &[]),
    ])
}

#[test]
fn liveness_prunes_the_parameters_that_nothing_reads() {
    let model = diamond_with_a_dead_join();
    let function = build_kestrel(&model);
    let roots = RootTable::collect(&function);
    let summaries = BlockSummaries::compute(&function, &roots);
    let dominators = DominatorTree::compute(&function);
    let frontiers = DominanceFrontiers::compute(&function, &dominators);
    let join = BlockRef::from_index(3);

    // Every root is live after the exit, so `R0` flows out and is live at the
    // join.
    let everything = RootLiveness::compute(&function, &roots, &summaries);
    let all = ParamPlacement::compute(
        &function,
        &dominators,
        &frontiers,
        &roots,
        &summaries,
        &everything,
    );
    assert_eq!(all.roots_at(join), [Reg::R0]);

    // With nothing live after the exit, `R0` is dead at the join.
    let nothing = RootLiveness::compute_seeded(&function, &roots, &summaries, |_, _| {});
    let pruned = ParamPlacement::compute(
        &function,
        &dominators,
        &frontiers,
        &roots,
        &summaries,
        &nothing,
    );
    assert_eq!(pruned.roots_at(join), []);
    assert_eq!(pruned.param_count(), 0);
}

#[test]
fn a_function_with_one_empty_block_needs_no_parameter() {
    let model = fixtures::empty_entry();
    let kestrel = run_recipe::<Kestrel, _>(&build_kestrel(&model));
    assert_eq!(kestrel.placement.param_count(), 0);
    let tessera = run_recipe::<Tessera, _>(&build_tessera(&model));
    assert_eq!(tessera.placement.param_count(), 0);
    assert!(tessera.roots.is_empty());
}
