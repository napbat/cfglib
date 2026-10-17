//! The README recipe, end to end, over both storages.
//!
//! `run_recipe` is the README snippet. Every fact it returns is compared with
//! the brute-force oracles, once per dialect, and the two dialects are also
//! compared with each other in neutral terms.

use std::collections::BTreeSet;

use cfglib::ir::rtl::{BlockSummaries, FunctionView, ParamPlacement, RootLiveness, RootTable};
use cfglib::{DenseBits, DenseId, DominanceFrontiers, DominatorTree, verify_view};

use crate::fixtures;
use crate::kestrel::Kestrel;
use crate::liveness::assert_liveness_matches;
use crate::model::{Model, Target, build_kestrel, build_tessera};
use crate::oracle;
use crate::placement::assert_placement_matches;
use crate::summaries::assert_summaries_match;
use crate::tessera::Tessera;

/// Every fact of the recipe for one function.
pub struct Recipe<D: Target, F: FunctionView<D>> {
    pub roots: RootTable<D>,
    pub summaries: BlockSummaries,
    pub liveness: RootLiveness,
    pub dominators: DominatorTree<F::NodeId>,
    pub frontiers: DominanceFrontiers<F::NodeId>,
    pub placement: ParamPlacement<D>,
}

/// The snippet of the README, with `function` as the only input.
pub fn run_recipe<D: Target, F: FunctionView<D>>(function: &F) -> Recipe<D, F> {
    let roots = RootTable::collect(function);
    let summaries = BlockSummaries::compute(function, &roots);
    let liveness = RootLiveness::compute(function, &roots, &summaries);
    let dominators = DominatorTree::compute(function);
    let frontiers = DominanceFrontiers::compute(function, &dominators);
    let placement = ParamPlacement::compute(
        function,
        &dominators,
        &frontiers,
        &roots,
        &summaries,
        &liveness,
    );
    Recipe {
        roots,
        summaries,
        liveness,
        dominators,
        frontiers,
        placement,
    }
}

/// The neutral ids of the roots that `row` holds.
pub fn neutral_set<D: Target>(roots: &RootTable<D>, row: &DenseBits) -> BTreeSet<u8> {
    row.ones()
        .map(|index| D::id_of(roots.root(index)))
        .collect()
}

/// Runs the recipe on `function` and compares every fact with the oracles.
pub fn assert_recipe_matches<D: Target, F: FunctionView<D>>(
    function: &F,
    model: &Model,
    label: &str,
) {
    let recipe = run_recipe::<D, F>(function);
    let named: BTreeSet<u8> = recipe.roots.roots().iter().map(|&r| D::id_of(r)).collect();
    assert_eq!(named, model.named_roots(), "{label}: root table");
    assert!(
        recipe
            .roots
            .roots()
            .windows(2)
            .all(|pair| pair[0] < pair[1]),
        "{label}: the table is strictly ascending in the dialect order"
    );

    assert_summaries_match(function, &recipe.roots, &recipe.summaries, model, label);
    assert_liveness_matches(function, &recipe.roots, &recipe.liveness, model, label);

    let reachable = oracle::reachable(model);
    let dominators = oracle::dominators(model);
    let frontiers = oracle::frontiers(model);
    for (node, _) in model.live() {
        let id = F::NodeId::from_index(node);
        assert_eq!(
            recipe.dominators.is_reachable(id),
            reachable[node],
            "{label}: reachability of block {node}"
        );
        if reachable[node] {
            for (dominator, _) in model.live() {
                assert_eq!(
                    recipe
                        .dominators
                        .dominates(F::NodeId::from_index(dominator), id),
                    dominators[node].contains(&dominator),
                    "{label}: does block {dominator} dominate block {node}"
                );
            }
            let actual: BTreeSet<usize> = recipe
                .frontiers
                .frontier(id)
                .iter()
                .map(|block| block.index())
                .collect();
            assert_eq!(actual, frontiers[node], "{label}: frontier of block {node}");
        }
    }
    assert_placement_matches(&recipe, model, label);
}

#[test]
fn every_fixture_storage_satisfies_the_graph_view_contract_in_both_dialects() {
    // The Tessera storage keeps forward adjacency only and derives its
    // predecessors with `scan_predecessors`, as the view documentation
    // recommends, so parallel edges must stay parallel there too.
    for (name, model) in fixtures::all() {
        let kestrel = verify_view(&build_kestrel(&model));
        assert!(kestrel.is_ok(), "{name}: pooled storage: {kestrel:?}");
        let tessera = verify_view(&build_tessera(&model));
        assert!(tessera.is_ok(), "{name}: forward-only storage: {tessera:?}");
    }
}

#[test]
fn recipe_matches_the_oracles_on_every_fixture_in_both_dialects() {
    for (name, model) in fixtures::all() {
        assert_recipe_matches::<Kestrel, _>(&build_kestrel(&model), &model, name);
        assert_recipe_matches::<Tessera, _>(&build_tessera(&model), &model, name);
    }
}

#[test]
fn both_dialects_give_the_same_facts_in_neutral_terms() {
    for (name, model) in fixtures::all() {
        let kestrel = build_kestrel(&model);
        let tessera = build_tessera(&model);
        let left = run_recipe::<Kestrel, _>(&kestrel);
        let right = run_recipe::<Tessera, _>(&tessera);
        assert_eq!(left.roots.len(), right.roots.len(), "{name}: table size");
        assert_eq!(left.placement.param_count(), right.placement.param_count());
        for (node, _) in model.live() {
            let (kestrel_id, tessera_id) = (
                <crate::kestrel_store::KestrelFunction as cfglib::GraphView>::NodeId::from_index(
                    node,
                ),
                u32::from_index(node),
            );
            let kestrel_placed: BTreeSet<u8> = left
                .placement
                .roots_at(kestrel_id)
                .iter()
                .map(|&r| Kestrel::id_of(r))
                .collect();
            let tessera_placed: BTreeSet<u8> = right
                .placement
                .roots_at(tessera_id)
                .iter()
                .map(|&r| Tessera::id_of(r))
                .collect();
            assert_eq!(kestrel_placed, tessera_placed, "{name}: block {node}");
            assert_eq!(
                neutral_set(&left.roots, left.liveness.live_in(kestrel_id)),
                neutral_set(&right.roots, right.liveness.live_in(tessera_id)),
                "{name}: live_in of block {node}"
            );
        }
    }
}

#[test]
fn the_root_order_is_the_order_of_each_dialect_not_a_shared_one() {
    let model = fixtures::loop_with_exits();
    let kestrel = run_recipe::<Kestrel, _>(&build_kestrel(&model));
    let tessera = run_recipe::<Tessera, _>(&build_tessera(&model));
    let ascending: Vec<u8> = kestrel
        .roots
        .roots()
        .iter()
        .map(|&r| Kestrel::id_of(r))
        .collect();
    let descending: Vec<u8> = tessera
        .roots
        .roots()
        .iter()
        .map(|&r| Tessera::id_of(r))
        .collect();
    assert_eq!(ascending, [0, 1, 2, 3, 4, 6, 9]);
    assert_eq!(descending, [9, 6, 4, 3, 2, 1, 0]);
}

#[test]
fn the_readme_example_places_a_parameter_for_the_root_that_two_arms_define() {
    // `join` of the README snippet is the last block of the diamond.
    let model = fixtures::diamond();
    let join = 3;

    let kestrel = build_kestrel(&model);
    let recipe = run_recipe::<Kestrel, _>(&kestrel);
    let id = <crate::kestrel_store::KestrelFunction as cfglib::GraphView>::NodeId::from_index(join);
    assert_eq!(recipe.placement.roots_at(id), [crate::kestrel::Reg::R0]);
    assert_eq!(recipe.placement.param_count(), 1);

    let tessera = build_tessera(&model);
    let recipe = run_recipe::<Tessera, _>(&tessera);
    let needed_at_join = recipe.placement.roots_at(3_u32);
    assert_eq!(needed_at_join, [crate::tessera::Slot::of(0)]);
    assert_eq!(recipe.placement.param_count(), 1);
}

#[test]
fn removed_and_unreachable_blocks_follow_the_documented_rules() {
    let model = fixtures::loop_with_exits();
    let function = build_kestrel(&model);
    let recipe = run_recipe::<Kestrel, _>(&function);
    let id = |index| {
        <crate::kestrel_store::KestrelFunction as cfglib::GraphView>::NodeId::from_index(index)
    };

    // Slot 6 was removed and block 8 is unreachable.
    assert_eq!(recipe.placement.roots_at(id(6)), []);
    assert!(!recipe.dominators.is_reachable(id(8)));
    // The removed slot's stale records name `VF` and `WIDE`, and nothing else
    // does, so neither is in the table.
    assert!(recipe.roots.index_of(crate::kestrel::Reg::Over).is_none());
    assert!(recipe.roots.index_of(crate::kestrel::Reg::Wide).is_none());
    // The unreachable block still gets liveness: it reads `R3` after writing
    // only `R2`, so `R3` is live on its entry.
    let r3 = recipe
        .roots
        .index_of(crate::kestrel::Reg::R3)
        .expect("named");
    assert!(recipe.liveness.live_in(id(8)).get(r3));
}
