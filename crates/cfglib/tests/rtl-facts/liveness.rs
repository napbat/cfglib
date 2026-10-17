//! `RootLiveness`: whole-root liveness and the exit seeds.
//!
//! The oracle is a backward fixed point over single instructions on the
//! neutral model, so it shares nothing with the block summaries that the
//! library solves over.

use std::collections::BTreeSet;
use std::panic::{AssertUnwindSafe, catch_unwind};

use cfglib::DenseId;
use cfglib::ir::rtl::{BlockSummaries, FunctionView, RootLiveness, RootTable};

use crate::fixtures;
use crate::kestrel::{Kestrel, Reg};
use crate::kestrel_store::{BlockRef, KestrelFunction};
use crate::model::{Access, Model, Target, block, build_kestrel, build_tessera, call, inst};
use crate::oracle;
use crate::recipe::neutral_set;
use crate::tessera::Tessera;

fn compare<D: Target, F: FunctionView<D>>(
    function: &F,
    roots: &RootTable<D>,
    actual: &RootLiveness,
    expected: &oracle::Liveness,
    model: &Model,
    label: &str,
) {
    for (index, _) in model.live() {
        let id = F::NodeId::from_index(index);
        assert_eq!(
            neutral_set(roots, actual.live_in(id)),
            expected.live_in[index],
            "{label}: live_in of block {index}"
        );
        assert_eq!(
            neutral_set(roots, actual.live_out(id)),
            expected.live_out[index],
            "{label}: live_out of block {index}"
        );
    }
    for index in 0..function.node_bound() {
        if !model.is_live(index) {
            let id = F::NodeId::from_index(index);
            assert_eq!(
                actual.live_in(id).count_ones(),
                0,
                "{label}: removed {index}"
            );
            assert_eq!(
                actual.live_out(id).count_ones(),
                0,
                "{label}: removed {index}"
            );
        }
    }
}

/// Compares `RootLiveness::compute`, whose seed is every named root, with the
/// oracle under the same seed.
pub fn assert_liveness_matches<D: Target, F: FunctionView<D>>(
    function: &F,
    roots: &RootTable<D>,
    liveness: &RootLiveness,
    model: &Model,
    label: &str,
) {
    let named = model.named_roots();
    let expected = oracle::liveness(model, &|_| named.clone());
    compare(function, roots, liveness, &expected, model, label);
}

/// A seed that differs per exit block and names roots the function lacks.
fn seed_for(block: usize) -> BTreeSet<u8> {
    let first = u8::try_from((block * 3 + 1) % 10).expect("below ten");
    let second = u8::try_from((block * 7 + 2) % 10).expect("below ten");
    BTreeSet::from([first, second])
}

fn assert_per_block_seeds_match<D: Target, F: FunctionView<D>>(
    function: &F,
    model: &Model,
    label: &str,
) {
    let roots = RootTable::collect(function);
    let summaries = BlockSummaries::compute(function, &roots);
    let actual = RootLiveness::compute_seeded(function, &roots, &summaries, |block, seed| {
        for id in seed_for(block.index()) {
            seed.insert(D::root_of(id));
        }
    });
    let named = model.named_roots();
    let expected = oracle::liveness(model, &|index| {
        seed_for(index).intersection(&named).copied().collect()
    });
    compare(function, &roots, &actual, &expected, model, label);
}

#[test]
fn per_exit_seeds_match_the_oracle_on_every_fixture_in_both_dialects() {
    for (name, model) in fixtures::all() {
        assert_per_block_seeds_match::<Kestrel, _>(&build_kestrel(&model), &model, name);
        assert_per_block_seeds_match::<Tessera, _>(&build_tessera(&model), &model, name);
    }
}

#[test]
fn an_empty_seed_matches_the_oracle_and_leaves_only_exposed_reads_live() {
    for (name, model) in fixtures::all() {
        let function = build_kestrel(&model);
        let roots = RootTable::collect(&function);
        let summaries = BlockSummaries::compute(&function, &roots);
        let actual = RootLiveness::compute_seeded(&function, &roots, &summaries, |_, _| {});
        let expected = oracle::liveness(&model, &|_| BTreeSet::new());
        compare(&function, &roots, &actual, &expected, &model, name);
    }

    // The diamond, derived by hand: the join reads `R0` and `R1`; the left arm
    // overwrites `R0`; the entry overwrites both.
    let model = fixtures::diamond();
    let function = build_kestrel(&model);
    let roots = RootTable::collect(&function);
    let summaries = BlockSummaries::compute(&function, &roots);
    let liveness = RootLiveness::compute_seeded(&function, &roots, &summaries, |_, _| {});
    let live_in = |block| neutral_set(&roots, liveness.live_in(BlockRef::from_index(block)));
    assert_eq!(live_in(0), BTreeSet::new());
    assert_eq!(live_in(1), BTreeSet::from([1]));
    assert_eq!(live_in(2), BTreeSet::from([0, 1]));
    assert_eq!(live_in(3), BTreeSet::from([0, 1]));
    let live_out = |block| neutral_set(&roots, liveness.live_out(BlockRef::from_index(block)));
    assert_eq!(live_out(3), BTreeSet::new());
}

#[test]
fn a_loop_is_solved_to_a_fixed_point() {
    // Derived by hand. Block 1 reads `R0` and then overwrites it, so `R0` is
    // live around the loop and into the exit; block 0 overwrites `R1` only.
    let model = fixtures::self_loop();
    let function = build_kestrel(&model);
    let roots = RootTable::collect(&function);
    let summaries = BlockSummaries::compute(&function, &roots);
    let liveness = RootLiveness::compute(&function, &roots, &summaries);
    let sets = |block: usize| {
        let id = BlockRef::from_index(block);
        (
            neutral_set(&roots, liveness.live_in(id)),
            neutral_set(&roots, liveness.live_out(id)),
        )
    };
    let both = BTreeSet::from([0, 1]);
    assert_eq!(sets(0), (BTreeSet::from([0]), both.clone()));
    assert_eq!(sets(1), (both.clone(), both.clone()));
    assert_eq!(sets(2), (both.clone(), both));
}

#[test]
fn a_call_makes_every_root_live_before_it() {
    // `call` reads every root, so `R0` is live on entry although the only
    // instruction after it overwrites `R0` and the seed is empty.
    let model = Model::new(vec![block(
        vec![call(vec![]), inst(vec![Access::WriteFull(0)])],
        &[],
    )]);
    let function = build_kestrel(&model);
    let roots = RootTable::collect(&function);
    let summaries = BlockSummaries::compute(&function, &roots);
    let liveness = RootLiveness::compute_seeded(&function, &roots, &summaries, |_, _| {});
    let entry = BlockRef::from_index(0);
    assert_eq!(
        neutral_set(&roots, liveness.live_in(entry)),
        BTreeSet::from([0])
    );
}

#[test]
fn the_seed_closure_runs_once_for_each_exit_block_including_unreachable_ones() {
    for (name, model) in fixtures::all() {
        let function = build_kestrel(&model);
        let roots = RootTable::collect(&function);
        let summaries = BlockSummaries::compute(&function, &roots);
        let mut called = Vec::new();
        let _ = RootLiveness::compute_seeded(&function, &roots, &summaries, |block, _| {
            called.push(block.index());
        });
        called.sort_unstable();
        assert_eq!(called, oracle::exits(&model), "{name}: exits");
    }
}

#[test]
fn the_exit_seed_starts_empty_and_reports_what_it_changes() {
    let model = fixtures::loop_with_exits();
    let function = build_kestrel(&model);
    let roots = RootTable::collect(&function);
    let summaries = BlockSummaries::compute(&function, &roots);
    let mut reports = Vec::new();
    let liveness = RootLiveness::compute_seeded(&function, &roots, &summaries, |block, seed| {
        if block.index() == 5 {
            reports.push(seed.insert(Reg::R1));
            reports.push(seed.insert(Reg::R1));
            // `Over` appears only in a removed block, so the function does
            // not name it and the seed ignores it.
            reports.push(seed.insert(Reg::Over));
            seed.insert_all();
            reports.push(seed.insert(Reg::R2));
        }
    });
    assert_eq!(reports, [true, false, false, false]);
    let exit = BlockRef::from_index(5);
    assert_eq!(liveness.live_out(exit).count_ones(), roots.len());
    // The other exit was seeded with nothing, so only its own read is live.
    let other = BlockRef::from_index(7);
    assert_eq!(liveness.live_out(other).count_ones(), 0);
}

#[test]
fn seeding_every_root_at_every_exit_is_the_default_solve() {
    for (name, model) in fixtures::all() {
        let function = build_kestrel(&model);
        let roots = RootTable::collect(&function);
        let summaries = BlockSummaries::compute(&function, &roots);
        let seeded = RootLiveness::compute_seeded(&function, &roots, &summaries, |_, seed| {
            seed.insert_all();
        });
        assert_eq!(
            seeded,
            RootLiveness::compute(&function, &roots, &summaries),
            "{name}"
        );
    }
}

#[test]
fn an_unreachable_block_still_has_liveness() {
    let model = fixtures::loop_with_exits();
    let function = build_kestrel(&model);
    let roots = RootTable::collect(&function);
    let summaries = BlockSummaries::compute(&function, &roots);
    let liveness = RootLiveness::compute(&function, &roots, &summaries);
    // Block 8 overwrites `R2` and reads `R3`, and is unreachable.
    let unreachable = BlockRef::from_index(8);
    let r2 = roots.index_of(Reg::R2).expect("named");
    let r3 = roots.index_of(Reg::R3).expect("named");
    assert!(liveness.live_in(unreachable).get(r3));
    assert!(!liveness.live_in(unreachable).get(r2));
}

#[test]
fn asking_for_liveness_past_the_node_bound_panics() {
    let function = KestrelFunction::new();
    let roots = RootTable::collect(&function);
    let summaries = BlockSummaries::compute(&function, &roots);
    let liveness = RootLiveness::compute(&function, &roots, &summaries);
    let asked = catch_unwind(AssertUnwindSafe(|| {
        liveness.live_in(BlockRef::from_index(1)).len()
    }));
    assert!(asked.is_err(), "block 1 is past the node bound of one");
}
