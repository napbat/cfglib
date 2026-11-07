//! The dense-row liveness and reaching definitions against the `BTreeSet`
//! drivers on random graphs with unwind edges, removed blocks, unreachable
//! code and universes wider than one word.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, DenseLiveness, DenseReachingDefs, DenseRow, Liveness, ReachingDef, ReachingDefs,
};

use crate::analysis_oracles::{Ins, Program};
use crate::graphs::{Rng, Shape};

/// A random program over `variables` variables whose instructions each
/// write up to `writes` of them, so a wide program has rows of several words.
fn program(rng: &mut Rng, variables: usize, writes: usize) -> Program {
    let shape = Shape::random(rng, false, true);
    let pick = |rng: &mut Rng, most: usize| -> Vec<u8> {
        let count = rng.below(most + 1);
        let mut picked: Vec<u8> = (0..count)
            .map(|_| u8::try_from(rng.below(variables)).expect("fewer than 256 variables"))
            .collect();
        picked.sort_unstable();
        picked.dedup();
        picked
    };
    let code = (0..shape.blocks)
        .map(|_| {
            (0..rng.below(12))
                .map(|_| Ins {
                    uses: pick(rng, 3),
                    defs: pick(rng, writes),
                    throws: rng.below(4) == 0,
                })
                .collect()
        })
        .collect();
    Program { shape, code }
}

/// The narrow programs of the oracle tests, alternating with wide ones.
fn programs(seed: u64) -> impl Iterator<Item = Program> {
    let mut rng = Rng::new(seed);
    (0..400).map(move |round| {
        if round % 2 == 0 {
            Program::random(&mut rng, true)
        } else {
            program(&mut rng, 150, 6)
        }
    })
}

fn slots(cfg: &Cfg<Ins>) -> impl Iterator<Item = BlockId> {
    (0..cfg.block_bound()).map(BlockId::from_index)
}

fn variables(live: &DenseLiveness<u8>, row: DenseRow<'_>) -> BTreeSet<u8> {
    row.ones().map(|bit| *live.variable(bit)).collect()
}

fn definitions(rd: &DenseReachingDefs<u8>, row: DenseRow<'_>) -> BTreeSet<ReachingDef<u8>> {
    row.ones().map(|bit| rd.definition(bit).clone()).collect()
}

fn assert_liveness_equal(cfg: &Cfg<Ins>, dense: &DenseLiveness<u8>, sets: &Liveness<u8>) {
    assert!(dense.variables().is_sorted(), "bit order is variable order");
    for block in slots(cfg) {
        assert_eq!(
            &variables(dense, dense.live_in(block)),
            sets.live_in(block),
            "in {block}"
        );
        assert_eq!(
            &variables(dense, dense.live_out(block)),
            sets.live_out(block),
            "out {block}"
        );
        for variable in 0..=u8::MAX {
            assert_eq!(
                dense.is_live_in(&variable, block),
                sets.is_live_in(&variable, block)
            );
            assert_eq!(
                dense.is_live_out(&variable, block),
                sets.is_live_out(&variable, block)
            );
        }
    }
}

#[test]
fn dense_liveness_equals_the_set_driver_on_random_graphs() {
    let mut wide = 0;
    for program in programs(51) {
        let cfg = program.cfg();
        let dense = DenseLiveness::compute(&cfg);
        assert_liveness_equal(&cfg, &dense, &Liveness::compute(&cfg));
        wide += usize::from(dense.variables().len() > 64);

        let seeds = |block: BlockId| {
            if program.shape.successors(block.index()).is_empty() {
                vec![0, 3, 200]
            } else {
                Vec::new()
            }
        };
        let dense = DenseLiveness::compute_with_exits(&cfg, seeds);
        assert_liveness_equal(&cfg, &dense, &Liveness::compute_with_exits(&cfg, seeds));
    }
    assert!(wide > 100, "{wide} solves had rows of more than one word");
}

#[test]
fn dense_reaching_definitions_equal_the_set_driver_on_random_graphs() {
    let mut wide = 0;
    for program in programs(52) {
        let cfg = program.cfg();
        let dense = DenseReachingDefs::compute(&cfg);
        let sets = ReachingDefs::compute(&cfg);
        assert!(
            dense.definitions().is_sorted(),
            "bit order is definition order"
        );
        wide += usize::from(dense.definitions().len() > 64);
        for block in slots(&cfg) {
            let reaching_in = definitions(&dense, dense.reaching_in(block));
            assert_eq!(&reaching_in, sets.reaching_in(block), "in {block}");
            let reaching_out = definitions(&dense, dense.reaching_out(block));
            assert_eq!(&reaching_out, sets.reaching_out(block), "out {block}");
            for definition in &reaching_in {
                assert!(
                    dense.reaching_in(block).get(
                        dense
                            .index_of(definition)
                            .expect("a reaching definition has a bit")
                    )
                );
            }
            for variable in 0..=u8::MAX {
                assert_eq!(
                    dense.defs_of_at_entry(&variable, block),
                    sets.defs_of_at_entry(&variable, block),
                    "{variable} at {block}"
                );
            }
        }
    }
    assert!(wide > 100, "{wide} solves had rows of more than one word");
}
