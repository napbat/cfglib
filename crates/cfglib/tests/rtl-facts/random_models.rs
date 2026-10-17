//! Many small random functions, each compared with every oracle.
//!
//! The generator is a fixed xorshift sequence, so a failure names the seed and
//! reproduces. Functions have loops, self loops, unreachable blocks, removed
//! slots with stale records, empty blocks, calls, traps, and several exits.

use std::collections::BTreeSet;

use crate::kestrel::Kestrel;
use crate::model::{
    Access, Ending, Inst, Model, ROOT_COUNT, block, build_kestrel, build_tessera, class, inst,
    removed, written_roots,
};
use crate::oracle;
use crate::recipe::assert_recipe_matches;
use crate::tessera::Tessera;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: usize) -> usize {
        let bound = u64::try_from(bound).expect("small bound");
        usize::try_from(self.next() % bound).expect("below the bound")
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    fn root(&mut self) -> u8 {
        u8::try_from(self.below(usize::from(ROOT_COUNT))).expect("below ten")
    }

    /// Up to three distinct roots of one class.
    fn candidates(&mut self) -> Vec<u8> {
        let first = self.root();
        let mut ids: Vec<u8> = (0..ROOT_COUNT)
            .filter(|&id| class(id) == class(first))
            .collect();
        let keep = 1 + self.below(ids.len().min(3));
        let mut chosen = Vec::new();
        while chosen.len() < keep {
            chosen.push(ids.remove(self.below(ids.len())));
        }
        chosen
    }

    fn access(&mut self) -> Access {
        match self.below(9) {
            0 => Access::Read(self.root()),
            1 => Access::ReadIndexed(self.candidates()),
            2 => Access::ReadSparse(self.root()),
            3 => Access::WriteFull(self.root()),
            4 => Access::WritePartial([0, 1, 2, 3, 8][self.below(5)]),
            5 => Access::WriteIndexed(self.candidates()),
            6 => Access::WriteSparse(self.root()),
            7 => Access::RaiseBinding(self.root()),
            _ => Access::Noise,
        }
    }

    fn insts(&mut self) -> Vec<Inst> {
        (0..self.below(4))
            .map(|_| {
                let mut taken = BTreeSet::new();
                let accesses = (0..self.below(4))
                    .map(|_| match self.access() {
                        // A record never writes one root twice.
                        write if written_roots(&write).iter().any(|id| taken.contains(id)) => {
                            Access::Noise
                        }
                        write => {
                            taken.extend(written_roots(&write));
                            write
                        }
                    })
                    .collect();
                let mut one = inst(accesses);
                if self.chance(15) {
                    one.ending = Ending::Call;
                }
                one
            })
            .collect()
    }
}

fn writes(inst: &Inst) -> bool {
    inst.accesses.iter().any(|access| {
        matches!(
            access,
            Access::WriteFull(_)
                | Access::WritePartial(_)
                | Access::WriteIndexed(_)
                | Access::WriteSparse(_)
        )
    })
}

/// A random function of two to nine slots.
pub fn random_model(seed: u64) -> Model {
    let mut rng = Rng::new(seed);
    let count = 2 + rng.below(8);
    let gone: Vec<bool> = (0..count).map(|slot| slot != 0 && rng.chance(15)).collect();
    let live: Vec<usize> = (0..count).filter(|&slot| !gone[slot]).collect();
    let slots = (0..count)
        .map(|slot| {
            let mut insts = rng.insts();
            if gone[slot] {
                return removed(insts);
            }
            let mut successors = Vec::new();
            for _ in 0..rng.below(4) {
                let target = live[rng.below(live.len())];
                if !successors.contains(&target) {
                    successors.push(target);
                }
            }
            if successors.is_empty()
                && let Some(last) = insts.last_mut()
                && !writes(last)
                && last.ending == Ending::Advance
                && rng.chance(30)
            {
                last.ending = Ending::Trap;
            }
            if successors.len() == 1
                && let Some(last) = insts.last_mut()
                && last.ending == Ending::Advance
                && rng.chance(40)
            {
                last.ending = Ending::Jump;
            }
            block(insts, &successors)
        })
        .collect();
    Model::new(slots)
}

#[test]
fn every_fact_matches_its_oracle_on_random_functions_in_both_dialects() {
    for seed in 0..400 {
        let model = random_model(seed);
        let label = format!("seed {seed}");
        assert_recipe_matches::<Kestrel, _>(&build_kestrel(&model), &model, &label);
        assert_recipe_matches::<Tessera, _>(&build_tessera(&model), &model, &label);
    }
}

#[test]
fn the_generator_covers_the_shapes_the_facts_must_handle() {
    let mut counts = [0_usize; 8];
    for seed in 0..400 {
        let model = random_model(seed);
        let reachable = oracle::reachable(&model);
        let dominators = oracle::dominators(&model);
        let named = model.named_roots();
        let live = oracle::liveness(&model, &|_| named.clone());
        let insts = || model.live().flat_map(|(_, block)| &block.insts);
        counts[0] += model.slots.len() - model.live().count();
        counts[1] += model.live().filter(|(index, _)| !reachable[*index]).count();
        counts[2] += model
            .live()
            .filter(|(index, block)| {
                reachable[*index]
                    && block
                        .successors
                        .iter()
                        .any(|target| dominators[*index].contains(target))
            })
            .count();
        counts[3] += insts().filter(|inst| inst.ending == Ending::Call).count();
        counts[4] += insts().filter(|inst| inst.ending == Ending::Trap).count();
        counts[5] += oracle::exits(&model).len();
        counts[6] += oracle::placement(&model, &live.live_in)
            .iter()
            .map(BTreeSet::len)
            .sum::<usize>();
        counts[7] += model
            .live()
            .filter(|(_, block)| block.insts.is_empty())
            .count();
    }
    let names = [
        "removed slots",
        "unreachable blocks",
        "back edges",
        "calls",
        "traps",
        "exits",
        "placed parameters",
        "empty blocks",
    ];
    for (name, count) in names.into_iter().zip(counts) {
        assert!(count >= 20, "only {count} {name} in 400 random functions");
    }
}
