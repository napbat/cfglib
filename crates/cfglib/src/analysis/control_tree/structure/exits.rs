//! Where each loop of the forest is left, found once for all loops.
//!
//! A loop with no recognized test takes its follow from its exits: the exit
//! edge out of its lowest block, preferring one that the enclosing bound
//! admits. Scanning a loop's blocks for that costs the size of the loop, and
//! loops nested `n` deep share most of their blocks, so scanning each one is
//! quadratic in `n`. The answers are a property of the graph and the forest,
//! so they are found for every loop in one pass over the edges.
//!
//! An edge from block `b` to block `t` is an exit of the loops that hold `b`
//! and not `t`: the loops of the chain from the innermost loop of `b` up to,
//! not including, the first that holds `t`. Two things are read off it:
//!
//! - The first exit of each loop, in the order of the edges (lowest block,
//!   then edge order), is the first edge that reaches the loop. The edges come
//!   in that order, and a loop of the chain that already has its first exit is
//!   skipped by a union-find over the chain, so every loop is assigned once.
//! - The local exits of a loop are the ones that land in the loop directly
//!   around it: it is the last loop of the chain. Each edge is local to one
//!   loop, so the lists hold one entry for each exit edge at most. A bound
//!   that admits exactly the loop around this one admits exactly those, so the
//!   first local exit the rest of the bound admits is the answer.
//!
//! Only properly nested loops are covered. An explicit loop (an irreducible
//! one) has no place in the chain, and the caller scans its blocks.

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::graph::loop_forest::{LoopForest, LoopId};

/// The exits of every properly nested loop of a forest.
pub(super) struct LoopExits {
    /// The target of the first exit edge of each loop, by loop position.
    first: Vec<Option<BlockId>>,
    /// The targets of the exit edges that land in the loop directly around
    /// each loop, in edge order, by loop position.
    local: Vec<Vec<BlockId>>,
}

/// No loop, in a table of loop positions.
const NONE: u32 = u32::MAX;

impl LoopExits {
    /// Find the exits of every nested loop of `forest`, reading the edges out
    /// of the blocks of `cfg` that the entry reaches.
    pub(super) fn compute<I, E>(cfg: &Cfg<I, E>, forest: &LoopForest, reachable: &[bool]) -> Self {
        let ids: Vec<LoopId> = forest.loops().collect();
        let count = ids.len();
        let enclosing: Vec<u32> = ids
            .iter()
            .map(|&lp| forest.enclosing(lp).map_or(NONE, position))
            .collect();
        let lifting = Lifting::new(&enclosing);
        // The loop after the last one: a loop with nothing around it climbs
        // into it, and nothing climbs out of it.
        let root = position_of(count);
        let mut next: Vec<u32> = (0..=root).collect();
        let mut first: Vec<Option<BlockId>> = alloc::vec![None; count];
        let mut local: Vec<Vec<BlockId>> = alloc::vec![Vec::new(); count];

        for block in cfg.block_ids().filter(|block| reachable[block.index()]) {
            let Some(innermost) = forest.innermost(block) else {
                continue;
            };
            for edge in cfg.outgoing(block) {
                let edge = cfg.edge(edge);
                let target = edge.target();
                if edge.kind().is_exceptional() || forest.contains(innermost, target) {
                    continue;
                }
                let mut at = find(&mut next, position(innermost));
                while at != root && !forest.contains(ids[at as usize], target) {
                    first[at as usize] = Some(target);
                    let above = match enclosing[at as usize] {
                        NONE => root,
                        outer => outer,
                    };
                    next[at as usize] = above;
                    at = find(&mut next, above);
                }
                let top = lifting.highest(position(innermost), |lp| {
                    !forest.contains(ids[lp as usize], target)
                });
                if enclosing[top as usize] != NONE {
                    local[top as usize].push(target);
                }
            }
        }
        Self { first, local }
    }

    /// The target of the first exit of `lp`, whatever the bound admits.
    pub(super) fn first(&self, lp: LoopId) -> Option<BlockId> {
        self.first[lp.index()]
    }

    /// The targets of the exits of `lp` that land in the loop around it, in
    /// edge order.
    pub(super) fn local(&self, lp: LoopId) -> &[BlockId] {
        &self.local[lp.index()]
    }
}

fn position(lp: LoopId) -> u32 {
    position_of(lp.index())
}

fn position_of(index: usize) -> u32 {
    u32::try_from(index).expect("a loop position fits in u32")
}

/// The first loop at or above `at` that has no first exit yet, or the root.
fn find(next: &mut [u32], mut at: u32) -> u32 {
    while next[at as usize] != at {
        next[at as usize] = next[next[at as usize] as usize];
        at = next[at as usize];
    }
    at
}

/// The loops around each loop, `2^k` steps up: how far a loop climbs while a
/// test that holds for the loops below and fails above keeps holding.
struct Lifting {
    steps: Vec<Vec<u32>>,
}

impl Lifting {
    fn new(enclosing: &[u32]) -> Self {
        let mut steps = alloc::vec![enclosing.to_vec()];
        let levels = enclosing.len().bit_width();
        for _ in 1..levels {
            let below = steps.last().expect("one level exists");
            let above = below
                .iter()
                .map(|&lp| if lp == NONE { NONE } else { below[lp as usize] })
                .collect();
            steps.push(above);
        }
        Self { steps }
    }

    /// The highest loop at or above `from` for which `holds` is true, given
    /// that it is true for `from` and for each loop below the answer.
    fn highest(&self, from: u32, holds: impl Fn(u32) -> bool) -> u32 {
        let mut at = from;
        for level in self.steps.iter().rev() {
            let above = level[at as usize];
            if above != NONE && holds(above) {
                at = above;
            }
        }
        at
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::collections::BTreeSet;
    use alloc::rc::Rc;
    use alloc::vec::Vec;

    use super::super::bound::Bound;
    use super::super::loops::{LoopRef, first_exit_into, scan_endless_follow};
    use super::LoopExits;
    use crate::block::BlockId;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::graph::dominator::DominatorTree;
    use crate::graph::loop_forest::LoopForest;
    use crate::test_util::rng::Rng;

    /// A graph of up to fourteen blocks whose edges are drawn at random: jumps,
    /// branches, switches, and back edges, so loops nest, share headers, and
    /// leave each other at random blocks, and some are irreducible.
    fn random_graph(random: &mut Rng) -> Cfg<u8> {
        let count = 4 + random.below(11);
        let mut cfg: Cfg<u8> = Cfg::new();
        let ids: Vec<BlockId> = core::iter::once(cfg.entry())
            .chain((1..count).map(|_| cfg.new_block()))
            .collect();
        for &block in &ids {
            match random.below(10) {
                0 => {}
                1..=4 => {
                    let kind =
                        [EdgeKind::Fallthrough, EdgeKind::Jump, EdgeKind::Back][random.below(3)];
                    cfg.add_edge(block, ids[random.below(count)], kind);
                }
                5..=8 => {
                    cfg.add_edge(block, ids[random.below(count)], EdgeKind::ConditionalTrue);
                    cfg.add_edge(block, ids[random.below(count)], EdgeKind::ConditionalFalse);
                }
                _ => {
                    for _ in 0..=random.below(3) {
                        cfg.add_edge(block, ids[random.below(count)], EdgeKind::SwitchCase);
                    }
                }
            }
        }
        cfg
    }

    #[test]
    fn the_exits_table_agrees_with_the_scan_of_the_loop() {
        let mut random = Rng::new(0xE8_17_5A_11);
        let mut answered = 0_usize;
        for sample in 0..30_000 {
            let cfg = random_graph(&mut random);
            let dom = DominatorTree::compute(&cfg);
            let forest = Rc::new(LoopForest::compute_all_tagged(&cfg, &dom));
            let mut reachable = alloc::vec![false; cfg.block_bound()];
            for block in cfg.block_ids() {
                reachable[block.index()] = dom.is_reachable(block);
            }
            let exits = LoopExits::compute(&cfg, &forest, &reachable);
            // A set that admits some blocks and not others, to reject some of
            // the local exits of a loop.
            let set: BTreeSet<BlockId> = cfg.block_ids().filter(|_| random.chance(50)).collect();
            for lp in forest.loops().filter(|&lp| forest.is_nested(lp)) {
                let natural = LoopRef {
                    forest: &forest,
                    lp,
                };
                let whole = Bound::Whole;
                assert_eq!(
                    exits.first(lp),
                    scan_endless_follow(&cfg, natural, &reachable, &whole),
                    "sample {sample}: first exit of {lp:?}"
                );
                let Some(around) = forest.enclosing(lp) else {
                    continue;
                };
                let beside = Bound::Set(Rc::new(set.clone()));
                for outer in [whole, beside] {
                    let within = outer.within_loop(&forest, around);
                    let (_, rest) = within.within().expect("a loop bound");
                    let table = exits
                        .local(lp)
                        .iter()
                        .copied()
                        .find(|&target| rest.contains(target))
                        .or_else(|| exits.first(lp));
                    assert_eq!(
                        table,
                        scan_endless_follow(&cfg, natural, &reachable, &within),
                        "sample {sample}: follow of {lp:?} inside {around:?}"
                    );
                    answered += 1;
                }
            }
        }
        assert!(answered > 1_000, "the graphs have nested loops: {answered}");
    }

    #[test]
    fn the_exits_into_a_set_agree_with_the_scan_of_the_loop() {
        let mut random = Rng::new(0x5E_7C_A9_03);
        let mut admitted = 0_usize;
        for sample in 0..30_000 {
            let cfg = random_graph(&mut random);
            let dom = DominatorTree::compute(&cfg);
            let forest = Rc::new(LoopForest::compute_all_tagged(&cfg, &dom));
            let mut reachable = alloc::vec![false; cfg.block_bound()];
            for block in cfg.block_ids() {
                reachable[block.index()] = dom.is_reachable(block);
            }
            let set: BTreeSet<BlockId> = cfg.block_ids().filter(|_| random.chance(40)).collect();
            for lp in forest.loops() {
                let natural = LoopRef {
                    forest: &forest,
                    lp,
                };
                let beside = Bound::Set(Rc::new(set.clone()));
                let mut bounds = alloc::vec![beside.clone()];
                if let Some(around) = forest.loops().find(|&other| other != lp) {
                    bounds.push(beside.within_loop(&forest, around));
                }
                for allowed in bounds {
                    let read = first_exit_into(&cfg, natural, &reachable, &allowed, &set)
                        .inspect(|_| admitted += 1)
                        .or_else(|| scan_endless_follow(&cfg, natural, &reachable, &Bound::Whole));
                    assert_eq!(
                        read,
                        scan_endless_follow(&cfg, natural, &reachable, &allowed),
                        "sample {sample}: follow of {lp:?} inside a set"
                    );
                }
            }
        }
        assert!(admitted > 1_000, "some exits are admitted: {admitted}");
    }
}
