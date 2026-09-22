//! Dead-code analysis — what is dead, reported without removing it.
//!
//! [`DeadCode::compute`] is the single source of truth for deadness:
//! [`dead_code_elimination`](crate::dead_code_elimination) applies its
//! instruction list, and [`remove_dead_code`](crate::remove_dead_code)
//! additionally consumes the structure the removals leave behind. Computing
//! the analysis alone serves the reporting shapes — dead-code diagnostics,
//! IDE greying, metrics — where knowing what is dead matters and deleting it
//! is someone else's decision.

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::{EffectInfo, ProgramPoint, fixpoint};
use crate::graph::traverse::{TraversalDirection, reachable};

/// Everything dead in a CFG, found without mutating it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeadCode {
    /// Instructions whose definitions are never used, in block order and
    /// ascending instruction order within a block.
    ///
    /// An instruction with declared side effects is never listed, and a dead
    /// instruction's uses keep nothing alive, so a chain feeding only a dead
    /// instruction is listed with it. Instructions of unreachable blocks are
    /// liveness-dead like any other (no live-out ever reaches them).
    ///
    /// [`ProgramPoint`] indices are positions, not identities: any
    /// instruction edit invalidates the analysis.
    pub instructions: Vec<ProgramPoint>,
    /// Blocks with no path from the entry, in allocation order.
    ///
    /// Reported separately from [`instructions`](Self::instructions) because
    /// they are dead for a different reason: no path executes them, side
    /// effects notwithstanding.
    pub unreachable_blocks: Vec<BlockId>,
}

impl DeadCode {
    /// Analyze `cfg` for dead instructions and unreachable blocks.
    ///
    /// Nothing leaves the graph: a definition survives only because
    /// something inside reads it. [`compute_with_exits`](Self::compute_with_exits)
    /// is the same analysis told what leaves.
    ///
    /// Requiring [`EffectInfo`] is deliberate, exactly as in
    /// [`dead_code_elimination`](crate::dead_code_elimination): a consumer
    /// must state an effect vocabulary before anything is called dead, which
    /// prevents silently classifying side-effecting statements as removable.
    ///
    /// # Panics
    ///
    /// Panics only if the unbounded fixpoint solve reports a step-limit
    /// error, which the unbounded configuration cannot produce.
    #[must_use]
    pub fn compute<I: EffectInfo, E>(cfg: &Cfg<I, E>) -> Self {
        Self::compute_with_exits(cfg, |_| Vec::new())
    }

    /// Analyze `cfg` with `live_out` stating what leaves each block.
    ///
    /// A function's own graph never says that a value outlives it — what a
    /// return hands back, the storage a caller reads again, the state
    /// something reached through a non-returning exit observes — so the
    /// definitions that produce those values read as dead. `live_out`
    /// states them: its variables are live at the end of the named block,
    /// and everything their definitions transitively read stays live with
    /// them. A caller normally answers for the blocks with no successors
    /// and hands back an empty vector everywhere else.
    ///
    /// # Panics
    ///
    /// Panics only if the unbounded fixpoint solve reports a step-limit
    /// error, which the unbounded configuration cannot produce.
    #[must_use]
    pub fn compute_with_exits<I: EffectInfo, E>(
        cfg: &Cfg<I, E>,
        live_out: impl Fn(BlockId) -> Vec<I::Variable>,
    ) -> Self {
        use crate::dataflow::liveness::SeededLivenessProblem;

        let problem = SeededLivenessProblem::new(&live_out);
        let liveness = fixpoint::solve_problem(cfg, &problem)
            .expect("an unbounded solve cannot exceed a step limit");

        let mut instructions = Vec::new();
        for block_id in cfg.block_ids() {
            let block = cfg.block(block_id);
            // The solved out-fact carries what flows back from the
            // successors; the seed is what leaves here, exactly as the
            // problem's own transfer joined it.
            let mut live = liveness.fact_out(block_id).clone();
            live.extend(live_out(block_id));
            let insts = block.instructions();
            let mut dead = Vec::new();

            for (index, inst) in insts.iter().enumerate().rev() {
                let has_side_effect = !inst.effects().is_empty();
                let defs_live = inst.defs().iter().any(|def| live.contains(def));

                if !has_side_effect && !inst.defs().is_empty() && !defs_live {
                    dead.push(index);
                } else {
                    for def in inst.defs() {
                        live.remove(def);
                    }
                    for used in inst.uses() {
                        live.insert(used.clone());
                    }
                }
            }

            instructions.extend(dead.into_iter().rev().map(|inst_idx| ProgramPoint {
                block: block_id,
                inst_idx,
            }));
        }

        let reached = reachable(cfg, [cfg.entry()], TraversalDirection::Outgoing);
        let unreachable_blocks = cfg
            .block_ids()
            .filter(|block| !reached[block.index()])
            .collect();

        Self {
            instructions,
            unreachable_blocks,
        }
    }

    /// Whether nothing dead was found.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty() && self.unreachable_blocks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::EdgeKind;
    use crate::test_util::{DfInst, TestEffect, df_def, df_impure, df_use};

    #[test]
    fn a_dead_definition_is_reported_and_a_live_one_is_not() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_def("dead", 0));
        cfg.block_mut(cfg.entry()).push(df_def("live", 1));
        cfg.block_mut(exit).push(df_use("use", 1));
        cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);

        let dead = DeadCode::compute(&cfg);
        assert_eq!(
            dead.instructions,
            alloc::vec![ProgramPoint {
                block: cfg.entry(),
                inst_idx: 0,
            }]
        );
        assert!(dead.unreachable_blocks.is_empty());
    }

    #[test]
    fn a_chain_feeding_only_a_dead_instruction_dies_with_it() {
        // a = ...; b = a; nothing uses b — both are dead in one analysis.
        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).push(df_def("a", 0));
        cfg.block_mut(cfg.entry()).push({
            let mut inst = df_def("b", 1);
            inst.uses.push(0);
            inst
        });

        let dead = DeadCode::compute(&cfg);
        let indices: Vec<usize> = dead.instructions.iter().map(|p| p.inst_idx).collect();
        assert_eq!(indices, alloc::vec![0, 1]);
    }

    #[test]
    fn a_side_effecting_instruction_is_never_reported() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let mut store = df_impure("volatile", TestEffect::MemoryWrite);
        store.defs.push(0);
        cfg.block_mut(cfg.entry()).push(store);

        let dead = DeadCode::compute(&cfg);
        assert!(dead.is_empty());
    }

    #[test]
    fn an_unreachable_block_is_reported_separately() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let orphan = cfg.new_block();
        cfg.block_mut(orphan).push(df_def("stranded", 0));

        let dead = DeadCode::compute(&cfg);
        assert_eq!(dead.unreachable_blocks, alloc::vec![orphan]);
        // Its non-effectful definition is liveness-dead as well.
        assert_eq!(
            dead.instructions,
            alloc::vec![ProgramPoint {
                block: orphan,
                inst_idx: 0,
            }]
        );
    }

    /// A definition nothing inside the graph reads is dead, and the same
    /// definition is live once an exit says the value leaves. The chain
    /// feeding it lives with it.
    #[test]
    fn a_definition_the_exit_seed_observes_is_not_dead() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_def("source", 0));
        cfg.block_mut(exit).push({
            let mut inst = df_def("leaves", 1);
            inst.uses.push(0);
            inst
        });
        cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);

        let unseeded = DeadCode::compute(&cfg);
        assert_eq!(
            unseeded.instructions,
            alloc::vec![ProgramPoint {
                block: exit,
                inst_idx: 0,
            }],
            "nothing inside the graph reads what the exit produces"
        );

        let seeded = DeadCode::compute_with_exits(&cfg, |block| {
            if block == exit {
                alloc::vec![1]
            } else {
                Vec::new()
            }
        });
        assert!(
            seeded.instructions.is_empty(),
            "the seeded definition and its source both survive: {:?}",
            seeded.instructions
        );
    }

    /// A seed names one variable, not the whole exit: an unrelated
    /// definition in the same block stays dead.
    #[test]
    fn an_exit_seed_keeps_only_what_it_names() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).push(df_def("kept", 0));
        cfg.block_mut(cfg.entry()).push(df_def("dropped", 1));

        let dead = DeadCode::compute_with_exits(&cfg, |_| alloc::vec![0]);
        assert_eq!(
            dead.instructions,
            alloc::vec![ProgramPoint {
                block: cfg.entry(),
                inst_idx: 1,
            }]
        );
    }

    #[test]
    fn a_clean_cfg_reports_nothing() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).push(df_def("keep", 0));
        cfg.block_mut(cfg.entry()).push(df_use("use", 0));
        assert!(DeadCode::compute(&cfg).is_empty());
    }
}
