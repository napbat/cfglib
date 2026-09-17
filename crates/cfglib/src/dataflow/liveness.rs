//! Liveness analysis.
//!
//! A **backward** data flow analysis that computes, for each program
//! point, the set of variables whose values may be read in the future
//! before being overwritten.
//!
//! A variable is **live** at a point if there exists a path from that
//! point to a use of the variable with no intervening definition.

extern crate alloc;
use alloc::collections::BTreeSet;

use super::fixpoint::{self, Direction, Facts, Problem};
use super::{InstrInfo, VariableId};
use crate::block::BlockId;
use crate::cfg::Cfg;

/// The liveness problem.
pub struct LivenessProblem;

impl<I: InstrInfo, E> Problem<I, E> for LivenessProblem {
    type Fact = BTreeSet<I::Variable>;

    fn direction(&self) -> Direction {
        Direction::Backward
    }

    fn bottom(&self) -> Self::Fact {
        BTreeSet::new()
    }

    fn entry_fact(&self) -> Self::Fact {
        // Nothing is live after program exit.
        BTreeSet::new()
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact {
        a.union(b).cloned().collect()
    }

    /// Backward transfer: `live_in` = uses ∪ (`live_out` − defs).
    ///
    /// Walk the block's instructions in **reverse** to compute the
    /// set of variables live at the block's entry.
    fn transfer(&self, cfg: &Cfg<I, E>, block: BlockId, live_out: &Self::Fact) -> Self::Fact {
        let mut live = live_out.clone();
        for inst in cfg.block(block).instructions().iter().rev() {
            backward_transfer(&mut live, inst);
        }
        live
    }
}

/// One instruction's backward step: kill its defs, then gen its uses.
fn backward_transfer<I: InstrInfo>(live: &mut BTreeSet<I::Variable>, instruction: &I) {
    for variable in instruction.defs() {
        live.remove(variable);
    }
    for variable in instruction.uses() {
        live.insert(variable.clone());
    }
}

/// Result of a liveness analysis with convenient query methods.
///
/// # Examples
///
/// ```
/// # use cfglib::{Cfg, EdgeKind, InstrInfo};
/// # #[derive(Debug, Clone)]
/// # struct Inst { uses: Vec<u16>, defs: Vec<u16> }
/// # impl InstrInfo for Inst {
/// #     type Variable = u16;
/// #     fn uses(&self) -> &[u16] { &self.uses }
/// #     fn defs(&self) -> &[u16] { &self.defs }
/// # }
/// use cfglib::dataflow::liveness::Liveness;
///
/// let mut cfg = Cfg::<Inst>::new();
/// let b0 = cfg.entry();
/// let b1 = cfg.new_block();
/// cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
///
/// let r0 = 0;
/// cfg.block_mut(b0).push(Inst { uses: vec![], defs: vec![r0] });
/// cfg.block_mut(b1).push(Inst { uses: vec![r0], defs: vec![] });
///
/// let live = Liveness::compute(&cfg);
/// // r0 is live-out of b0 (used in b1).
/// assert!(live.is_live_out(&r0, b0));
/// ```
pub struct Liveness<V> {
    inner: Facts<BTreeSet<V>>,
}

impl<V: VariableId> Liveness<V> {
    /// Run liveness analysis on the given CFG.
    ///
    /// # Panics
    ///
    /// Panics only if the unbounded fixpoint solve reports a step-limit
    /// error, which the unbounded configuration cannot produce.
    #[must_use]
    pub fn compute<I: InstrInfo<Variable = V>, E>(cfg: &Cfg<I, E>) -> Self {
        let result = fixpoint::solve_problem(cfg, &LivenessProblem)
            .expect("an unbounded solve cannot exceed a step limit");
        Self { inner: result }
    }

    /// Variables live at the **entry** of a block.
    #[must_use]
    pub fn live_in(&self, block: BlockId) -> &BTreeSet<V> {
        self.inner.fact_in(block)
    }

    /// Variables live at the **exit** of a block.
    #[must_use]
    pub fn live_out(&self, block: BlockId) -> &BTreeSet<V> {
        self.inner.fact_out(block)
    }

    /// Check if a variable is live at a block's entry.
    #[must_use]
    pub fn is_live_in(&self, variable: &V, block: BlockId) -> bool {
        self.live_in(block).contains(variable)
    }

    /// Check if a variable is live at a block's exit.
    #[must_use]
    pub fn is_live_out(&self, variable: &V, block: BlockId) -> bool {
        self.live_out(block).contains(variable)
    }

    /// Variables live immediately **before** each instruction of `block`.
    ///
    /// Element `i` is the live set at the point just before instruction `i`;
    /// for a non-empty block the first element equals
    /// [`live_in`](Self::live_in). Computed on demand by replaying the
    /// block's backward transfer from [`live_out`](Self::live_out), exactly
    /// as the block-level fixpoint did.
    #[must_use]
    pub fn live_before_instructions<I: InstrInfo<Variable = V>, E>(
        &self,
        cfg: &Cfg<I, E>,
        block: BlockId,
    ) -> alloc::vec::Vec<BTreeSet<V>> {
        let instructions = cfg.block(block).instructions();
        let mut sets = alloc::vec![BTreeSet::new(); instructions.len()];
        let mut live = self.live_out(block).clone();
        for (index, instruction) in instructions.iter().enumerate().rev() {
            backward_transfer(&mut live, instruction);
            sets[index].clone_from(&live);
        }
        sets
    }

    /// Variables live immediately **after** each instruction of `block`.
    ///
    /// Element `i` is the live set at the point just after instruction `i`;
    /// for a non-empty block the last element equals
    /// [`live_out`](Self::live_out). A definition at instruction `i` whose
    /// variable is absent from element `i` is a dead store within this
    /// analysis's precision.
    #[must_use]
    pub fn live_after_instructions<I: InstrInfo<Variable = V>, E>(
        &self,
        cfg: &Cfg<I, E>,
        block: BlockId,
    ) -> alloc::vec::Vec<BTreeSet<V>> {
        let instructions = cfg.block(block).instructions();
        let mut sets = alloc::vec![BTreeSet::new(); instructions.len()];
        let mut live = self.live_out(block).clone();
        for (index, instruction) in instructions.iter().enumerate().rev() {
            sets[index].clone_from(&live);
            backward_transfer(&mut live, instruction);
        }
        sets
    }

    /// All variables that are live somewhere in the program.
    #[must_use]
    pub fn all_live_variables<I: InstrInfo<Variable = V>, E>(
        &self,
        cfg: &Cfg<I, E>,
    ) -> BTreeSet<V> {
        let mut all = BTreeSet::new();
        for block_id in cfg.block_ids() {
            all.extend(self.live_in(block_id).iter().cloned());
            all.extend(self.live_out(block_id).iter().cloned());
        }
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::CfgBuilder;
    use crate::flow::FlowEffect;
    use crate::test_util::{df_def as def, df_ff, df_use as use_, df_with_effect};
    use alloc::vec;

    #[test]
    fn liveness_linear_use_makes_live() {
        // bb0: def r0; use r0
        // r0 should be live-in (because use reads it) and NOT live-out
        // (nothing after the block reads it).
        let cfg = CfgBuilder::build(vec![def("def_r0", 0), use_("use_r0", 0)]).unwrap();
        let live = Liveness::compute(&cfg);
        // r0 is used in the block → live-in should contain r0
        // (the def kills it, but the use is after the def so
        //  backward: use generates, def kills → net: not live-in
        //  actually: backward walk: use_r0 gens r0, def_r0 kills r0 → live_in = {})
        // But there's nothing after, so live_out = {}
        assert!(!live.is_live_out(&0, cfg.entry()));
    }

    #[test]
    fn liveness_use_without_def_is_live_in() {
        // bb0: use r0 (no def) → r0 should be live-in
        let cfg = CfgBuilder::build(vec![use_("use_r0", 0)]).unwrap();
        let live = Liveness::compute(&cfg);
        assert!(live.is_live_in(&0, cfg.entry()));
    }

    #[test]
    fn liveness_dead_def() {
        // bb0: def r0 (never used) → r0 should NOT be live anywhere
        let cfg = CfgBuilder::build(vec![def("def_r0", 0)]).unwrap();
        let live = Liveness::compute(&cfg);
        assert!(!live.is_live_in(&0, cfg.entry()));
        assert!(!live.is_live_out(&0, cfg.entry()));
    }

    #[test]
    fn liveness_across_blocks() {
        // bb0: def r0; if
        // bb1: use r0
        // bb2: (nothing)
        // bb3: endif
        // r0 should be live-out of bb0 because bb1 uses it.
        let cfg = CfgBuilder::build(vec![
            def("def_r0", 0),
            df_with_effect(df_ff("if"), FlowEffect::ConditionalOpen),
            use_("use_r0", 0),
            df_with_effect(df_ff("else"), FlowEffect::ConditionalAlternate),
            df_with_effect(df_ff("endif"), FlowEffect::ConditionalClose),
        ])
        .unwrap();
        let live = Liveness::compute(&cfg);
        assert!(
            live.is_live_out(&0, cfg.entry()),
            "r0 is live-out of entry because the true branch uses it"
        );
    }

    #[test]
    fn liveness_empty_single_block() {
        // Empty CFG — no instructions, nothing should be live.
        use crate::test_util::DfInst;
        let cfg = CfgBuilder::build(alloc::vec::Vec::<DfInst>::new()).unwrap();
        let live = Liveness::compute(&cfg);
        assert!(!live.is_live_in(&0, cfg.entry()));
        assert!(!live.is_live_out(&0, cfg.entry()));
    }

    #[test]
    fn liveness_self_loop() {
        // Self-loop: use r0 in a block that loops to itself.
        // r0 is never defined → live-in = {r0}.
        use crate::cfg::Cfg;
        use crate::edge::EdgeKind;
        use crate::test_util::DfInst;

        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(use_("use_r0", 0));
        cfg.add_edge(cfg.entry(), cfg.entry(), EdgeKind::Back);
        let live = Liveness::compute(&cfg);
        assert!(
            live.is_live_in(&0, cfg.entry()),
            "r0 used but not defined in self-loop"
        );
    }

    #[test]
    fn instruction_granular_sets_refine_the_block_facts() {
        // bb0: def r0; use r0; def r0
        // The final def is a dead store: r0 is not live after it. The first
        // def is live until the use consumes it.
        let cfg =
            CfgBuilder::build(vec![def("first", 0), use_("read", 0), def("dead", 0)]).unwrap();
        let live = Liveness::compute(&cfg);
        let entry = cfg.entry();

        let before = live.live_before_instructions(&cfg, entry);
        let after = live.live_after_instructions(&cfg, entry);
        assert_eq!(before.len(), 3);
        assert_eq!(after.len(), 3);

        assert_eq!(before[0], live.live_in(entry).clone());
        assert_eq!(after[2], live.live_out(entry).clone());

        assert!(after[0].contains(&0), "the first def feeds the use");
        assert!(before[1].contains(&0), "the use reads a live value");
        assert!(!after[2].contains(&0), "the final def is a dead store");
    }

    #[test]
    fn liveness_all_live_variables() {
        let cfg = CfgBuilder::build(vec![use_("use_r0", 0), use_("use_r1", 1), def("def_r2", 2)])
            .unwrap();
        let live = Liveness::compute(&cfg);
        let all = live.all_live_variables(&cfg);
        assert!(all.contains(&0));
        assert!(all.contains(&1));
        // r2 is defined but never used, so not live.
        assert!(!all.contains(&2));
    }
}
