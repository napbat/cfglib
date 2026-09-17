//! Reaching definitions analysis.
//!
//! A **forward** data flow analysis that computes, for each program point,
//! the set of definitions (writes) that may reach it without being killed
//! (overwritten) along the way.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::fixpoint::{self, Direction, Facts, Problem};
use super::{DefSite, InstrInfo, VariableId};
use crate::block::BlockId;
use crate::cfg::Cfg;

/// A reaching definition: which IR variable was defined, and where.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReachingDef<V> {
    /// The IR variable that was written.
    pub variable: V,
    /// Where the write happened.
    pub site: DefSite,
}

/// The reaching definitions problem.
pub struct ReachingDefsProblem;

impl<I: InstrInfo, E> Problem<I, E> for ReachingDefsProblem {
    type Fact = BTreeSet<ReachingDef<I::Variable>>;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self) -> Self::Fact {
        BTreeSet::new()
    }

    fn entry_fact(&self) -> Self::Fact {
        BTreeSet::new()
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact {
        a.union(b).cloned().collect()
    }

    fn transfer(&self, cfg: &Cfg<I, E>, block: BlockId, input: &Self::Fact) -> Self::Fact {
        let mut out = input.clone();
        let insts = cfg.block(block).instructions();

        for (idx, inst) in insts.iter().enumerate() {
            let defs = inst.defs();
            if !defs.is_empty() {
                let site = DefSite {
                    block,
                    inst_idx: idx,
                };
                // Kill: remove all previous defs of the same variables.
                for variable in defs {
                    out.retain(|rd| &rd.variable != variable);
                }
                // Gen: add the new defs.
                for variable in defs {
                    out.insert(ReachingDef {
                        variable: variable.clone(),
                        site,
                    });
                }
            }
        }

        out
    }
}

/// Result of a reaching definitions analysis with convenient query methods.
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
/// use cfglib::dataflow::reaching::ReachingDefs;
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
/// let rd = ReachingDefs::compute(&cfg);
/// // The def of r0 in b0 reaches b1.
/// assert!(!rd.reaching_in(b1).is_empty());
/// ```
pub struct ReachingDefs<V> {
    inner: Facts<BTreeSet<ReachingDef<V>>>,
}

impl<V: VariableId> ReachingDefs<V> {
    /// Run reaching definitions on the given CFG.
    ///
    /// # Panics
    ///
    /// Panics only if the unbounded fixpoint solve reports a step-limit
    /// error, which the unbounded configuration cannot produce.
    #[must_use]
    pub fn compute<I: InstrInfo<Variable = V>, E>(cfg: &Cfg<I, E>) -> Self {
        let result = fixpoint::solve_problem(cfg, &ReachingDefsProblem)
            .expect("an unbounded solve cannot exceed a step limit");
        Self { inner: result }
    }

    /// Definitions reaching the **entry** of a block (before any
    /// instruction in the block executes).
    #[must_use]
    pub fn reaching_in(&self, block: BlockId) -> &BTreeSet<ReachingDef<V>> {
        self.inner.fact_in(block)
    }

    /// Definitions reaching the **exit** of a block (after all
    /// instructions in the block have executed).
    #[must_use]
    pub fn reaching_out(&self, block: BlockId) -> &BTreeSet<ReachingDef<V>> {
        self.inner.fact_out(block)
    }

    /// All definitions of a specific variable reaching a block's entry.
    #[must_use]
    pub fn defs_of_at_entry(&self, variable: &V, block: BlockId) -> Vec<DefSite> {
        self.reaching_in(block)
            .iter()
            .filter(|rd| &rd.variable == variable)
            .map(|rd| rd.site)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::CfgBuilder;
    use crate::flow::FlowEffect;
    use crate::test_util::{
        df_def as def, df_ff as ff, df_use as use_, df_with_effect as with_effect,
    };
    use alloc::vec;

    #[test]
    fn reaching_linear_single_def() {
        // bb0: def r0; use r0
        let cfg = CfgBuilder::build(vec![def("def_r0", 0), use_("use_r0", 0)]).unwrap();
        let rd = ReachingDefs::compute(&cfg);
        let out = rd.reaching_out(cfg.entry());
        assert_eq!(out.len(), 1);
        assert!(out.iter().any(|r| r.variable == 0));
    }

    #[test]
    fn reaching_linear_kill_redefinition() {
        // bb0: def r0; def r0 (again) — first def should be killed
        let cfg = CfgBuilder::build(vec![def("def1", 0), def("def2", 0)]).unwrap();
        let rd = ReachingDefs::compute(&cfg);
        let out = rd.reaching_out(cfg.entry());
        assert_eq!(out.len(), 1);
        let rd_item = out.iter().next().unwrap();
        assert_eq!(rd_item.site.inst_idx, 1); // second def survives
    }

    #[test]
    fn reaching_linear_two_variables() {
        // bb0: def r0; def r1 — both should reach the exit
        let cfg = CfgBuilder::build(vec![def("def_r0", 0), def("def_r1", 1)]).unwrap();
        let rd = ReachingDefs::compute(&cfg);
        let out = rd.reaching_out(cfg.entry());
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn reaching_branch_merges_both_defs() {
        // bb0: if
        // bb1 (true): def r0
        // bb2 (false): def r0
        // bb3 (merge): use r0 — both defs should reach
        let cfg = CfgBuilder::build(vec![
            with_effect(ff("if"), FlowEffect::ConditionalOpen),
            def("def_true", 0),
            with_effect(ff("else"), FlowEffect::ConditionalAlternate),
            def("def_false", 0),
            with_effect(ff("endif"), FlowEffect::ConditionalClose),
            use_("use_r0", 0),
        ])
        .unwrap();
        let merge_block = cfg.block_ids().last().expect("the fixture has blocks");
        let rd = ReachingDefs::compute(&cfg);
        let defs_at_merge = rd.defs_of_at_entry(&0, merge_block);
        assert_eq!(
            defs_at_merge.len(),
            2,
            "both branch defs should reach merge"
        );
    }

    #[test]
    fn reaching_loop_def_reaches_through_backedge() {
        // bb0: def r0; loop; use r0; def r0; endloop
        let cfg = CfgBuilder::build(vec![
            def("init", 0),
            with_effect(ff("loop"), FlowEffect::LoopOpen),
            use_("read", 0),
            def("update", 0),
            with_effect(ff("endloop"), FlowEffect::LoopClose),
        ])
        .unwrap();
        let rd = ReachingDefs::compute(&cfg);
        // The loop header should have defs reaching from both
        // the pre-loop init and the loop body update (via back-edge).
        let header = BlockId::from_raw(1);
        let defs = rd.defs_of_at_entry(&0, header);
        assert!(!defs.is_empty(), "at least the init def reaches the header");
    }
}
