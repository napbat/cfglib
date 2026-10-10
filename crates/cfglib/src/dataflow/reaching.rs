//! Reaching definitions analysis.
//!
//! A **forward** data flow analysis that computes, for each program point,
//! the set of definitions (writes) that may reach it without being killed
//! (overwritten) along the way.
//!
//! An unwind that leaves its block before a throwing instruction carries only
//! the definitions completed before that instruction, so the solve runs on
//! the [edge-sensitive solver](super::edge_fixpoint).

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::edge_fixpoint::{self, EdgeProblem};
use super::fixpoint::{Direction, Facts};
use super::{DefSite, InstrInfo, VariableId, departs_before_throws};
use crate::block::BlockId;
use crate::cfg::{Cfg, CfgEdge};

/// A reaching definition: which IR variable was defined, and where.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReachingDef<V> {
    /// The IR variable that was written.
    pub variable: V,
    /// Where the write happened.
    pub site: DefSite,
}

/// The reaching definitions problem.
///
/// A normal edge carries the definitions reaching its source's end. An
/// edge that leaves before throwing instructions carries the definitions
/// reaching the point before each of them, so a throwing instruction's own
/// definitions and every later one in its block never reach the handler.
pub struct ReachingDefsProblem;

impl ReachingDefsProblem {
    /// Replays `block` over the definitions reaching its entry, handing the
    /// definitions that reach each throwing instruction to `before_unwind`.
    fn replay<I: InstrInfo, E>(
        cfg: &Cfg<I, E>,
        block: BlockId,
        input: &BTreeSet<ReachingDef<I::Variable>>,
        mut before_unwind: impl FnMut(&BTreeSet<ReachingDef<I::Variable>>),
    ) -> BTreeSet<ReachingDef<I::Variable>> {
        let mut reaching = input.clone();
        for (inst_idx, instruction) in cfg.block(block).instructions().iter().enumerate() {
            if instruction.may_unwind() {
                before_unwind(&reaching);
            }
            let defs = instruction.defs();
            if defs.is_empty() {
                continue;
            }
            let site = DefSite { block, inst_idx };
            // Kill: remove all previous defs of the same variables.
            for variable in defs {
                reaching.retain(|definition| &definition.variable != variable);
            }
            // Gen: add the new defs.
            for variable in defs {
                reaching.insert(ReachingDef {
                    variable: variable.clone(),
                    site,
                });
            }
        }
        reaching
    }
}

impl<I: InstrInfo, E> EdgeProblem<Cfg<I, E>> for ReachingDefsProblem {
    type Fact = BTreeSet<ReachingDef<I::Variable>>;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _cfg: &Cfg<I, E>) -> Self::Fact {
        BTreeSet::new()
    }

    fn meet(&self, left: &Self::Fact, right: &Self::Fact) -> Self::Fact {
        left.union(right).cloned().collect()
    }

    fn transfer_node(&self, cfg: &Cfg<I, E>, block: BlockId, input: &Self::Fact) -> Self::Fact {
        Self::replay(cfg, block, input, |_| {})
    }

    fn transfer_edge(
        &self,
        cfg: &Cfg<I, E>,
        edge: CfgEdge<'_, E>,
        input: &Self::Fact,
        output: &Self::Fact,
    ) -> Self::Fact {
        if !departs_before_throws(cfg, edge) {
            return output.clone();
        }
        let mut departing = BTreeSet::new();
        Self::replay(cfg, edge.source(), input, |reaching| {
            departing.extend(reaching.iter().cloned());
        });
        departing
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
    /// Blocks unreachable from the entry receive no definitions, and
    /// contribute none to the blocks they branch to.
    ///
    /// # Panics
    ///
    /// Panics only if the unbounded fixpoint solve reports a step-limit
    /// error, which the unbounded configuration cannot produce.
    #[must_use]
    pub fn compute<I: InstrInfo<Variable = V>, E>(cfg: &Cfg<I, E>) -> Self {
        let reachable = cfg.depth_first_preorder();
        let facts = edge_fixpoint::solve_edge_problem_from(cfg, &ReachingDefsProblem, &reachable)
            .expect("an unbounded solve cannot exceed a step limit");
        Self {
            inner: facts.into_block_facts(),
        }
    }

    /// Definitions reaching the **entry** of a block (before any
    /// instruction in the block executes).
    ///
    /// An unwind predecessor contributes only the definitions completed
    /// before its throwing instructions.
    #[must_use]
    pub fn reaching_in(&self, block: BlockId) -> &BTreeSet<ReachingDef<V>> {
        self.inner.fact_in(block)
    }

    /// Definitions reaching the **exit** of a block after all of its
    /// instructions complete normally.
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

    #[test]
    fn a_handler_receives_only_the_definitions_completed_before_each_throw() {
        use crate::edge::EdgeKind;

        // entry: x = prior
        // protected: x = load (throws); y = 1; call (throws); x = after
        // `protected` falls through to `normal` and unwinds to `handler`.
        let mut cfg = Cfg::new();
        let protected = cfg.new_block();
        let normal = cfg.new_block();
        let handler = cfg.new_block();
        cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
        cfg.add_edge(protected, normal, EdgeKind::Fallthrough);
        cfg.add_edge(protected, handler, EdgeKind::ExceptionUnwind);
        cfg.block_mut(cfg.entry()).push(def("prior", 0));
        cfg.block_mut(protected).instructions_mut().extend([
            with_effect(def("load", 0), FlowEffect::MayThrow),
            def("y", 1),
            with_effect(ff("call"), FlowEffect::MayThrow),
            def("after", 0),
        ]);

        let rd = ReachingDefs::compute(&cfg);
        let at = |block, inst_idx| DefSite { block, inst_idx };
        assert_eq!(
            rd.defs_of_at_entry(&0, handler),
            [at(cfg.entry(), 0), at(protected, 0)],
            "the faulting load reaches only the later throw; the last write none"
        );
        assert_eq!(rd.defs_of_at_entry(&1, handler), [at(protected, 1)]);
        assert_eq!(rd.defs_of_at_entry(&0, normal), [at(protected, 3)]);
        assert_eq!(rd.defs_of_at_entry(&1, normal), [at(protected, 1)]);
    }

    #[test]
    fn an_unreachable_predecessor_contributes_no_definitions() {
        use crate::edge::EdgeKind;

        let mut cfg = Cfg::new();
        let unreachable = cfg.new_block();
        let merge = cfg.new_block();
        cfg.add_edge(cfg.entry(), merge, EdgeKind::Fallthrough);
        cfg.add_edge(unreachable, merge, EdgeKind::Fallthrough);
        cfg.block_mut(cfg.entry()).push(def("reached", 0));
        cfg.block_mut(unreachable).push(def("never", 0));

        let rd = ReachingDefs::compute(&cfg);
        assert_eq!(
            rd.defs_of_at_entry(&0, merge),
            [DefSite {
                block: cfg.entry(),
                inst_idx: 0,
            }]
        );
    }
}
