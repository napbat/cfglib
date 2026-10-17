//! Phi webs for renamed SSA values.
//!
//! Values connected by phis form congruence classes that are useful for copy
//! coalescing and register allocation. A phi's inputs are its edge operands
//! and, in a branch-target entry, its [start
//! value](crate::SsaPhi::start_value): the start value joins the web of the
//! phi's result like any operand.

extern crate alloc;
use crate::VariableId;
use crate::dataflow::ssa::{SsaForm, SsaInstruction, SsaPhi, SsaValue};
use crate::union_find::DisjointSet;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

/// A congruence class of SSA values connected by phis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhiWeb<V> {
    /// SSA values in this congruence class.
    pub values: BTreeSet<SsaValue<V>>,
}

/// Result of phi-web computation.
///
/// [`webs`](Self::webs) and [`web_of`](Self::web_of) are two views of one
/// partition: `web_of[value] == index` exactly when `webs[index]` contains
/// `value`. [`PhiWebs::compute`] and [`PhiWebs::compute_live`] produce them
/// together; code that edits either field must keep the other in agreement.
#[derive(Debug, Clone)]
pub struct PhiWebs<V> {
    /// All phi webs found in the SSA form.
    pub webs: Vec<PhiWeb<V>>,
    /// Map from an SSA value to its web index.
    pub web_of: BTreeMap<SsaValue<V>, usize>,
}

impl<V: VariableId> PhiWebs<V> {
    /// Compute phi congruence classes from a renamed SSA form.
    #[must_use]
    pub fn compute(ssa: &SsaForm<V>) -> Self {
        let phis: Vec<&SsaPhi<V>> = ssa.phis().map(|(_, phi)| phi).collect();
        Self::from_phis(&phis)
    }

    /// Compute congruence classes over **live** phis only: a phi is live
    /// when its result reaches an instruction use, directly or through
    /// other live phis.
    ///
    /// Placement is not liveness-pruned, so a join can carry a phi whose
    /// merged value nothing ever reads; including such a phi would unite
    /// lifetimes that never actually flow together. Use this form when the
    /// webs decide variable identity (splitting, coalescing for
    /// destruction) rather than describing the raw SSA structure.
    #[must_use]
    pub fn compute_live(ssa: &SsaForm<V>) -> Self {
        let phis: Vec<&SsaPhi<V>> = ssa.phis().map(|(_, phi)| phi).collect();
        let phi_of: BTreeMap<&SsaValue<V>, usize> = phis
            .iter()
            .enumerate()
            .map(|(index, phi)| (&phi.result, index))
            .collect();
        let mut live = vec![false; phis.len()];
        // Each phi result is defined once, so each phi is taken off the
        // demand stack at most once and its operands become demanded in turn.
        // A start value is a live-in, never a phi result, so demanding it
        // would end at once and the loop leaves it out.
        let mut demanded: Vec<&SsaValue<V>> = ssa
            .blocks()
            .iter()
            .flat_map(|block| &block.instructions)
            .flat_map(SsaInstruction::uses)
            .collect();
        while let Some(value) = demanded.pop() {
            let Some(&index) = phi_of.get(value) else {
                continue;
            };
            if core::mem::replace(&mut live[index], true) {
                continue;
            }
            demanded.extend(phis[index].operands().iter().map(|(_, operand)| operand));
        }
        let live_phis: Vec<&SsaPhi<V>> = phis
            .into_iter()
            .zip(&live)
            .filter_map(|(phi, &live)| live.then_some(phi))
            .collect();
        Self::from_phis(&live_phis)
    }

    fn from_phis(phis: &[&SsaPhi<V>]) -> Self {
        let mut all_values = Vec::new();
        let mut value_to_index = BTreeMap::new();

        for phi in phis {
            let start = phi.start_value();
            for value in core::iter::once(&phi.result).chain(inputs(phi, start.as_ref())) {
                if !value_to_index.contains_key(value) {
                    let index = all_values.len();
                    value_to_index.insert(value.clone(), index);
                    all_values.push(value.clone());
                }
            }
        }

        let mut union_find = DisjointSet::new(all_values.len());
        for phi in phis {
            let result_index = value_to_index[&phi.result];
            let start = phi.start_value();
            for operand in inputs(phi, start.as_ref()) {
                union_find.union(result_index, value_to_index[operand]);
            }
        }

        let mut root_to_web = BTreeMap::new();
        let mut webs: Vec<PhiWeb<V>> = Vec::new();
        let mut web_of = BTreeMap::new();

        for (index, value) in all_values.into_iter().enumerate() {
            let root = union_find.find(index);
            let web_index = if let Some(existing) = root_to_web.get(&root) {
                *existing
            } else {
                let new_index = webs.len();
                webs.push(PhiWeb {
                    values: BTreeSet::new(),
                });
                root_to_web.insert(root, new_index);
                new_index
            };
            webs[web_index].values.insert(value.clone());
            web_of.insert(value, web_index);
        }

        PhiWebs { webs, web_of }
    }
}

/// Every value that flows into `phi`: its edge operands, then `start`, which
/// is the phi's [start value](SsaPhi::start_value) that the caller holds.
///
/// The web of a phi depends on where a value arrives from only in that it
/// arrives, so a start value counts like any operand.
fn inputs<'a, V>(
    phi: &'a SsaPhi<V>,
    start: Option<&'a SsaValue<V>>,
) -> impl Iterator<Item = &'a SsaValue<V>> {
    phi.operands().iter().map(|(_, value)| value).chain(start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;

    use crate::edge::EdgeKind;
    use crate::graph::dominator::DominatorTree;
    use crate::test_util::{DfInst, df_def, df_use};

    #[test]
    fn empty_ssa_has_no_webs() {
        let cfg = Cfg::<DfInst>::new();
        let dom = DominatorTree::compute(&cfg);
        let ssa = SsaForm::compute(&cfg, &dom);
        assert_eq!(
            PhiWebs::compute(&ssa).webs,
            [] as [crate::dataflow::phi_web::PhiWeb<u16>; 0]
        );
    }

    #[test]
    fn dead_loop_header_phi_is_excluded_from_live_webs() {
        // The loop body redefines the variable before any use, so the phi
        // placed at the header merges a value nothing reads. Live webs must
        // not unite the pre-loop and in-loop lifetimes through it.
        let mut cfg = Cfg::<DfInst>::new();
        let top = cfg.new_block();
        let header = cfg.new_block();
        let body = cfg.new_block();
        let exit = cfg.new_block();
        cfg.block_mut(top).push(df_def("pre", 0));
        cfg.block_mut(top).push(df_use("pre_use", 0));
        cfg.block_mut(body).push(df_def("loop", 0));
        cfg.block_mut(body).push(df_use("loop_use", 0));
        cfg.add_edge(cfg.entry(), top, EdgeKind::Fallthrough);
        cfg.add_edge(top, header, EdgeKind::Fallthrough);
        cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(body, header, EdgeKind::Fallthrough);

        let dom = DominatorTree::compute(&cfg);
        let ssa = SsaForm::compute(&cfg, &dom);
        assert_eq!(
            PhiWebs::compute(&ssa).webs.len(),
            1,
            "unpruned placement carries the dead header phi"
        );
        assert!(
            PhiWebs::compute_live(&ssa).webs.is_empty(),
            "no instruction reads a phi result, so no web is live"
        );
    }

    #[test]
    fn diamond_phi_forms_one_web() {
        let mut cfg = Cfg::<DfInst>::new();
        let left = cfg.new_block();
        let right = cfg.new_block();
        let merge = cfg.new_block();
        cfg.block_mut(left).push(df_def("left", 0));
        cfg.block_mut(right).push(df_def("right", 0));
        cfg.block_mut(merge).push(df_use("merged", 0));
        cfg.add_edge(cfg.entry(), left, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), right, EdgeKind::ConditionalFalse);
        cfg.add_edge(left, merge, EdgeKind::Fallthrough);
        cfg.add_edge(right, merge, EdgeKind::Fallthrough);

        let dom = DominatorTree::compute(&cfg);
        let ssa = SsaForm::compute(&cfg, &dom);
        let webs = PhiWebs::compute(&ssa);
        assert_eq!(webs.webs.len(), 1);
        assert_eq!(webs.webs[0].values.len(), 3);
    }

    /// Give each block of `cfg` definitions and uses over three variables so
    /// that merges take phis and only some of the phi results are read.
    fn filled(mut cfg: Cfg<DfInst>) -> Cfg<DfInst> {
        for (index, block) in cfg.block_ids().collect::<Vec<_>>().into_iter().enumerate() {
            let variable = u16::try_from(index % 3).expect("a small index fits in u16");
            cfg.block_mut(block).push(df_def("def", variable));
            if index % 4 == 3 {
                cfg.block_mut(block).push(df_use("use", (variable + 1) % 3));
            }
        }
        cfg
    }

    fn nested_loops() -> Cfg<DfInst> {
        let mut cfg = Cfg::<DfInst>::new();
        let [outer, inner, body, latch, exit] = core::array::from_fn(|_| cfg.new_block());
        cfg.add_edge(cfg.entry(), outer, EdgeKind::Fallthrough);
        cfg.add_edge(outer, inner, EdgeKind::ConditionalTrue);
        cfg.add_edge(outer, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(inner, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(inner, latch, EdgeKind::ConditionalFalse);
        cfg.add_edge(body, inner, EdgeKind::Back);
        cfg.add_edge(latch, outer, EdgeKind::Back);
        filled(cfg)
    }

    /// [`nested_loops`] with the outer header as the entry, so the entry's
    /// phis also merge the value entering the function.
    fn entry_nested_loops() -> Cfg<DfInst> {
        let mut cfg = Cfg::<DfInst>::new();
        let [inner, body, latch, exit] = core::array::from_fn(|_| cfg.new_block());
        let entry = cfg.entry();
        cfg.add_edge(entry, inner, EdgeKind::ConditionalTrue);
        cfg.add_edge(entry, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(inner, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(inner, latch, EdgeKind::ConditionalFalse);
        cfg.add_edge(body, inner, EdgeKind::Back);
        cfg.add_edge(latch, entry, EdgeKind::Back);
        filled(cfg)
    }

    fn family() -> Vec<Cfg<DfInst>> {
        let mut family: Vec<_> = (1..6)
            .map(|regions| filled(crate::test_util::shapes::diamond_chain(regions)))
            .collect();
        family.push(nested_loops());
        family.push(entry_nested_loops());
        family
    }

    /// Whether `start` reaches an instruction use through phis, by searching
    /// forward from the value rather than propagating demand back from uses.
    fn reaches_a_use(ssa: &SsaForm<u16>, start: &SsaValue<u16>) -> bool {
        let mut seen = BTreeSet::new();
        let mut pending = vec![start.clone()];
        while let Some(value) = pending.pop() {
            if !seen.insert(value.clone()) {
                continue;
            }
            let used = ssa
                .blocks()
                .iter()
                .flat_map(|block| &block.instructions)
                .any(|instruction| instruction.uses().contains(&value));
            if used {
                return true;
            }
            pending.extend(
                ssa.phis()
                    .filter(|(_, phi)| phi.operands().iter().any(|(_, operand)| *operand == value))
                    .map(|(_, phi)| phi.result.clone()),
            );
        }
        false
    }

    /// The webs of the live phis, merged by repeated overlap.
    fn expected_live_webs(ssa: &SsaForm<u16>) -> BTreeSet<BTreeSet<SsaValue<u16>>> {
        let mut webs: Vec<BTreeSet<SsaValue<u16>>> = Vec::new();
        for (_, phi) in ssa
            .phis()
            .filter(|(_, phi)| reaches_a_use(ssa, &phi.result))
        {
            let start = phi.start_value();
            let mut merged: BTreeSet<_> = core::iter::once(phi.result.clone())
                .chain(phi.operands().iter().map(|(_, value)| value.clone()))
                .chain(start)
                .collect();
            webs.retain(|web| {
                let overlaps = !web.is_disjoint(&merged);
                if overlaps {
                    merged.extend(web.iter().cloned());
                }
                !overlaps
            });
            webs.push(merged);
        }
        webs.into_iter().collect()
    }

    #[test]
    fn live_webs_unite_exactly_the_phis_whose_results_reach_a_use() {
        let (mut some_dead, mut some_live) = (false, false);
        for cfg in family() {
            let ssa = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
            let live = PhiWebs::compute_live(&ssa);
            let actual: BTreeSet<_> = live.webs.iter().map(|web| web.values.clone()).collect();
            assert_eq!(
                actual,
                expected_live_webs(&ssa),
                "{} blocks",
                cfg.block_count()
            );
            some_dead |= live.web_of.len() < PhiWebs::compute(&ssa).web_of.len();
            some_live |= !live.webs.is_empty();
        }
        assert!(
            some_dead && some_live,
            "the family must mix dead and live phis"
        );
    }

    #[test]
    fn the_web_list_and_the_value_index_describe_one_partition() {
        for cfg in family() {
            let ssa = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
            for webs in [PhiWebs::compute(&ssa), PhiWebs::compute_live(&ssa)] {
                let listed: usize = webs.webs.iter().map(|web| web.values.len()).sum();
                assert_eq!(listed, webs.web_of.len(), "no value is in two webs");
                for (index, web) in webs.webs.iter().enumerate() {
                    assert!(!web.values.is_empty());
                    assert!(web.values.iter().all(|value| webs.web_of[value] == index));
                }
            }
        }
    }

    /// The entry loop: the entry reads variable 0 when `reads` and the latch
    /// redefines it.
    fn entry_loop(reads: bool) -> Cfg<DfInst> {
        let mut cfg = Cfg::<DfInst>::new();
        let [latch, exit] = core::array::from_fn(|_| cfg.new_block());
        let entry = cfg.entry();
        cfg.add_edge(entry, latch, EdgeKind::ConditionalTrue);
        cfg.add_edge(entry, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(latch, entry, EdgeKind::Back);
        if reads {
            cfg.block_mut(entry).push(df_use("read", 0));
        }
        cfg.block_mut(latch).push(df_def("redefine", 0));
        cfg
    }

    #[test]
    fn a_start_value_joins_the_web_of_its_entry_phi() {
        let cfg = entry_loop(true);
        let ssa = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
        let phi = &ssa.block(cfg.entry()).phis[0];
        let expected = BTreeSet::from([
            phi.result.clone(),
            phi.operands()[0].1.clone(),
            SsaValue::live_in(0),
        ]);

        for webs in [PhiWebs::compute(&ssa), PhiWebs::compute_live(&ssa)] {
            let [web] = webs.webs.as_slice() else {
                panic!("one phi, one web: {:?}", webs.webs);
            };
            assert_eq!(web.values, expected);
            assert_eq!(
                webs.web_of[&SsaValue::live_in(0)],
                webs.web_of[&phi.result],
                "the start value is in the web of the phi it feeds"
            );
        }
    }

    #[test]
    fn a_dead_entry_phi_keeps_its_start_value_out_of_the_live_webs() {
        let cfg = entry_loop(false);
        let ssa = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));

        let all = PhiWebs::compute(&ssa);
        assert!(all.web_of.contains_key(&SsaValue::live_in(0)));
        assert!(
            PhiWebs::compute_live(&ssa).webs.is_empty(),
            "no instruction reads the phi, so no web is live"
        );
    }
}
