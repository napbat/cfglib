//! Phi webs for renamed SSA values.
//!
//! Values connected by phis form congruence classes that are useful for copy
//! coalescing and register allocation.

extern crate alloc;
use crate::dataflow::VariableId;
use crate::dataflow::ssa::{SsaForm, SsaValue};
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
        let phis: Vec<&crate::dataflow::ssa::SsaPhi<V>> = ssa.phis().map(|(_, phi)| phi).collect();
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
        let mut used: BTreeSet<SsaValue<V>> = ssa
            .blocks()
            .iter()
            .flat_map(|block| block.instructions.iter())
            .flat_map(|instruction| instruction.uses.iter().cloned())
            .collect();
        let phis: Vec<&crate::dataflow::ssa::SsaPhi<V>> = ssa.phis().map(|(_, phi)| phi).collect();
        let mut live = vec![false; phis.len()];
        loop {
            let mut changed = false;
            for (index, phi) in phis.iter().enumerate() {
                if !live[index] && used.contains(&phi.result) {
                    live[index] = true;
                    for (_, operand) in &phi.operands {
                        used.insert(operand.clone());
                    }
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let live_phis: Vec<&crate::dataflow::ssa::SsaPhi<V>> = phis
            .into_iter()
            .zip(&live)
            .filter_map(|(phi, &live)| live.then_some(phi))
            .collect();
        Self::from_phis(&live_phis)
    }

    fn from_phis(phis: &[&crate::dataflow::ssa::SsaPhi<V>]) -> Self {
        let mut all_values = Vec::new();
        let mut value_to_index = BTreeMap::new();

        for phi in phis {
            for value in
                core::iter::once(&phi.result).chain(phi.operands.iter().map(|(_, value)| value))
            {
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
            for (_, operand) in &phi.operands {
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
}
