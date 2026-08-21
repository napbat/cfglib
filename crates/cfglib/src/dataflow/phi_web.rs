//! Phi webs for renamed SSA values.
//!
//! Values connected by phis form congruence classes that are useful for copy
//! coalescing and register allocation.

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::dataflow::VariableId;
use crate::dataflow::ssa::{SsaForm, SsaValue};

#[derive(Debug, Clone)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, mut index: usize) -> usize {
        while self.parent[index] != index {
            self.parent[index] = self.parent[self.parent[index]];
            index = self.parent[index];
        }
        index
    }

    fn union(&mut self, left: usize, right: usize) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return;
        }

        match self.rank[left_root].cmp(&self.rank[right_root]) {
            Ordering::Less => self.parent[left_root] = right_root,
            Ordering::Greater => self.parent[right_root] = left_root,
            Ordering::Equal => {
                self.parent[right_root] = left_root;
                self.rank[left_root] += 1;
            }
        }
    }
}

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

/// Compute phi congruence classes from a renamed SSA form.
#[must_use]
pub fn compute_phi_webs<V: VariableId>(ssa: &SsaForm<V>) -> PhiWebs<V> {
    let mut all_values = Vec::new();
    let mut value_to_index = BTreeMap::new();

    for (_, phi) in ssa.phis() {
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

    let mut union_find = UnionFind::new(all_values.len());
    for (_, phi) in ssa.phis() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::dataflow::ssa::build_ssa;
    use crate::edge::EdgeKind;
    use crate::graph::dominator::DominatorTree;
    use crate::test_util::{DfInst, df_def, df_use};

    #[test]
    fn empty_ssa_has_no_webs() {
        let cfg = Cfg::<DfInst>::new();
        let dom = DominatorTree::compute(&cfg);
        let ssa = build_ssa(&cfg, &dom);
        assert_eq!(compute_phi_webs(&ssa).webs.len(), 0);
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
        let ssa = build_ssa(&cfg, &dom);
        let webs = compute_phi_webs(&ssa);
        assert_eq!(webs.webs.len(), 1);
        assert_eq!(webs.webs[0].values.len(), 3);
    }
}
