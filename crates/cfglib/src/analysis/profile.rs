//! Profile annotations — edge/block weight management for
//! profile-guided analysis and optimization.
//!
//! Provides utilities to annotate CFG edges with branch probabilities
//! or execution counts, and to derive block frequencies.

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeId;

/// Profile data for a CFG.
#[derive(Debug, Clone)]
pub struct CfgProfile {
    /// Block frequencies.
    pub block_weights: BTreeMap<BlockId, f64>,
    /// Edge weights (branch probabilities or counts).
    pub edge_weights: BTreeMap<EdgeId, f64>,
}

impl CfgProfile {
    /// Build a profile from edge weights already set on the CFG.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, CfgProfile, EdgeKind, set_uniform_edge_weights};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let b1 = cfg.new_block();
    /// cfg.add_edge(cfg.entry(), b1, EdgeKind::Fallthrough);
    ///
    /// set_uniform_edge_weights(&mut cfg);
    /// let profile = CfgProfile::from_edge_weights(&cfg);
    /// assert!(profile.hottest_block().is_some());
    /// ```
    #[must_use]
    pub fn from_edge_weights<I>(cfg: &Cfg<I>) -> Self {
        let mut edge_weights = BTreeMap::new();
        let mut block_weights = BTreeMap::new();

        for edge in cfg.edges() {
            if let Some(w) = edge.weight() {
                edge_weights.insert(edge.id(), w);
            }
        }

        // Derive block weights: sum of incoming edge weights.
        // Entry block gets weight 1.0 if no incoming edges have weights.
        block_weights.insert(cfg.entry(), 1.0);
        for block_id in cfg.block_ids() {
            let bid = block_id;
            if bid == cfg.entry() {
                continue;
            }
            let in_weight: f64 = cfg
                .incoming(bid)
                .filter_map(|eid| edge_weights.get(&eid))
                .sum();
            if in_weight > 0.0 {
                block_weights.insert(bid, in_weight);
            }
        }

        Self {
            block_weights,
            edge_weights,
        }
    }

    /// Get the hottest block (highest frequency).
    #[must_use]
    pub fn hottest_block(&self) -> Option<(BlockId, f64)> {
        self.block_weights
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(&bid, &w)| (bid, w))
    }

    /// Get the coldest block (lowest frequency).
    #[must_use]
    pub fn coldest_block(&self) -> Option<(BlockId, f64)> {
        self.block_weights
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(&bid, &w)| (bid, w))
    }

    /// Hot blocks above a frequency threshold.
    #[must_use]
    pub fn hot_blocks(&self, threshold: f64) -> Vec<BlockId> {
        self.block_weights
            .iter()
            .filter(|&(_, w)| *w >= threshold)
            .map(|(&bid, _)| bid)
            .collect()
    }
}

/// Set uniform edge weights on `cfg` (equal probability for all successors).
pub fn set_uniform_edge_weights<I>(cfg: &mut Cfg<I>) {
    let block_ids: Vec<BlockId> = cfg.block_ids().collect();
    for bid in block_ids {
        let succs: Vec<_> = cfg.outgoing(bid).collect();
        if succs.is_empty() {
            continue;
        }
        let w = 1.0 / crate::usize_to_f64(succs.len());
        for eid in succs {
            cfg.edge_mut(eid).set_weight(Some(w));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::test_util::ff;

    #[test]
    fn uniform_weights() {
        let mut cfg = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("br"));
        cfg.block_mut(a).instructions_mut().push(ff("a"));
        cfg.block_mut(b).instructions_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        set_uniform_edge_weights(&mut cfg);
        let profile = CfgProfile::from_edge_weights(&cfg);
        assert_eq!(profile.edge_weights.len(), 2);
        for &w in profile.edge_weights.values() {
            assert!((w - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn hottest_block() {
        let mut profile = CfgProfile {
            block_weights: BTreeMap::new(),
            edge_weights: BTreeMap::new(),
        };
        let a = BlockId::from_raw(0);
        let b = BlockId::from_raw(1);
        profile.block_weights.insert(a, 10.0);
        profile.block_weights.insert(b, 100.0);
        let (hot, _) = profile.hottest_block().unwrap();
        assert_eq!(hot, b);
    }

    #[test]
    fn hot_blocks_filter() {
        let mut profile = CfgProfile {
            block_weights: BTreeMap::new(),
            edge_weights: BTreeMap::new(),
        };
        profile.block_weights.insert(BlockId::from_raw(0), 1.0);
        profile.block_weights.insert(BlockId::from_raw(1), 50.0);
        profile.block_weights.insert(BlockId::from_raw(2), 100.0);
        let hot = profile.hot_blocks(50.0);
        assert_eq!(hot.len(), 2);
    }
}
