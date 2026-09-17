//! Purity classification.
//!
//! Determines whether a block or entire CFG is **pure** (no observable
//! side effects) or **impure** based on the instruction-level side
//! effect declarations.
//!
//! An instruction type declares its side effects through
//! [`EffectInfo`] in its own effect vocabulary — machine memory/IO for a
//! binary adapter, allocation/panics/channel sends for a source language.

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::EffectInfo;

/// Purity verdict for a block or CFG, over a consumer effect vocabulary `E`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Purity<E> {
    /// No side effects at all.
    Pure,
    /// Has side effects — carries the sorted, deduplicated set of observed
    /// effect kinds.
    Impure(Vec<E>),
}

impl<E> Purity<E> {
    /// Returns `true` if pure.
    #[must_use]
    pub fn is_pure(&self) -> bool {
        matches!(self, Purity::Pure)
    }

    /// Returns `true` if impure.
    #[must_use]
    pub fn is_impure(&self) -> bool {
        !self.is_pure()
    }
}

fn collect_effects<E: Clone + Ord>(mut all: Vec<E>) -> Purity<E> {
    if all.is_empty() {
        Purity::Pure
    } else {
        all.sort();
        all.dedup();
        Purity::Impure(all)
    }
}

/// Analyze purity of a single block.
#[must_use]
pub fn block_purity<I: EffectInfo>(cfg: &Cfg<I>, block: BlockId) -> Purity<I::Effect> {
    let mut all = Vec::new();
    for inst in cfg.block(block).instructions() {
        all.extend_from_slice(inst.effects());
    }
    collect_effects(all)
}

/// Analyze purity of the entire CFG.
#[must_use]
pub fn cfg_purity<I: EffectInfo>(cfg: &Cfg<I>) -> Purity<I::Effect> {
    let mut all = Vec::new();
    for b in cfg.blocks() {
        for inst in b.instructions() {
            all.extend_from_slice(inst.effects());
        }
    }
    collect_effects(all)
}

/// Collect per-block purity for every block in the CFG.
#[must_use]
pub fn block_purities<I: EffectInfo>(cfg: &Cfg<I>) -> Vec<(BlockId, Purity<I::Effect>)> {
    cfg.block_ids()
        .map(|block| (block, block_purity(cfg, block)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::CfgBuilder;
    use crate::flow::FlowEffect;
    use crate::test_util::{TestEffect, df_impure as impure, df_pure as pure, df_with_effect};
    use alloc::vec;

    #[test]
    fn pure_cfg() {
        let cfg = CfgBuilder::build(vec![pure("add"), pure("mul")]).unwrap();
        assert!(cfg_purity(&cfg).is_pure());
    }

    #[test]
    fn impure_cfg() {
        let cfg = CfgBuilder::build(vec![
            impure("load", TestEffect::MemoryRead),
            impure("store", TestEffect::MemoryWrite),
        ])
        .unwrap();
        let p = cfg_purity(&cfg);
        assert!(p.is_impure());
        if let Purity::Impure(effs) = p {
            assert_eq!(effs, vec![TestEffect::MemoryRead, TestEffect::MemoryWrite]);
        }
    }

    #[test]
    fn mixed_block_purity() {
        let cfg = CfgBuilder::build(vec![
            pure("add"),
            df_with_effect(pure("if"), FlowEffect::ConditionalOpen),
            impure("store", TestEffect::MemoryWrite),
            df_with_effect(pure("else"), FlowEffect::ConditionalAlternate),
            pure("nop"),
            df_with_effect(pure("endif"), FlowEffect::ConditionalClose),
        ])
        .unwrap();
        // Entry block (has "add") should be pure.
        assert!(block_purity(&cfg, cfg.entry()).is_pure());
        // The whole CFG is impure because one branch stores.
        assert!(cfg_purity(&cfg).is_impure());
    }
}
