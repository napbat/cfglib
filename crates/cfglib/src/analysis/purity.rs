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
use crate::dataflow::EffectInfo;
use crate::dataflow::cfg_view::CfgView;

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

/// The purity of the blocks `blocks` taken together.
fn purity_of_blocks<G>(
    cfg: &G,
    blocks: impl IntoIterator<Item = BlockId>,
) -> Purity<<G::Inst as EffectInfo>::Effect>
where
    G: CfgView,
    G::Inst: EffectInfo,
{
    let mut all = Vec::new();
    for block in blocks {
        for inst in cfg.instructions(block) {
            all.extend_from_slice(inst.effects());
        }
    }
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
pub fn block_purity<G>(cfg: &G, block: BlockId) -> Purity<<G::Inst as EffectInfo>::Effect>
where
    G: CfgView,
    G::Inst: EffectInfo,
{
    purity_of_blocks(cfg, [block])
}

/// Analyze purity of the entire CFG.
#[must_use]
pub fn cfg_purity<G>(cfg: &G) -> Purity<<G::Inst as EffectInfo>::Effect>
where
    G: CfgView,
    G::Inst: EffectInfo,
{
    purity_of_blocks(cfg, cfg.node_ids())
}

/// Collect per-block purity for every block in the CFG.
#[must_use]
pub fn block_purities<G>(cfg: &G) -> Vec<(BlockId, Purity<<G::Inst as EffectInfo>::Effect>)>
where
    G: CfgView,
    G::Inst: EffectInfo,
{
    cfg.node_ids()
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
            impure("store", TestEffect::MemoryWrite),
            impure("load", TestEffect::MemoryRead),
            impure("store", TestEffect::MemoryWrite),
        ])
        .unwrap();
        assert_eq!(
            cfg_purity(&cfg),
            Purity::Impure(vec![TestEffect::MemoryRead, TestEffect::MemoryWrite]),
            "the effects come back sorted and without repeats"
        );
    }

    fn mixed_cfg() -> crate::cfg::Cfg<crate::test_util::DfInst> {
        CfgBuilder::build(vec![
            pure("add"),
            df_with_effect(pure("if"), FlowEffect::ConditionalOpen),
            impure("store", TestEffect::MemoryWrite),
            df_with_effect(pure("else"), FlowEffect::ConditionalAlternate),
            impure("load", TestEffect::MemoryRead),
            df_with_effect(pure("endif"), FlowEffect::ConditionalClose),
        ])
        .unwrap()
    }

    #[test]
    fn mixed_block_purity() {
        let cfg = mixed_cfg();
        // Entry block (has "add") should be pure.
        assert!(block_purity(&cfg, cfg.entry()).is_pure());
        // The whole CFG is impure because the branches store and load.
        assert_eq!(
            cfg_purity(&cfg),
            Purity::Impure(vec![TestEffect::MemoryRead, TestEffect::MemoryWrite])
        );
    }

    #[test]
    fn cfg_purity_is_the_union_of_its_blocks() {
        let cfg = mixed_cfg();
        let mut union = Vec::new();
        for (_, purity) in block_purities(&cfg) {
            if let Purity::Impure(effects) = purity {
                union.extend(effects);
            }
        }
        union.sort();
        union.dedup();
        assert_eq!(cfg_purity(&cfg), Purity::Impure(union));
    }
}
