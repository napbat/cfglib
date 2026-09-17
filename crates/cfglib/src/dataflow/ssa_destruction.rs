//! SSA destruction support.
//!
//! Converts renamed phis into parallel copies on incoming CFG edges. The
//! consumer remains responsible for materializing those copies in its native
//! instruction representation and for sequencing copy cycles.

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::dataflow::VariableId;
use crate::dataflow::ssa::{SsaForm, SsaValue};

/// A copy to be materialized on a specific CFG edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhiCopy<V> {
    /// Predecessor block containing the source side of the edge.
    pub from_block: BlockId,
    /// Block containing the lowered phi.
    pub to_block: BlockId,
    /// SSA value defined by the phi.
    pub destination: SsaValue<V>,
    /// SSA value supplied by `from_block`.
    pub source: SsaValue<V>,
}

/// Compute all edge copies needed to eliminate the phis in `ssa`.
#[must_use]
pub fn eliminate_phis<V: VariableId>(ssa: &SsaForm<V>) -> Vec<PhiCopy<V>> {
    let mut copies = Vec::new();

    for (block, phi) in ssa.phis() {
        for (predecessor, source) in &phi.operands {
            copies.push(PhiCopy {
                from_block: *predecessor,
                to_block: block,
                destination: phi.result.clone(),
                source: source.clone(),
            });
        }
    }

    copies
}

/// Group phi copies by the predecessor block where they must be emitted.
///
/// Copies in one group form a parallel assignment and may need a temporary
/// when the native instruction representation lowers a cycle.
#[must_use]
pub fn copies_by_predecessor<V>(copies: &[PhiCopy<V>]) -> Vec<(BlockId, Vec<&PhiCopy<V>>)> {
    let mut by_predecessor: BTreeMap<BlockId, Vec<&PhiCopy<V>>> = BTreeMap::new();
    for copy in copies {
        by_predecessor
            .entry(copy.from_block)
            .or_default()
            .push(copy);
    }
    by_predecessor.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;

    use crate::edge::EdgeKind;
    use crate::graph::dominator::DominatorTree;
    use crate::test_util::{DfInst, df_def, df_use};
    use alloc::vec;

    #[test]
    fn diamond_phis_lower_to_renamed_copies() {
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
        let copies = eliminate_phis(&ssa);
        let merge_copies: Vec<_> = copies
            .iter()
            .filter(|copy| copy.to_block == merge)
            .collect();

        assert_eq!(merge_copies.len(), 2);
        assert!(merge_copies.iter().all(|copy| {
            copy.destination.variable == 0
                && copy.source.variable == 0
                && copy.destination != copy.source
        }));
    }

    #[test]
    fn copies_are_grouped_by_predecessor() {
        let copies = vec![
            PhiCopy {
                from_block: BlockId::from_raw(0),
                to_block: BlockId::from_raw(2),
                destination: SsaValue::new(0_u16, 3),
                source: SsaValue::new(0_u16, 1),
            },
            PhiCopy {
                from_block: BlockId::from_raw(1),
                to_block: BlockId::from_raw(2),
                destination: SsaValue::new(0_u16, 3),
                source: SsaValue::new(0_u16, 2),
            },
        ];

        let grouped = copies_by_predecessor(&copies);
        assert_eq!(grouped.len(), 2);
        assert!(grouped.iter().all(|(_, group)| group.len() == 1));
    }
}
