//! Switch table reconstruction.
//!
//! Detects multi-way dispatch sites and recovers structured
//! [`EdgeKind::SwitchCase`] edges from them. The branch-target token `T` is
//! consumer-typed: a raw address for binary jump tables (x86 tables, ARM
//! TBB/TBH), a CST node or label id for source-level computed gotos and
//! lowered `match` dispatch.
//!
//! Two entry points compose:
//!
//! - [`detect_switch_tables`] scans block terminators via the opt-in
//!   [`SwitchSource`] trait, for consumers whose instructions know their own
//!   target tables.
//! - [`recover_switch_tables`] rewires the CFG from a list of
//!   [`JumpTable`]s — hand-built when table discovery needs external context
//!   (loader state, memory dumps) the instruction cannot see.

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;

/// A recovered multi-way dispatch site over branch-target tokens `T`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct JumpTable<T> {
    /// The block containing the dispatch.
    pub block: BlockId,
    /// Known case targets (resolved by the consumer).
    pub targets: Vec<T>,
    /// Optional default / fallthrough target.
    pub default_target: Option<T>,
}

/// A dispatch table: the case targets plus an optional default target.
pub type SwitchTargets<T> = (Vec<T>, Option<T>);

/// Opt-in: instructions that are multi-way indirect branches with a
/// statically recoverable target table.
pub trait SwitchSource {
    /// Consumer branch-target token.
    type Target: Clone;

    /// The resolved case targets and optional default, when this
    /// instruction is a table dispatch. `None` for everything else.
    fn switch_targets(&self) -> Option<SwitchTargets<Self::Target>>;
}

/// Scan block terminators for switch sources.
///
/// Returns one [`JumpTable`] per block whose final instruction reports
/// targets through [`SwitchSource::switch_targets`].
#[must_use]
pub fn detect_switch_tables<I: SwitchSource>(cfg: &Cfg<I>) -> Vec<JumpTable<I::Target>> {
    let mut tables = Vec::new();
    for block in cfg.blocks() {
        let Some(last) = block.instructions().last() else {
            continue;
        };
        let Some((targets, default_target)) = last.switch_targets() else {
            continue;
        };
        tables.push(JumpTable {
            block: block.id(),
            targets,
            default_target,
        });
    }
    tables
}

/// Result of switch table reconstruction.
#[derive(Debug, Clone)]
pub struct SwitchRecovery {
    /// Block that was converted from indirect jump to switch.
    pub block: BlockId,
    /// Number of case edges added.
    pub num_cases: usize,
}

/// Reconstruct switch tables from detected jump-table patterns.
///
/// For each [`JumpTable`], removes the existing `IndirectJump` edge(s) from
/// the block and replaces them with `SwitchCase` edges to each resolved
/// target (plus an `Unconditional` edge for the default target).
///
/// The `resolve` function maps target tokens to block IDs — the consumer
/// provides it because token-to-block mapping is frontend-specific (an
/// address-to-block map for binaries, a CST-node-to-block map for source).
pub fn recover_switch_tables<I, T>(
    cfg: &mut Cfg<I>,
    tables: &[JumpTable<T>],
    mut resolve: impl FnMut(&T) -> Option<BlockId>,
) -> Vec<SwitchRecovery> {
    let mut results = Vec::new();

    for table in tables {
        // Remove existing IndirectJump edges from this block.
        let edges_to_remove: Vec<_> = cfg
            .successor_edges(table.block)
            .iter()
            .filter(|&&eid| cfg.edge(eid).kind() == EdgeKind::IndirectJump)
            .copied()
            .collect();
        for eid in edges_to_remove {
            cfg.remove_edge(eid);
        }

        let mut num_cases = 0;

        // Add SwitchCase edges for each resolved target.
        for target in &table.targets {
            if let Some(target_block) = resolve(target) {
                cfg.add_edge(table.block, target_block, EdgeKind::SwitchCase);
                num_cases += 1;
            }
        }

        // Add default target if present.
        if let Some(default_block) = table.default_target.as_ref().and_then(&mut resolve) {
            cfg.add_edge(table.block, default_block, EdgeKind::Unconditional);
        }

        results.push(SwitchRecovery {
            block: table.block,
            num_cases,
        });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;
    use alloc::vec;

    /// A machine-flavoured terminator: an indirect jump through a table of
    /// raw addresses.
    struct TableJump {
        targets: Option<(Vec<u64>, Option<u64>)>,
    }

    impl SwitchSource for TableJump {
        type Target = u64;

        fn switch_targets(&self) -> Option<(Vec<u64>, Option<u64>)> {
            self.targets.clone()
        }
    }

    #[test]
    fn detect_and_recover_address_table() {
        let mut cfg: Cfg<TableJump> = Cfg::new();
        let case_a = cfg.new_block();
        let case_b = cfg.new_block();
        let default = cfg.new_block();
        cfg.add_edge(cfg.entry(), case_a, EdgeKind::IndirectJump);
        cfg.block_mut(cfg.entry()).push(TableJump {
            targets: Some((vec![0x1000, 0x2000], Some(0x3000))),
        });

        let tables = detect_switch_tables(&cfg);
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].targets, vec![0x1000, 0x2000]);

        let address_map: BTreeMap<u64, BlockId> =
            [(0x1000, case_a), (0x2000, case_b), (0x3000, default)]
                .into_iter()
                .collect();
        let recovered =
            recover_switch_tables(&mut cfg, &tables, |addr| address_map.get(addr).copied());
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].num_cases, 2);
        // The IndirectJump edge is gone; SwitchCase + default edges exist.
        assert!(cfg.edges().all(|e| e.kind() != EdgeKind::IndirectJump));
        assert_eq!(
            cfg.edges()
                .filter(|e| e.kind() == EdgeKind::SwitchCase)
                .count(),
            2
        );
        assert_eq!(
            cfg.edges()
                .filter(|e| e.kind() == EdgeKind::Unconditional)
                .count(),
            1
        );
    }

    #[test]
    fn source_tokens_resolve_switch_targets() {
        /// A source-flavoured dispatch whose targets are CST node ids.
        struct ComputedGoto {
            case_nodes: Vec<u32>,
        }

        impl SwitchSource for ComputedGoto {
            type Target = u32;

            fn switch_targets(&self) -> Option<(Vec<u32>, Option<u32>)> {
                (!self.case_nodes.is_empty()).then(|| (self.case_nodes.clone(), None))
            }
        }

        let mut cfg: Cfg<ComputedGoto> = Cfg::new();
        let arm = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(ComputedGoto {
            case_nodes: vec![7],
        });

        let tables = detect_switch_tables(&cfg);
        let recovered =
            recover_switch_tables(&mut cfg, &tables, |node| (*node == 7).then_some(arm));
        assert_eq!(recovered[0].num_cases, 1);
        assert!(cfg.edges().any(|e| e.kind() == EdgeKind::SwitchCase));
    }
}
