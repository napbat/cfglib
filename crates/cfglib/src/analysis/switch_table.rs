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
use crate::dataflow::cfg_view::CfgView;
use crate::edge::EdgeKind;

/// A dispatch table over branch-target tokens `T`: the case targets plus an
/// optional default target.
///
/// This is what an instruction reports about itself
/// ([`SwitchSource::switch_targets`]) and, with the block that holds the
/// dispatch, what a [`JumpTable`] records.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SwitchTargets<T> {
    /// Known case targets (resolved by the consumer).
    pub targets: Vec<T>,
    /// Optional default / fallthrough target.
    pub default_target: Option<T>,
}

/// A recovered multi-way dispatch site over branch-target tokens `T`.
///
/// With the `serde` feature the dispatch fields are written beside `block`
/// (`{"block":..,"targets":..,"default_target":..}`), not nested.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct JumpTable<T> {
    /// The block containing the dispatch.
    pub block: BlockId,
    /// The case targets and the default target of the dispatch.
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub dispatch: SwitchTargets<T>,
}

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
pub fn detect_switch_tables<G>(cfg: &G) -> Vec<JumpTable<<G::Inst as SwitchSource>::Target>>
where
    G: CfgView,
    G::Inst: SwitchSource,
{
    let mut tables = Vec::new();
    for block_id in cfg.node_ids() {
        if let Some(last) = cfg.instructions(block_id).last()
            && let Some(dispatch) = last.switch_targets()
        {
            tables.push(JumpTable {
                block: block_id,
                dispatch,
            });
        }
    }
    tables
}

/// Result of switch table reconstruction.
#[derive(Debug, Clone)]
pub struct SwitchRecovery {
    /// Block that was converted from indirect jump to switch.
    pub block: BlockId,
    /// Number of case edges added.
    pub case_count: usize,
}

/// Reconstruct switch tables from detected jump-table patterns.
///
/// This function rewires the CFG, so it takes a [`Cfg`] and not a
/// [`CfgView`]. The graph can carry any edge payload `E`; the edges the
/// function adds carry `E::default()`, and edges it does not remove keep
/// their payload.
///
/// For each [`JumpTable`], removes the existing `IndirectJump` edge(s) from
/// the block and replaces them with `SwitchCase` edges to each resolved
/// target (plus an `Unconditional` edge for the default target).
///
/// The `resolve` function maps target tokens to block IDs — the consumer
/// provides it because token-to-block mapping is frontend-specific (an
/// address-to-block map for binaries, a CST-node-to-block map for source).
pub fn recover_switch_tables<I, T, E: Default>(
    cfg: &mut Cfg<I, E>,
    tables: &[JumpTable<T>],
    mut resolve: impl FnMut(&T) -> Option<BlockId>,
) -> Vec<SwitchRecovery> {
    let mut results = Vec::new();

    for table in tables {
        let edges_to_remove: Vec<_> = cfg
            .outgoing(table.block)
            .filter(|&eid| cfg.edge(eid).kind() == EdgeKind::IndirectJump)
            .collect();
        for eid in edges_to_remove {
            cfg.remove_edge(eid);
        }

        let mut case_count = 0;

        for target in &table.dispatch.targets {
            if let Some(target_block) = resolve(target) {
                cfg.add_edge(table.block, target_block, EdgeKind::SwitchCase);
                case_count += 1;
            }
        }

        if let Some(default) = &table.dispatch.default_target
            && let Some(default_block) = resolve(default)
        {
            cfg.add_edge(table.block, default_block, EdgeKind::Unconditional);
        }

        results.push(SwitchRecovery {
            block: table.block,
            case_count,
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
        targets: Option<SwitchTargets<u64>>,
    }

    impl SwitchSource for TableJump {
        type Target = u64;

        fn switch_targets(&self) -> Option<SwitchTargets<u64>> {
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
            targets: Some(SwitchTargets {
                targets: vec![0x1000, 0x2000],
                default_target: Some(0x3000),
            }),
        });

        let tables = detect_switch_tables(&cfg);
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].dispatch.targets, vec![0x1000, 0x2000]);
        assert_eq!(tables[0].dispatch.default_target, Some(0x3000));

        let address_map: BTreeMap<u64, BlockId> =
            [(0x1000, case_a), (0x2000, case_b), (0x3000, default)]
                .into_iter()
                .collect();
        let recovered =
            recover_switch_tables(&mut cfg, &tables, |addr| address_map.get(addr).copied());
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].case_count, 2);
        // The IndirectJump edge is gone; the cases and the default replace it.
        assert_eq!(
            cfg.edges()
                .map(|edge| (edge.kind(), edge.target()))
                .collect::<Vec<_>>(),
            [
                (EdgeKind::SwitchCase, case_a),
                (EdgeKind::SwitchCase, case_b),
                (EdgeKind::Unconditional, default),
            ]
        );
    }

    #[test]
    fn recovery_works_on_graphs_with_an_edge_payload() {
        let mut cfg = Cfg::<TableJump, u8>::with_edge_payload();
        let case = cfg.new_block();
        let other = cfg.new_block();
        cfg.add_edge_with_payload(cfg.entry(), case, EdgeKind::IndirectJump, 1);
        let kept = cfg.add_edge_with_payload(case, other, EdgeKind::Fallthrough, 9);
        cfg.block_mut(cfg.entry()).push(TableJump {
            targets: Some(SwitchTargets {
                targets: vec![0x10],
                default_target: None,
            }),
        });

        let tables = detect_switch_tables(&cfg);
        let recovered = recover_switch_tables(&mut cfg, &tables, |address| {
            (*address == 0x10).then_some(case)
        });

        assert_eq!(recovered[0].case_count, 1);
        let case_edge = cfg
            .edges()
            .find(|edge| edge.kind() == EdgeKind::SwitchCase)
            .expect("the resolved case is wired");
        assert_eq!(
            case_edge.payload(),
            &0,
            "an added edge has the default payload"
        );
        assert_eq!(
            cfg.edge(kept).payload(),
            &9,
            "an unrelated edge keeps its payload"
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

            fn switch_targets(&self) -> Option<SwitchTargets<u32>> {
                (!self.case_nodes.is_empty()).then(|| SwitchTargets {
                    targets: self.case_nodes.clone(),
                    default_target: None,
                })
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
        assert_eq!(recovered[0].case_count, 1);
        assert!(cfg.edges().any(|e| e.kind() == EdgeKind::SwitchCase));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn a_jump_table_keeps_its_flat_document_shape() {
        extern crate std;

        let table = JumpTable {
            block: BlockId::from_raw(3),
            dispatch: SwitchTargets {
                targets: vec![10_u32, 20],
                default_target: Some(30),
            },
        };
        let json = serde_json::to_string(&table).expect("serialization cannot fail");
        assert_eq!(json, r#"{"block":3,"targets":[10,20],"default_target":30}"#);
        let decoded: JumpTable<u32> = serde_json::from_str(&json).expect("the flat form decodes");
        assert_eq!(decoded, table);
    }
}
