//! Basic CFG cleanup passes with metadata-preserving rewrite records.
//!
//! Every pass here removes blocks for real: a bypassed, merged, or
//! unreachable block leaves the graph through [`Cfg::remove_block`] rather
//! than being emptied in place, so [`Cfg::block_count`] and
//! [`Cfg::contains_block`] answer honestly afterwards. The slots stay
//! reserved until the caller compacts, which is what keeps the identities in
//! the returned [`Rewrite`] meaningful.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::rewrite::Rewrite;

/// Remove blocks unreachable from the entry block.
///
/// This compatibility entry point returns only the number of blocks removed.
/// Use [`remove_unreachable_mapped`] when identities matter.
pub fn remove_unreachable<I, E>(cfg: &mut Cfg<I, E>) -> usize {
    remove_unreachable_mapped(cfg).0
}

/// Remove unreachable blocks and report every removed block and edge.
///
/// The removed slots keep their instructions readable until the next
/// [`Cfg::compact`], so a caller can still inspect what it dropped.
pub fn remove_unreachable_mapped<I, E>(cfg: &mut Cfg<I, E>) -> (usize, Rewrite) {
    let mut reachable = vec![false; cfg.block_bound()];
    for block in cfg.depth_first_preorder() {
        reachable[block.index()] = true;
    }

    let doomed: Vec<BlockId> = cfg
        .block_ids()
        .filter(|block| !reachable[block.index()])
        .collect();

    let mut mapping = Rewrite::new();
    for block in &doomed {
        let (_, removal) = cfg.remove_block_mapped(*block);
        mapping.compose(removal);
    }
    (doomed.len(), mapping)
}

/// Merge blocks connected by a sole-successor, sole-predecessor edge.
///
/// The entry block is never consumed as a merge target.
///
/// This compatibility entry point returns only the number of merges. Use
/// [`merge_blocks_mapped`] to retain the identity relationship.
pub fn merge_blocks<I, E>(cfg: &mut Cfg<I, E>) -> usize {
    merge_blocks_inner(cfg, None)
}

/// Merge linear blocks while preserving every surviving edge identity and
/// payload. The entry block is never consumed as a merge target.
pub fn merge_blocks_mapped<I, E>(cfg: &mut Cfg<I, E>) -> (usize, Rewrite) {
    let mut mapping = Rewrite::new();
    let merged = merge_blocks_inner(cfg, Some(&mut mapping));
    (merged, mapping)
}

fn merge_blocks_inner<I, E>(cfg: &mut Cfg<I, E>, mut mapping: Option<&mut Rewrite>) -> usize {
    let mut merged = 0;
    let order = cfg.depth_first_preorder();
    for source in order {
        if !cfg.contains_block(source) {
            continue;
        }
        while let Some(connecting) = sole_outgoing(cfg, source) {
            let target = cfg.edge(connecting).target();
            if target == source || target == cfg.entry() {
                break;
            }
            if cfg.incoming(target).count() != 1 {
                break;
            }

            let target_instructions = core::mem::take(cfg.block_mut(target).instructions_mut());
            cfg.block_mut(source)
                .instructions_mut()
                .extend(target_instructions);

            cfg.remove_edge(connecting);
            if let Some(mapping) = mapping.as_deref_mut() {
                mapping.record_edge(connecting, []);
            }
            cfg.move_outgoing_edges(target, source);
            let moved: Vec<_> = cfg.outgoing(source).collect();
            cfg.remove_block(target);
            if let Some(mapping) = mapping.as_deref_mut() {
                for edge in moved {
                    mapping.record_edge(edge, [edge]);
                }
                mapping.record_block(target, [source]);
            }

            merged += 1;
        }
    }
    merged
}

/// The only outgoing edge of `block`, when it has exactly one.
fn sole_outgoing<I, E>(cfg: &Cfg<I, E>, block: BlockId) -> Option<crate::EdgeId> {
    let mut outgoing = cfg.outgoing(block);
    let first = outgoing.next()?;
    outgoing.next().is_none().then_some(first)
}

/// Bypass empty blocks with one fallthrough-like successor.
///
/// This compatibility entry point returns only the count. Use
/// [`remove_empty_blocks_mapped`] when clients retain graph identities.
pub fn remove_empty_blocks<I, E>(cfg: &mut Cfg<I, E>) -> usize {
    remove_empty_blocks_inner(cfg, None)
}

/// Bypass empty blocks without reallocating their incoming edges.
pub fn remove_empty_blocks_mapped<I, E>(cfg: &mut Cfg<I, E>) -> (usize, Rewrite) {
    let mut mapping = Rewrite::new();
    let removed = remove_empty_blocks_inner(cfg, Some(&mut mapping));
    (removed, mapping)
}

fn remove_empty_blocks_inner<I, E>(
    cfg: &mut Cfg<I, E>,
    mut mapping: Option<&mut Rewrite>,
) -> usize {
    let mut removed = 0;
    let order = cfg.depth_first_preorder();
    for id in order {
        if id == cfg.entry() || !cfg.contains_block(id) || !cfg.block(id).is_empty() {
            continue;
        }
        let Some(outgoing) = sole_outgoing(cfg, id) else {
            continue;
        };
        let edge = cfg.edge(outgoing);
        if !matches!(edge.kind(), EdgeKind::Fallthrough | EdgeKind::Unconditional) {
            continue;
        }
        let target = edge.target();
        if target == id {
            continue;
        }

        let bypassed: Vec<_> = cfg.incoming(id).collect();
        cfg.redirect_edges_to(id, target);
        cfg.remove_edge(outgoing);
        cfg.remove_block(id);
        if let Some(mapping) = mapping.as_deref_mut() {
            for edge in bypassed {
                mapping.record_edge(edge, [edge]);
            }
            mapping.record_edge(outgoing, []);
            mapping.record_block(id, []);
        }
        removed += 1;
    }
    removed
}

/// Run all simplification passes until no more changes occur.
///
/// Returns the total number of transformations applied. Use
/// [`simplify_mapped`] to retain the composed identity mapping.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, simplify};
///
/// let mut cfg = Cfg::<u32>::new();
/// let b1 = cfg.new_block();
/// let _unreachable = cfg.new_block();
/// cfg.add_edge(cfg.entry(), b1, EdgeKind::Fallthrough);
///
/// assert!(simplify(&mut cfg) > 0);
/// ```
pub fn simplify<I, E>(cfg: &mut Cfg<I, E>) -> usize {
    simplify_mapped(cfg).0
}

/// Run all simplification passes and compose their rewrite maps.
pub fn simplify_mapped<I, E>(cfg: &mut Cfg<I, E>) -> (usize, Rewrite) {
    let mut total = 0;
    let mut mapping = Rewrite::new();
    loop {
        let (unreachable, unreachable_map) = remove_unreachable_mapped(cfg);
        mapping.compose(unreachable_map);
        let (empty, empty_map) = remove_empty_blocks_mapped(cfg);
        mapping.compose(empty_map);
        let (merged, merged_map) = merge_blocks_mapped(cfg);
        mapping.compose(merged_map);
        let round = unreachable + empty + merged;
        if round == 0 {
            break;
        }
        total += round;
    }
    (total, mapping)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::test_util::{diamond_cfg, ff};

    #[test]
    fn remove_unreachable_noop_when_all_reachable() {
        let mut cfg = diamond_cfg();
        let removed = remove_unreachable(&mut cfg);
        assert_eq!(removed, 0);
    }

    #[test]
    fn remove_unreachable_removes_disconnected_block() {
        let mut cfg = diamond_cfg();
        let orphan = cfg.new_block();
        cfg.block_mut(orphan).push(ff("dead"));
        let removed = remove_unreachable(&mut cfg);
        assert_eq!(removed, 1);
        assert!(!cfg.contains_block(orphan));
    }

    #[test]
    fn merge_blocks_merges_linear_chain() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(ff("a"));
        cfg.block_mut(b).push(ff("b"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        let merged = merge_blocks(&mut cfg);
        assert_eq!(merged, 1);
        assert_eq!(cfg.block(cfg.entry()).instructions().len(), 2);
    }

    #[test]
    fn merge_blocks_does_not_merge_when_multiple_predecessors() {
        let mut cfg = diamond_cfg();
        let merged = merge_blocks(&mut cfg);
        assert_eq!(merged, 0);
    }

    #[test]
    fn merge_blocks_skips_self_loop() {
        let mut cfg = Cfg::new();
        cfg.block_mut(cfg.entry()).push(ff("a"));
        cfg.add_edge(cfg.entry(), cfg.entry(), EdgeKind::Back);
        let merged = merge_blocks(&mut cfg);
        assert_eq!(merged, 0);
    }

    #[test]
    fn merge_preserves_parallel_weighted_edges_and_back_edge_identity() {
        let mut cfg = Cfg::<u32>::new();
        let source = cfg.entry();
        let target = cfg.new_block();
        let sink = cfg.new_block();
        cfg.block_mut(source).push(0);
        cfg.block_mut(target).push(1);
        cfg.block_mut(sink).push(2);
        cfg.add_edge(source, target, EdgeKind::Fallthrough);
        let first = cfg.add_weighted_edge(target, sink, EdgeKind::ConditionalTrue, 0.25);
        let second = cfg.add_weighted_edge(target, sink, EdgeKind::ConditionalFalse, 0.75);
        let back = cfg.add_weighted_edge(target, source, EdgeKind::Back, 0.875);

        assert_eq!(merge_blocks(&mut cfg), 1);

        assert_eq!(cfg.outgoing(target).collect::<Vec<_>>(), &[]);
        assert_eq!(
            cfg.outgoing(source).collect::<Vec<_>>(),
            &[first, second, back]
        );
        assert_eq!(cfg.edge(first).source(), source);
        assert_eq!(cfg.edge(first).target(), sink);
        assert_eq!(cfg.edge(first).kind(), EdgeKind::ConditionalTrue);
        assert_eq!(cfg.edge(first).weight(), Some(0.25));
        assert_eq!(cfg.edge(second).kind(), EdgeKind::ConditionalFalse);
        assert_eq!(cfg.edge(second).weight(), Some(0.75));
        assert_eq!(cfg.edge(back).source(), source);
        assert_eq!(cfg.edge(back).target(), source);
        assert_eq!(cfg.edge(back).kind(), EdgeKind::Back);
        assert_eq!(cfg.edge(back).weight(), Some(0.875));
        assert_eq!(cfg.edge_count(), 3);
    }

    #[test]
    fn merge_never_consumes_the_entry_block() {
        let mut cfg = Cfg::<u32>::new();
        let entry = cfg.entry();
        let back_edge_source = cfg.new_block();
        let branch = cfg.new_block();
        let exit = cfg.new_block();
        for (index, block) in [entry, back_edge_source, branch, exit]
            .into_iter()
            .enumerate()
        {
            cfg.block_mut(block)
                .push(u32::try_from(index).expect("test block index fits in u32"));
        }
        let to_back_edge = cfg.add_edge(entry, back_edge_source, EdgeKind::ConditionalTrue);
        let to_branch = cfg.add_edge(entry, branch, EdgeKind::ConditionalFalse);
        let back = cfg.add_edge(back_edge_source, entry, EdgeKind::Back);
        cfg.add_edge(branch, exit, EdgeKind::Fallthrough);

        assert_eq!(merge_blocks(&mut cfg), 1);

        assert_eq!(cfg.block(entry).instructions(), &[0]);
        assert_eq!(
            cfg.outgoing(entry).collect::<Vec<_>>(),
            &[to_back_edge, to_branch]
        );
        assert_eq!(cfg.edge(back).source(), back_edge_source);
        assert_eq!(cfg.edge(back).target(), entry);
        assert_eq!(cfg.block(branch).instructions(), &[2, 3]);
        assert_eq!(cfg.outgoing(branch).count(), 0);
        assert!(crate::verify(&cfg).is_ok());
    }

    #[test]
    fn remove_empty_blocks_bypasses_empty_block() {
        let mut cfg = Cfg::new();
        let empty = cfg.new_block();
        let target = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(ff("entry"));
        cfg.block_mut(target).push(ff("target"));
        cfg.add_edge(cfg.entry(), empty, EdgeKind::Fallthrough);
        cfg.add_edge(empty, target, EdgeKind::Fallthrough);
        let removed = remove_empty_blocks(&mut cfg);
        assert_eq!(removed, 1);
        let succs: Vec<_> = cfg.successors(cfg.entry()).collect();
        assert_eq!(succs.len(), 1);
        assert_eq!(succs[0], target);
    }

    #[test]
    fn remove_empty_blocks_does_not_remove_entry() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(b).push(ff("b"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        let removed = remove_empty_blocks(&mut cfg);
        assert_eq!(removed, 0);
    }

    #[test]
    fn simplify_runs_all_passes() {
        let mut cfg = Cfg::new();
        let empty = cfg.new_block();
        let b = cfg.new_block();
        let orphan = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(ff("entry"));
        cfg.block_mut(b).push(ff("b"));
        cfg.block_mut(orphan).push(ff("dead"));
        cfg.add_edge(cfg.entry(), empty, EdgeKind::Fallthrough);
        cfg.add_edge(empty, b, EdgeKind::Fallthrough);
        let total = simplify(&mut cfg);
        assert!(
            total > 0,
            "simplify should perform at least 1 transformation"
        );
        assert!(!cfg.contains_block(orphan));
    }
}
