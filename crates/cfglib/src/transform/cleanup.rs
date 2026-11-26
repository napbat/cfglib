//! Basic CFG cleanup passes with metadata-preserving rewrite records.
//!
//! Every pass here removes blocks for real: a bypassed, merged, or
//! unreachable block leaves the graph through [`Cfg::remove_block`] rather
//! than being emptied in place, so [`Cfg::block_count`] and
//! [`Cfg::contains_block`] answer honestly afterwards. The slots stay
//! reserved until the caller compacts, which is what keeps the identities in
//! the returned [`Rewrite`] meaningful.
//!
//! Every pass has a `_mapped` form that returns its count and that record.
//! Block merging and empty-block bypass also have a plain form that records
//! nothing: they move every edge of a block, so a record would cost as much
//! as the move itself.
//!
//! The passes also keep the CFG's exception metadata true. A block that a
//! handler, filter, or cleanup names by a single reference is never removed
//! or merged away, and blocks are only merged when the same regions protect
//! both, so [`Cfg::compact`] and [`EhModel`](crate::EhModel) stay usable
//! after any sequence of them.

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::traverse::{depth_first_preorder, reachable};
use crate::graph::view::Direction;
use crate::rewrite::{Recorder, Rewrite};
use crate::transform::contract::contract;

/// Remove blocks unreachable from the entry block and report every removed
/// block and edge.
///
/// Returns the number of blocks removed and the record of what went. The
/// removed slots keep their instructions readable until the next
/// [`Cfg::compact`], so a caller can still inspect what it dropped.
///
/// Blocks that exception metadata names (handler entries, filter blocks,
/// cleanup resume points and continuation targets) count as roots, along with
/// everything they reach: a handler no modeled edge enters is still a handler.
/// Call [`Cfg::remove_unreachable_regions`] first to drop regions that nothing
/// can raise into, together with their handler blocks.
pub fn remove_unreachable_mapped<I, E>(cfg: &mut Cfg<I, E>) -> (usize, Rewrite) {
    let roots: Vec<BlockId> = core::iter::once(cfg.entry())
        .chain(cfg.anchors().filter(|&anchor| cfg.contains_block(anchor)))
        .collect();
    let reachable = reachable(&*cfg, roots, Direction::Forward);

    let doomed: Vec<BlockId> = cfg
        .block_ids()
        .filter(|block| !reachable[block.index()])
        .collect();

    // Only removals: recording each block's retirement straight into one
    // record is the composition of the per-block records.
    let mut rewrite = Rewrite::new();
    for &block in &doomed {
        for edge in cfg.outgoing(block).chain(cfg.incoming(block)) {
            rewrite.record_edge(edge, []);
        }
        rewrite.record_block(block, []);
        cfg.remove_block(block);
    }
    (doomed.len(), rewrite)
}

/// Merge blocks connected by a sole-successor, sole-predecessor edge.
///
/// Returns the number of merges. Use [`merge_blocks_mapped`] to retain the
/// identity relationship.
///
/// Each merge is a [`contract_edge`](crate::contract_edge): the entry block
/// is never consumed as a merge target, an exception edge is never
/// contracted, and the two blocks must lie in the same protected regions and
/// handler bodies with no handler, filter, or cleanup naming the block that
/// goes away.
pub fn merge_blocks<I, E>(cfg: &mut Cfg<I, E>) -> usize {
    merge(cfg, &mut ())
}

/// Merge blocks like [`merge_blocks`] and report the identities it touched.
///
/// Returns the number of merges and a record in which every surviving edge
/// keeps its identity and payload, each connecting edge is removed, and each
/// consumed block maps to the block it was merged into.
pub fn merge_blocks_mapped<I, E>(cfg: &mut Cfg<I, E>) -> (usize, Rewrite) {
    let mut rewrite = Rewrite::new();
    let merged = merge(cfg, &mut rewrite);
    (merged, rewrite)
}

fn merge<I, E>(cfg: &mut Cfg<I, E>, record: &mut impl Recorder) -> usize {
    let mut merged = 0;
    let order = depth_first_preorder(&*cfg, cfg.entry(), Direction::Forward);
    for source in order {
        if !cfg.contains_block(source) {
            continue;
        }
        while let Some(connecting) = cfg.sole_outgoing(source) {
            let target = cfg.edge(connecting).target();
            if !contract(cfg, source, target, record) {
                break;
            }
            merged += 1;
        }
    }
    merged
}

/// Bypass empty blocks with one fallthrough-like successor.
///
/// Returns the number of blocks removed. Use [`remove_empty_blocks_mapped`]
/// when clients retain graph identities. A block that exception metadata
/// names (a handler entry, filter block, or cleanup block) is never bypassed.
pub fn remove_empty_blocks<I, E>(cfg: &mut Cfg<I, E>) -> usize {
    bypass_empty(cfg, &mut ())
}

/// Bypass empty blocks like [`remove_empty_blocks`] and report the
/// identities it touched.
///
/// Returns the number of blocks removed and the record: the bypassed block
/// and its outgoing edge are removed, and every edge that entered it keeps
/// its identity, now entering the successor.
pub fn remove_empty_blocks_mapped<I, E>(cfg: &mut Cfg<I, E>) -> (usize, Rewrite) {
    let mut rewrite = Rewrite::new();
    let removed = bypass_empty(cfg, &mut rewrite);
    (removed, rewrite)
}

fn bypass_empty<I, E, R: Recorder>(cfg: &mut Cfg<I, E>, record: &mut R) -> usize {
    let mut removed = 0;
    let order = depth_first_preorder(&*cfg, cfg.entry(), Direction::Forward);
    for id in order {
        if id == cfg.entry()
            || !cfg.contains_block(id)
            || !cfg.block(id).is_empty()
            || cfg.is_anchor(id)
        {
            continue;
        }
        let Some(outgoing) = cfg.sole_outgoing(id) else {
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

        if R::RECORDS {
            for bypassed in cfg.incoming(id) {
                record.record_edge(bypassed, [bypassed]);
            }
        }
        record.record_edge(outgoing, []);
        record.record_block(id, []);
        cfg.redirect_edges_to(id, target);
        cfg.remove_edge(outgoing);
        cfg.remove_block(id);
        removed += 1;
    }
    removed
}

/// Run all simplification passes until no more changes occur, and compose
/// their records.
///
/// Returns the total number of transformations applied and one record from
/// the identities before the first pass to those after the last.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, simplify_mapped};
///
/// let mut cfg = Cfg::<u32>::new();
/// let b1 = cfg.new_block();
/// let unreachable = cfg.new_block();
/// cfg.add_edge(cfg.entry(), b1, EdgeKind::Fallthrough);
///
/// let (changed, rewrite) = simplify_mapped(&mut cfg);
/// assert!(changed > 0);
/// assert_eq!(rewrite.blocks(unreachable), Some([].as_slice()));
/// ```
pub fn simplify_mapped<I, E>(cfg: &mut Cfg<I, E>) -> (usize, Rewrite) {
    let mut total = 0;
    let mut rewrite = Rewrite::new();
    loop {
        let (unreachable, unreachable_rewrite) = remove_unreachable_mapped(cfg);
        rewrite.compose(&unreachable_rewrite);
        let (empty, empty_rewrite) = remove_empty_blocks_mapped(cfg);
        rewrite.compose(&empty_rewrite);
        let (merged, merged_rewrite) = merge_blocks_mapped(cfg);
        rewrite.compose(&merged_rewrite);
        let round = unreachable + empty + merged;
        if round == 0 {
            break;
        }
        total += round;
    }
    (total, rewrite)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::{Edge, EdgeKind};
    use crate::test_util::{diamond_cfg, ff, probability};

    #[test]
    fn remove_unreachable_noop_when_all_reachable() {
        let mut cfg = diamond_cfg();
        let removed = remove_unreachable_mapped(&mut cfg).0;
        assert_eq!(removed, 0);
    }

    #[test]
    fn remove_unreachable_removes_disconnected_block() {
        let mut cfg = diamond_cfg();
        let orphan = cfg.new_block();
        cfg.block_mut(orphan).push(ff("dead"));
        let removed = remove_unreachable_mapped(&mut cfg).0;
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
        let first = cfg.add_edge(
            target,
            sink,
            Edge::new(EdgeKind::ConditionalTrue).with_weight(probability(0.25)),
        );
        let second = cfg.add_edge(
            target,
            sink,
            Edge::new(EdgeKind::ConditionalFalse).with_weight(probability(0.75)),
        );
        let back = cfg.add_edge(
            target,
            source,
            Edge::new(EdgeKind::Back).with_weight(probability(0.875)),
        );

        assert_eq!(merge_blocks(&mut cfg), 1);

        assert_eq!(cfg.outgoing(target).collect::<Vec<_>>(), &[]);
        assert_eq!(
            cfg.outgoing(source).collect::<Vec<_>>(),
            &[first, second, back]
        );
        assert_eq!(cfg.edge(first).source(), source);
        assert_eq!(cfg.edge(first).target(), sink);
        assert_eq!(cfg.edge(first).kind(), EdgeKind::ConditionalTrue);
        assert_eq!(cfg.edge(first).weight(), Some(probability(0.25)));
        assert_eq!(cfg.edge(second).kind(), EdgeKind::ConditionalFalse);
        assert_eq!(cfg.edge(second).weight(), Some(probability(0.75)));
        assert_eq!(cfg.edge(back).source(), source);
        assert_eq!(cfg.edge(back).target(), source);
        assert_eq!(cfg.edge(back).kind(), EdgeKind::Back);
        assert_eq!(cfg.edge(back).weight(), Some(probability(0.875)));
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
        let total = simplify_mapped(&mut cfg).0;
        assert!(
            total > 0,
            "simplify should perform at least 1 transformation"
        );
        assert!(!cfg.contains_block(orphan));
    }

    #[test]
    fn the_plain_and_mapped_merges_leave_the_same_graph() {
        let build = || {
            let mut cfg = Cfg::<u32>::new();
            let mut previous = cfg.entry();
            for value in 0..6 {
                let next = cfg.new_block();
                cfg.block_mut(next).push(value);
                cfg.add_edge(previous, next, EdgeKind::Fallthrough);
                previous = next;
            }
            let empty = cfg.new_block();
            let exit = cfg.new_block();
            cfg.block_mut(exit).push(99);
            cfg.add_edge(previous, empty, EdgeKind::Jump);
            cfg.add_edge(empty, exit, EdgeKind::Fallthrough);
            cfg.add_edge(exit, exit, EdgeKind::Back);
            cfg
        };
        let (mut plain, mut mapped) = (build(), build());
        assert_eq!(
            remove_empty_blocks(&mut plain),
            remove_empty_blocks_mapped(&mut mapped).0
        );
        assert_eq!(merge_blocks(&mut plain), merge_blocks_mapped(&mut mapped).0);
        assert_eq!(plain, mapped);
    }
}
