//! Basic CFG cleanup passes — unreachable block removal, block merging,
//! empty block bypass, and combined simplification.
//!
//! All passes mutate the graph in-place and return the number of
//! blocks affected.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;

/// Remove blocks unreachable from the entry block.
///
/// Unreachable blocks have their instructions cleared and all
/// incident edges removed, turning them into dead slots in the
/// arena. Returns the number of unreachable blocks cleaned up.
pub fn remove_unreachable<I>(cfg: &mut Cfg<I>) -> usize {
    let reachable = cfg.dfs_preorder();
    let n = cfg.num_blocks();
    let mut is_reachable = vec![false; n];
    for &id in &reachable {
        is_reachable[id.index()] = true;
    }

    let mut removed = 0;
    for (i, &reachable) in is_reachable.iter().enumerate() {
        if !reachable {
            let id = BlockId::from_index(i);
            let has_insts = !cfg.block(id).instructions().is_empty();
            let has_edges =
                !cfg.successor_edges(id).is_empty() || !cfg.predecessor_edges(id).is_empty();
            if !has_insts && !has_edges {
                continue; // Already dead — nothing to clean up.
            }
            // Clear instructions.
            cfg.block_mut(id).instructions_vec_mut().clear();
            // Remove all outgoing edges.
            let out: Vec<_> = cfg.successor_edges(id).to_vec();
            for eid in out {
                cfg.remove_edge(eid);
            }
            // Remove all incoming edges.
            let inc: Vec<_> = cfg.predecessor_edges(id).to_vec();
            for eid in inc {
                cfg.remove_edge(eid);
            }
            removed += 1;
        }
    }
    removed
}

/// Merge a block with its sole successor when:
/// - the block has exactly one outgoing edge, and
/// - that edge's target has exactly one incoming edge.
///
/// The entry block is never consumed as a merge target.
///
/// Returns the number of merges performed.
pub fn merge_blocks<I>(cfg: &mut Cfg<I>) -> usize {
    let mut merged = 0;
    let order = cfg.dfs_preorder();
    for id in order {
        while let [connecting] = cfg.successor_edges(id) {
            let connecting = *connecting;
            let target = cfg.edge(connecting).target();
            if target == id || target == cfg.entry() {
                break;
            }
            if cfg.predecessor_edges(target).len() != 1 {
                break;
            }
            // Merge: append target's instructions to id.
            let target_insts = core::mem::take(cfg.block_mut(target).instructions_vec_mut());
            cfg.block_mut(id)
                .instructions_vec_mut()
                .extend(target_insts);

            // Remove the connecting edge.
            cfg.remove_edge(connecting);

            // Transfer target's outgoing edges to id.
            cfg.move_outgoing_edges(target, id);

            merged += 1;
        }
    }
    merged
}

/// Bypass empty blocks that have a single unconditional/fallthrough outgoing
/// edge by redirecting their incoming edges to that edge's target.
///
/// Returns the number of blocks bypassed.
pub fn remove_empty_blocks<I>(cfg: &mut Cfg<I>) -> usize {
    let mut removed = 0;
    let order = cfg.dfs_preorder();
    for id in order {
        if id == cfg.entry() || !cfg.block(id).is_empty() {
            continue;
        }
        let outgoing = match cfg.successor_edges(id) {
            [edge] => *edge,
            _ => continue,
        };
        let edge = cfg.edge(outgoing);
        if !matches!(edge.kind(), EdgeKind::Fallthrough | EdgeKind::Unconditional) {
            continue;
        }
        let target = edge.target();
        // Redirect all predecessors of `id` to `target`.
        cfg.redirect_edges_to(id, target);
        // Remove the outgoing edge.
        cfg.remove_edge(outgoing);
        removed += 1;
    }
    removed
}

/// Run all simplification passes until no more changes occur.
///
/// Returns the total number of transformations applied.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, simplify};
///
/// let mut cfg = Cfg::<u32>::new();
/// let b0 = cfg.entry();
/// let b1 = cfg.new_block();
/// let b2 = cfg.new_block(); // unreachable
/// cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
///
/// let changes = simplify(&mut cfg);
/// assert!(changes > 0); // removed unreachable b2
/// ```
pub fn simplify<I>(cfg: &mut Cfg<I>) -> usize {
    let mut total = 0;
    loop {
        let r = remove_unreachable(cfg);
        let e = remove_empty_blocks(cfg);
        let m = merge_blocks(cfg);
        let round = r + e + m;
        if round == 0 {
            break;
        }
        total += round;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

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

        assert_eq!(cfg.successor_edges(target), &[]);
        assert_eq!(cfg.successor_edges(source), &[first, second, back]);
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
        assert_eq!(cfg.num_edges(), 3);
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
        assert_eq!(cfg.successor_edges(entry), &[to_back_edge, to_branch]);
        assert_eq!(cfg.edge(back).source(), back_edge_source);
        assert_eq!(cfg.edge(back).target(), entry);
        assert_eq!(cfg.block(branch).instructions(), &[2, 3]);
        assert_eq!(cfg.successor_edges(branch).len(), 0);
        assert!(crate::verify(&cfg).is_ok());
    }
}
