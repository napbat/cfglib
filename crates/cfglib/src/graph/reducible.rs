//! Irreducible-to-reducible CFG transformation via node splitting.
//!
//! An irreducible CFG contains cycles with multiple entry points.
//! [`make_reducible`] eliminates these by duplicating the secondary
//! entry nodes so that every cycle has a single dominating header.
//!
//! The algorithm is iterative: after each round of splitting, the
//! dominator tree is recomputed and the CFG is re-checked. The loop
//! terminates when the CFG is reducible.

extern crate alloc;
use alloc::vec::Vec;
use smallvec::SmallVec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::dominator::DominatorTree;
use crate::graph::structure::find_irreducible_entry;
use crate::graph::traverse::{TraversalDirection, reachable};

/// Transform an irreducible CFG into a reducible one by node splitting.
///
/// Returns the number of blocks that were duplicated. If the CFG is
/// already reducible, returns 0 and makes no changes.
///
/// **Caution**: node splitting can cause exponential code growth in
/// pathological cases. For most real-world binaries the duplication
/// is modest.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind};
/// use cfglib::graph::reducible::make_reducible;
///
/// // A simple reducible CFG returns 0 (no changes).
/// let mut cfg = Cfg::<u32>::new();
/// let b1 = cfg.new_block();
/// cfg.add_edge(cfg.entry(), b1, EdgeKind::Fallthrough);
/// assert_eq!(make_reducible(&mut cfg), 0);
/// ```
pub fn make_reducible<I: Clone>(cfg: &mut Cfg<I>) -> usize {
    let mut total_split = 0;

    loop {
        let dom = DominatorTree::compute(cfg);
        // Find an irreducible entry and split ONE per iteration.
        // After each split the dominator tree is stale, so we
        // must recompute before picking the next target.
        if let Some(target) = find_irreducible_entry(cfg, &dom) {
            split_node(cfg, target);
            total_split += 1;
        } else {
            break;
        }
    }

    total_split
}

/// Duplicate block `target` — create a copy and redirect external
/// predecessors to the copy, keeping cycle-internal predecessors
/// on the original. This breaks the irreducible entry by giving
/// external entries their own copy of the block.
fn split_node<I: Clone>(cfg: &mut Cfg<I>, target: BlockId) {
    // Duplicate the target's instructions into a new block.
    let copy = cfg.new_block();
    let insts = cfg.block(target).instructions().to_vec();
    for inst in insts {
        cfg.blocks[copy.index()].instructions.push(inst);
    }

    // Partition predecessors: keep edges from blocks that target
    // can reach (they're in a cycle with target), redirect the rest
    // to the copy (they're external entries).
    let cycle_reachable = reachable(cfg, [target], TraversalDirection::Outgoing);
    let mut redirected = SmallVec::<[crate::edge::EdgeId; 4]>::new();
    {
        let edges = &mut cfg.edges;
        cfg.preds[target.index()].retain(|eid| {
            let eid = *eid;
            let edge = edges[eid.index()].as_mut().unwrap();
            // If target can reach the source, they're in a cycle — keep it.
            if cycle_reachable[edge.source.index()] {
                true
            } else {
                edge.target = copy;
                redirected.push(eid);
                false
            }
        });
    }
    cfg.preds[copy.index()].extend(redirected);

    // Clone outgoing edges from target to copy. Original edge identities stay
    // attached to `target`; the copy receives fresh identities in the same
    // adjacency order with all semantic metadata retained.
    let outgoing: Vec<(BlockId, EdgeKind, Option<f64>)> = cfg
        .successor_edges(target)
        .iter()
        .map(|&eid| {
            let e = cfg.edge(eid);
            (e.target(), e.kind(), e.weight())
        })
        .collect();

    for (succ, kind, weight) in outgoing {
        if let Some(weight) = weight {
            cfg.add_weighted_edge(copy, succ, kind, weight);
        } else {
            cfg.add_edge(copy, succ, kind);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::graph::dominator::DominatorTree;
    use crate::graph::structure::is_reducible;
    use crate::test_util::ff;

    #[test]
    fn already_reducible_is_noop() {
        // Simple diamond: entry → A → merge, entry → B → merge.
        let mut cfg = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        let merge = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("entry"));
        cfg.block_mut(a).instructions_vec_mut().push(ff("a"));
        cfg.block_mut(b).instructions_vec_mut().push(ff("b"));
        cfg.block_mut(merge)
            .instructions_vec_mut()
            .push(ff("merge"));
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        cfg.add_edge(a, merge, EdgeKind::Fallthrough);
        cfg.add_edge(b, merge, EdgeKind::Fallthrough);

        let dom = DominatorTree::compute(&cfg);
        assert!(is_reducible(&cfg, &dom));
        let splits = make_reducible(&mut cfg);
        assert_eq!(splits, 0);
    }

    #[test]
    fn irreducible_cycle_is_fixed() {
        // Build an irreducible CFG:
        //   entry → A, entry → B
        //   A → B, B → A   (cycle with two entries)
        let mut cfg = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("entry"));
        cfg.block_mut(a).instructions_vec_mut().push(ff("a"));
        cfg.block_mut(b).instructions_vec_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        cfg.add_edge(a, b, EdgeKind::Fallthrough);
        cfg.add_edge(b, a, EdgeKind::Fallthrough);

        let dom = DominatorTree::compute(&cfg);
        assert!(!is_reducible(&cfg, &dom), "should be irreducible before");

        let splits = make_reducible(&mut cfg);
        assert!(splits > 0, "should have split at least one node");

        let dom2 = DominatorTree::compute(&cfg);
        assert!(is_reducible(&cfg, &dom2), "should be reducible after");
    }

    #[test]
    fn multiple_splits_recompute_the_next_irreducible_entry() {
        let mut cfg: Cfg<()> = Cfg::new();
        let cycle: Vec<_> = (0..4).map(|_| cfg.new_block()).collect();
        let external = cfg.new_block();

        cfg.add_edge(cfg.entry(), cycle[0], EdgeKind::ConditionalTrue);
        cfg.add_edge(cycle[0], cycle[1], EdgeKind::Fallthrough);
        cfg.add_edge(cycle[1], cycle[2], EdgeKind::Fallthrough);
        cfg.add_edge(cycle[2], cycle[3], EdgeKind::Fallthrough);
        cfg.add_edge(cycle[3], cycle[0], EdgeKind::Back);
        cfg.add_edge(cfg.entry(), external, EdgeKind::ConditionalFalse);
        cfg.add_edge(external, cycle[1], EdgeKind::Unconditional);

        let before = DominatorTree::compute(&cfg);
        assert!(!is_reducible(&cfg, &before));
        let original_blocks = cfg.num_blocks();

        assert_eq!(make_reducible(&mut cfg), 3);
        assert_eq!(cfg.num_blocks(), original_blocks + 3);
        let after = DominatorTree::compute(&cfg);
        assert!(is_reducible(&cfg, &after));
    }

    #[test]
    fn splitting_preserves_outgoing_edge_metadata_and_order() {
        // B is the first irreducible entry witnessed by the DFS. Splitting it
        // must retain B's original edges and reproduce their complete metadata
        // on the new copy in adjacency order.
        let mut cfg = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("entry"));
        cfg.block_mut(a).instructions_vec_mut().push(ff("a"));
        cfg.block_mut(b).instructions_vec_mut().push(ff("b"));
        cfg.block_mut(exit).instructions_vec_mut().push(ff("exit"));

        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        let redirected = cfg.add_weighted_edge(cfg.entry(), b, EdgeKind::ConditionalFalse, 0.125);
        cfg.add_edge(a, b, EdgeKind::Fallthrough);
        let back = cfg.add_weighted_edge(b, a, EdgeKind::Back, 0.75);
        let leave = cfg.add_weighted_edge(b, exit, EdgeKind::SwitchCase, 0.25);
        let original_block_count = cfg.num_blocks();

        let dom = DominatorTree::compute(&cfg);
        assert_eq!(find_irreducible_entry(&cfg, &dom), Some(b));
        assert_eq!(make_reducible(&mut cfg), 1);

        let copy = BlockId::from_index(original_block_count);
        assert_eq!(cfg.edge(redirected).target(), copy);
        assert_eq!(cfg.edge(redirected).weight(), Some(0.125));
        assert_eq!(cfg.successor_edges(b), &[back, leave]);

        let copied = cfg.successor_edges(copy);
        assert_eq!(copied.len(), 2);
        assert_ne!(copied[0], back);
        assert_ne!(copied[1], leave);
        assert_eq!(cfg.edge(copied[0]).source(), copy);
        assert_eq!(cfg.edge(copied[0]).target(), a);
        assert_eq!(cfg.edge(copied[0]).kind(), EdgeKind::Back);
        assert_eq!(cfg.edge(copied[0]).weight(), Some(0.75));
        assert_eq!(cfg.edge(copied[1]).source(), copy);
        assert_eq!(cfg.edge(copied[1]).target(), exit);
        assert_eq!(cfg.edge(copied[1]).kind(), EdgeKind::SwitchCase);
        assert_eq!(cfg.edge(copied[1]).weight(), Some(0.25));
    }
}
