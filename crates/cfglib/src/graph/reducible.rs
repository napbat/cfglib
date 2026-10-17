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
    let instructions = cfg.block(target).instructions().to_vec();
    cfg.block_mut(copy).instructions_mut().extend(instructions);

    // Partition predecessors: keep edges from blocks that target
    // can reach (they're in a cycle with target), redirect the rest
    // to the copy (they're external entries).
    let cycle_reachable = reachable(cfg, [target], TraversalDirection::Outgoing);
    let external: SmallVec<[crate::edge::EdgeId; 4]> = cfg
        .incoming(target)
        .filter(|&edge| !cycle_reachable[cfg.edge(edge).source().index()])
        .collect();
    for edge in external {
        cfg.redirect_edge_target(edge, copy);
    }

    // Clone outgoing edges from target to copy. Original edge identities stay
    // attached to `target`; the copy receives fresh identities in the same
    // adjacency order with all semantic metadata retained.
    let outgoing: Vec<(BlockId, EdgeKind, Option<f64>)> = cfg
        .outgoing(target)
        .map(|eid| {
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
            .instructions_mut()
            .push(ff("entry"));
        cfg.block_mut(a).instructions_mut().push(ff("a"));
        cfg.block_mut(b).instructions_mut().push(ff("b"));
        cfg.block_mut(merge).instructions_mut().push(ff("merge"));
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
            .instructions_mut()
            .push(ff("entry"));
        cfg.block_mut(a).instructions_mut().push(ff("a"));
        cfg.block_mut(b).instructions_mut().push(ff("b"));
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
    fn every_irreducible_entry_is_split_in_turn() {
        // Two independent two-entry cycles: whichever the search meets first,
        // fixing it leaves the other, so the next round has to find it again.
        let mut cfg: Cfg<()> = Cfg::new();
        let entry = cfg.entry();
        for _ in 0..2 {
            let [a, b] = [(); 2].map(|()| cfg.new_block());
            cfg.add_edge(entry, a, EdgeKind::ConditionalTrue);
            cfg.add_edge(entry, b, EdgeKind::ConditionalFalse);
            cfg.add_edge(a, b, EdgeKind::Fallthrough);
            cfg.add_edge(b, a, EdgeKind::Fallthrough);
        }
        let before = DominatorTree::compute(&cfg);
        assert!(!is_reducible(&cfg, &before));
        let original_blocks = cfg.block_count();

        let splits = make_reducible(&mut cfg);

        assert!(splits >= 2, "each cycle needs a split, got {splits}");
        assert_eq!(cfg.block_count(), original_blocks + splits);
        let after = DominatorTree::compute(&cfg);
        assert!(is_reducible(&cfg, &after));
    }

    #[test]
    fn a_cycle_entered_in_its_middle_is_made_reducible() {
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
        let original_blocks = cfg.block_count();

        let splits = make_reducible(&mut cfg);

        assert!(splits >= 1);
        assert_eq!(cfg.block_count(), original_blocks + splits);
        let after = DominatorTree::compute(&cfg);
        assert!(is_reducible(&cfg, &after));
    }

    #[test]
    fn splitting_preserves_outgoing_edge_metadata_and_order() {
        // entry -> {a, b} and a <-> b, every edge weighted and each cycle block
        // also leaving to `exit`, so whichever block the search picks has two
        // outgoing edges to clone and an entering edge to redirect.
        let mut cfg = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        let exit = cfg.new_block();
        for (block, name) in [(cfg.entry(), "entry"), (a, "a"), (b, "b"), (exit, "exit")] {
            cfg.block_mut(block).instructions_mut().push(ff(name));
        }

        let into_a = cfg.add_weighted_edge(cfg.entry(), a, EdgeKind::ConditionalTrue, 0.375);
        let into_b = cfg.add_weighted_edge(cfg.entry(), b, EdgeKind::ConditionalFalse, 0.125);
        cfg.add_weighted_edge(a, b, EdgeKind::Fallthrough, 0.5);
        cfg.add_weighted_edge(a, exit, EdgeKind::SwitchCase, 0.625);
        cfg.add_weighted_edge(b, a, EdgeKind::Back, 0.75);
        cfg.add_weighted_edge(b, exit, EdgeKind::SwitchCase, 0.25);
        let original_block_bound = cfg.block_bound();

        let dom = DominatorTree::compute(&cfg);
        let target = find_irreducible_entry(&cfg, &dom).expect("the cycle has two entries");
        let entering = if target == a { into_a } else { into_b };
        let outgoing: Vec<_> = cfg.outgoing(target).collect();
        let signature = |cfg: &Cfg<crate::test_util::MockInst>, edge| {
            let edge = cfg.edge(edge);
            (edge.target(), edge.kind(), edge.weight())
        };
        let expected: Vec<_> = outgoing.iter().map(|&edge| signature(&cfg, edge)).collect();
        let entering_weight = cfg.edge(entering).weight();
        assert_eq!(make_reducible(&mut cfg), 1);

        let copy = BlockId::from_index(original_block_bound);
        assert_eq!(cfg.edge(entering).target(), copy);
        assert_eq!(cfg.edge(entering).weight(), entering_weight);
        assert_eq!(
            cfg.outgoing(target).collect::<Vec<_>>(),
            outgoing,
            "the original keeps its edges and their order"
        );

        let copied: Vec<_> = cfg.outgoing(copy).collect();
        assert_eq!(copied.len(), outgoing.len());
        for ((&copied, &original), expected) in copied.iter().zip(&outgoing).zip(&expected) {
            assert_ne!(copied, original);
            assert_eq!(cfg.edge(copied).source(), copy);
            assert_eq!(&signature(&cfg, copied), expected);
        }
    }
}
