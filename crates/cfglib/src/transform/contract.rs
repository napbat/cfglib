//! Edge contraction and node splitting.
//!
//! Graph rewriting primitives that complement the existing block
//! merging in [`super::cleanup`].

extern crate alloc;

use crate::block::BlockId;
use crate::cfg::Cfg;

/// Contract an edge by merging the target block into the source block.
///
/// The edge `(source → target)` is removed, the target's instructions
/// are appended to the source, and all outgoing edges of the target
/// are redirected to originate from the source.
///
/// Returns `true` if contraction was performed, `false` if the edge
/// cannot be contracted (e.g., target has other predecessors, or
/// source has other successors). The CFG entry is never accepted as a target.
///
/// Requires `I: Clone` because instruction vectors are manipulated.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, contract_edge};
///
/// let mut cfg = Cfg::<u32>::new();
/// let b0 = cfg.entry();
/// let b1 = cfg.new_block();
/// cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
///
/// // b0 has 1 succ, b1 has 1 pred — contractible.
/// assert!(contract_edge(&mut cfg, b0, b1));
/// ```
pub fn contract_edge<I: Clone>(cfg: &mut Cfg<I>, source: BlockId, target: BlockId) -> bool {
    if target == cfg.entry() {
        return false;
    }

    // Target must have exactly one predecessor (source).
    let [incoming] = cfg.predecessor_edges(target) else {
        return false;
    };
    let incoming = *incoming;
    // Source must have exactly one successor (target).
    let [connecting] = cfg.successor_edges(source) else {
        return false;
    };
    let connecting = *connecting;
    // Don't contract self-loops.
    if source == target
        || cfg.edge(incoming).source() != source
        || cfg.edge(connecting).target() != target
    {
        return false;
    }

    // Append target's instructions to source.
    let target_instrs = cfg.block(target).instructions().to_vec();
    cfg.block_mut(source)
        .instructions_vec_mut()
        .extend(target_instrs);

    // Copy label if source doesn't have one.
    let inherited_label = if cfg.block(source).label().is_none() {
        cfg.block(target).label().map(alloc::string::String::from)
    } else {
        None
    };
    if let Some(label) = inherited_label {
        cfg.block_mut(source).set_label(label);
    }

    // Remove the edge source → target.
    cfg.remove_edge(connecting);

    // Redirect target's outgoing edges to source.
    cfg.move_outgoing_edges(target, source);

    true
}

/// Split a block at a given instruction index, creating a new block.
///
/// This is a thin wrapper around [`Cfg::split_block`] that also
/// reconnects edges properly.
///
/// Returns the new block containing instructions from `at` onward.
pub fn split_node<I: Clone>(cfg: &mut Cfg<I>, block: BlockId, at: usize) -> BlockId {
    cfg.split_block(block, at)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::EdgeKind;
    use crate::test_util::ff;

    #[test]
    fn contract_linear_chain() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("a"));
        cfg.block_mut(b).instructions_vec_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);

        let entry = cfg.entry();
        let ok = contract_edge(&mut cfg, entry, b);
        assert!(ok);
        assert_eq!(cfg.block(entry).instructions().len(), 2);
    }

    #[test]
    fn contract_refuses_multi_pred() {
        let mut cfg: Cfg<crate::test_util::MockInst> = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        let merge = cfg.new_block();
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        cfg.add_edge(a, merge, EdgeKind::Fallthrough);
        cfg.add_edge(b, merge, EdgeKind::Fallthrough);

        // merge has 2 predecessors — cannot contract.
        assert!(!contract_edge(&mut cfg, a, merge));
    }

    #[test]
    fn contract_refuses_to_consume_the_entry_block() {
        let mut cfg = Cfg::<u32>::new();
        let entry = cfg.entry();
        let source = cfg.new_block();
        cfg.block_mut(entry).push(0);
        cfg.block_mut(source).push(1);
        let outgoing = cfg.add_edge(entry, source, EdgeKind::Fallthrough);
        let back = cfg.add_edge(source, entry, EdgeKind::Back);

        assert!(!contract_edge(&mut cfg, source, entry));

        assert_eq!(cfg.block(entry).instructions(), &[0]);
        assert_eq!(cfg.block(source).instructions(), &[1]);
        assert_eq!(cfg.successor_edges(entry), &[outgoing]);
        assert_eq!(cfg.successor_edges(source), &[back]);
        assert_eq!(cfg.edge(back).target(), entry);
        assert!(crate::verify(&cfg).is_ok());
    }

    #[test]
    fn contract_preserves_parallel_weighted_edges_and_back_edge_identity() {
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

        assert!(contract_edge(&mut cfg, source, target));

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
    fn split_node_works() {
        let mut cfg = Cfg::new();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .extend([ff("a"), ff("b"), ff("c")]);

        let entry = cfg.entry();
        let new_block = split_node(&mut cfg, entry, 1);
        assert_eq!(cfg.block(entry).instructions().len(), 1);
        assert_eq!(cfg.block(new_block).instructions().len(), 2);
    }
}
