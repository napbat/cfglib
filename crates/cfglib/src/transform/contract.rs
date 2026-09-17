//! Edge contraction and instruction-neutral node splitting.

extern crate alloc;

use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::{Cfg, SplitPointError};
use crate::edge::EdgeId;
use crate::rewrite::Rewrite;

/// Contract an edge by merging `target` into `source`.
///
/// `target` is removed: its instructions, its label, and its outgoing edges
/// move to `source`, and the block itself stops being a block of the CFG,
/// exactly as [`merge_blocks`](crate::merge_blocks) leaves the chain it
/// folded. Its slot stays reserved, so every surviving identity is unchanged.
///
/// This compatibility entry point reports only whether contraction happened.
/// Use [`contract_edge_mapped`] when clients retain graph identities. The CFG
/// entry is never accepted as `target`, and self-loops are not contracted.
pub fn contract_edge<I, E>(cfg: &mut Cfg<I, E>, source: BlockId, target: BlockId) -> bool {
    contract_edge_inner(cfg, source, target, None)
}

/// Contract an edge while retaining all surviving outgoing edge identities
/// and payloads.
///
/// Returns `None` unless `source` has exactly one outgoing edge, that edge
/// targets `target`, and it is `target`'s sole incoming edge. The connecting
/// edge maps to removal, `target` is removed and maps to `source`, and
/// redirected outgoing edges map to themselves. The CFG entry is never
/// accepted as `target`, and self-loops are not contracted.
pub fn contract_edge_mapped<I, E>(
    cfg: &mut Cfg<I, E>,
    source: BlockId,
    target: BlockId,
) -> Option<Rewrite> {
    let mut mapping = Rewrite::new();
    contract_edge_inner(cfg, source, target, Some(&mut mapping)).then_some(mapping)
}

fn contract_edge_inner<I, E>(
    cfg: &mut Cfg<I, E>,
    source: BlockId,
    target: BlockId,
    mut mapping: Option<&mut Rewrite>,
) -> bool {
    if source == target || target == cfg.entry() {
        return false;
    }
    let source_outgoing: Vec<EdgeId> = cfg.outgoing(source).collect();
    let target_incoming: Vec<EdgeId> = cfg.incoming(target).collect();
    if source_outgoing.len() != 1 || target_incoming.len() != 1 {
        return false;
    }
    let connecting = source_outgoing[0];
    if target_incoming[0] != connecting || cfg.edge(connecting).target() != target {
        return false;
    }

    let target_label = cfg.block(target).label().map(alloc::string::String::from);
    let target_instructions = core::mem::take(cfg.block_mut(target).instructions_mut());
    cfg.block_mut(source)
        .instructions_mut()
        .extend(target_instructions);
    if cfg.block(source).label().is_none() {
        if let Some(label) = target_label {
            cfg.block_mut(source).set_label(label);
        }
    }

    cfg.remove_edge(connecting);
    if let Some(mapping) = mapping.as_deref_mut() {
        mapping.record_edge(connecting, []);
    }
    cfg.move_outgoing_edges(target, source);
    let moved: Vec<EdgeId> = cfg.outgoing(source).collect();
    cfg.remove_block(target);
    if let Some(mapping) = mapping {
        for edge in moved {
            mapping.record_edge(edge, [edge]);
        }
        mapping.record_block(target, [source]);
    }
    true
}

/// Split a block at one instruction index with a default edge payload.
pub fn split_node<I, E>(cfg: &mut Cfg<I, E>, block: BlockId, at: usize) -> BlockId
where
    E: Default,
{
    cfg.split_block(block, at)
}

/// Split a block at one instruction index and return its rewrite mapping.
pub fn split_node_with_payload_mapped<I, E>(
    cfg: &mut Cfg<I, E>,
    block: BlockId,
    at: usize,
    payload: E,
) -> (BlockId, Rewrite) {
    cfg.split_block_with_payload_mapped(block, at, payload)
}

/// Split a block at several original instruction boundaries.
///
/// This frontend-neutral primitive is suitable for materializing individual
/// throwing instructions, invoke sites, or any other consumer-selected
/// program points without encoding instruction semantics in cfglib.
///
/// # Errors
///
/// Returns [`SplitPointError`] when a point is out of bounds or the points are
/// not strictly increasing.
pub fn split_node_at_points<I, E>(
    cfg: &mut Cfg<I, E>,
    block: BlockId,
    points: impl IntoIterator<Item = (usize, E)>,
) -> Result<(alloc::vec::Vec<BlockId>, Rewrite), SplitPointError> {
    cfg.split_block_at_points_with_payloads(block, points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::EdgeKind;
    use crate::test_util::ff;

    #[test]
    fn contract_retains_outgoing_identity_and_payload() {
        let mut cfg = Cfg::<_, &'static str>::with_edge_payload();
        let target = cfg.new_block();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(ff("a"));
        cfg.block_mut(target).push(ff("b"));
        let connecting =
            cfg.add_edge_with_payload(cfg.entry(), target, EdgeKind::Fallthrough, "join");
        let outgoing = cfg.add_edge_with_payload(target, exit, EdgeKind::Jump, "provenance");

        let entry = cfg.entry();
        let mapping = contract_edge_mapped(&mut cfg, entry, target).unwrap();
        assert_eq!(mapping.edges(connecting), Some([].as_slice()));
        assert_eq!(mapping.edges(outgoing), Some([outgoing].as_slice()));
        assert_eq!(mapping.blocks(target), Some([entry].as_slice()));
        assert_eq!(cfg.edge(outgoing).source(), entry);
        assert_eq!(cfg.edge(outgoing).payload(), &"provenance");
        assert!(!cfg.contains_block(target), "the contracted block is gone");
        assert_eq!(cfg.block_count(), 2);
        assert_eq!(cfg.block_bound(), 3, "its slot stays reserved");
    }

    #[test]
    fn contract_removes_the_block_it_folds_like_a_merge() {
        let mut cfg = Cfg::<u32>::new();
        let source = cfg.entry();
        let target = cfg.new_block();
        let sink = cfg.new_block();
        cfg.block_mut(source).push(0);
        cfg.block_mut(target).push(1);
        cfg.block_mut(sink).push(2);
        cfg.add_edge(source, target, EdgeKind::Fallthrough);
        cfg.add_edge(target, sink, EdgeKind::Jump);

        assert!(contract_edge(&mut cfg, source, target));

        assert!(!cfg.contains_block(target));
        assert_eq!(cfg.block_count(), 2);
        assert_eq!(cfg.block(source).instructions(), &[0, 1]);
        assert_eq!(
            cfg.block_ids().collect::<Vec<_>>(),
            [source, sink],
            "a contracted block is no longer a block of the CFG"
        );
        assert!(crate::verify(&cfg).is_ok());
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
        assert_eq!(cfg.outgoing(entry).collect::<Vec<_>>(), &[outgoing]);
        assert_eq!(cfg.outgoing(source).collect::<Vec<_>>(), &[back]);
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

        assert!(!cfg.contains_block(target));
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
    fn split_points_are_atomic_and_instruction_neutral() {
        let mut cfg = Cfg::<_, u8>::with_edge_payload();
        let entry = cfg.entry();
        cfg.block_mut(entry)
            .instructions_mut()
            .extend([ff("a"), ff("may_throw"), ff("c")]);

        let error = split_node_at_points(&mut cfg, entry, [(2, 2), (1, 1)]).unwrap_err();
        assert!(matches!(
            error,
            SplitPointError::NotStrictlyIncreasing {
                previous: 2,
                point: 1
            }
        ));
        assert_eq!(cfg.block_count(), 1, "validation precedes mutation");

        let (blocks, mapping) = split_node_at_points(&mut cfg, entry, [(1, 10), (2, 20)]).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(cfg.block(blocks[1]).instructions()[0].1, "may_throw");
        assert_eq!(mapping.blocks(entry), Some(blocks.as_slice()));
        let first_edge = cfg.outgoing(blocks[0]).next().unwrap();
        let second_edge = cfg.outgoing(blocks[1]).next().unwrap();
        assert_eq!(cfg.edge(first_edge).payload(), &10);
        assert_eq!(cfg.edge(second_edge).payload(), &20);
    }
}
