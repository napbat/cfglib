extern crate alloc;
extern crate std;
use super::*;
use crate::edge::{Edge, EdgeKind};
use crate::graph::store::NodeOrder;
use crate::test_util::MockInst;
use alloc::vec;
use alloc::vec::Vec;

#[test]
fn edge_weight_roundtrip() {
    let mut cfg = Cfg::<MockInst>::new();
    let b0 = cfg.entry();
    let b1 = cfg.new_block();
    let eid = cfg.add_edge(
        b0,
        b1,
        Edge::new(EdgeKind::ConditionalTrue).with_weight(0.75),
    );
    assert_eq!(cfg.edge(eid).weight(), Some(0.75));
    // Default edge should have no weight.
    let eid2 = cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
    assert_eq!(cfg.edge(eid2).weight(), None);
}

#[test]
fn subgraph_extraction() {
    let mut cfg = Cfg::<MockInst>::new();
    let b0 = cfg.entry();
    let b1 = cfg.new_block();
    let b2 = cfg.new_block();
    cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
    cfg.add_edge(b1, b2, EdgeKind::Fallthrough);

    // Extract first two blocks.
    let (sub, rewrite) = cfg.subgraph_mapped(&[b0, b1]);
    assert_eq!(sub.block_count(), 2);
    // The subgraph should have an edge from block 0 to block 1.
    let succs: Vec<BlockId> = sub.successors(sub.entry()).collect();
    assert_eq!(succs.len(), 1);
    assert_eq!(rewrite.blocks(b2), Some([].as_slice()));
}

#[test]
fn subgraph_empty_input() {
    let (sub, _) = Cfg::<MockInst>::new().subgraph_mapped(&[]);
    assert_eq!(sub.block_count(), 1); // Cfg::new() always has an entry
}

#[test]
fn remove_edge_tombstones_correctly() {
    let mut cfg = Cfg::<MockInst>::new();
    let b0 = cfg.entry();
    let b1 = cfg.new_block();
    let b2 = cfg.new_block();
    let e1 = cfg.add_edge(b0, b1, EdgeKind::ConditionalTrue);
    let e2 = cfg.add_edge(b0, b2, EdgeKind::ConditionalFalse);

    // Both edges are live.
    assert_eq!(cfg.edge_count(), 2);
    assert_eq!(cfg.edges().count(), 2);

    // Remove one edge.
    assert_eq!(cfg.edge(e1).kind(), EdgeKind::ConditionalTrue);
    assert!(cfg.remove_edge(e1));

    // edges() should now skip the removed slot.
    assert_eq!(cfg.edges().count(), 1);
    let remaining: Vec<_> = cfg.edges().map(|edge| edge.id()).collect();
    assert_eq!(remaining[0], e2);

    // Successor list should only contain e2.
    assert_eq!(cfg.outgoing(b0).count(), 1);
    assert_eq!(cfg.outgoing(b0).next().unwrap(), e2);

    // Double-remove reports that there was nothing to do.
    assert!(!cfg.remove_edge(e1));
}

#[test]
fn split_block_preserves_outgoing_edge_identity_and_metadata() {
    let mut cfg = Cfg::<u32>::new();
    let source = cfg.entry();
    let sink = cfg.new_block();
    cfg.block_mut(source).instructions_mut().extend([1, 2]);
    let outgoing = cfg.add_edge(
        source,
        sink,
        Edge::new(EdgeKind::ConditionalTrue).with_weight(0.75),
    );

    let split = cfg.split_block(source, 1, EdgeKind::Fallthrough);
    let [fallthrough] = cfg.outgoing(source).collect::<Vec<_>>()[..] else {
        panic!("split source should have one fallthrough edge");
    };

    assert_eq!(cfg.outgoing(split).collect::<Vec<_>>(), &[outgoing]);
    assert_eq!(cfg.edge(outgoing).source(), split);
    assert_eq!(cfg.edge(outgoing).target(), sink);
    assert_eq!(cfg.edge(outgoing).kind(), EdgeKind::ConditionalTrue);
    assert_eq!(cfg.edge(outgoing).weight(), Some(0.75));
    assert_eq!(cfg.edge(fallthrough).source(), source);
    assert_eq!(cfg.edge(fallthrough).target(), split);
    assert_eq!(cfg.incoming(sink).collect::<Vec<_>>(), &[outgoing]);
}

#[test]
fn split_points_are_atomic_and_instruction_neutral() {
    let mut cfg = Cfg::<_, u8>::with_edge_payload();
    let entry = cfg.entry();
    cfg.block_mut(entry)
        .instructions_mut()
        .extend(["a", "may_throw", "c"]);

    let point = |at, payload| (at, Edge::new(EdgeKind::Fallthrough).with_payload(payload));
    let error = cfg
        .split_block_at_points(entry, [point(2, 2), point(1, 1)])
        .unwrap_err();
    assert!(matches!(
        error,
        SplitPointError::NotStrictlyIncreasing {
            previous: 2,
            point: 1
        }
    ));
    assert_eq!(cfg.block_count(), 1, "validation precedes mutation");

    let exit = cfg.new_block();
    let leaving = cfg.add_edge(entry, exit, Edge::new(EdgeKind::Jump).with_payload(7));
    let (blocks, rewrite) = cfg
        .split_block_at_points(entry, [point(1, 10), point(2, 20)])
        .unwrap();
    assert_eq!(blocks.len(), 3);
    assert_eq!(cfg.block(blocks[1]).instructions(), &["may_throw"]);
    let first_edge = cfg.outgoing(blocks[0]).next().unwrap();
    let second_edge = cfg.outgoing(blocks[1]).next().unwrap();
    assert_eq!(cfg.edge(first_edge).payload(), &10);
    assert_eq!(cfg.edge(second_edge).payload(), &20);
    assert_eq!(cfg.edge(leaving).source(), blocks[2]);

    // The record keys only identities that existed before the split.
    assert_eq!(rewrite.blocks(entry), Some(blocks.as_slice()));
    assert_eq!(rewrite.blocks(blocks[1]), None);
    assert_eq!(rewrite.created_blocks(), &blocks[1..]);
    assert_eq!(rewrite.created_edges(), &[first_edge, second_edge]);
    assert_eq!(rewrite.edges(leaving), Some([leaving].as_slice()));
}

#[test]
fn splitting_at_no_point_changes_nothing() {
    let mut cfg = Cfg::<u32>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).instructions_mut().extend([1, 2]);
    let (blocks, rewrite) = cfg
        .split_block_at_points(entry, core::iter::empty::<(usize, EdgeKind)>())
        .unwrap();
    assert_eq!(blocks, [entry]);
    assert!(rewrite.is_empty());
    assert_eq!(cfg.block_count(), 1);
}

#[test]
fn redirect_edges_moves_predecessors_in_order() {
    let mut cfg = Cfg::<MockInst>::new();
    let old = cfg.new_block();
    let new_target = cfg.new_block();
    let first = cfg.add_edge(cfg.entry(), new_target, EdgeKind::Fallthrough);
    let second = cfg.add_edge(cfg.entry(), old, EdgeKind::ConditionalTrue);
    let third = cfg.add_edge(
        cfg.entry(),
        old,
        Edge::new(EdgeKind::ConditionalFalse).with_weight(0.25),
    );

    cfg.redirect_edges_to(old, new_target);

    assert_eq!(cfg.incoming(old).collect::<Vec<_>>(), &[]);
    assert_eq!(
        cfg.incoming(new_target).collect::<Vec<_>>(),
        &[first, second, third]
    );
    assert_eq!(cfg.edge(second).target(), new_target);
    assert_eq!(cfg.edge(third).target(), new_target);
    assert_eq!(cfg.edge(third).weight(), Some(0.25));
}

#[test]
fn redirect_edges_to_same_block_is_a_noop() {
    let mut cfg = Cfg::<MockInst>::new();
    let target = cfg.new_block();
    let edge = cfg.add_edge(cfg.entry(), target, EdgeKind::Fallthrough);

    cfg.redirect_edges_to(target, target);

    assert_eq!(cfg.incoming(target).collect::<Vec<_>>(), &[edge]);
    assert_eq!(cfg.edge(edge).target(), target);
}

#[test]
fn redirect_edges_rejects_an_invalid_target_before_mutating() {
    let mut cfg = Cfg::<MockInst>::new();
    let old_target = cfg.new_block();
    let edge = cfg.add_edge(cfg.entry(), old_target, EdgeKind::Fallthrough);
    let invalid = BlockId::from_index(cfg.block_bound());

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cfg.redirect_edges_to(old_target, invalid);
    }));

    assert!(panic.is_err());
    assert_eq!(cfg.incoming(old_target).collect::<Vec<_>>(), &[edge]);
    assert_eq!(cfg.edge(edge).target(), old_target);
}

#[test]
fn move_outgoing_edges_preserves_order_identity_and_metadata() {
    let mut cfg = Cfg::<MockInst>::new();
    let old = cfg.new_block();
    let new_source = cfg.new_block();
    let sink = cfg.new_block();
    let existing = cfg.add_edge(new_source, sink, EdgeKind::Fallthrough);
    let first = cfg.add_edge(
        old,
        sink,
        Edge::new(EdgeKind::ConditionalTrue).with_weight(0.25),
    );
    let second = cfg.add_edge(
        old,
        sink,
        Edge::new(EdgeKind::ConditionalFalse).with_weight(0.75),
    );
    let becomes_self = cfg.add_edge(
        old,
        new_source,
        Edge::new(EdgeKind::Back).with_weight(0.875),
    );
    let old_self = cfg.add_edge(
        old,
        old,
        Edge::new(EdgeKind::Unconditional).with_weight(0.5),
    );

    cfg.move_outgoing_edges(old, new_source);

    assert_eq!(cfg.outgoing(old).collect::<Vec<_>>(), &[]);
    assert_eq!(
        cfg.outgoing(new_source).collect::<Vec<_>>(),
        &[existing, first, second, becomes_self, old_self]
    );
    assert_eq!(cfg.edge(first).source(), new_source);
    assert_eq!(cfg.edge(first).target(), sink);
    assert_eq!(cfg.edge(first).kind(), EdgeKind::ConditionalTrue);
    assert_eq!(cfg.edge(first).weight(), Some(0.25));
    assert_eq!(cfg.edge(second).weight(), Some(0.75));
    assert_eq!(cfg.edge(becomes_self).source(), new_source);
    assert_eq!(cfg.edge(becomes_self).target(), new_source);
    assert_eq!(cfg.edge(becomes_self).weight(), Some(0.875));
    assert_eq!(cfg.edge(old_self).source(), new_source);
    assert_eq!(cfg.edge(old_self).target(), old);
    assert_eq!(cfg.edge(old_self).weight(), Some(0.5));
    assert_eq!(
        cfg.incoming(sink).collect::<Vec<_>>(),
        &[existing, first, second]
    );
    assert_eq!(
        cfg.incoming(new_source).collect::<Vec<_>>(),
        &[becomes_self]
    );
    assert_eq!(cfg.incoming(old).collect::<Vec<_>>(), &[old_self]);
}

#[test]
fn redirecting_onto_a_block_keeps_the_edges_added_after_its_last_redirect() {
    let mut cfg = Cfg::<MockInst>::new();
    let a = cfg.entry();
    let b = cfg.new_block();
    let c = cfg.new_block();
    let d = cfg.new_block();
    let first = cfg.add_edge(a, b, EdgeKind::Unconditional);
    let resident = cfg.add_edge(c, d, EdgeKind::Unconditional);
    cfg.redirect_edge_source(first, c);
    let appended = cfg.add_edge(c, b, EdgeKind::Unconditional);
    let arriving = cfg.add_edge(d, a, EdgeKind::Unconditional);

    cfg.redirect_edge_source(arriving, c);

    assert_eq!(
        cfg.outgoing(c).collect::<Vec<_>>(),
        &[resident, first, appended, arriving]
    );
    assert_eq!(cfg.incoming(b).collect::<Vec<_>>(), &[first, appended]);
    let report = crate::verify(&cfg);
    assert!(report.is_ok(), "{:?}", report.errors);
}

#[test]
fn exit_blocks_iterator() {
    let mut cfg = Cfg::<MockInst>::new();
    let b1 = cfg.new_block();
    cfg.add_edge(cfg.entry(), b1, EdgeKind::Fallthrough);
    // b1 has no outgoing edges — it's an exit block.
    let exits: Vec<BlockId> = cfg.exit_blocks().collect();
    assert_eq!(exits, vec![b1]);
}

#[test]
fn is_dense_is_true_exactly_when_compaction_changes_no_identity() {
    let mut cfg = Cfg::<MockInst>::new();
    let kept = cfg.new_block();
    let middle = cfg.new_block();
    let last = cfg.new_block();
    cfg.add_edge(cfg.entry(), kept, EdgeKind::Fallthrough);
    let doomed = cfg.add_edge(kept, middle, EdgeKind::Fallthrough);
    cfg.add_edge(middle, last, EdgeKind::Fallthrough);
    assert!(cfg.is_dense(), "adding never invalidates an identity");

    // A removed slot below a live one: compaction renumbers.
    cfg.remove_edge(doomed);
    cfg.remove_block(middle);
    let predicted = cfg.is_dense();
    assert!(!predicted);
    assert_eq!(cfg.compact(NodeOrder::Insertion).is_identity(), predicted);
    assert!(cfg.is_dense());

    // A removed trailing slot is still a removed slot: compaction drops it.
    let trailing = cfg.new_block();
    cfg.remove_block(trailing);
    let predicted = cfg.is_dense();
    assert!(!predicted);
    assert_eq!(cfg.compact(NodeOrder::Insertion).is_identity(), predicted);

    // Every edge removed: the edge slots are all removed ones.
    let only = cfg.edges().next().map(|edge| edge.id());
    cfg.remove_edge(only.expect("the chain kept one edge"));
    let predicted = cfg.is_dense();
    assert!(!predicted);
    assert_eq!(cfg.compact(NodeOrder::Insertion).is_identity(), predicted);
    assert!(cfg.is_dense());
}
