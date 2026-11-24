//! Tests of cycle equivalence and the program structure tree.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use super::{CycleEquivalence, ProgramStructureTree, SeseEdge, SeseRegionId};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::{EdgeId, EdgeKind};
use crate::graph::store::Graph;
use crate::graph::view::Rooted;
use reference::{Boundary, Reference};

mod exhaustive;
mod paper;
mod reference;

/// A graph over `node_count` blocks, block 0 the entry, with `edges` in order.
fn build(node_count: usize, edges: &[(usize, usize)]) -> (Cfg<()>, Vec<BlockId>, Vec<EdgeId>) {
    let mut cfg = Cfg::new();
    let mut blocks = vec![cfg.entry()];
    blocks.extend((1..node_count).map(|_| cfg.new_block()));
    let ids = edges
        .iter()
        .map(|&(from, to)| cfg.add_edge(blocks[from], blocks[to], EdgeKind::Unconditional))
        .collect();
    (cfg, blocks, ids)
}

/// The regions of `tree` as boundary pairs, in preorder.
fn boundaries(tree: &ProgramStructureTree) -> Vec<(Boundary, Boundary)> {
    tree.regions()
        .map(|region| (tree.entry_edge(region), tree.exit_edge(region)))
        .collect()
}

/// The region with the given boundaries.
fn region_between(tree: &ProgramStructureTree, entry: Boundary, exit: Boundary) -> SeseRegionId {
    tree.regions()
        .find(|&region| tree.entry_edge(region) == entry && tree.exit_edge(region) == exit)
        .expect("the tree has a region with these boundaries")
}

/// Check what holds of every tree whatever graph produced it: regions in
/// preorder, nested or disjoint, each block in at most one innermost region,
/// and the three block queries agreeing.
fn assert_tree_invariants<I>(tree: &ProgramStructureTree, cfg: &Cfg<I>) {
    for region in tree.regions() {
        if let Some(parent) = tree.parent(region) {
            assert!(parent < region, "a parent precedes its child in preorder");
        }
        let blocks = tree.blocks(region);
        assert!(
            !blocks.is_empty(),
            "a canonical region holds its entry's target"
        );
        let unique: BTreeSet<BlockId> = blocks.iter().copied().collect();
        assert_eq!(unique.len(), blocks.len(), "a block is listed once");
        for &block in blocks {
            assert!(tree.contains(region, block));
            let mut inner = tree.innermost(block).expect("a listed block has a region");
            while inner != region {
                inner = tree
                    .parent(inner)
                    .expect("the innermost region lies inside every region holding the block");
            }
        }
        for block in cfg.block_ids() {
            assert_eq!(tree.contains(region, block), unique.contains(&block));
        }
        if let Some(parent) = tree.parent(region) {
            let outer: BTreeSet<BlockId> = tree.blocks(parent).iter().copied().collect();
            assert!(unique.is_subset(&outer), "a region lies inside its parent");
            assert!(unique.len() < outer.len(), "a parent is strictly larger");
        }
    }
    for first in tree.regions() {
        for second in tree.regions() {
            let a: BTreeSet<BlockId> = tree.blocks(first).iter().copied().collect();
            let b: BTreeSet<BlockId> = tree.blocks(second).iter().copied().collect();
            assert!(
                a.is_disjoint(&b) || a.is_subset(&b) || b.is_subset(&a),
                "regions nest or are disjoint"
            );
        }
    }
}

/// Check the classes and the tree of `cfg` against the definitions.
fn assert_matches_reference<I>(cfg: &Cfg<I>) {
    let reference = Reference::new(cfg);
    let classes = CycleEquivalence::compute(cfg);
    let tree = ProgramStructureTree::compute(cfg);
    assert_tree_invariants(&tree, cfg);

    let edges: Vec<Boundary> = reference.boundaries().collect();
    let mut seen = BTreeSet::new();
    for (a, &first) in edges.iter().enumerate() {
        let class = classes
            .class(first)
            .expect("an edge of the closed graph has a class");
        assert!(class < classes.class_count());
        seen.insert(class);
        for (b, &second) in edges.iter().enumerate() {
            assert_eq!(
                classes.class(first) == classes.class(second),
                reference.equivalent(a, b),
                "cycle equivalence of {first:?} and {second:?}"
            );
        }
    }
    assert_eq!(seen.len(), classes.class_count(), "every class has an edge");
    for edge in cfg.edges() {
        let reached = reference.reaches(edge.source().index());
        assert_eq!(classes.class(SeseEdge::Edge(edge.id())).is_some(), reached);
    }

    let expected = reference.canonical_regions();
    let expected_pairs: BTreeSet<(Boundary, Boundary)> = expected
        .iter()
        .map(|(entry, exit, _)| (*entry, *exit))
        .collect();
    let actual_pairs: BTreeSet<(Boundary, Boundary)> = boundaries(&tree).into_iter().collect();
    assert_eq!(actual_pairs, expected_pairs, "canonical regions");
    assert_eq!(tree.len(), expected.len());
    for (entry, exit, blocks) in &expected {
        let region = region_between(&tree, *entry, *exit);
        let actual: BTreeSet<usize> = tree.blocks(region).iter().map(|b| b.index()).collect();
        let wanted: BTreeSet<usize> = blocks.iter().copied().collect();
        assert_eq!(actual, wanted, "blocks of ({entry:?}, {exit:?})");
        let (entering, leaving) = reference.crossings(blocks);
        assert_eq!(
            entering,
            [*entry],
            "the only edge into ({entry:?}, {exit:?})"
        );
        assert_eq!(
            leaving,
            [*exit],
            "the only edge out of ({entry:?}, {exit:?})"
        );
    }
    for block in cfg.block_ids() {
        let smallest = expected
            .iter()
            .filter(|(_, _, blocks)| blocks.contains(&block.index()))
            .min_by_key(|(_, _, blocks)| blocks.len())
            .map(|(entry, exit, _)| region_between(&tree, *entry, *exit));
        assert_eq!(
            tree.innermost(block),
            smallest,
            "innermost region of {block:?}"
        );
    }
}

#[test]
fn a_lone_block_is_one_region_from_entry_to_exit() {
    let (cfg, blocks, _) = build(1, &[]);
    let tree = ProgramStructureTree::compute(&cfg);
    assert_eq!(
        boundaries(&tree),
        [(SeseEdge::Entry, SeseEdge::Exit(blocks[0]))]
    );
    assert_eq!(tree.blocks(SeseRegionId(0)), [blocks[0]]);
    let classes = CycleEquivalence::compute(&cfg);
    assert_eq!(classes.class_count(), 1);
    assert_matches_reference(&cfg);
}

#[test]
fn a_chain_is_a_sequence_of_one_block_regions() {
    let (cfg, blocks, edges) = build(3, &[(0, 1), (1, 2)]);
    let tree = ProgramStructureTree::compute(&cfg);
    assert_eq!(
        boundaries(&tree),
        [
            (SeseEdge::Entry, SeseEdge::Edge(edges[0])),
            (SeseEdge::Edge(edges[0]), SeseEdge::Edge(edges[1])),
            (SeseEdge::Edge(edges[1]), SeseEdge::Exit(blocks[2])),
        ]
    );
    for (index, &block) in blocks.iter().enumerate() {
        assert_eq!(
            tree.innermost(block),
            Some(SeseRegionId(u32::try_from(index).unwrap()))
        );
        assert_eq!(
            tree.parent(SeseRegionId(u32::try_from(index).unwrap())),
            None
        );
    }
}

#[test]
fn a_loop_headed_by_the_root_is_entered_over_the_virtual_entry() {
    // entry <-> body; entry -> exit.
    let (cfg, blocks, edges) = build(3, &[(0, 1), (1, 0), (0, 2)]);
    let tree = ProgramStructureTree::compute(&cfg);
    let whole_loop = region_between(&tree, SeseEdge::Entry, SeseEdge::Edge(edges[2]));
    assert_eq!(tree.blocks(whole_loop), [blocks[0], blocks[1]]);
    let body = region_between(&tree, SeseEdge::Edge(edges[0]), SeseEdge::Edge(edges[1]));
    assert_eq!(tree.parent(body), Some(whole_loop));
    assert_eq!(tree.innermost(blocks[1]), Some(body));
    assert_matches_reference(&cfg);
}

#[test]
fn an_endless_loop_leaves_over_the_cut() {
    // entry -> header <-> body, and nothing leaves the loop.
    let (cfg, blocks, edges) = build(3, &[(0, 1), (1, 2), (2, 1)]);
    let tree = ProgramStructureTree::compute(&cfg);
    // The latch closes the loop, so the cut joins it to the virtual exit.
    let endless = region_between(&tree, SeseEdge::Edge(edges[0]), SeseEdge::Exit(blocks[2]));
    assert_eq!(tree.blocks(endless), [blocks[1], blocks[2]]);
    assert_matches_reference(&cfg);
}

#[test]
fn two_exits_each_end_their_own_region() {
    let (cfg, blocks, edges) = build(3, &[(0, 1), (0, 2)]);
    let tree = ProgramStructureTree::compute(&cfg);
    assert_eq!(
        boundaries(&tree),
        [
            (SeseEdge::Edge(edges[0]), SeseEdge::Exit(blocks[1])),
            (SeseEdge::Edge(edges[1]), SeseEdge::Exit(blocks[2])),
        ]
    );
    assert_eq!(
        tree.innermost(blocks[0]),
        None,
        "the branch is in no region"
    );
    assert_matches_reference(&cfg);
}

#[test]
fn parallel_edges_stay_distinct() {
    // entry => next twice; next -> exit.
    let (cfg, _, edges) = build(3, &[(0, 1), (0, 1), (1, 2)]);
    let classes = CycleEquivalence::compute(&cfg);
    let class = |edge| classes.class(SeseEdge::Edge(edge));
    assert_ne!(class(edges[0]), class(edges[1]));
    assert_eq!(classes.class(SeseEdge::Entry), class(edges[2]));
    assert_matches_reference(&cfg);
}

#[test]
fn a_self_loop_is_a_class_of_its_own() {
    let (cfg, _, edges) = build(2, &[(0, 1), (1, 1)]);
    let classes = CycleEquivalence::compute(&cfg);
    let spin = classes.class(SeseEdge::Edge(edges[1]));
    assert!(spin.is_some());
    assert_ne!(spin, classes.class(SeseEdge::Edge(edges[0])));
    assert_ne!(spin, classes.class(SeseEdge::Entry));
    assert_matches_reference(&cfg);
}

#[test]
fn unreachable_and_removed_code_is_in_no_region() {
    // entry -> a -> exit; dead -> a; a removed block and a removed edge.
    let (mut cfg, blocks, edges) = build(5, &[(0, 1), (1, 2), (3, 1), (0, 2), (4, 2)]);
    assert!(cfg.remove_edge(edges[3]));
    assert!(cfg.remove_block(blocks[4]));
    let classes = CycleEquivalence::compute(&cfg);
    assert_eq!(classes.class(SeseEdge::Edge(edges[2])), None, "dead source");
    assert_eq!(
        classes.class(SeseEdge::Edge(edges[3])),
        None,
        "removed edge"
    );
    assert_eq!(classes.class(SeseEdge::Edge(EdgeId::from_index(99))), None);
    assert_eq!(
        classes.class(SeseEdge::Exit(blocks[1])),
        None,
        "not an exit"
    );
    let tree = ProgramStructureTree::compute(&cfg);
    assert_eq!(tree.innermost(blocks[3]), None);
    assert_eq!(tree.innermost(blocks[4]), None);
    assert_eq!(tree.innermost(BlockId::from_index(99)), None);
    assert_eq!(tree.len(), 3, "the live chain");
    assert_matches_reference(&cfg);
}

#[test]
fn any_rooted_edge_view_is_accepted() {
    let mut graph = Graph::<(), ()>::new();
    let nodes: Vec<_> = (0..4).map(|_| graph.add_node(())).collect();
    let into_loop = graph.add_edge(nodes[0], nodes[1], ());
    graph.add_edge(nodes[1], nodes[2], ());
    graph.add_edge(nodes[2], nodes[1], ());
    let out_of_loop = graph.add_edge(nodes[1], nodes[3], ());
    let rooted = Rooted::new(&graph, nodes[0]);
    let tree = ProgramStructureTree::compute(&rooted);
    let found = tree
        .regions()
        .find(|&region| tree.entry_edge(region) == SeseEdge::Edge(into_loop))
        .expect("the loop is a region");
    assert_eq!(tree.exit_edge(found), SeseEdge::Edge(out_of_loop));
    assert_eq!(tree.blocks(found), [nodes[1], nodes[2]]);
    let classes = CycleEquivalence::compute(&rooted);
    assert_eq!(
        classes.class(SeseEdge::Edge(into_loop)),
        classes.class(SeseEdge::Edge(out_of_loop))
    );
}

#[test]
#[should_panic(expected = "program structure root is not a node of the graph")]
fn a_root_beyond_the_bound_panics() {
    let graph = Graph::<(), ()>::new();
    let missing = crate::graph::store::NodeId::from_index(3);
    let _ = ProgramStructureTree::compute(&Rooted::new(&graph, missing));
}

/// Blocks in a chain of `length`.
fn chain(length: usize) -> Cfg<()> {
    let edges: Vec<(usize, usize)> = (1..length).map(|next| (next - 1, next)).collect();
    build(length, &edges).0
}

/// If/else diamonds nested `depth` deep in their `then` arms, and the blocks
/// that branch at each level, the innermost `then` block last.
fn diamond_nest(depth: usize) -> (Cfg<()>, Vec<BlockId>) {
    let mut cfg = Cfg::new();
    let mut branches = vec![cfg.entry()];
    let mut joins = Vec::with_capacity(depth);
    for level in 0..depth {
        let then_block = cfg.new_block();
        let else_block = cfg.new_block();
        let join = cfg.new_block();
        cfg.add_edge(branches[level], then_block, EdgeKind::ConditionalTrue);
        cfg.add_edge(branches[level], else_block, EdgeKind::ConditionalFalse);
        cfg.add_edge(else_block, join, EdgeKind::Fallthrough);
        if let Some(&outer) = joins.last() {
            cfg.add_edge(join, outer, EdgeKind::Fallthrough);
        }
        branches.push(then_block);
        joins.push(join);
    }
    let innermost = *branches.last().expect("the entry branches");
    if let Some(&join) = joins.last() {
        cfg.add_edge(innermost, join, EdgeKind::Fallthrough);
    }
    (cfg, branches)
}

#[test]
fn a_chain_of_a_hundred_thousand_blocks_needs_no_call_stack() {
    let cfg = chain(100_000);
    let tree = ProgramStructureTree::compute(&cfg);
    assert_eq!(tree.len(), 100_000);
    assert!(tree.regions().all(|region| tree.parent(region).is_none()));
}

#[test]
fn diamonds_nested_a_hundred_thousand_deep_need_no_call_stack() {
    const DEPTH: usize = 100_000;
    let (cfg, branches) = diamond_nest(DEPTH);
    let tree = ProgramStructureTree::compute(&cfg);
    // The whole procedure, one diamond per inner level, each `else` arm, and
    // the innermost `then` block.
    assert_eq!(tree.len(), 2 * DEPTH + 1);
    let mut depth = 0;
    let mut region = tree.innermost(branches[DEPTH]);
    while let Some(inside) = region {
        depth += 1;
        region = tree.parent(inside);
    }
    assert_eq!(depth, DEPTH + 1);
}

#[test]
fn small_diamond_nests_match_the_definitions() {
    for depth in 0..4 {
        assert_matches_reference(&diamond_nest(depth).0);
    }
}
