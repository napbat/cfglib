//! The examples of Johnson, Pearson and Pingali, "The Program Structure Tree:
//! Computing Control Regions in Linear Time", PLDI 1994.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use super::{Boundary, assert_matches_reference, build, region_between};
use crate::block::BlockId;
use crate::graph::program_structure::{CycleEquivalence, ProgramStructureTree, SeseEdge};

/// Check that `groups` is the partition of the listed edges into classes, and
/// that there are no other classes.
fn assert_classes(classes: &CycleEquivalence, groups: &[&[Boundary]]) {
    let mut seen = BTreeSet::new();
    for group in groups {
        let class = classes.class(group[0]).expect("a listed edge has a class");
        for &edge in *group {
            assert_eq!(
                classes.class(edge),
                Some(class),
                "{edge:?} with {:?}",
                group[0]
            );
        }
        assert!(
            seen.insert(class),
            "the class of {:?} is listed once",
            group[0]
        );
    }
    assert_eq!(seen.len(), classes.class_count());
}

/// The blocks of Figure 1(a), top to bottom, left arm before right arm.
mod block {
    pub(super) const START: usize = 0;
    pub(super) const TOP: usize = 1;
    pub(super) const LOOP_HEADER: usize = 2;
    pub(super) const IN_F: usize = 3;
    pub(super) const BRANCH: usize = 4;
    pub(super) const IN_I: usize = 5;
    pub(super) const IN_J: usize = 6;
    pub(super) const MERGE: usize = 7;
    pub(super) const LATCH: usize = 8;
    pub(super) const IN_C: usize = 9;
    pub(super) const DIAMOND: usize = 10;
    pub(super) const IN_H: usize = 11;
    pub(super) const LOOPED_ARM: usize = 12;
    pub(super) const JOIN: usize = 13;
    pub(super) const TAIL: usize = 14;
    pub(super) const IN_E: usize = 15;
    pub(super) const BOTTOM: usize = 16;
    pub(super) const END: usize = 17;
    pub(super) const COUNT: usize = 18;
}

use block::{
    BOTTOM, BRANCH, COUNT, DIAMOND, END, IN_C, IN_E, IN_F, IN_H, IN_I, IN_J, JOIN, LATCH,
    LOOP_HEADER, LOOPED_ARM, MERGE, START, TAIL, TOP,
};

/// The edges of Figure 1(a).
const FIGURE_ONE: [(usize, usize); 22] = [
    (START, TOP),
    (TOP, LOOP_HEADER),
    (TOP, IN_C),
    (LOOP_HEADER, IN_F),
    (IN_F, BRANCH),
    (BRANCH, IN_I),
    (BRANCH, IN_J),
    (IN_I, MERGE),
    (IN_J, MERGE),
    (MERGE, LATCH),
    (LATCH, LOOP_HEADER),
    (LATCH, BOTTOM),
    (IN_C, DIAMOND),
    (DIAMOND, IN_H),
    (DIAMOND, LOOPED_ARM),
    (IN_H, JOIN),
    (LOOPED_ARM, JOIN),
    (JOIN, TAIL),
    (TAIL, LOOPED_ARM),
    (TAIL, IN_E),
    (IN_E, BOTTOM),
    (BOTTOM, END),
];

/// One region of Figure 1(b): its name, entry and exit edges, the name of
/// its parent, and its blocks.
type Named = (
    char,
    (usize, usize),
    (usize, usize),
    Option<char>,
    &'static [usize],
);

/// The program structure tree of Figure 1(b).
const FIGURE_ONE_TREE: [Named; 10] = [
    (
        'a',
        (START, TOP),
        (BOTTOM, END),
        None,
        &[
            TOP,
            LOOP_HEADER,
            IN_F,
            BRANCH,
            IN_I,
            IN_J,
            MERGE,
            LATCH,
            IN_C,
            DIAMOND,
            IN_H,
            LOOPED_ARM,
            JOIN,
            TAIL,
            IN_E,
            BOTTOM,
        ],
    ),
    (
        'b',
        (TOP, LOOP_HEADER),
        (LATCH, BOTTOM),
        Some('a'),
        &[LOOP_HEADER, IN_F, BRANCH, IN_I, IN_J, MERGE, LATCH],
    ),
    ('c', (TOP, IN_C), (IN_C, DIAMOND), Some('a'), &[IN_C]),
    (
        'd',
        (IN_C, DIAMOND),
        (TAIL, IN_E),
        Some('a'),
        &[DIAMOND, IN_H, LOOPED_ARM, JOIN, TAIL],
    ),
    ('e', (TAIL, IN_E), (IN_E, BOTTOM), Some('a'), &[IN_E]),
    ('f', (LOOP_HEADER, IN_F), (IN_F, BRANCH), Some('b'), &[IN_F]),
    (
        'g',
        (IN_F, BRANCH),
        (MERGE, LATCH),
        Some('b'),
        &[BRANCH, IN_I, IN_J, MERGE],
    ),
    ('h', (DIAMOND, IN_H), (IN_H, JOIN), Some('d'), &[IN_H]),
    ('i', (BRANCH, IN_I), (IN_I, MERGE), Some('g'), &[IN_I]),
    ('j', (BRANCH, IN_J), (IN_J, MERGE), Some('g'), &[IN_J]),
];

/// Figure 1: a control flow graph and its program structure tree.
#[test]
fn figure_one_has_the_regions_of_its_program_structure_tree() {
    let (cfg, blocks, edges) = build(COUNT, &FIGURE_ONE);
    let edge = |pair: (usize, usize)| {
        let index = FIGURE_ONE
            .iter()
            .position(|&known| known == pair)
            .expect("the figure has this edge");
        SeseEdge::Edge(edges[index])
    };
    let tree = ProgramStructureTree::compute(&cfg);
    let region_named = |name: char| {
        let &(_, entry, exit, _, _) = FIGURE_ONE_TREE
            .iter()
            .find(|region| region.0 == name)
            .expect("the figure names this region");
        region_between(&tree, edge(entry), edge(exit))
    };

    for &(name, _, _, parent, members) in &FIGURE_ONE_TREE {
        let region = region_named(name);
        assert_eq!(
            tree.parent(region),
            parent.map(region_named),
            "parent of {name}"
        );
        let actual: BTreeSet<BlockId> = tree.blocks(region).iter().copied().collect();
        let wanted: BTreeSet<BlockId> = members.iter().map(|&index| blocks[index]).collect();
        assert_eq!(actual, wanted, "blocks of {name}");
    }
    // The figure leaves out the start and end nodes, which the closed graph
    // makes regions of their own.
    let start = region_between(&tree, SeseEdge::Entry, edge((START, TOP)));
    let end = region_between(&tree, edge((BOTTOM, END)), SeseEdge::Exit(blocks[END]));
    assert_eq!(tree.blocks(start), [blocks[START]]);
    assert_eq!(tree.blocks(end), [blocks[END]]);
    assert_eq!(tree.len(), FIGURE_ONE_TREE.len() + 2);
    // The arm a back edge enters has two entries, so it is no region.
    assert_eq!(tree.innermost(blocks[LOOPED_ARM]), Some(region_named('d')));
    assert_matches_reference(&cfg);
}

/// The undirected spanning trees of Figure 3, as a chain closed through the
/// virtual exit, with their backedges directed up the chain: the virtual entry,
/// the seven tree edges and the virtual exit, then the backedges.
fn chain_with_backedges(
    backedges: &[(usize, usize)],
) -> (Vec<Boundary>, Vec<Boundary>, CycleEquivalence) {
    let mut edge_list: Vec<(usize, usize)> = (1..8).map(|next| (next - 1, next)).collect();
    edge_list.extend_from_slice(backedges);
    let (cfg, blocks, edges) = build(8, &edge_list);
    assert_matches_reference(&cfg);
    let mut boundaries: Vec<Boundary> = edges.iter().map(|&edge| SeseEdge::Edge(edge)).collect();
    let back = boundaries.split_off(7);
    let mut chain = vec![SeseEdge::Entry];
    chain.extend(boundaries);
    chain.push(SeseEdge::Exit(blocks[7]));
    (chain, back, CycleEquivalence::compute(&cfg))
}

/// Figure 3(a): properly nested brackets, `z` over the second to fourth tree
/// edges, `w` over the third and `y` over the sixth.
#[test]
fn figure_three_a_structured_loops() {
    let (chain, back, classes) = chain_with_backedges(&[(4, 1), (3, 2), (6, 5)]);
    assert_classes(
        &classes,
        &[
            &[chain[0], chain[1], chain[5], chain[7], chain[8]],
            &[chain[2], chain[4]],
            &[chain[3]],
            &[chain[6]],
            &[back[0]],
            &[back[1]],
            &[back[2]],
        ],
    );
}

/// Figure 3(b): crossing brackets, where tree edges `a` (the third) and `c`
/// (the fifth) share a class and `b` (the second) does not.
#[test]
fn figure_three_b_unstructured_loops() {
    let (chain, back, classes) = chain_with_backedges(&[(5, 1), (6, 2), (4, 3)]);
    assert_classes(
        &classes,
        &[
            &[chain[0], chain[1], chain[7], chain[8]],
            &[chain[2]],
            &[chain[3], chain[5]],
            &[chain[4]],
            &[chain[6]],
            &[back[0]],
            &[back[1]],
            &[back[2]],
        ],
    );
}
