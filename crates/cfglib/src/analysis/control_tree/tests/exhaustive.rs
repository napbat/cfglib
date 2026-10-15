//! Every small control-flow graph against the invariants of a control tree.
//!
//! The graphs are every combination of terminators, one per block, over up to
//! four blocks. The structure each yields is free to differ, but the tree
//! must cover the instructions of every reachable block exactly once, name
//! a goto only where the report does, and be the same tree whether a block
//! is emptied by the caller or really empty.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::test_util::{DfInst, df_ff, df_pred};
use crate::{ControlNode, ControlTree, LiftReport};

/// How one block ends.
#[derive(Clone, Copy, Debug)]
enum Terminator {
    Exit,
    One(usize, EdgeKind),
    Branch(usize, usize),
    Switch { cases: u32, default: Option<usize> },
}

impl Terminator {
    fn edges(self, count: usize) -> Vec<(usize, EdgeKind)> {
        match self {
            Self::Exit => Vec::new(),
            Self::One(target, kind) => vec![(target, kind)],
            Self::Branch(on_true, on_false) => vec![
                (on_true, EdgeKind::ConditionalTrue),
                (on_false, EdgeKind::ConditionalFalse),
            ],
            Self::Switch { cases, default } => (0..count)
                .filter(|target| cases & (1 << target) != 0)
                .map(|target| (target, EdgeKind::SwitchCase))
                .chain(default.map(|target| (target, EdgeKind::Fallthrough)))
                .collect(),
        }
    }
}

/// Every terminator over `count` blocks, switches included on request.
fn terminators(count: usize, kinds: &[EdgeKind], switches: bool) -> Vec<Terminator> {
    let mut all = vec![Terminator::Exit];
    for target in 0..count {
        all.extend(kinds.iter().map(|&kind| Terminator::One(target, kind)));
    }
    for on_true in 0..count {
        for on_false in 0..count {
            all.push(Terminator::Branch(on_true, on_false));
        }
    }
    if switches {
        for cases in 1..(1_u32 << count) {
            all.push(Terminator::Switch {
                cases,
                default: None,
            });
            all.extend((0..count).map(|target| Terminator::Switch {
                cases,
                default: Some(target),
            }));
        }
    }
    all
}

/// The graph of `terminators`, block `b` holding `instructions[b]` instructions
/// under `predicate`.
fn graph(
    terminators: &[Terminator],
    instructions: &[usize],
    predicate: fn(usize, usize) -> Option<(u16, bool)>,
) -> Cfg<DfInst> {
    let count = terminators.len();
    let mut cfg: Cfg<DfInst> = Cfg::new();
    let mut blocks = vec![cfg.entry()];
    blocks.extend((1..count).map(|_| cfg.new_block()));
    for (block, &length) in instructions.iter().enumerate() {
        for index in 0..length {
            let inst = match predicate(block, index) {
                Some((variable, when_true)) => df_pred("p", variable, when_true),
                None => df_ff("i"),
            };
            cfg.block_mut(blocks[block]).push(inst);
        }
    }
    for (block, terminator) in terminators.iter().enumerate() {
        for (target, kind) in terminator.edges(count) {
            cfg.add_edge(blocks[block], blocks[target], kind);
        }
    }
    cfg
}

fn unpredicated(_: usize, _: usize) -> Option<(u16, bool)> {
    None
}

/// Alternating polarity on the first two instructions of odd blocks.
fn alternating(block: usize, index: usize) -> Option<(u16, bool)> {
    (block % 2 == 1).then_some((1, index == 0))
}

fn reachable(cfg: &Cfg<DfInst>) -> Vec<BlockId> {
    cfg.reverse_postorder()
}

/// Check the invariants of one graph's tree and report.
fn check(cfg: &Cfg<DfInst>, tree: &ControlTree, report: &LiftReport, context: &str) {
    // Every node is reachable from the root exactly once.
    assert_eq!(tree.preorder().count(), tree.node_count(), "{context}");

    // The instructions of every reachable block appear exactly once, and no
    // other instruction appears.
    let mut seen = Vec::new();
    tree.for_each_instruction(cfg, |point, _| seen.push((point.block, point.inst_idx)));
    seen.sort_unstable();
    let mut expected: Vec<(BlockId, usize)> = reachable(cfg)
        .into_iter()
        .flat_map(|block| {
            (0..cfg.block(block).instructions().len()).map(move |index| (block, index))
        })
        .collect();
    expected.sort_unstable();
    assert_eq!(seen, expected, "coverage of {context}");

    // Gotos are exactly the reported ones.
    let mut gotos: Vec<BlockId> = tree
        .preorder()
        .filter_map(|id| match *tree.node(id) {
            ControlNode::Goto { target } => Some(target),
            _ => None,
        })
        .collect();
    gotos.sort_unstable();
    let mut reported: Vec<BlockId> = report.gotos.iter().map(|goto| goto.target).collect();
    reported.sort_unstable();
    assert_eq!(gotos, reported, "gotos of {context}");

    // A goto target has a label, or is reported as dangling.
    let labels: Vec<BlockId> = tree
        .preorder()
        .filter_map(|id| match *tree.node(id) {
            ControlNode::Label { block, .. } => Some(block),
            _ => None,
        })
        .collect();
    for target in &reported {
        assert!(
            labels.contains(target) != report.unresolved_labels.contains(target),
            "label of goto {target} in {context}"
        );
    }
    for block in &report.swept_blocks {
        assert!(labels.contains(block), "swept {block} in {context}");
    }
    assert_eq!(
        report.is_fully_structured(),
        report.gotos.is_empty()
            && report.unstructured_regions.is_empty()
            && report.unresolved_labels.is_empty()
    );
}

/// Call `visit` with every assignment of a terminator to each of `count`
/// blocks.
fn for_each_graph(count: usize, terminators: &[Terminator], mut visit: impl FnMut(&[Terminator])) {
    let mut choice = vec![0_usize; count];
    loop {
        let picked: Vec<Terminator> = choice.iter().map(|&i| terminators[i]).collect();
        visit(&picked);
        let mut digit = 0;
        while digit < count {
            choice[digit] += 1;
            if choice[digit] < terminators.len() {
                break;
            }
            choice[digit] = 0;
            digit += 1;
        }
        if digit == count {
            return;
        }
    }
}

const ONE_SUCCESSOR: [EdgeKind; 4] = [
    EdgeKind::Fallthrough,
    EdgeKind::Unconditional,
    EdgeKind::Jump,
    EdgeKind::Back,
];

#[test]
fn every_graph_of_up_to_three_blocks_is_covered_exactly_once() {
    for count in 1..=3 {
        let all = terminators(count, &ONE_SUCCESSOR, true);
        for_each_graph(count, &all, |picked| {
            let cfg = graph(picked, &vec![1; count], unpredicated);
            let (tree, report) = ControlTree::compute_with_report(&cfg);
            check(&cfg, &tree, &report, &alloc::format!("{picked:?}"));
        });
    }
}

#[test]
fn every_graph_of_four_blocks_is_covered_exactly_once() {
    let all = terminators(4, &[EdgeKind::Fallthrough, EdgeKind::Jump], false);
    for_each_graph(4, &all, |picked| {
        let cfg = graph(picked, &[1; 4], unpredicated);
        let (tree, report) = ControlTree::compute_with_report(&cfg);
        check(&cfg, &tree, &report, &alloc::format!("{picked:?}"));
    });
}

/// Call `visit` with every instruction count in `0..=most` for each of `count`
/// blocks.
fn for_each_length(count: usize, most: usize, mut visit: impl FnMut(&[usize])) {
    let mut lengths = vec![0_usize; count];
    loop {
        visit(&lengths);
        let mut digit = 0;
        while digit < count && lengths[digit] == most {
            lengths[digit] = 0;
            digit += 1;
        }
        if digit == count {
            return;
        }
        lengths[digit] += 1;
    }
}

#[test]
fn blocks_without_instructions_are_covered_too() {
    // Two blocks take every terminator and zero to two instructions; three
    // blocks take jumps, branches, and zero or one instruction.
    for (count, most, kinds, switches) in [
        (1, 2, &ONE_SUCCESSOR[..], true),
        (2, 2, &ONE_SUCCESSOR[..], true),
        (3, 1, &[EdgeKind::Jump][..], false),
    ] {
        let all = terminators(count, kinds, switches);
        for_each_length(count, most, |lengths| {
            for_each_graph(count, &all, |picked| {
                let cfg = graph(picked, lengths, unpredicated);
                let (tree, report) = ControlTree::compute_with_report(&cfg);
                check(
                    &cfg,
                    &tree,
                    &report,
                    &alloc::format!("{picked:?} with {lengths:?} instructions"),
                );
            });
        });
    }
}

#[test]
fn predicated_trees_cover_every_instruction_exactly_once() {
    for (count, switches) in [(1, true), (2, true), (3, false)] {
        let all = terminators(count, &[EdgeKind::Fallthrough, EdgeKind::Jump], switches);
        for_each_graph(count, &all, |picked| {
            let cfg = graph(picked, &vec![2; count], alternating);
            let (tree, report) = ControlTree::compute_predicated_with_report(&cfg);
            check(&cfg, &tree, &report, &alloc::format!("{picked:?}"));
            // Regionizing changes no control structure, so the report is the
            // plain structure's report.
            assert_eq!(
                report,
                ControlTree::compute_with_report(&cfg).1,
                "{picked:?}"
            );
            assert_eq!(tree, ControlTree::compute_predicated(&cfg), "{picked:?}");
        });
    }
}

#[test]
fn treating_blocks_as_empty_is_the_same_as_clearing_them() {
    for (count, switches) in [(1, true), (2, true), (3, false)] {
        let all = terminators(count, &[EdgeKind::Fallthrough, EdgeKind::Jump], switches);
        for_each_graph(count, &all, |picked| {
            for emptied in 0..(1_u32 << count) {
                let full = graph(picked, &vec![1; count], unpredicated);
                let lengths: Vec<usize> = (0..count)
                    .map(|block| usize::from(emptied & (1 << block) == 0))
                    .collect();
                let cleared = graph(picked, &lengths, unpredicated);
                let blocks: Vec<BlockId> = (0..count)
                    .filter(|block| emptied & (1 << block) != 0)
                    .map(BlockId::from_index)
                    .collect();
                assert_eq!(
                    ControlTree::compute_with_report_treating_empty(&full, &blocks),
                    ControlTree::compute_with_report(&cleared),
                    "{picked:?} with {blocks:?} emptied"
                );
            }
        });
    }
}
