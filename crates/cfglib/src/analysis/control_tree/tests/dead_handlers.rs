//! Handlers no exception edge leads to, inside and outside loops.
//!
//! A catch pad is dead when no call unwinds into it, yet its body still has
//! to appear in the tree exactly once, with every goto label resolved.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use super::exhaustive::{Terminator, for_each_graph, graph, terminators, unpredicated};
use super::invariants::{check, describe, isolate_handlers, owned_blocks};
use super::pseudocode;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};
use crate::test_util::rng::Rng;
use crate::test_util::{DfInst, MockInst, ff};
use crate::{ControlTree, GotoReason};
use EdgeKind::{Back, ConditionalFalse, ConditionalTrue, Fallthrough};

const NAMES: [&str; 9] = ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"];

/// The seven blocks of a loop whose body holds a try, with a catch pad that
/// nothing unwinds into:
///
/// ```text
/// b0 -> b1;  b1 -> b2 (true), b6 (false);  b2 -> b3;  b3 -> b1 (back)
/// b4 -> b5 (true), b3 (false);  b5 -> b3
/// try over {b2}; catch pad b4 with extent {b4, b5}
/// ```
///
/// `looped` ends `b3` in the back edge; otherwise `b3` falls to the exit.
fn dead_pad(looped: bool) -> Cfg<MockInst> {
    pad_graph(looped, &[4, 5], false)
}

/// [`dead_pad`] with the catch extent `extent`, and a block `b7` with no
/// predecessor that enters the pad's tail `b5` when `orphan` is set.
fn pad_graph(looped: bool, extent: &[usize], orphan: bool) -> Cfg<MockInst> {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let b: Vec<BlockId> = core::iter::once(cfg.entry())
        .chain((1..if orphan { 8 } else { 7 }).map(|_| cfg.new_block()))
        .collect();
    for (index, &block) in b.iter().enumerate() {
        cfg.block_mut(block).push(ff(NAMES[index]));
    }
    cfg.add_edge(b[0], b[1], Fallthrough);
    cfg.add_edge(b[1], b[2], ConditionalTrue);
    cfg.add_edge(b[1], b[6], ConditionalFalse);
    cfg.add_edge(b[2], b[3], Fallthrough);
    if looped {
        cfg.add_edge(b[3], b[1], Back);
    } else {
        cfg.add_edge(b[3], b[6], Fallthrough);
    }
    cfg.add_edge(b[4], b[5], ConditionalTrue);
    cfg.add_edge(b[4], b[3], ConditionalFalse);
    cfg.add_edge(b[5], b[3], Fallthrough);
    if orphan {
        cfg.add_edge(b[7], b[5], Fallthrough);
    }
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [b[2]].into_iter().collect(),
        handlers: vec![Handler {
            entry: b[4],
            body: HandlerBody::known(extent.iter().map(|&block| b[block])),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });
    cfg
}

#[test]
fn a_dead_catch_pad_inside_a_loop_keeps_its_body() {
    let cfg = dead_pad(true);
    let (tree, report) = ControlTree::compute_with_report(&cfg);
    check(&cfg, &tree, &report, "dead pad in a loop");
    assert_eq!(
        pseudocode(&cfg, &tree),
        "b0\nwhile {\n    b1\n    break if !cond;\n    try {\n        b2\n    } catch {\n        \
         if (b4) {\n            b5\n            goto .bb3;\n        } else {\n            goto .bb3;\n        \
         }\n    }\n    .bb3:\n        b3\n}\nb6\n"
    );
    assert!(report.unresolved_labels.is_empty(), "{report:?}");
    assert!(report.swept_blocks.is_empty(), "{report:?}");
    assert!(report.unstructured_regions.is_empty(), "{report:?}");
    // The pad's tail reaches the loop's latch from outside the handler extent.
    assert!(
        report
            .gotos
            .iter()
            .all(|goto| goto.reason == GotoReason::BoundaryEscape)
    );
}

#[test]
fn a_dead_catch_pad_is_structured_the_same_in_and_out_of_a_loop() {
    // The handler's own text is the same with and without the back edge.
    let handler_text = |looped: bool| {
        let cfg = dead_pad(looped);
        let tree = ControlTree::compute(&cfg);
        let text = pseudocode(&cfg, &tree);
        let start = text.find("catch {").expect("a catch handler");
        let end = text.find(".bb3:").expect("the latch label");
        alloc::string::String::from(text[start..end].trim_start())
    };
    assert_eq!(handler_text(true), handler_text(false));
}

#[test]
fn a_live_catch_pad_inside_a_loop_matches_the_dead_one() {
    let mut cfg = dead_pad(true);
    let (pad, try_block) = (
        cfg.block_ids().nth(4).unwrap(),
        cfg.block_ids().nth(2).unwrap(),
    );
    cfg.add_edge(try_block, pad, EdgeKind::ExceptionHandler);
    let (live, report) = ControlTree::compute_with_report(&cfg);
    check(&cfg, &live, &report, "live pad in a loop");
    let dead_cfg = dead_pad(true);
    let dead = ControlTree::compute(&dead_cfg);
    assert_eq!(pseudocode(&cfg, &live), pseudocode(&dead_cfg, &dead));
}

#[test]
fn a_dead_pad_tail_outside_the_extent_is_swept_under_its_label() {
    // b4 -> b5 -> b3, but the extent is {b4}; b7 also enters b5, so the pad
    // cannot inline it and the transfer is a goto. Nothing but the pad reaches
    // b5, so only the sweep can emit it.
    let cfg = pad_graph(true, &[4], true);
    let b: Vec<BlockId> = cfg.block_ids().collect();
    let (tree, report) = ControlTree::compute_with_report(&cfg);
    check(&cfg, &tree, &report, "dead pad tail");
    assert_eq!(report.swept_blocks, [b[5]], "{report:?}");
    assert!(report.unresolved_labels.is_empty(), "{report:?}");
    let text = pseudocode(&cfg, &tree);
    assert_eq!(text.matches("b5\n").count(), 1, "{text}");
}

#[test]
fn a_block_only_a_known_extent_claims_is_emitted() {
    // b5 has no edge at all, but the catch extent names it.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let b: Vec<BlockId> = core::iter::once(cfg.entry())
        .chain((1..6).map(|_| cfg.new_block()))
        .collect();
    for (index, &block) in b.iter().enumerate() {
        cfg.block_mut(block).push(ff(NAMES[index]));
    }
    cfg.add_edge(b[0], b[1], Fallthrough);
    cfg.add_edge(b[1], b[2], Fallthrough);
    cfg.add_edge(b[3], b[2], Fallthrough);
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [b[1]].into_iter().collect(),
        handlers: vec![Handler {
            entry: b[3],
            body: HandlerBody::known([b[3], b[5]]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });
    let (tree, report) = ControlTree::compute_with_report(&cfg);
    check(&cfg, &tree, &report, "orphan in a known extent");
    assert_eq!(owned_blocks(&cfg), [b[0], b[1], b[2], b[3], b[5]]);
    // The extent is the handler's own code, so the block lands inside it.
    assert!(report.swept_blocks.is_empty(), "{report:?}");
    assert_eq!(
        pseudocode(&cfg, &tree),
        "b0\ntry {\n    b1\n} catch {\n    b3\n    b5\n}\nb2\n"
    );
}

#[test]
fn one_dead_pad_named_as_entry_and_filter_is_one_root() {
    // The pad b3 is the entry of one handler and the filter funclet of
    // another, and it sits inside a loop.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let b: Vec<BlockId> = core::iter::once(cfg.entry())
        .chain((1..6).map(|_| cfg.new_block()))
        .collect();
    for (index, &block) in b.iter().enumerate() {
        cfg.block_mut(block).push(ff(NAMES[index]));
    }
    cfg.add_edge(b[0], b[1], Fallthrough);
    cfg.add_edge(b[1], b[2], ConditionalTrue);
    cfg.add_edge(b[1], b[5], ConditionalFalse);
    cfg.add_edge(b[2], b[1], Back);
    cfg.add_edge(b[3], b[4], ConditionalTrue);
    cfg.add_edge(b[3], b[2], ConditionalFalse);
    cfg.add_edge(b[4], b[2], Fallthrough);
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [b[2]].into_iter().collect(),
        handlers: vec![
            Handler {
                entry: b[3],
                body: HandlerBody::known([b[3], b[4]]),
                kind: HandlerKind::Catch,
            },
            Handler {
                entry: b[4],
                body: HandlerBody::known([b[4]]),
                kind: HandlerKind::Filter { filter_block: b[3] },
            },
        ],
        parent: None,
    });
    let (tree, report) = ControlTree::compute_with_report(&cfg);
    check(&cfg, &tree, &report, "pad named twice");
}

/// A real function's graph, minimized from a Rust-to-IL lowering: five
/// regions, three levels of loops, and catch and finally pads that nothing
/// unwinds into. Edge lines are `GE source target kind`; region lines are
/// `GR id parent protected handler:kind:extent;...`.
const REAL_FUNCTION: &str = "\
GRAPH blocks=49 entry=0\n\
GE 0 1 Fallthrough\n\
GE 1 3 ConditionalTrue\n\
GE 1 4 ConditionalFalse\n\
GE 3 4 Fallthrough\n\
GE 4 5 ConditionalTrue\n\
GE 4 6 ConditionalFalse\n\
GE 5 7 Fallthrough\n\
GE 6 7 Fallthrough\n\
GE 7 8 ConditionalTrue\n\
GE 7 9 ConditionalFalse\n\
GE 8 10 ConditionalTrue\n\
GE 8 11 ConditionalFalse\n\
GE 9 12 Fallthrough\n\
GE 10 11 Fallthrough\n\
GE 11 9 Fallthrough\n\
GE 12 13 ConditionalTrue\n\
GE 12 14 ConditionalFalse\n\
GE 13 15 ConditionalTrue\n\
GE 13 16 ConditionalFalse\n\
GE 14 23 ConditionalTrue\n\
GE 14 24 ConditionalFalse\n\
GE 15 16 Fallthrough\n\
GE 16 19 Fallthrough\n\
GE 17 21 ConditionalTrue\n\
GE 17 22 ConditionalFalse\n\
GE 18 12 Back\n\
GE 19 2 Unconditional\n\
GE 19 20 Fallthrough\n\
GE 20 18 ExceptionLeave\n\
GE 21 22 Fallthrough\n\
GE 22 18 Fallthrough\n\
GE 23 27 Fallthrough\n\
GE 24 28 ConditionalTrue\n\
GE 24 29 ConditionalFalse\n\
GE 25 2 ExceptionUnwind\n\
GE 26 24 Fallthrough\n\
GE 27 25 ExceptionUnwind\n\
GE 28 30 ConditionalTrue\n\
GE 28 31 ConditionalFalse\n\
GE 29 34 Fallthrough\n\
GE 30 31 Fallthrough\n\
GE 31 29 Fallthrough\n\
GE 32 2 ExceptionUnwind\n\
GE 32 33 ExceptionLeave\n\
GE 33 40 ConditionalTrue\n\
GE 33 41 ConditionalFalse\n\
GE 34 35 ConditionalTrue\n\
GE 34 36 ConditionalFalse\n\
GE 35 39 Fallthrough\n\
GE 36 32 ExceptionLeave\n\
GE 37 32 ExceptionUnwind\n\
GE 38 36 Fallthrough\n\
GE 39 37 ExceptionUnwind\n\
GE 40 41 Fallthrough\n\
GE 41 42 ConditionalTrue\n\
GE 41 43 ConditionalFalse\n\
GE 42 44 ConditionalTrue\n\
GE 42 45 ConditionalFalse\n\
GE 43 2 Fallthrough\n\
GE 44 45 Fallthrough\n\
GE 45 48 Fallthrough\n\
GE 46 2 ExceptionUnwind\n\
GE 47 43 Fallthrough\n\
GE 48 46 ExceptionUnwind\n\
GR 0 - 19,20 17:Catch:17,21,22\n\
GR 1 - 27 25:Finally:25\n\
GR 2 - 34,35,36,37,38,39 32:Finally:32\n\
GR 3 2 39 37:Finally:37\n\
GR 4 - 48 46:Finally:46";

fn edge_kind(name: &str) -> EdgeKind {
    match name {
        "Fallthrough" => Fallthrough,
        "ConditionalTrue" => ConditionalTrue,
        "ConditionalFalse" => ConditionalFalse,
        "Unconditional" => EdgeKind::Unconditional,
        "Back" => Back,
        "ExceptionLeave" => EdgeKind::ExceptionLeave,
        "ExceptionUnwind" => EdgeKind::ExceptionUnwind,
        "ExceptionHandler" => EdgeKind::ExceptionHandler,
        other => panic!("unknown edge kind {other}"),
    }
}

fn numbers(text: &str, ids: &[BlockId]) -> Vec<BlockId> {
    text.split(',')
        .filter(|part| !part.is_empty())
        .map(|part| ids[part.parse::<usize>().unwrap()])
        .collect()
}

/// Build a graph from the line format of [`REAL_FUNCTION`], one instruction
/// per block named by its index.
fn from_text(text: &str) -> Cfg<MockInst> {
    let mut lines = text.lines();
    let header = lines.next().expect("a header line");
    let count: usize = header
        .split_whitespace()
        .find_map(|field| field.strip_prefix("blocks="))
        .expect("a block count")
        .parse()
        .unwrap();
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let ids: Vec<BlockId> = core::iter::once(cfg.entry())
        .chain((1..count).map(|_| cfg.new_block()))
        .collect();
    let names: Vec<&'static str> = (0..count)
        .map(|index| &*alloc::string::String::leak(alloc::format!("b{index}")))
        .collect();
    for (&block, &name) in ids.iter().zip(&names) {
        cfg.block_mut(block).push(ff(name));
    }
    for line in lines {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields[0] {
            "GE" => {
                let source: usize = fields[1].parse().unwrap();
                let target: usize = fields[2].parse().unwrap();
                cfg.add_edge(ids[source], ids[target], edge_kind(fields[3]));
            }
            "GR" => {
                let parent =
                    (fields[2] != "-").then(|| RegionId::from_raw(fields[2].parse().unwrap()));
                let handlers = fields[4]
                    .split(';')
                    .map(|handler| {
                        let parts: Vec<&str> = handler.split(':').collect();
                        Handler {
                            entry: ids[parts[0].parse::<usize>().unwrap()],
                            kind: match parts[1] {
                                "Catch" => HandlerKind::Catch,
                                "Finally" => HandlerKind::Finally,
                                other => panic!("unknown handler kind {other}"),
                            },
                            body: HandlerBody::known(numbers(parts[2], &ids)),
                        }
                    })
                    .collect();
                cfg.add_region(Region {
                    id: RegionId::from_raw(fields[1].parse().unwrap()),
                    protected_blocks: numbers(fields[3], &ids).into_iter().collect(),
                    handlers,
                    parent,
                });
            }
            other => panic!("unknown line {other}"),
        }
    }
    cfg
}

#[test]
fn a_real_function_with_dead_pads_in_loops_loses_no_block() {
    let cfg = from_text(REAL_FUNCTION);
    let (tree, report) = ControlTree::compute_with_report(&cfg);
    check(&cfg, &tree, &report, "real function");
    assert!(report.unresolved_labels.is_empty(), "{report:?}");
    assert!(report.unstructured_regions.is_empty(), "{report:?}");
    // The catch pad's blocks are in the tree, inside its handler. The pad's
    // entry b17 is the handler's branch, which the header prints.
    let text = pseudocode(&cfg, &tree);
    for pad_block in [21, 22] {
        assert_eq!(
            text.matches(&alloc::format!("b{pad_block}\n")).count(),
            1,
            "{text}"
        );
    }
}

/// Every instruction of the owned blocks, with a region that guards `p`, a
/// handler at `h` whose extent is `h` and `extent`, and an exception edge to
/// the pad when `live`.
fn with_region(
    cfg: &mut Cfg<DfInst>,
    protected: usize,
    handler: usize,
    extent: &[usize],
    live: bool,
) {
    let ids: Vec<BlockId> = cfg.block_ids().collect();
    if live {
        cfg.add_edge(ids[protected], ids[handler], EdgeKind::ExceptionHandler);
    }
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [ids[protected]].into_iter().collect(),
        handlers: vec![Handler {
            entry: ids[handler],
            body: HandlerBody::known(
                core::iter::once(handler)
                    .chain(extent.iter().copied())
                    .map(|block| ids[block]),
            ),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });
}

#[test]
fn every_small_graph_with_a_handler_keeps_every_owned_block() {
    // Three blocks, every terminator, every region: any guarded block, any
    // pad, any extent, with and without an exception edge into the pad, and
    // with any one block empty. Gotos to an empty block dangle by design;
    // `check` holds every other label to resolve. A block the caller treats
    // as empty gives the tree of a graph where it really is.
    let count = 3;
    let all = terminators(count, &[Fallthrough, Back], false);
    for_each_graph(count, &all, |picked: &[Terminator]| {
        for protected in 0..count {
            // A pad and its extent lie outside the blocks the region guards.
            for handler in (0..count).filter(|&block| block != protected) {
                let others: Vec<usize> = (0..count)
                    .filter(|&block| block != handler && block != protected)
                    .collect();
                for subset in 0..(1_u32 << others.len()) {
                    let extent: Vec<usize> = others
                        .iter()
                        .enumerate()
                        .filter(|&(bit, _)| subset & (1 << bit) != 0)
                        .map(|(_, &block)| block)
                        .collect();
                    for live in [false, true] {
                        let build = |lengths: &[usize]| {
                            let mut cfg = graph(picked, lengths, unpredicated);
                            with_region(&mut cfg, protected, handler, &extent, live);
                            isolate_handlers(&mut cfg);
                            cfg
                        };
                        for emptied in [None, Some(0), Some(1), Some(2)] {
                            let mut lengths = vec![1; count];
                            if let Some(block) = emptied {
                                lengths[block] = 0;
                            }
                            let cfg = build(&lengths);
                            let context = alloc::format!(
                                "{picked:?} guarding {protected}, pad {handler} extent \
                                 {extent:?} live {live} emptied {emptied:?}"
                            );
                            let (tree, report) = ControlTree::compute_with_report(&cfg);
                            check(&cfg, &tree, &report, &context);
                            if let Some(block) = emptied {
                                let full = build(&vec![1; count]);
                                let hole = full.block_ids().nth(block).unwrap();
                                assert_eq!(
                                    ControlTree::compute_with_report_treating_empty(&full, &[hole]),
                                    (tree, report),
                                    "{context}"
                                );
                            }
                        }
                    }
                }
            }
        }
    });
}

/// A graph of five to nine blocks with one or two regions whose pads are
/// often dead, so pads and their extents land inside loops, branches, and
/// each other. Only a pad's own extent and exception edges enter it. A block
/// is empty with `empty_percent` chance.
fn random_graph_with_regions(random: &mut Rng, empty_percent: usize) -> Cfg<DfInst> {
    let count = 5 + random.below(5);
    let mut cfg: Cfg<DfInst> = Cfg::new();
    let ids: Vec<BlockId> = core::iter::once(cfg.entry())
        .chain((1..count).map(|_| cfg.new_block()))
        .collect();
    for &block in &ids {
        if empty_percent == 0 || !random.chance(empty_percent) {
            cfg.block_mut(block).push(crate::test_util::df_ff("i"));
        }
    }
    for &block in &ids {
        match random.below(20) {
            0..=1 => {}
            2..=9 => {
                let kind = [Fallthrough, EdgeKind::Jump, Back][random.below(3)];
                cfg.add_edge(block, ids[random.below(count)], kind);
            }
            10..=18 => {
                cfg.add_edge(block, ids[random.below(count)], ConditionalTrue);
                cfg.add_edge(block, ids[random.below(count)], ConditionalFalse);
            }
            _ => {
                for _ in 0..=random.below(3) {
                    cfg.add_edge(block, ids[random.below(count)], EdgeKind::SwitchCase);
                }
            }
        }
    }
    for _ in 0..=random.below(2) {
        let protected: BTreeSet<BlockId> = (0..=random.below(2))
            .map(|_| ids[random.below(count)])
            .collect();
        // A pad, its extent, and its filter lie outside the blocks the region
        // guards.
        let outside: Vec<BlockId> = ids
            .iter()
            .copied()
            .filter(|block| !protected.contains(block))
            .collect();
        if outside.is_empty() {
            continue;
        }
        let pick = |random: &mut Rng| outside[random.below(outside.len())];
        let handlers = (0..=random.below(2))
            .map(|_| {
                let entry = pick(random);
                if random.chance(60) {
                    for &block in &protected {
                        cfg.add_edge(block, entry, EdgeKind::ExceptionHandler);
                    }
                }
                let kind = match random.below(4) {
                    0 => HandlerKind::Catch,
                    1 => HandlerKind::CatchAll,
                    2 => HandlerKind::Finally,
                    _ => HandlerKind::Filter {
                        filter_block: pick(random),
                    },
                };
                let mut extent: BTreeSet<BlockId> = [entry].into_iter().collect();
                for _ in 0..random.below(4) {
                    extent.insert(pick(random));
                }
                let body = if random.chance(80) {
                    HandlerBody::known(extent)
                } else {
                    HandlerBody::unknown()
                };
                Handler { entry, body, kind }
            })
            .collect();
        let parent =
            (!cfg.regions().is_empty() && random.chance(30)).then(|| RegionId::from_raw(0));
        cfg.add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks: protected,
            handlers,
            parent,
        });
    }
    isolate_handlers(&mut cfg);
    cfg
}

#[test]
fn random_graphs_with_handlers_keep_every_owned_block() {
    let mut random = Rng::new(0x9E37_79B9_7F4A_7C15);
    for sample in 0..40_000 {
        let cfg = random_graph_with_regions(&mut random, 0);
        let (tree, report) = ControlTree::compute_with_report(&cfg);
        check(
            &cfg,
            &tree,
            &report,
            &alloc::format!("sample {sample}: {}", describe(&cfg)),
        );
    }
}

#[test]
fn random_graphs_with_handlers_and_empty_blocks_keep_every_label() {
    // Gotos to an empty block dangle by design, and the report lists them;
    // every other label resolves. Treating the empty blocks as empty is the
    // same as clearing them.
    let mut random = Rng::new(0xD1B5_4A32_D192_ED03);
    for sample in 0..40_000 {
        let mut cfg = random_graph_with_regions(&mut random, 30);
        let context = alloc::format!("sample {sample}: {}", describe(&cfg));
        let (tree, report) = ControlTree::compute_with_report(&cfg);
        check(&cfg, &tree, &report, &context);
        let emptied: Vec<BlockId> = cfg
            .block_ids()
            .filter(|&block| cfg.block(block).instructions().is_empty())
            .collect();
        for &block in &emptied {
            cfg.block_mut(block).push(crate::test_util::df_ff("i"));
        }
        assert_eq!(
            ControlTree::compute_with_report_treating_empty(&cfg, &emptied),
            (tree, report),
            "{context}"
        );
    }
}
