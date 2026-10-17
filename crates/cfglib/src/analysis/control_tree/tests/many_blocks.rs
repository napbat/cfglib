//! Graphs with a hundred thousand blocks that the structurer has to sweep, jump
//! into, or claim for a region: the work has to stay linear in the number of
//! blocks, not quadratic in it.
//!
//! Each test checks that every instruction of the graph appears in the tree
//! exactly once, which is what a decompiler relies on, and that no label is
//! left without its block.

extern crate alloc;
use alloc::vec::Vec;

use super::small_stack::on_small_stack;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};
use crate::test_util::{DfInst, df_ff};
use crate::{ControlTree, LiftReport};

const BLOCKS: usize = 100_000;

fn new_block(cfg: &mut Cfg<DfInst>) -> BlockId {
    let block = cfg.new_block();
    cfg.block_mut(block).push(df_ff("work"));
    cfg.block_mut(block).push(df_ff("more"));
    block
}

/// Structure `cfg` on a small stack and check that every instruction is in the
/// tree once. Returns the report.
fn structure_and_cover(build: fn(usize) -> Cfg<DfInst>) -> LiftReport {
    on_small_stack(move || {
        let cfg = build(BLOCKS);
        let (tree, report) = ControlTree::compute_with_report(&cfg);
        let mut seen = 0;
        tree.for_each_instruction(&cfg, |_, _| seen += 1);
        let total: usize = cfg
            .block_ids()
            .map(|block| cfg.block(block).instructions().len())
            .sum();
        assert_eq!(seen, total, "every instruction once");
        assert_eq!(report.unresolved_labels, []);
        report
    })
}

/// A switch whose `count` arms each jump to one shared block that the switch
/// has no merge at, so every arm names a block with `count` predecessors.
fn jumps_into_one_block(count: usize) -> Cfg<DfInst> {
    let mut cfg = Cfg::new();
    let dispatch = cfg.entry();
    cfg.block_mut(dispatch).push(df_ff("switch"));
    let shared = new_block(&mut cfg);
    let other = new_block(&mut cfg);
    for _ in 0..count {
        let arm = new_block(&mut cfg);
        cfg.add_edge(dispatch, arm, EdgeKind::SwitchCase);
        cfg.add_edge(arm, shared, EdgeKind::Jump);
    }
    cfg.add_edge(dispatch, other, EdgeKind::Fallthrough);
    cfg
}

/// A switch whose arms `x(i)` jump to the blocks `u(i)` and `u(i + 1)`, so no
/// `u` has one predecessor: the walk leaves each of them for the sweep.
fn swept_blocks(count: usize) -> Cfg<DfInst> {
    let mut cfg = Cfg::new();
    let dispatch = cfg.entry();
    cfg.block_mut(dispatch).push(df_ff("switch"));
    let other = new_block(&mut cfg);
    let targets: Vec<BlockId> = (0..=count).map(|_| new_block(&mut cfg)).collect();
    for index in 0..count {
        let arm = new_block(&mut cfg);
        cfg.add_edge(dispatch, arm, EdgeKind::SwitchCase);
        cfg.add_edge(arm, targets[index], EdgeKind::Jump);
        cfg.add_edge(arm, targets[index + 1], EdgeKind::Jump);
    }
    cfg.add_edge(dispatch, other, EdgeKind::Fallthrough);
    cfg
}

/// A chain of `count` blocks, each protected by a region of its own whose
/// catch pad no call unwinds into. The pad `p(i)` jumps to the blocks `d(i)`
/// and `d(i + 1)`, which nothing else reaches: the sweep emits them for the
/// gotos, in the order of their identities.
fn dead_pads(count: usize) -> Cfg<DfInst> {
    let mut cfg = Cfg::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(df_ff("start"));
    let chain: Vec<BlockId> = (0..count).map(|_| new_block(&mut cfg)).collect();
    let pads: Vec<BlockId> = (0..count).map(|_| new_block(&mut cfg)).collect();
    let dead: Vec<BlockId> = (0..=count).map(|_| new_block(&mut cfg)).collect();
    cfg.add_edge(entry, chain[0], EdgeKind::Fallthrough);
    for index in 0..count {
        if let Some(&next) = chain.get(index + 1) {
            cfg.add_edge(chain[index], next, EdgeKind::Fallthrough);
        }
        cfg.add_edge(pads[index], dead[index], EdgeKind::Jump);
        cfg.add_edge(pads[index], dead[index + 1], EdgeKind::Jump);
        cfg.add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks: [chain[index]].into_iter().collect(),
            handlers: alloc::vec![Handler {
                entry: pads[index],
                body: HandlerBody::known([pads[index]]),
                kind: HandlerKind::Catch,
            }],
            parent: None,
        });
    }
    cfg
}

/// `count` endless loops nested in each other, each with its header protected
/// by a region of its own (one block, one catch pad that jumps to the way out
/// of the function): the regions are as many as the loops, and each loop sits
/// in the body of a `try` that holds one block of it.
fn endless_loops_each_in_a_try(count: usize) -> Cfg<DfInst> {
    let mut cfg = Cfg::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(df_ff("start"));
    let headers: Vec<BlockId> = (0..count).map(|_| new_block(&mut cfg)).collect();
    let tests: Vec<BlockId> = (0..count).map(|_| new_block(&mut cfg)).collect();
    let latches: Vec<BlockId> = (0..count).map(|_| new_block(&mut cfg)).collect();
    let out = new_block(&mut cfg);
    cfg.add_edge(entry, headers[0], EdgeKind::Fallthrough);
    for index in 0..count {
        if let Some(&inner) = headers.get(index + 1) {
            cfg.add_edge(headers[index], inner, EdgeKind::Fallthrough);
        }
        let leave = if index == 0 { out } else { tests[index - 1] };
        cfg.add_edge(tests[index], leave, EdgeKind::ConditionalTrue);
        cfg.add_edge(tests[index], latches[index], EdgeKind::ConditionalFalse);
        cfg.add_edge(latches[index], headers[index], EdgeKind::Jump);
    }
    cfg.add_edge(headers[count - 1], tests[count - 1], EdgeKind::Fallthrough);
    for &header in &headers {
        let pad = new_block(&mut cfg);
        cfg.add_edge(header, pad, EdgeKind::ExceptionHandler);
        cfg.add_edge(pad, out, EdgeKind::Jump);
        cfg.add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks: [header].into_iter().collect(),
            handlers: alloc::vec![Handler {
                entry: pad,
                body: HandlerBody::known([pad]),
                kind: HandlerKind::Catch,
            }],
            parent: None,
        });
    }
    cfg
}

#[test]
fn a_hundred_thousand_jumps_into_one_block_structure_in_linear_time() {
    let report = structure_and_cover(jumps_into_one_block);
    assert!(
        report.gotos.len() > BLOCKS / 2,
        "the arms jump by goto: {}",
        report.gotos.len()
    );
}

#[test]
fn a_hundred_thousand_swept_blocks_are_emitted_in_linear_time() {
    let report = structure_and_cover(swept_blocks);
    assert!(
        report.swept_blocks.len() > BLOCKS / 2,
        "the refused blocks are swept: {}",
        report.swept_blocks.len()
    );
}

#[test]
fn a_hundred_thousand_dead_pads_and_their_gotos_are_emitted_in_linear_time() {
    let report = structure_and_cover(dead_pads);
    assert_eq!(report.unstructured_regions, []);
    assert!(
        report.swept_blocks.len() > BLOCKS / 2,
        "the dead blocks are swept: {}",
        report.swept_blocks.len()
    );
}

#[test]
fn a_hundred_thousand_nested_endless_loops_each_in_a_try_are_followed_in_linear_time() {
    let report = structure_and_cover(endless_loops_each_in_a_try);
    assert_eq!(report.unstructured_regions, []);
}
