//! The structuring variant that treats blocks as if they were empty, and the
//! spans a tree hands back.

use std::collections::BTreeSet;

use cfglib::{BlockId, Cfg, ControlNode, ControlTree, ProgramPoint};

use crate::builders::{Flow, pseudocode};
use crate::coverage::{assert_spans_tile_each_block, reachable_blocks};
use crate::implied_flow::assert_flow_matches_graph_with;
use crate::reading::top_level;
use crate::tree_shape::assert_well_formed;

/// A loop whose `break` and `continue` each compile to a block holding one
/// `jmp`: the trampolines a source-level lifter wants structured away.
struct Trampolines {
    flow: Flow,
    break_hop: BlockId,
    continue_hop: BlockId,
}

fn loop_with_trampolines() -> Trampolines {
    // while i < n { if done(i) { break }; if skip(i) { continue }; work(i); i += 1 }
    let mut flow = Flow::starting_with("i = 0");
    let header = flow.stmt("i < n");
    let done = flow.stmt("done(i)");
    let break_hop = flow.stmt("jmp after_loop");
    let skip = flow.stmt("skip(i)");
    let continue_hop = flow.stmt("jmp loop_top");
    let work = flow.stmt("work(i)");
    let step = flow.stmt("i += 1");
    let exit = flow.stmt("return");
    flow.fall(flow.entry(), header);
    flow.branch(header, done, exit);
    flow.branch(done, break_hop, skip);
    flow.jump(break_hop, exit);
    flow.branch(skip, continue_hop, work);
    flow.jump(continue_hop, header);
    flow.fall(work, step);
    flow.jump(step, header);
    Trampolines {
        flow,
        break_hop,
        continue_hop,
    }
}

#[test]
fn treating_a_jump_trampoline_as_empty_structures_it_as_a_plain_break_or_continue() {
    let Trampolines {
        flow,
        break_hop,
        continue_hop,
    } = loop_with_trampolines();
    let empty = [break_hop, continue_hop];

    let (as_is, as_is_report) = ControlTree::compute_with_report(&flow.cfg);
    let (tree, report) = ControlTree::compute_with_report_treating_empty(&flow.cfg, &empty);

    assert!(as_is_report.is_fully_structured(), "{as_is_report:?}");
    assert!(report.is_fully_structured(), "{report:?}");
    assert_eq!(
        pseudocode(&flow.cfg, &as_is),
        "i = 0\nwhile {\n    i < n\n    break if !cond;\n    if (done(i)) {\n        jmp after_loop\n        break;\n    } else {\n        if (skip(i)) {\n            jmp loop_top\n            continue;\n        } else {\n            work(i)\n            i += 1\n            continue;\n        }\n    }\n}\nreturn\n"
    );
    assert_eq!(
        pseudocode(&flow.cfg, &tree),
        "i = 0\nwhile {\n    i < n\n    break if !cond;\n    if (done(i)) {\n        break;\n    } else {\n        if (skip(i)) {\n            continue;\n        } else {\n            work(i)\n            i += 1\n            continue;\n        }\n    }\n}\nreturn\n"
    );
}

#[test]
fn treating_blocks_as_empty_equals_structuring_a_copy_whose_blocks_were_cleared() {
    let Trampolines {
        flow,
        break_hop,
        continue_hop,
    } = loop_with_trampolines();
    let empty = [break_hop, continue_hop];

    let mut cleared = flow.cfg.clone();
    for block in empty {
        cleared.block_mut(block).instructions_mut().clear();
    }
    let (expected_tree, expected_report) = ControlTree::compute_with_report(&cleared);
    let (tree, report) = ControlTree::compute_with_report_treating_empty(&flow.cfg, &empty);

    assert_eq!(tree, expected_tree);
    assert_eq!(report, expected_report);
    assert!(
        flow.cfg
            .block(break_hop)
            .instructions()
            .contains(&"jmp after_loop"),
        "the graph is not changed"
    );
}

#[test]
fn treating_no_block_as_empty_is_compute_with_report() {
    let Trampolines { flow, .. } = loop_with_trampolines();

    assert_eq!(
        ControlTree::compute_with_report_treating_empty(&flow.cfg, &[]),
        ControlTree::compute_with_report(&flow.cfg)
    );
}

#[test]
fn trees_treating_blocks_as_empty_imply_the_flow_through_those_blocks() {
    let Trampolines {
        flow,
        break_hop,
        continue_hop,
    } = loop_with_trampolines();
    let empty: BTreeSet<BlockId> = [break_hop, continue_hop].into_iter().collect();

    let (tree, report) =
        ControlTree::compute_with_report_treating_empty(&flow.cfg, &[break_hop, continue_hop]);

    assert_well_formed(&tree);
    let covered = reachable_blocks(&flow.cfg);
    assert_flow_matches_graph_with(&flow.cfg, &tree, &report, &covered, &|block| {
        empty.contains(&block) || flow.cfg.block(block).is_empty()
    });
    let mut kept = covered;
    for block in &empty {
        kept.remove(block);
    }
    assert_spans_tile_each_block(&flow.cfg, &tree, &kept);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn treating_a_block_the_graph_lacks_as_empty_panics() {
    let flow = Flow::starting_with("only");

    let _ = ControlTree::compute_with_report_treating_empty(&flow.cfg, &[BlockId::from_raw(9)]);
}

#[test]
fn a_span_reads_back_its_slice_of_the_instructions() {
    let mut cfg = Cfg::<&'static str>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry)
        .instructions_mut()
        .extend(["p0 = lane < 8", "y = a", "store y"]);
    let tree = ControlTree::compute(&cfg);

    let spans = crate::coverage::leaf_spans(&tree);
    assert_eq!(spans.len(), 1, "{tree:?}");
    let span = spans[0];
    assert_eq!(span.block(), entry);
    assert_eq!(span.range(), 0..3);
    assert_eq!(span.len(), span.end() - span.start());
    assert_eq!(span.instructions(&cfg), cfg.block(entry).instructions());
    let points: Vec<ProgramPoint> = span.points().collect();
    assert_eq!(
        points,
        (0..3)
            .map(|inst_idx| ProgramPoint {
                block: entry,
                inst_idx
            })
            .collect::<Vec<_>>()
    );
    assert!(matches!(top_level(&tree)[..], [ControlNode::Return(_)]));
}
