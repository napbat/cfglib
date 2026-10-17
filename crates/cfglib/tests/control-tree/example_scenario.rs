//! The scenario of `examples/control-tree.rs`, as a test: a machine-shaped
//! `while` loop with an `if / else` in its body. Cargo builds the example under
//! `cargo test` but never runs it, so its assertion guards nothing; this does.

use cfglib::{Cfg, ControlNode, ControlTree, EdgeKind};

use crate::golden::assert_golden;
use crate::oracles::structure_and_check;
use crate::tree_shape::assert_fully_structured;

/// `total = 0; while i < n { if odd(i) { total += i } else { total -= 1 }; i += 1 }; return total`
fn example_graph() -> Cfg<&'static str> {
    let mut cfg = Cfg::<&str>::new();
    let header = cfg.new_block();
    let odd = cfg.new_block();
    let add = cfg.new_block();
    let even = cfg.new_block();
    let step = cfg.new_block();
    let exit = cfg.new_block();

    cfg.block_mut(cfg.entry()).push("total = 0");
    cfg.block_mut(header).push("i < n");
    cfg.block_mut(odd).push("odd(i)");
    cfg.block_mut(add).push("total += i");
    cfg.block_mut(even).push("total -= 1");
    cfg.block_mut(step).push("i += 1");
    cfg.block_mut(exit).push("return total");

    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, odd, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(odd, add, EdgeKind::ConditionalTrue);
    cfg.add_edge(add, step, EdgeKind::Fallthrough);
    cfg.add_edge(odd, even, EdgeKind::ConditionalFalse);
    cfg.add_edge(even, step, EdgeKind::Fallthrough);
    cfg.add_edge(step, header, EdgeKind::Jump);
    cfg
}

#[test]
fn the_example_graph_is_fully_structured() {
    let cfg = example_graph();

    let (tree, report) = ControlTree::compute_with_report(&cfg);

    assert!(report.is_fully_structured(), "{report:?}");
    assert_fully_structured(&tree, &report);
}

#[test]
fn the_example_prints_the_pseudocode_of_its_source() {
    let cfg = example_graph();
    let (tree, _) = structure_and_check(&cfg);

    let pseudocode = tree
        .to_pseudocode(|sink, point| {
            sink.write_str(cfg.block(point.block).instructions()[point.inst_idx])
        })
        .expect("writing to a String cannot fail");

    assert_golden("control-tree/while_if_else.pseudo", &pseudocode);
}

#[test]
fn the_example_summary_line_counts_eight_nodes_and_five_straight_line_leaves() {
    let cfg = example_graph();
    let (tree, _) = structure_and_check(&cfg);

    let statements = tree
        .preorder()
        .filter(|&id| {
            matches!(
                tree.node(id),
                ControlNode::Block(_) | ControlNode::Return(_)
            )
        })
        .count();

    // Block(init), Loop, If, Block(add), Block(even), Block(step), Return(exit)
    // and the root sequence that holds init, the loop, and the return.
    assert_eq!(tree.node_count(), 8);
    assert_eq!(statements, 5);
}
