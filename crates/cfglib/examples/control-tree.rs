//! Structure a control-flow graph into a control tree and print it.
//!
//! The graph is the machine shape of
//!
//! ```text
//! total = 0
//! while i < n {
//!     if odd(i) { total += i } else { total -= 1 }
//!     i += 1
//! }
//! return total
//! ```
//!
//! The tree holds no instruction, so the example prints each one by reading
//! it back from the graph. Run it with `cargo run --example control-tree`.

use cfglib::{Cfg, ControlNode, ControlTree, EdgeKind};

fn main() {
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

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert!(report.is_fully_structured(), "{report:?}");

    let pseudocode = tree
        .to_pseudocode(|sink, point| {
            sink.write_str(cfg.block(point.block).instructions()[point.inst_idx])
        })
        .expect("writing to a String cannot fail");
    print!("{pseudocode}");

    let statements = tree
        .preorder()
        .filter(|&id| {
            matches!(
                tree.node(id),
                ControlNode::Block(_) | ControlNode::Return(_)
            )
        })
        .count();
    println!(
        "// {} nodes, {statements} straight-line leaves",
        tree.node_count()
    );
}
