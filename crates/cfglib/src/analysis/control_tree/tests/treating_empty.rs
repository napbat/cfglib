//! Structuring with blocks the caller treats as empty.

use super::{contains_block, has_node_kind};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::flow::FlowEffect;
use crate::test_util::{MockInst, ff};
use crate::{BlockSpan, ControlNode, ControlTree};

/// `entry: cond; true -> body; false -> exit`, a loop whose body leaves
/// through `hop`, a block holding a lone jump to the loop exit:
/// `header --true--> body --true--> hop --jump--> exit`.
fn loop_with_jump_trampoline(hop_instruction: bool) -> (Cfg<MockInst>, BlockId) {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let header = cfg.new_block();
    let body = cfg.new_block();
    let hop = cfg.new_block();
    let tail = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(header).push(ff("condition"));
    cfg.block_mut(body).push(ff("maybe_break"));
    if hop_instruction {
        cfg.block_mut(hop).push(ff("jmp"));
    }
    cfg.block_mut(tail).push(ff("tail_inst"));
    cfg.block_mut(exit)
        .push(MockInst(FlowEffect::Return, "after_loop"));
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, hop, EdgeKind::ConditionalTrue);
    cfg.add_edge(body, tail, EdgeKind::ConditionalFalse);
    cfg.add_edge(hop, exit, EdgeKind::Jump);
    cfg.add_edge(tail, header, EdgeKind::Jump);
    (cfg, hop)
}

#[test]
fn treating_a_block_as_empty_matches_a_graph_where_it_is_empty() {
    let (full, hop) = loop_with_jump_trampoline(true);
    let (empty, _) = loop_with_jump_trampoline(false);

    let ignored = ControlTree::compute_with_report_treating_empty(&full, &[hop]);
    assert_eq!(ignored, ControlTree::compute_with_report(&empty));

    // The jump block is a trampoline only once it is treated as empty: the
    // body's exit is then a plain `break`, and the hop block has no leaf.
    assert!(
        has_node_kind(&ignored.0, |n| matches!(
            n,
            ControlNode::Break { label: None }
        )),
        "{:?}",
        ignored.0
    );
    assert!(!contains_block(&ignored.0, hop), "{:?}", ignored.0);
    assert!(ignored.1.is_fully_structured(), "{:?}", ignored.1);
    let plain = ControlTree::compute(&full);
    assert!(contains_block(&plain, hop), "{plain:?}");
    assert_ne!(plain, ignored.0);
}

#[test]
fn treating_nothing_as_empty_is_compute_with_report() {
    let (cfg, _) = loop_with_jump_trampoline(true);
    assert_eq!(
        ControlTree::compute_with_report_treating_empty(&cfg, &[]),
        ControlTree::compute_with_report(&cfg)
    );
}

#[test]
fn an_emptied_condition_block_has_an_empty_condition_span() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let then_block = cfg.new_block();
    let merge = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("test"));
    cfg.block_mut(then_block).push(ff("then"));
    cfg.block_mut(merge).push(ff("merge"));
    cfg.add_edge(cfg.entry(), then_block, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), merge, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_block, merge, EdgeKind::Fallthrough);

    let (tree, _) = ControlTree::compute_with_report_treating_empty(&cfg, &[cfg.entry()]);
    let condition = tree
        .preorder()
        .find_map(|id| match *tree.node(id) {
            ControlNode::IfThenElse { condition, .. } => Some(condition),
            _ => None,
        })
        .expect("the branch stays a branch");
    assert_eq!(condition, BlockSpan::new(cfg.entry(), 0, 0));
    assert!(condition.is_empty());
}
