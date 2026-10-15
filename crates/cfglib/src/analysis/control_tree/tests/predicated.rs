//! Predicated regionization: guards, witnesses, and the report.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::pseudocode;
use crate::builder::CfgBuilder;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::test_util::{df_ff, df_pred};
use crate::{ControlNode, ControlTree};

#[test]
fn compute_predicated_regionizes_same_predicate_runs() {
    let cfg = CfgBuilder::build(vec![
        df_ff("plain"),
        df_pred("guarded_a", 3, true),
        df_pred("guarded_b", 3, true),
        df_pred("negated", 3, false),
        df_ff("after"),
    ])
    .unwrap();

    let tree = ControlTree::compute_predicated(&cfg);
    let pseudo = pseudocode(&cfg, &tree);
    assert!(pseudo.contains("@guarded(guarded_a)"), "{pseudo}");
    assert!(pseudo.contains("@guarded(!negated)"), "{pseudo}");
    // Same-predicate instructions share one region.
    assert_eq!(pseudo.matches("@guarded(").count(), 2, "{pseudo}");
    // Unpredicated instructions stay outside regions.
    assert!(pseudo.starts_with("plain\n"), "{pseudo}");
}

#[test]
fn predicated_guards_name_their_witness_and_split_the_block() {
    let cfg = CfgBuilder::build(vec![
        df_ff("plain"),
        df_pred("guarded_a", 3, true),
        df_pred("guarded_b", 3, true),
        df_pred("negated", 3, false),
        df_ff("after"),
    ])
    .unwrap();
    let block = cfg.entry();

    let tree = ControlTree::compute_predicated(&cfg);
    let guards: Vec<_> = tree
        .preorder()
        .filter_map(|id| match *tree.node(id) {
            ControlNode::Guarded {
                predicate,
                when_true,
                body,
            } => Some((predicate, when_true, tree.body(body).to_vec())),
            _ => None,
        })
        .collect();
    assert_eq!(guards.len(), 2, "{tree:?}");
    assert_eq!((guards[0].0.block, guards[0].0.inst_idx), (block, 1));
    assert!(guards[0].1);
    assert_eq!((guards[1].0.block, guards[1].0.inst_idx), (block, 3));
    assert!(!guards[1].1);
    assert!(matches!(
        tree.node(guards[0].2[0]),
        ControlNode::Block(span) if span.range() == (1..3)
    ));

    // The four segments cover the block exactly once, in order.
    let mut points = Vec::new();
    tree.for_each_instruction(&cfg, |point, _| points.push(point.inst_idx));
    assert_eq!(points, [0, 1, 2, 3, 4]);
}

#[test]
fn predicated_report_matches_the_plain_report() {
    let mut cfg: Cfg<crate::test_util::DfInst> = Cfg::new();
    let target = cfg.new_block();
    let pad = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(df_ff("src"));
    cfg.block_mut(target).push(df_ff("dst"));
    cfg.block_mut(pad).push(df_ff("pad"));
    cfg.add_edge(cfg.entry(), target, EdgeKind::Jump);
    cfg.add_region(crate::Region {
        id: crate::RegionId::from_raw(0),
        protected_blocks: [target].into_iter().collect(),
        handlers: vec![crate::Handler {
            entry: pad,
            body: crate::HandlerBody::known([pad]),
            kind: crate::HandlerKind::CatchAll,
        }],
        parent: None,
    });

    let plain = ControlTree::compute_with_report(&cfg).1;
    assert!(!plain.is_fully_structured(), "{plain:?}");
    let (_, predicated) = ControlTree::compute_predicated_with_report(&cfg);
    assert_eq!(predicated, plain);
}
