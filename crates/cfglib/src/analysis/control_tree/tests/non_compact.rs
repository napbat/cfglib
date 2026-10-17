//! A graph with removed block slots structures like its compacted copy.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::pseudocode;
use crate::ControlTree;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};
use crate::test_util::{MockInst, ff};

/// A loop around a try, with a removed block before every named block.
fn with_gaps() -> Cfg<MockInst> {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let mut gaps = Vec::new();
    let mut named = |cfg: &mut Cfg<MockInst>, name: &'static str| {
        gaps.push(cfg.new_block());
        let block = cfg.new_block();
        cfg.block_mut(block).push(ff(name));
        block
    };
    let header = named(&mut cfg, "header");
    let body = named(&mut cfg, "body");
    let pad = named(&mut cfg, "pad");
    let exit = named(&mut cfg, "exit");
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, header, EdgeKind::Jump);
    cfg.add_edge(body, pad, EdgeKind::ExceptionHandler);
    cfg.add_edge(pad, header, EdgeKind::Jump);
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [body].into_iter().collect(),
        handlers: vec![Handler {
            entry: pad,
            body: HandlerBody::known([pad]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });
    for gap in gaps {
        cfg.remove_block(gap);
    }
    cfg
}

#[test]
fn removed_block_slots_do_not_change_the_structure() {
    let mut cfg = with_gaps();
    assert!(!cfg.is_compact());
    let (sparse, sparse_report) = ControlTree::compute_with_report(&cfg);
    let sparse_text = pseudocode(&cfg, &sparse);

    cfg.compact();
    let (dense, dense_report) = ControlTree::compute_with_report(&cfg);
    assert_eq!(sparse_text, pseudocode(&cfg, &dense));
    assert_eq!(sparse_report.gotos.len(), dense_report.gotos.len());
    assert_eq!(sparse.node_count(), dense.node_count());
}
