//! Integration tests: loops in code the entry cannot reach, through the public
//! control-tree facade.
//!
//! A catch pad that no exception edge leads to is still laid out, and a loop
//! in it is a loop: the tree has to say so rather than jump to the loop's own
//! head.

use cfglib::{
    BlockId, Cfg, ControlNode, ControlTree, EdgeKind, Handler, HandlerBody, HandlerKind, Region,
    RegionId,
};

#[derive(Debug, Clone, Copy)]
enum Closing {
    /// The loop header branches back to itself.
    SelfEdge,
    /// A second block returns to the header by an edge tagged `Back`.
    Tagged,
    /// A second block returns to the header by an ordinary jump.
    Untagged,
}

/// `entry -> live`, a try over `{entry}` whose catch pad is entered by no edge,
/// and in the pad a loop closed as `closing` says. Returns the graph and the
/// loop's header.
fn dead_pad_with_a_loop(closing: Closing) -> (Cfg<&'static str>, BlockId) {
    let mut cfg = Cfg::<&str>::new();
    let entry = cfg.entry();
    let [live, pad, header, body, after] = [(); 5].map(|()| cfg.new_block());
    for (block, name) in [
        (entry, "entry"),
        (live, "live"),
        (pad, "pad"),
        (header, "header"),
        (body, "body"),
        (after, "after"),
    ] {
        cfg.block_mut(block).push(name);
    }
    cfg.add_edge(entry, live, EdgeKind::Fallthrough);
    cfg.add_edge(pad, header, EdgeKind::Fallthrough);
    cfg.add_edge(header, after, EdgeKind::ConditionalFalse);
    match closing {
        Closing::SelfEdge => {
            cfg.add_edge(header, header, EdgeKind::ConditionalTrue);
        }
        Closing::Tagged | Closing::Untagged => {
            cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
            let kind = match closing {
                Closing::Tagged => EdgeKind::Back,
                _ => EdgeKind::Fallthrough,
            };
            cfg.add_edge(body, header, kind);
        }
    }
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [entry].into_iter().collect(),
        handlers: vec![Handler {
            entry: pad,
            body: HandlerBody::known([pad, header, body, after]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });
    (cfg, header)
}

#[test]
fn a_loop_in_a_dead_catch_pad_is_a_loop() {
    for closing in [Closing::SelfEdge, Closing::Tagged, Closing::Untagged] {
        let (cfg, header) = dead_pad_with_a_loop(closing);
        let (tree, report) = ControlTree::compute_with_report(&cfg);

        let loops: Vec<BlockId> = tree
            .preorder()
            .filter_map(|id| match tree.node(id) {
                ControlNode::Loop { header, .. } => Some(*header),
                _ => None,
            })
            .collect();
        assert_eq!(loops, [header], "{closing:?}");
        assert!(report.is_fully_structured(), "{closing:?}: {report:?}");

        // Every instruction appears once: the pad's code is not repeated to
        // make the loop.
        let text = tree
            .to_pseudocode(|sink, point| {
                sink.write_str(cfg.block(point.block).instructions()[point.inst_idx])
            })
            .expect("writing to a String cannot fail");
        for name in ["entry", "live", "pad", "header", "after"] {
            assert_eq!(
                text.matches(name).count(),
                1,
                "{closing:?}: {name} in\n{text}"
            );
        }
    }
}

#[test]
fn a_dead_block_that_returns_into_live_code_makes_no_loop_of_it() {
    // `live` returns, and dead code with a tag reaches back into it.
    let mut cfg = Cfg::<&str>::new();
    let dead = cfg.new_block();
    cfg.block_mut(cfg.entry()).push("live");
    cfg.block_mut(dead).push("dead");
    cfg.add_edge(dead, cfg.entry(), EdgeKind::Back);

    let (tree, report) = ControlTree::compute_with_report(&cfg);

    assert!(
        tree.preorder()
            .all(|id| !matches!(tree.node(id), ControlNode::Loop { .. })),
        "a block that returns is not in a loop"
    );
    assert!(report.is_fully_structured(), "{report:?}");
}
