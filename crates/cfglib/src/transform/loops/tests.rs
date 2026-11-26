use super::*;
use crate::builder::CfgBuilder;
use crate::edge::Edge;
use crate::flow::{Flow, FlowEffect};
use crate::test_util::{MockInst, ff, probability};

/// The loops of `cfg` as it is now.
fn loops_now<I, E>(cfg: &Cfg<I, E>) -> LoopForest {
    LoopForest::compute(cfg, &DominatorTree::compute(cfg))
}

#[test]
fn canonicalize_loops_adds_exits() {
    let mut cfg = CfgBuilder::build(alloc::vec![
        MockInst(FlowEffect::LoopOpen, "loop"),
        MockInst(FlowEffect::ConditionalBreak, "breakc"),
        ff("body"),
        MockInst(FlowEffect::LoopClose, "endloop"),
        MockInst(FlowEffect::Flow(Flow::Return), "ret"),
    ])
    .unwrap();
    let dom = DominatorTree::compute(&cfg);
    let canonical = canonicalize_loops(&mut cfg, &dom);
    assert_eq!(canonical.len(), 1);
    assert_ne!(canonical[0].exits.len(), 0);
}

/// `entry -> header` twice over (a weighted conditional arm and a weighted
/// switch arm through `side`), the loop `header <-> body`, and the exits.
struct TwoEntryLoop {
    cfg: Cfg<u32>,
    header: BlockId,
    side: BlockId,
    other: BlockId,
    conditional: EdgeId,
    unrelated: EdgeId,
    through_side: EdgeId,
    side_to_header: EdgeId,
}

fn two_entry_loop() -> TwoEntryLoop {
    let mut cfg = Cfg::<u32>::new();
    let entry = cfg.entry();
    let [header, body, side, other] = [(); 4].map(|()| cfg.new_block());
    let conditional = cfg.add_edge(
        entry,
        header,
        Edge::new(EdgeKind::ConditionalTrue).with_weight(probability(0.25)),
    );
    let unrelated = cfg.add_edge(entry, other, EdgeKind::ConditionalFalse);
    let through_side = cfg.add_edge(entry, side, EdgeKind::SwitchCase);
    let side_to_header = cfg.add_edge(
        side,
        header,
        Edge::new(EdgeKind::Fallthrough).with_weight(probability(0.75)),
    );
    cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(body, header, EdgeKind::Back);
    cfg.add_edge(header, other, EdgeKind::ConditionalFalse);
    TwoEntryLoop {
        cfg,
        header,
        side,
        other,
        conditional,
        unrelated,
        through_side,
        side_to_header,
    }
}

#[test]
fn inserting_a_preheader_redirects_the_entering_edges_in_place() {
    let TwoEntryLoop {
        mut cfg,
        header,
        side,
        other,
        conditional,
        unrelated,
        through_side,
        side_to_header,
    } = two_entry_loop();
    let entry = cfg.entry();
    let before: Vec<_> = cfg.outgoing(entry).collect();
    let loops = loops_now(&cfg);
    let lp = loops.loop_of_header(header).expect("header heads a loop");

    let preheader = insert_preheader(&mut cfg, &loops, lp).expect("two entries need a preheader");

    assert_eq!(
        cfg.outgoing(entry).collect::<Vec<_>>(),
        before,
        "the entry keeps its edges in order"
    );
    assert_eq!(before, [conditional, unrelated, through_side]);
    assert_eq!(cfg.edge(conditional).target(), preheader);
    assert_eq!(cfg.edge(conditional).kind(), EdgeKind::ConditionalTrue);
    assert_eq!(cfg.edge(conditional).weight(), Some(probability(0.25)));
    assert_eq!(cfg.edge(unrelated).target(), other);
    assert_eq!(cfg.edge(side_to_header).source(), side);
    assert_eq!(cfg.edge(side_to_header).target(), preheader);
    assert_eq!(cfg.edge(side_to_header).weight(), Some(probability(0.75)));

    let outside: Vec<_> = cfg
        .incoming(header)
        .filter(|&edge| !loops.contains(lp, cfg.edge(edge).source()))
        .collect();
    assert_eq!(outside.len(), 1);
    assert_eq!(cfg.edge(outside[0]).source(), preheader);
    assert_eq!(cfg.edge(outside[0]).kind(), EdgeKind::Fallthrough);
    assert_eq!(cfg.outgoing(preheader).count(), 1);
}

#[test]
fn a_header_with_a_sole_dedicated_predecessor_needs_no_preheader() {
    let mut cfg = Cfg::<u32>::new();
    let [header, body] = [(); 2].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(body, header, EdgeKind::Back);
    let loops = loops_now(&cfg);
    let lp = loops.loop_of_header(header).expect("header heads a loop");

    assert_eq!(insert_preheader(&mut cfg, &loops, lp), None);
    assert_eq!(cfg.block_count(), 3);
}

#[test]
fn a_sole_predecessor_that_also_branches_elsewhere_is_not_a_preheader() {
    let mut cfg = Cfg::<u32>::new();
    let [header, body, exit] = [(); 3].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), header, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(body, header, EdgeKind::Back);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    let dom = DominatorTree::compute(&cfg);

    let canonical = canonicalize_loops(&mut cfg, &dom);

    assert_eq!(canonical.len(), 1);
    assert_eq!(canonical[0].header, header);
    let preheader = canonical[0]
        .preheader
        .expect("the loop is entered from outside");
    assert_ne!(preheader, cfg.entry());
    assert_eq!(cfg.successors(preheader).collect::<Vec<_>>(), [header]);
    let loops = loops_now(&cfg);
    let lp = loops.loop_of_header(header).expect("header heads a loop");
    let outside: Vec<_> = cfg
        .predecessors(header)
        .filter(|&block| !loops.contains(lp, block))
        .collect();
    assert_eq!(outside, [preheader]);
}

#[test]
fn an_existing_dedicated_predecessor_is_reported_as_the_preheader() {
    let mut cfg = Cfg::<u32>::new();
    let [header, body] = [(); 2].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(body, header, EdgeKind::Back);
    let dom = DominatorTree::compute(&cfg);

    let canonical = canonicalize_loops(&mut cfg, &dom);

    assert_eq!(canonical[0].preheader, Some(cfg.entry()));
    assert_eq!(cfg.block_count(), 3);
}

#[test]
fn an_inner_preheader_joins_the_loops_around_it() {
    // entry -> outer; outer -> {inner, side}; side -> inner; inner <-> work;
    // work -> outer is the outer back edge; outer -> exit. The inner header has
    // two entries, both inside the outer loop.
    let mut cfg = Cfg::<u32>::new();
    let [outer, inner, side, work, exit] = [(); 5].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), outer, EdgeKind::Fallthrough);
    cfg.add_edge(outer, inner, EdgeKind::ConditionalTrue);
    cfg.add_edge(outer, side, EdgeKind::SwitchCase);
    cfg.add_edge(outer, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(side, inner, EdgeKind::Fallthrough);
    cfg.add_edge(inner, work, EdgeKind::ConditionalTrue);
    cfg.add_edge(work, inner, EdgeKind::Back);
    cfg.add_edge(inner, outer, EdgeKind::Back);
    let dom = DominatorTree::compute(&cfg);

    let canonical = canonicalize_loops(&mut cfg, &dom);

    assert_eq!(canonical.len(), 2);
    let [outer_loop, inner_loop] = [&canonical[0], &canonical[1]];
    assert_eq!(outer_loop.header, outer);
    assert_eq!(inner_loop.header, inner);
    let inner_preheader = inner_loop.preheader.expect("two entries");
    let loops = loops_now(&cfg);
    let [outer_now, inner_now] =
        [outer, inner].map(|header| loops.loop_of_header(header).expect("still a loop"));
    assert!(
        loops.contains(outer_now, inner_preheader),
        "the new block sits on the path through the outer loop"
    );
    assert!(!loops.contains(inner_now, inner_preheader));
    assert_eq!(outer_loop.exits, loops.exits(&cfg, outer_now));
    assert_eq!(inner_loop.exits, loops.exits(&cfg, inner_now));
}

#[test]
fn exits_are_reported_after_every_preheader_exists() {
    // entry -> {h1, h2}; h1 <-> b1; h1 -> h2. The second loop's preheader takes
    // over the edge that leaves the first loop, so that edge now ends there.
    let mut cfg = Cfg::<u32>::new();
    let [h1, b1, h2, b2, done] = [(); 5].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), h1, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), h2, EdgeKind::ConditionalFalse);
    cfg.add_edge(h1, b1, EdgeKind::ConditionalTrue);
    cfg.add_edge(b1, h1, EdgeKind::Back);
    cfg.add_edge(h1, h2, EdgeKind::ConditionalFalse);
    cfg.add_edge(h2, b2, EdgeKind::ConditionalTrue);
    cfg.add_edge(b2, h2, EdgeKind::Back);
    cfg.add_edge(h2, done, EdgeKind::ConditionalFalse);
    let dom = DominatorTree::compute(&cfg);

    let canonical = canonicalize_loops(&mut cfg, &dom);

    assert_eq!(canonical.len(), 2);
    let loops = loops_now(&cfg);
    for canonical_loop in &canonical {
        let lp = loops
            .loop_of_header(canonical_loop.header)
            .expect("still a loop");
        assert_eq!(
            canonical_loop.exits,
            loops.exits(&cfg, lp),
            "exits of the loop at {:?}",
            canonical_loop.header
        );
    }
    let first = canonical.iter().find(|c| c.header == h1).unwrap();
    let second_preheader = canonical
        .iter()
        .find(|c| c.header == h2)
        .and_then(|c| c.preheader)
        .expect("the second loop has two entries");
    assert_eq!(first.exits, [second_preheader]);
}
