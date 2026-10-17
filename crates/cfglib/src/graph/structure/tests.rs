use super::*;
use crate::builder::CfgBuilder;
use crate::flow::FlowEffect;
use crate::graph::dominator::DominatorTree;
use crate::test_util::{MockInst, ff};
use alloc::vec;

#[test]
fn no_loops_in_linear_cfg() {
    let cfg = CfgBuilder::build(vec![ff("a"), ff("b"), ff("c")]).unwrap();
    let dom = DominatorTree::compute(&cfg);
    let loops = detect_loops(&cfg, &dom);
    assert_eq!(loops.len(), 0);
}

#[test]
fn simple_loop_detected() {
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::LoopOpen, "loop"),
        ff("body"),
        MockInst(FlowEffect::LoopClose, "endloop"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    let dom = DominatorTree::compute(&cfg);
    let loops = detect_loops(&cfg, &dom);
    assert_eq!(loops.len(), 1);
    assert!(!loops[0].body.is_empty(), "loop body is non-empty");
}

#[test]
fn loops_with_one_header_merge_every_latch_body() {
    let mut cfg = Cfg::<MockInst>::new();
    let header = cfg.entry();
    let left_body = cfg.new_block();
    let left_latch = cfg.new_block();
    let right_body = cfg.new_block();
    let right_latch = cfg.new_block();
    let exit = cfg.new_block();

    cfg.add_edge(header, left_body, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, right_body, EdgeKind::ConditionalFalse);
    cfg.add_edge(header, exit, EdgeKind::SwitchCase);
    cfg.add_edge(left_body, left_latch, EdgeKind::Fallthrough);
    cfg.add_edge(left_latch, header, EdgeKind::Back);
    cfg.add_edge(right_body, right_latch, EdgeKind::Fallthrough);
    cfg.add_edge(right_latch, header, EdgeKind::Back);

    let dom = DominatorTree::compute(&cfg);
    let loops = detect_loops(&cfg, &dom);

    assert_eq!(loops.len(), 1);
    assert_eq!(loops[0].header, header);
    assert_eq!(loops[0].latches, BTreeSet::from([left_latch, right_latch]));
    assert_eq!(
        loops[0].body,
        BTreeSet::from([header, left_body, left_latch, right_body, right_latch])
    );
    assert_eq!(loops[0].depth, 0);
}

#[test]
fn nested_loops_have_depth() {
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::LoopOpen, "outer"),
        MockInst(FlowEffect::LoopOpen, "inner"),
        ff("body"),
        MockInst(FlowEffect::ConditionalBreak, "breakc_inner"),
        MockInst(FlowEffect::LoopClose, "end_inner"),
        MockInst(FlowEffect::ConditionalBreak, "breakc_outer"),
        MockInst(FlowEffect::LoopClose, "end_outer"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    let dom = DominatorTree::compute(&cfg);
    let loops = detect_loops(&cfg, &dom);
    assert_eq!(loops.len(), 2);
    // Outermost first (depth 0), inner second (depth 1).
    assert_eq!(loops[0].depth, 0);
    assert_eq!(loops[1].depth, 1);
}

#[test]
fn loop_with_break_still_detected() {
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::LoopOpen, "loop"),
        MockInst(FlowEffect::ConditionalBreak, "breakc"),
        ff("body"),
        MockInst(FlowEffect::LoopClose, "endloop"),
    ])
    .unwrap();
    let dom = DominatorTree::compute(&cfg);
    let loops = detect_loops(&cfg, &dom);
    assert_eq!(loops.len(), 1);
}

#[test]
fn tagged_back_edge_survives_without_dominance() {
    // Two entries into a tagged cycle: dominance alone cannot prove
    // b1 → b2 is a loop, but the builder tag says it is.
    let mut cfg = Cfg::<MockInst>::new();
    let b1 = cfg.new_block();
    let b2 = cfg.new_block();
    cfg.add_edge(cfg.entry(), b1, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), b2, EdgeKind::ConditionalFalse);
    cfg.add_edge(b1, b2, EdgeKind::Fallthrough);
    cfg.add_edge(b2, b1, EdgeKind::Back);

    let dom = DominatorTree::compute(&cfg);
    assert_eq!(find_back_edges(&cfg, &dom).len(), 0);
    let tagged = find_back_edges_tagged(&cfg, &dom);
    assert_eq!(tagged.len(), 1);
    assert_eq!(tagged[0].header, b1);
    let loops = detect_loops_tagged(&cfg, &dom);
    assert_eq!(loops.len(), 1);
    assert!(loops[0].body.contains(&b2));
}

#[test]
fn detected_depth_is_the_depth_in_the_nesting_tree() {
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::LoopOpen, "outer"),
        MockInst(FlowEffect::LoopOpen, "inner"),
        ff("body"),
        MockInst(FlowEffect::ConditionalBreak, "breakc_inner"),
        MockInst(FlowEffect::LoopClose, "end_inner"),
        MockInst(FlowEffect::ConditionalBreak, "breakc_outer"),
        MockInst(FlowEffect::LoopClose, "end_outer"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    let dom = DominatorTree::compute(&cfg);
    let loops = detect_loops(&cfg, &dom);
    let tree = crate::graph::loop_nest::LoopNestingTree::compute(&loops);
    assert_eq!(loops.len(), 2);
    for (index, lp) in loops.iter().enumerate() {
        assert_eq!(lp.depth, tree.depth(index), "loop at {:?}", lp.header);
    }
}

#[test]
fn irreducible_loops_over_one_cycle_are_siblings_not_each_others_parents() {
    // Two entries into a cycle tagged in both directions: two loops, one
    // body. Neither contains the other, so both are outermost.
    let mut cfg = Cfg::<u32>::new();
    let b1 = cfg.new_block();
    let b2 = cfg.new_block();
    cfg.add_edge(cfg.entry(), b1, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), b2, EdgeKind::ConditionalFalse);
    cfg.add_edge(b1, b2, EdgeKind::Back);
    cfg.add_edge(b2, b1, EdgeKind::Back);

    let dom = DominatorTree::compute(&cfg);
    let loops = detect_loops_tagged(&cfg, &dom);
    assert_eq!(loops.len(), 2);
    assert!(loops.iter().all(|lp| lp.depth == 0));
    let tree = crate::graph::loop_nest::LoopNestingTree::compute(&loops);
    assert_eq!(tree.roots(), [0, 1]);
}

#[test]
fn tagged_irreducible_loop_body_stays_inside_the_cycle() {
    // entry → {b1, b2} and a tagged b1 ↔ b2: the entry reaches the latch
    // without passing the header, but it is not part of the cycle.
    let mut cfg = Cfg::<u32>::new();
    let b1 = cfg.new_block();
    let b2 = cfg.new_block();
    cfg.add_edge(cfg.entry(), b1, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), b2, EdgeKind::ConditionalFalse);
    cfg.add_edge(b1, b2, EdgeKind::Back);
    cfg.add_edge(b2, b1, EdgeKind::Back);

    let dom = DominatorTree::compute(&cfg);
    let loops = detect_loops_tagged(&cfg, &dom);
    assert_eq!(loops.len(), 2);
    for lp in &loops {
        assert_eq!(lp.body, BTreeSet::from([b1, b2]), "header {:?}", lp.header);
    }
}

#[test]
fn tagged_irreducible_loop_keeps_every_block_of_the_cycle() {
    // entry → {h, y}, h → x → y → h, with y → h tagged: the cycle is
    // h, x, y even though h dominates neither x nor y.
    let mut cfg = Cfg::<u32>::new();
    let [h, x, y] = [(); 3].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), h, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), y, EdgeKind::ConditionalFalse);
    cfg.add_edge(h, x, EdgeKind::Fallthrough);
    cfg.add_edge(x, y, EdgeKind::Fallthrough);
    cfg.add_edge(y, h, EdgeKind::Back);

    let dom = DominatorTree::compute(&cfg);
    let loops = detect_loops_tagged(&cfg, &dom);
    assert_eq!(loops.len(), 1);
    assert_eq!(loops[0].header, h);
    assert_eq!(loops[0].body, BTreeSet::from([h, x, y]));
}

#[test]
fn dead_predecessors_are_not_part_of_a_loop_body() {
    let mut cfg = Cfg::<u32>::new();
    let [header, body, dead] = [(); 3].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, body, EdgeKind::Fallthrough);
    cfg.add_edge(body, header, EdgeKind::Back);
    cfg.add_edge(dead, body, EdgeKind::Fallthrough);

    let dom = DominatorTree::compute(&cfg);
    for loops in [detect_loops(&cfg, &dom), detect_loops_tagged(&cfg, &dom)] {
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].body, BTreeSet::from([header, body]));
    }
}

#[test]
fn unreachable_code_has_no_back_edges() {
    // A block that branches to itself is dominated by itself, but the
    // entry never reaches it, so it is no more a loop than the
    // unreachable two-cycle beside it.
    let mut cfg = Cfg::<u32>::new();
    let [dead_self, dead_a, dead_b, dead_tagged] = [(); 4].map(|()| cfg.new_block());
    cfg.add_edge(dead_self, dead_self, EdgeKind::ConditionalTrue);
    cfg.add_edge(dead_a, dead_b, EdgeKind::Fallthrough);
    cfg.add_edge(dead_b, dead_a, EdgeKind::Fallthrough);
    cfg.add_edge(dead_tagged, dead_a, EdgeKind::Back);

    let dom = DominatorTree::compute(&cfg);
    assert_eq!(find_back_edges(&cfg, &dom), Vec::new());
    assert_eq!(find_back_edges_tagged(&cfg, &dom), Vec::new());
    assert_eq!(detect_loops_tagged(&cfg, &dom), Vec::new());
}

#[test]
fn linear_cfg_is_reducible() {
    let cfg = CfgBuilder::build(vec![ff("a"), ff("b")]).unwrap();
    let dom = DominatorTree::compute(&cfg);
    assert!(is_reducible(&cfg, &dom));
}

#[test]
fn loop_cfg_is_reducible() {
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::LoopOpen, "loop"),
        ff("body"),
        MockInst(FlowEffect::LoopClose, "endloop"),
    ])
    .unwrap();
    let dom = DominatorTree::compute(&cfg);
    assert!(is_reducible(&cfg, &dom));
}

#[test]
fn if_else_no_loops() {
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::ConditionalOpen, "if"),
        ff("then"),
        MockInst(FlowEffect::ConditionalAlternate, "else"),
        ff("else_body"),
        MockInst(FlowEffect::ConditionalClose, "endif"),
    ])
    .unwrap();
    let dom = DominatorTree::compute(&cfg);
    let loops = detect_loops(&cfg, &dom);
    assert_eq!(loops.len(), 0);
    assert!(is_reducible(&cfg, &dom));
}

#[test]
fn loop_exit_blocks_found() {
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::LoopOpen, "loop"),
        MockInst(FlowEffect::ConditionalBreak, "breakc"),
        ff("body"),
        MockInst(FlowEffect::LoopClose, "endloop"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    let dom = DominatorTree::compute(&cfg);
    let loops = detect_loops(&cfg, &dom);
    assert_ne!(loops.len(), 0);
    let exits = loop_exit_blocks(&cfg, &loops[0]);
    assert!(
        !exits.is_empty(),
        "loop should have at least one exit block"
    );
}

#[test]
fn canonicalize_loops_adds_exits() {
    let mut cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::LoopOpen, "loop"),
        MockInst(FlowEffect::ConditionalBreak, "breakc"),
        ff("body"),
        MockInst(FlowEffect::LoopClose, "endloop"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    let dom = DominatorTree::compute(&cfg);
    let canonical = canonicalize_loops(&mut cfg, &dom);
    assert!(!canonical.is_empty());
    assert!(!canonical[0].exits.is_empty());
}

/// `entry -> header` twice over (a weighted conditional arm and a weighted
/// switch arm through `side`), the loop `header <-> body`, and the exits.
struct TwoEntryLoop {
    cfg: Cfg<u32>,
    header: BlockId,
    side: BlockId,
    other: BlockId,
    conditional: crate::edge::EdgeId,
    unrelated: crate::edge::EdgeId,
    through_side: crate::edge::EdgeId,
    side_to_header: crate::edge::EdgeId,
}

fn two_entry_loop() -> TwoEntryLoop {
    let mut cfg = Cfg::<u32>::new();
    let entry = cfg.entry();
    let [header, body, side, other] = [(); 4].map(|()| cfg.new_block());
    let conditional = cfg.add_weighted_edge(entry, header, EdgeKind::ConditionalTrue, 0.25);
    let unrelated = cfg.add_edge(entry, other, EdgeKind::ConditionalFalse);
    let through_side = cfg.add_edge(entry, side, EdgeKind::SwitchCase);
    let side_to_header = cfg.add_weighted_edge(side, header, EdgeKind::Fallthrough, 0.75);
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
        ..
    } = two_entry_loop();
    let entry = cfg.entry();
    let before: Vec<_> = cfg.outgoing(entry).collect();
    let dom = DominatorTree::compute(&cfg);
    let lp = detect_loops(&cfg, &dom).remove(0);

    let preheader = insert_preheader(&mut cfg, &lp).expect("two entries need a preheader");

    assert_eq!(
        cfg.outgoing(entry).collect::<Vec<_>>(),
        before,
        "the entry keeps its edges in order"
    );
    assert_eq!(before, [conditional, unrelated, through_side]);
    assert_eq!(cfg.edge(conditional).target(), preheader);
    assert_eq!(cfg.edge(conditional).kind(), EdgeKind::ConditionalTrue);
    assert_eq!(cfg.edge(conditional).weight(), Some(0.25));
    assert_eq!(cfg.edge(unrelated).target(), other);
    assert_eq!(cfg.edge(side_to_header).source(), side);
    assert_eq!(cfg.edge(side_to_header).target(), preheader);
    assert_eq!(cfg.edge(side_to_header).weight(), Some(0.75));

    let outside: Vec<_> = cfg
        .incoming(header)
        .filter(|&edge| !lp.body.contains(&cfg.edge(edge).source()))
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
    let dom = DominatorTree::compute(&cfg);
    let lp = detect_loops(&cfg, &dom).remove(0);

    assert_eq!(insert_preheader(&mut cfg, &lp), None);
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
    let preheader = canonical[0]
        .preheader
        .expect("the loop is entered from outside");
    assert_ne!(preheader, cfg.entry());
    assert_eq!(cfg.successors(preheader).collect::<Vec<_>>(), [header]);
    let outside: Vec<_> = cfg
        .predecessors(header)
        .filter(|block| !canonical[0].natural_loop.body.contains(block))
        .collect();
    assert_eq!(outside, [preheader]);
}

#[test]
fn a_loop_headed_by_the_entry_has_no_preheader() {
    let mut cfg = Cfg::<u32>::new();
    let [body, exit] = [(); 2].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), body, EdgeKind::ConditionalTrue);
    cfg.add_edge(body, cfg.entry(), EdgeKind::Back);
    cfg.add_edge(cfg.entry(), exit, EdgeKind::ConditionalFalse);
    let dom = DominatorTree::compute(&cfg);

    let canonical = canonicalize_loops(&mut cfg, &dom);

    assert_eq!(canonical.len(), 1);
    assert_eq!(canonical[0].preheader, None);
    assert_eq!(cfg.block_count(), 3, "nothing could be inserted");
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
    assert_eq!(outer_loop.natural_loop.header, outer);
    assert_eq!(inner_loop.natural_loop.header, inner);
    let inner_preheader = inner_loop.preheader.expect("two entries");
    assert!(
        outer_loop.natural_loop.body.contains(&inner_preheader),
        "the new block sits on the path through the outer loop"
    );
    assert!(!inner_loop.natural_loop.body.contains(&inner_preheader));
    let dom = DominatorTree::compute(&cfg);
    let recomputed = detect_loops_tagged(&cfg, &dom);
    let outer_now = recomputed.iter().find(|lp| lp.header == outer).unwrap();
    assert_eq!(outer_now.body, outer_loop.natural_loop.body);
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
    for canonical_loop in &canonical {
        assert_eq!(
            canonical_loop.exits,
            loop_exit_blocks(&cfg, &canonical_loop.natural_loop),
            "exits of the loop at {:?}",
            canonical_loop.natural_loop.header
        );
    }
    let first = canonical
        .iter()
        .find(|c| c.natural_loop.header == h1)
        .unwrap();
    let second_preheader = canonical
        .iter()
        .find(|c| c.natural_loop.header == h2)
        .and_then(|c| c.preheader)
        .expect("the second loop has two entries");
    assert_eq!(first.exits, BTreeSet::from([second_preheader]));
}

/// A cycle that survives removing every edge whose target dominates its
/// source is a cycle no node dominates: the graph is irreducible exactly
/// then. The check shares nothing with the walk `is_reducible` runs.
fn has_cycle_without_back_edges(
    count: usize,
    edges: &[(usize, usize)],
    dom: &DominatorTree<crate::graph::store::NodeId>,
) -> bool {
    use crate::graph::store::NodeId;

    let mut reach = alloc::vec![alloc::vec![false; count]; count];
    for &(from, to) in edges {
        let (from, to) = (NodeId::from_index(from), NodeId::from_index(to));
        if dom.is_reachable(from) && !dom.dominates(to, from) {
            reach[from.index()][to.index()] = true;
        }
    }
    for via in 0..count {
        for from in 0..count {
            for to in 0..count {
                reach[from][to] |= reach[from][via] && reach[via][to];
            }
        }
    }
    (0..count).any(|node| reach[node][node])
}

#[test]
fn reducibility_is_acyclicity_without_back_edges_on_every_small_digraph() {
    use crate::graph::store::{Graph, NodeId};
    use crate::graph::view::Rooted;

    let mut irreducible = 0_usize;
    for count in 1..=4_usize {
        let pairs: Vec<(usize, usize)> = (0..count)
            .flat_map(|from| (0..count).map(move |to| (from, to)))
            .collect();
        for mask in 0..1_u32 << pairs.len() {
            let edges: Vec<(usize, usize)> = pairs
                .iter()
                .enumerate()
                .filter(|&(bit, _)| mask & (1 << bit) != 0)
                .map(|(_, &pair)| pair)
                .collect();
            let mut graph = Graph::<(), ()>::new();
            for _ in 0..count {
                graph.add_node(());
            }
            for &(from, to) in &edges {
                graph.add_edge(NodeId::from_index(from), NodeId::from_index(to), ());
            }
            let view = Rooted::new(&graph, NodeId::from_index(0));
            let dom = DominatorTree::compute(&view);

            let expected = !has_cycle_without_back_edges(count, &edges, &dom);
            assert_eq!(
                is_reducible(&view, &dom),
                expected,
                "{count} nodes {edges:?}"
            );
            irreducible += usize::from(!expected);
        }
    }
    assert!(
        irreducible > 0,
        "the enumeration must reach irreducible graphs"
    );
}

#[test]
fn a_back_tag_on_an_edge_that_closes_no_cycle_is_not_a_back_edge() {
    // `entry -Back-> sink`: the sink returns, so it cannot reach its tail.
    let mut cfg = Cfg::<u32>::new();
    let sink = cfg.new_block();
    cfg.add_edge(cfg.entry(), sink, EdgeKind::Back);
    let dom = DominatorTree::compute(&cfg);

    assert_eq!(find_back_edges_tagged(&cfg, &dom), Vec::new());
    assert_eq!(detect_loops_tagged(&cfg, &dom), Vec::new());
}

#[test]
fn a_back_tag_leaving_a_cycle_for_a_block_outside_it_is_ignored() {
    // `entry -> a <-> b`, and a tag on `b -> exit`, which leaves the cycle.
    let mut cfg = Cfg::<u32>::new();
    let [a, b, exit] = [(); 3].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), a, EdgeKind::Fallthrough);
    cfg.add_edge(a, b, EdgeKind::ConditionalTrue);
    cfg.add_edge(b, a, EdgeKind::Back);
    cfg.add_edge(b, exit, EdgeKind::Back);
    let dom = DominatorTree::compute(&cfg);

    let backs = find_back_edges_tagged(&cfg, &dom);

    assert_eq!(backs, [BackEdge { tail: b, header: a }]);
    let loops = detect_loops_tagged(&cfg, &dom);
    assert_eq!(loops.len(), 1);
    assert_eq!(loops[0].body, BTreeSet::from([a, b]));
}

/// `entry -> exit` is the live program. The dead code is a source `source`
/// that enters `head`, a loop `head <-> tail` (closed by an ordinary jump, so
/// nothing but dominance within the dead code says it is one), and a way out
/// `head -> out`.
struct DeadLoop {
    cfg: Cfg<u32>,
    head: BlockId,
    tail: BlockId,
}

fn dead_loop() -> DeadLoop {
    let mut cfg = Cfg::<u32>::new();
    let [exit, source, head, tail, out] = [(); 5].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);
    cfg.add_edge(source, head, EdgeKind::Fallthrough);
    cfg.add_edge(head, tail, EdgeKind::ConditionalTrue);
    cfg.add_edge(head, out, EdgeKind::ConditionalFalse);
    cfg.add_edge(tail, head, EdgeKind::Fallthrough);
    DeadLoop { cfg, head, tail }
}

#[test]
fn without_dead_code_every_loop_is_the_one_detect_loops_tagged_finds() {
    let mut cfg = Cfg::<u32>::new();
    let [head, body, exit] = [(); 3].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), head, EdgeKind::Fallthrough);
    cfg.add_edge(head, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(head, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, head, EdgeKind::Back);
    let dom = DominatorTree::compute(&cfg);

    assert_eq!(
        detect_all_loops_tagged(&cfg, &dom),
        detect_loops_tagged(&cfg, &dom)
    );
}

#[test]
fn dominance_within_dead_code_finds_its_loops() {
    let DeadLoop { cfg, head, tail } = dead_loop();
    let dom = DominatorTree::compute(&cfg);
    assert_eq!(detect_loops_tagged(&cfg, &dom), Vec::new());

    let loops = detect_all_loops_tagged(&cfg, &dom);

    assert_eq!(loops.len(), 1);
    assert_eq!(loops[0].header, head);
    assert_eq!(loops[0].latches, BTreeSet::from([tail]));
    assert_eq!(loops[0].body, BTreeSet::from([head, tail]));
    assert_eq!(loops[0].depth, 0);
}

#[test]
fn dead_self_loops_and_tagged_loops_are_loops_too() {
    let mut cfg = Cfg::<u32>::new();
    let [exit, spin, tagged_head, tagged_tail] = [(); 4].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);
    cfg.add_edge(spin, spin, EdgeKind::ConditionalTrue);
    cfg.add_edge(tagged_head, tagged_tail, EdgeKind::ConditionalTrue);
    cfg.add_edge(tagged_tail, tagged_head, EdgeKind::Back);
    let dom = DominatorTree::compute(&cfg);

    let headers: Vec<BlockId> = detect_all_loops_tagged(&cfg, &dom)
        .iter()
        .map(|lp| lp.header)
        .collect();

    assert_eq!(headers, [spin, tagged_head]);
}

#[test]
fn nesting_among_dead_loops_counts_only_dead_loops() {
    // Dead: source -> outer -> inner <-> work, inner -> outer (a back edge to
    // outer), outer -> out. Live: a loop of its own, which must not count.
    let mut cfg = Cfg::<u32>::new();
    let [head, body, exit, source, outer, inner, work, out] = [(); 8].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), head, EdgeKind::Fallthrough);
    cfg.add_edge(head, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(head, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, head, EdgeKind::Back);
    cfg.add_edge(source, outer, EdgeKind::Fallthrough);
    cfg.add_edge(outer, inner, EdgeKind::ConditionalTrue);
    cfg.add_edge(outer, out, EdgeKind::ConditionalFalse);
    cfg.add_edge(inner, work, EdgeKind::ConditionalTrue);
    cfg.add_edge(work, inner, EdgeKind::Fallthrough);
    cfg.add_edge(inner, outer, EdgeKind::Fallthrough);
    let dom = DominatorTree::compute(&cfg);

    let loops = detect_all_loops_tagged(&cfg, &dom);

    let depth_of = |header| {
        loops
            .iter()
            .find(|lp| lp.header == header)
            .map(|lp| lp.depth)
    };
    assert_eq!(loops.len(), 3);
    assert_eq!(depth_of(head), Some(0));
    assert_eq!(depth_of(outer), Some(0));
    assert_eq!(depth_of(inner), Some(1));
    assert!(
        loops
            .windows(2)
            .all(|pair| (pair[0].depth, pair[0].header) <= (pair[1].depth, pair[1].header)),
        "outermost first, then by header"
    );
}

#[test]
fn dead_code_that_jumps_into_live_code_neither_joins_nor_closes_its_loops() {
    // A live loop `head <-> body`, a dead block feeding the body, and a dead
    // tagged edge into the live head, which closes no cycle.
    let mut cfg = Cfg::<u32>::new();
    let [head, body, exit, feeder] = [(); 4].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), head, EdgeKind::Fallthrough);
    cfg.add_edge(head, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(head, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, head, EdgeKind::Back);
    cfg.add_edge(feeder, body, EdgeKind::Jump);
    cfg.add_edge(feeder, head, EdgeKind::Back);
    let dom = DominatorTree::compute(&cfg);

    let all = detect_all_loops_tagged(&cfg, &dom);

    assert_eq!(all, detect_loops_tagged(&cfg, &dom));
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].body, BTreeSet::from([head, body]));
}

#[test]
fn a_dead_loop_that_feeds_a_live_loops_body_stays_apart_from_it() {
    // Live: entry -> head <-> body (the back edge tagged), head -> exit. Dead:
    // `source -> top <-> side`, and `side` jumps into the live body, which
    // takes the entry away from `body` as dominator in the whole-graph forest.
    let mut cfg = Cfg::<u32>::new();
    let [head, body, exit, source, top, side] = [(); 6].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), head, EdgeKind::Fallthrough);
    cfg.add_edge(head, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(head, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, head, EdgeKind::Back);
    cfg.add_edge(source, top, EdgeKind::Fallthrough);
    cfg.add_edge(top, side, EdgeKind::ConditionalTrue);
    cfg.add_edge(side, top, EdgeKind::Fallthrough);
    cfg.add_edge(side, body, EdgeKind::Jump);
    let dom = DominatorTree::compute(&cfg);

    let loops = detect_all_loops_tagged(&cfg, &dom);

    let find = |header| loops.iter().find(|lp| lp.header == header);
    let live = find(head).expect("the live loop is still found");
    assert_eq!(live.body, BTreeSet::from([head, body]));
    assert_eq!(live.depth, 0);
    let dead = find(top).expect("the dead loop is found");
    assert_eq!(dead.body, BTreeSet::from([top, side]));
    assert_eq!(dead.latches, BTreeSet::from([side]));
    assert_eq!(dead.depth, 0, "the live loop does not contain it");
    assert_eq!(loops.len(), 2);
}

/// On random graphs with dead code, loops of every closing style and edges
/// from dead code into live code: the live loops are exactly those of
/// [`detect_loops_tagged`]; a dead loop's header, latches and body are all dead
/// blocks; and a dead loop's depth counts the dead loops around it and no live
/// one.
#[test]
fn the_loops_of_dead_code_hold_only_dead_blocks_on_random_graphs() {
    let kinds = [
        EdgeKind::Fallthrough,
        EdgeKind::Back,
        EdgeKind::ConditionalTrue,
        EdgeKind::Jump,
        EdgeKind::Back,
    ];
    let mut with_a_dead_loop = 0;
    for seed in 0..4_000_u64 {
        let mut rng = crate::test_util::rng::Rng::new(seed);
        let count = 4 + rng.below(9);
        let mut cfg = Cfg::<u32>::new();
        let mut blocks = vec![cfg.entry()];
        blocks.extend((1..count).map(|_| cfg.new_block()));
        for _ in 0..count + rng.below(count * 2) {
            let (from, to) = (rng.below(count), rng.below(count));
            cfg.add_edge(blocks[from], blocks[to], kinds[rng.below(kinds.len())]);
        }
        let dom = DominatorTree::compute(&cfg);
        let all = detect_all_loops_tagged(&cfg, &dom);
        let label = alloc::format!("seed {seed}");

        let live: Vec<_> = all
            .iter()
            .filter(|lp| dom.is_reachable(lp.header))
            .cloned()
            .collect();
        assert_eq!(live, detect_loops_tagged(&cfg, &dom), "{label}");

        let dead: Vec<_> = all
            .iter()
            .filter(|lp| !dom.is_reachable(lp.header))
            .cloned()
            .collect();
        with_a_dead_loop += usize::from(!dead.is_empty());
        for lp in &dead {
            assert!(
                lp.body.iter().all(|&block| !dom.is_reachable(block)),
                "{label}"
            );
            assert!(
                lp.latches.iter().all(|&block| !dom.is_reachable(block)),
                "{label}"
            );
        }
        // `nesting` breaks a tie between equally large enclosing loops by list
        // order, and detection lists loops by header.
        let mut by_header = dead.clone();
        by_header.sort_by_key(|lp| lp.header);
        let depths = crate::graph::loop_nest::nesting(&by_header).depths;
        for (lp, depth) in by_header.iter().zip(depths) {
            let listed = dead.iter().find(|other| other.header == lp.header);
            assert_eq!(
                listed.map(|other| other.depth),
                Some(depth),
                "dead loop at {:?}, {label}",
                lp.header
            );
        }
    }
    assert!(
        with_a_dead_loop > 1_000,
        "only {with_a_dead_loop} graphs had a dead loop"
    );
}
