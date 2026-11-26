//! The series-parallel-loop decomposition against the grammar's CFG
//! construction and the definition of a path decomposition, over the
//! control-tree fixtures and over each construct the grammar lacks.

extern crate alloc;
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use super::exhaustive::{Terminator, for_each_graph, graph, terminators};
use super::small_stack::on_small_stack;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::traverse::reachable;
use crate::graph::view::Direction;
use crate::region::{Handler, HandlerBody, HandlerKind, Region};
use crate::{ControlNode, ControlTree, LoopKind, SplError, SplNode};
use checker::{check_decomposition, check_path};

mod checker;

/// What the decomposition of `cfg` must give, read from the tree and the
/// graph alone: `None` for a decomposition, or the first construct in
/// preorder that the grammar lacks, then a block the entry does not reach,
/// then one of several exits.
fn expected<I>(cfg: &Cfg<I>, tree: &ControlTree) -> Option<SplError> {
    let root = tree.root();
    let last = match *tree.node(root) {
        ControlNode::Sequence(body) => tree.body(body).last().copied(),
        _ => Some(root),
    };
    for id in tree.preorder() {
        let found = match *tree.node(id) {
            ControlNode::Goto { .. } => Some(SplError::Goto(id)),
            ControlNode::TryCatch { .. } => Some(SplError::TryCatch(id)),
            ControlNode::Label { block, body } => {
                let names_its_loop = matches!(
                    tree.body(body),
                    [only] if matches!(*tree.node(*only), ControlNode::Loop { header, .. } if header == block)
                );
                (!names_its_loop).then_some(SplError::Label(id))
            }
            ControlNode::Break { label: Some(_) } | ControlNode::Continue { label: Some(_) } => {
                Some(SplError::MultiLevelJump(id))
            }
            ControlNode::Return(span) if Some(id) != last => {
                Some(SplError::EarlyReturn(span.block()))
            }
            _ => None,
        };
        if found.is_some() {
            return found;
        }
    }
    let reached = reachable(cfg, [cfg.entry()], Direction::Forward);
    if let Some(dead) = cfg.block_ids().find(|block| !reached[block.index()]) {
        return Some(SplError::UnplacedBlock(dead));
    }
    let mut exits = cfg
        .block_ids()
        .filter(|&block| cfg.outgoing(block).next().is_none());
    exits.next()?;
    exits.next().map(SplError::EarlyReturn)
}

/// Whether the tree has a leaf, a test or a header for `block`.
fn names(tree: &ControlTree, block: BlockId) -> bool {
    tree.preorder().any(|id| match *tree.node(id) {
        ControlNode::Block(span) | ControlNode::Return(span) => span.block() == block,
        ControlNode::IfThenElse { condition, .. } | ControlNode::Switch { condition, .. } => {
            condition.block() == block
        }
        ControlNode::Loop { header, kind, .. } => {
            header == block
                || match kind {
                    LoopKind::While { condition, .. } => tree
                        .spans(condition)
                        .iter()
                        .any(|span| span.block() == block),
                    LoopKind::DoWhile { latch, .. } => latch.block() == block,
                    LoopKind::Endless => false,
                }
        }
        _ => false,
    })
}

/// Decompose `cfg` and check the result against [`expected`]: a decomposition
/// passes both checkers, and an error is the expected one. Returns whether
/// there is a decomposition.
///
/// The tree passes through a block without code, so where that block goes is
/// the decomposition's to find. An exit without code inside a loop or one arm
/// of a branch is an early return, as a `Return` there is, and a block
/// without code that two constructs enter, such as a tail two arms at
/// different depths share, has no place. Either is accepted only when
/// `refused_with_code`, the same graph with code in every block, has no
/// decomposition either: there the tree shows the goto or the return.
fn check<I>(cfg: &Cfg<I>, refused_with_code: &dyn Fn() -> bool, context: &str) -> bool {
    let tree = ControlTree::compute(cfg);
    match (expected(cfg, &tree), tree.spl_decomposition(cfg)) {
        (None, Ok(spl)) => {
            check_decomposition(&spl, cfg, context);
            check_path(&spl, cfg, context);
            true
        }
        (None, Err(SplError::EarlyReturn(exit))) => {
            assert!(
                !names(&tree, exit),
                "{context}: {exit} is the last statement"
            );
            assert!(
                cfg.outgoing(exit).next().is_none(),
                "{context}: {exit} exits"
            );
            assert!(refused_with_code(), "{context}: decomposes with code");
            false
        }
        (None, Err(SplError::UnplacedBlock(block))) => {
            assert!(!names(&tree, block), "{context}: {block} has a leaf");
            assert!(
                cfg.block(block).instructions().is_empty(),
                "{context}: {block} has no code"
            );
            assert!(
                cfg.incoming(block).nth(1).is_some(),
                "{context}: two edges enter {block}"
            );
            assert!(refused_with_code(), "{context}: decomposes with code");
            false
        }
        // Which of several exits is early depends on the decomposition.
        (Some(SplError::EarlyReturn(_)), Err(SplError::EarlyReturn(_))) => false,
        (Some(wanted), Err(found)) => {
            assert_eq!(found, wanted, "{context}");
            false
        }
        (wanted, found) => panic!("{context}: expected {wanted:?}, got {found:?}"),
    }
}

const ONE_SUCCESSOR: [EdgeKind; 4] = [
    EdgeKind::Fallthrough,
    EdgeKind::Unconditional,
    EdgeKind::Jump,
    EdgeKind::Back,
];

/// Check every graph of the given shape with every instruction count in
/// `0..=most` per block, and return how many decompose.
fn check_every_graph(count: usize, kinds: &[EdgeKind], switches: bool, most: usize) -> usize {
    let all = terminators(count, kinds, switches);
    let mut lengths = vec![0_usize; count];
    let mut decomposed = 0;
    loop {
        for_each_graph(count, &all, |picked: &[Terminator]| {
            let cfg = graph(picked, &lengths);
            let refused_with_code = || {
                let full = graph(picked, &vec![1; count]);
                ControlTree::compute(&full)
                    .spl_decomposition(&full)
                    .is_err()
            };
            let context = format!("{picked:?} with {lengths:?} instructions");
            if check(&cfg, &refused_with_code, &context) {
                decomposed += 1;
            }
        });
        let Some(digit) = lengths.iter().position(|&length| length < most) else {
            return decomposed;
        };
        lengths[digit] += 1;
        lengths[..digit].fill(0);
    }
}

#[test]
fn every_graph_of_up_to_three_blocks_decomposes_unless_an_obstacle_says_why() {
    let decomposed = (1..=3)
        .map(|count| check_every_graph(count, &ONE_SUCCESSOR, true, 1))
        .sum::<usize>();
    assert!(
        decomposed > 100_000,
        "most of these graphs are structured: {decomposed}"
    );
}

#[test]
fn every_graph_of_two_blocks_with_up_to_two_instructions_decomposes_unless_an_obstacle_says_why() {
    check_every_graph(1, &ONE_SUCCESSOR, true, 2);
    check_every_graph(2, &ONE_SUCCESSOR, true, 2);
}

#[test]
fn every_graph_of_four_blocks_decomposes_unless_an_obstacle_says_why() {
    check_every_graph(4, &[EdgeKind::Fallthrough, EdgeKind::Jump], false, 0);
}

/// A graph over `count` blocks that each hold an instruction, block 0 the
/// entry, with `edges` in order.
fn build(count: usize, edges: &[(usize, usize, EdgeKind)]) -> (Cfg<&'static str>, Vec<BlockId>) {
    let mut cfg = Cfg::new();
    let mut blocks = vec![cfg.entry()];
    blocks.extend((1..count).map(|_| cfg.new_block()));
    for &block in &blocks {
        cfg.block_mut(block).push("work");
    }
    for &(from, to, kind) in edges {
        cfg.add_edge(blocks[from], blocks[to], kind);
    }
    (cfg, blocks)
}

/// The decomposition of `cfg`, checked.
fn decompose(cfg: &Cfg<&'static str>) -> crate::SplDecomposition {
    let spl = ControlTree::compute(cfg)
        .spl_decomposition(cfg)
        .expect("the graph is structured");
    check_decomposition(&spl, cfg, "fixture");
    check_path(&spl, cfg, "fixture");
    spl
}

/// The kind of the one loop of `cfg`'s tree.
fn loop_kind(cfg: &Cfg<&'static str>) -> LoopKind {
    let tree = ControlTree::compute(cfg);
    tree.preorder()
        .find_map(|id| match *tree.node(id) {
            ControlNode::Loop { kind, .. } => Some(kind),
            _ => None,
        })
        .expect("the graph has a loop")
}

const T: EdgeKind = EdgeKind::ConditionalTrue;
const F: EdgeKind = EdgeKind::ConditionalFalse;
const FALL: EdgeKind = EdgeKind::Fallthrough;
const JUMP: EdgeKind = EdgeKind::Jump;

#[test]
fn a_straight_line_has_width_one() {
    let (cfg, blocks) = build(3, &[(0, 1, FALL), (1, 2, FALL)]);
    let spl = decompose(&cfg);
    assert_eq!(spl.blocks(), blocks);
    assert_eq!(spl.nesting_depth(), 0);
    assert_eq!(spl.path_decomposition().width(), 1);
}

#[test]
fn an_if_else_diamond_has_width_two() {
    let (cfg, blocks) = build(4, &[(0, 1, T), (0, 2, F), (1, 3, FALL), (2, 3, FALL)]);
    let spl = decompose(&cfg);
    assert_eq!(spl.nesting_depth(), 1);
    assert_eq!(spl.path_decomposition().width(), 2);
    let SplNode::Series { children } = *spl.node(spl.root()) else {
        panic!("a branch and its merge are a series");
    };
    let statements = spl.children(children);
    assert_eq!(*spl.node(statements[0]), SplNode::Block(blocks[0]));
    let SplNode::Parallel { branch, arms } = *spl.node(statements[1]) else {
        panic!("the branch's arms are a parallel");
    };
    assert_eq!(branch, blocks[0]);
    assert_eq!(arms.len(), 2);
}

#[test]
fn a_pre_tested_loop_tests_at_the_end_of_its_condition_chain() {
    // entry -> head -> test; test -> body -> head; test -> exit.
    let (cfg, blocks) = build(
        5,
        &[
            (0, 1, FALL),
            (1, 2, FALL),
            (2, 3, T),
            (2, 4, F),
            (3, 1, JUMP),
        ],
    );
    assert!(matches!(loop_kind(&cfg), LoopKind::While { condition, .. } if condition.len() == 2));
    let spl = decompose(&cfg);
    let header = spl
        .blocks()
        .iter()
        .position(|&block| block == blocks[1])
        .expect("the header has a leaf");
    assert_eq!(
        spl.blocks()[header + 1],
        blocks[2],
        "the chain is in the body"
    );
}

#[test]
fn a_post_tested_loop_tests_at_its_latch() {
    // entry -> head -> latch; latch -> head | exit.
    let (cfg, blocks) = build(4, &[(0, 1, FALL), (1, 2, FALL), (2, 1, T), (2, 3, F)]);
    assert!(matches!(loop_kind(&cfg), LoopKind::DoWhile { .. }));
    let spl = decompose(&cfg);
    assert_eq!(spl.blocks(), blocks);
}

#[test]
fn a_one_block_post_tested_loop_is_its_own_latch() {
    let (cfg, _) = build(3, &[(0, 1, FALL), (1, 1, T), (1, 2, F)]);
    assert!(matches!(loop_kind(&cfg), LoopKind::DoWhile { .. }));
    decompose(&cfg);
}

#[test]
fn an_endless_loop_leaves_only_through_its_breaks() {
    // entry -> head -> test; test -> exit (break) | body -> head.
    let (cfg, _) = build(
        5,
        &[
            (0, 1, FALL),
            (1, 2, FALL),
            (2, 4, T),
            (2, 3, F),
            (3, 1, JUMP),
        ],
    );
    decompose(&cfg);
    // And one that never leaves.
    let (spin, _) = build(3, &[(0, 1, FALL), (1, 2, FALL), (2, 1, JUMP)]);
    assert!(matches!(loop_kind(&spin), LoopKind::Endless));
    decompose(&spin);
}

#[test]
fn a_continue_and_a_break_in_a_loop_body() {
    // while (h) { if (a) continue; if (b) break; work }; exit
    let (cfg, _) = build(
        6,
        &[
            (0, 1, FALL),
            (1, 2, T),
            (1, 5, F),
            (2, 1, T),
            (2, 3, F),
            (3, 5, T),
            (3, 4, F),
            (4, 1, JUMP),
        ],
    );
    let spl = decompose(&cfg);
    let jumps: Vec<SplNode> = (0..spl.node_count())
        .map(|index| *spl.node(crate::SplNodeId::from_index(index)))
        .filter(|node| matches!(node, SplNode::Break | SplNode::Continue))
        .collect();
    assert!(jumps.contains(&SplNode::Break) && jumps.contains(&SplNode::Continue));
}

#[test]
fn blocks_without_code_get_leaves_where_flow_enters_them() {
    // entry -> empty -> branch; branch -> empty arm | arm; both -> empty merge
    // -> empty exit.
    let (mut cfg, blocks) = build(
        7,
        &[
            (0, 1, FALL),
            (1, 2, FALL),
            (2, 3, T),
            (2, 4, F),
            (3, 5, FALL),
            (4, 5, FALL),
            (5, 6, FALL),
        ],
    );
    for empty in [1, 3, 5, 6] {
        cfg.block_mut(blocks[empty]).instructions_mut().clear();
    }
    let spl = decompose(&cfg);
    assert_eq!(spl.blocks().last(), Some(&blocks[6]));
}

#[test]
fn a_switch_is_one_parallel_with_an_arm_per_target() {
    let (cfg, blocks) = build(
        5,
        &[
            (0, 1, EdgeKind::SwitchCase),
            (0, 2, EdgeKind::SwitchCase),
            (0, 3, FALL),
            (1, 4, FALL),
            (2, 4, FALL),
            (3, 4, FALL),
        ],
    );
    let spl = decompose(&cfg);
    let arms = spl
        .blocks()
        .iter()
        .filter(|block| [blocks[1], blocks[2], blocks[3]].contains(block))
        .count();
    assert_eq!(arms, 3);
    assert_eq!(spl.nesting_depth(), 1);
}

/// Decompose `cfg`, which has an obstacle, and return it after checking it
/// against [`expected`].
fn obstacle(cfg: &Cfg<&'static str>) -> SplError {
    let tree = ControlTree::compute(cfg);
    let found = tree
        .spl_decomposition(cfg)
        .expect_err("the graph has an obstacle");
    assert_eq!(Some(found), expected(cfg, &tree));
    found
}

/// The first node of `cfg`'s tree in preorder that `wanted` accepts.
fn first_node(
    cfg: &Cfg<&'static str>,
    wanted: impl Fn(&ControlNode) -> bool,
) -> crate::ControlNodeId {
    let tree = ControlTree::compute(cfg);
    tree.preorder()
        .find(|&id| wanted(tree.node(id)))
        .expect("the tree has the node")
}

#[test]
fn a_goto_has_no_decomposition() {
    // while (h) { if (b) goto y; }; x; y: return. The loop's second exit is
    // a goto ahead, which the label at `y` resolves.
    let (cfg, _) = build(
        5,
        &[
            (0, 1, FALL),
            (1, 2, T),
            (1, 3, F),
            (2, 1, T),
            (2, 4, F),
            (3, 4, FALL),
        ],
    );
    let goto = first_node(&cfg, |node| matches!(node, ControlNode::Goto { .. }));
    assert_eq!(obstacle(&cfg), SplError::Goto(goto));
}

#[test]
fn a_label_has_no_decomposition() {
    // Two entries into a cycle: entry -> a | b, a -> b, b -> a | exit. The
    // walk labels `a` before it jumps back to it.
    let (cfg, _) = build(
        4,
        &[(0, 1, T), (0, 2, F), (1, 2, FALL), (2, 1, T), (2, 3, F)],
    );
    let label = first_node(&cfg, |node| matches!(node, ControlNode::Label { .. }));
    assert_eq!(obstacle(&cfg), SplError::Label(label));
}

#[test]
fn a_try_region_has_no_decomposition() {
    let (mut cfg, blocks) = build(4, &[(0, 1, FALL), (1, 3, FALL), (2, 3, FALL)]);
    cfg.add_edge(blocks[1], blocks[2], EdgeKind::ExceptionHandler);
    cfg.add_region(Region {
        protected_blocks: [blocks[1]].into_iter().collect(),
        handlers: vec![Handler {
            entry: blocks[2],
            body: HandlerBody::known([blocks[2]]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    })
    .unwrap();
    assert!(matches!(obstacle(&cfg), SplError::TryCatch(_)));
}

#[test]
fn a_break_out_of_two_loops_has_no_decomposition() {
    // outer: while (o) { inner: while (i) { if (c) break outer; } }; exit
    let (cfg, _) = build(
        5,
        &[
            (0, 1, FALL),
            (1, 2, T),
            (1, 4, F),
            (2, 3, T),
            (2, 1, F),
            (3, 4, T),
            (3, 2, F),
        ],
    );
    assert!(matches!(obstacle(&cfg), SplError::MultiLevelJump(_)));
}

#[test]
fn a_return_inside_a_loop_has_no_decomposition() {
    // while (h) { if (c) return; } return
    let (cfg, blocks) = build(
        5,
        &[(0, 1, FALL), (1, 2, T), (1, 4, F), (2, 3, T), (2, 1, F)],
    );
    assert_eq!(obstacle(&cfg), SplError::EarlyReturn(blocks[3]));
}

#[test]
fn two_exit_blocks_have_no_decomposition() {
    let (cfg, _) = build(3, &[(0, 1, T), (0, 2, F)]);
    assert!(matches!(obstacle(&cfg), SplError::EarlyReturn(_)));
}

#[test]
fn a_block_the_entry_does_not_reach_has_no_leaf() {
    let (cfg, blocks) = build(3, &[(0, 1, FALL), (2, 1, FALL)]);
    assert_eq!(obstacle(&cfg), SplError::UnplacedBlock(blocks[2]));
}

/// The result for `edges` with no code in `empty`, and with code everywhere.
fn with_and_without_code(
    count: usize,
    edges: &[(usize, usize, EdgeKind)],
    empty: &[usize],
) -> (Result<(), SplError>, Result<(), SplError>, Vec<BlockId>) {
    let (full, blocks) = build(count, edges);
    let mut emptied = full.clone();
    for &block in empty {
        emptied.block_mut(blocks[block]).instructions_mut().clear();
    }
    let decompose = |cfg: &Cfg<&'static str>| {
        let tree = ControlTree::compute(cfg);
        assert_eq!(expected(cfg, &tree), None, "the tree has no obstacle");
        tree.spl_decomposition(cfg).map(|_| ())
    };
    let with_code = ControlTree::compute(&full)
        .spl_decomposition(&full)
        .map(|_| ());
    (decompose(&emptied), with_code, blocks)
}

#[test]
fn an_empty_tail_two_depths_share_has_no_place() {
    // if (a) { if (b) { tail } } else { tail }; exit, with `tail` empty: the
    // tree passes through it, but the grammar needs it in two arms.
    let (emptied, with_code, blocks) = with_and_without_code(
        4,
        &[(0, 1, T), (0, 2, F), (1, 2, T), (1, 3, F), (2, 3, FALL)],
        &[2],
    );
    assert_eq!(emptied, Err(SplError::UnplacedBlock(blocks[2])));
    assert!(with_code.is_err(), "with code, the tree has a goto");
}

#[test]
fn an_empty_exit_inside_a_loop_is_an_early_return() {
    // while (a) { if (b) continue; return; } spin forever, with the return
    // empty: the tree has no `Return`, but the exit is inside the loop.
    let (emptied, with_code, blocks) = with_and_without_code(
        4,
        &[(0, 1, T), (0, 2, F), (1, 0, T), (1, 3, F), (2, 2, FALL)],
        &[0, 1, 2, 3],
    );
    assert_eq!(emptied, Err(SplError::EarlyReturn(blocks[3])));
    assert_eq!(with_code, Err(SplError::EarlyReturn(blocks[3])));
}

#[test]
fn an_exception_edge_has_no_production() {
    // entry -> a -> b, and the entry unwinds into `b`, which no region
    // declares as a handler.
    let (mut cfg, blocks) = build(3, &[(0, 1, FALL), (1, 2, FALL)]);
    let unwind = cfg.add_edge(blocks[0], blocks[2], EdgeKind::ExceptionUnwind);
    let tree = ControlTree::compute(&cfg);
    assert_eq!(expected(&cfg, &tree), None, "the tree has no obstacle");
    assert_eq!(
        tree.spl_decomposition(&cfg),
        Err(SplError::UnplacedEdge(unwind))
    );
}

/// `if (t0) { if (t1) { ... leaf } }; merge`, `depth` tests deep.
fn nested_then_arms(depth: usize) -> Cfg<&'static str> {
    let mut edges = Vec::new();
    let (leaf, merge) = (depth, depth + 1);
    for test in 0..depth {
        let next = if test + 1 < depth { test + 1 } else { leaf };
        edges.push((test, next, T));
        edges.push((test, merge, F));
    }
    edges.push((leaf, merge, FALL));
    build(depth + 2, &edges).0
}

/// `while (h0) { while (h1) { ... work } }`, `depth` loops deep.
fn nested_loops(depth: usize) -> Cfg<&'static str> {
    let mut edges = vec![(0, 1, FALL)];
    let (work, exit) = (depth + 1, depth + 2);
    for level in 1..=depth {
        let body = if level < depth { level + 1 } else { work };
        let out = if level == 1 { exit } else { level - 1 };
        edges.push((level, body, T));
        edges.push((level, out, F));
    }
    edges.push((work, depth, JUMP));
    build(depth + 3, &edges).0
}

/// A graph of a nest `depth` levels deep.
type Nest = fn(usize) -> Cfg<&'static str>;

#[test]
fn a_hundred_thousand_nested_constructs_decompose_on_a_small_stack() {
    // The bags of a path decomposition hold O(n·d) blocks, quadratic in a
    // nest, so it is taken at a smaller depth.
    const DEPTH: usize = 100_000;
    const PATH_DEPTH: usize = 1_000;
    // Each shape with the blocks beyond its tests, and the nodes a level
    // nests: a branch, or a loop around its test's branch.
    let shapes: [(Nest, usize, usize); 2] = [(nested_then_arms, 2, 1), (nested_loops, 3, 2)];
    for (build, extra, per_level) in shapes {
        let (depth, blocks, width, path_depth) = on_small_stack(move || {
            let decompose = |depth| {
                let cfg = build(depth);
                ControlTree::compute(&cfg)
                    .spl_decomposition(&cfg)
                    .expect("the nest is structured")
            };
            let spl = decompose(DEPTH);
            let shallow = decompose(PATH_DEPTH);
            let width = shallow.path_decomposition().width();
            (
                spl.nesting_depth(),
                spl.blocks().len(),
                width,
                shallow.nesting_depth(),
            )
        });
        assert_eq!(depth, per_level * DEPTH);
        assert_eq!(blocks, DEPTH + extra);
        assert!(width <= 2 * path_depth + 1);
    }
}

#[test]
fn small_nests_pass_both_checkers() {
    for depth in 1..6 {
        decompose(&nested_then_arms(depth));
        decompose(&nested_loops(depth));
    }
}
