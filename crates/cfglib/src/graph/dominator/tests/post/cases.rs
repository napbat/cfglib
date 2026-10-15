//! The worked examples of exits and exit-free regions, and the cases that pin
//! one rule each.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::reference::{PostReference, assert_matches};
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::cdg::control_dependence_graph;
use crate::graph::dominator::tests::support::{EdgeList, node};
use crate::graph::dominator::tests::{BoundedDiamond, BoundedNode};
use crate::graph::dominator::{PostDominatorScratch, PostDominatorTree, PostNode};
use crate::graph::store::{Graph, NodeId};
use crate::graph::view::{GraphView, Rooted};

/// The index of the block called `name`.
fn index_of(names: &[&str], name: &str) -> u32 {
    let position = names
        .iter()
        .position(|&candidate| candidate == name)
        .unwrap_or_else(|| panic!("no block is called {name}"));
    node(position)
}

/// A graph over the blocks `names`, the first of which is the entry, with the
/// edges written by name.
fn graph_of(names: &[&str], edges: &[(&str, &str)]) -> EdgeList {
    let edges = edges
        .iter()
        .map(|&(from, to)| (index_of(names, from), index_of(names, to)))
        .collect();
    EdgeList::new(names.len(), edges)
}

/// What a worked example must produce. Blocks are listed by name, and `"X*"`
/// names the virtual exit.
struct Expected<'a> {
    exits: &'a [&'a str],
    cut_roots: &'a [&'a str],
    /// The immediate post-dominator of every block.
    ipdoms: &'a [(&'a str, &'a str)],
    /// The control dependences, as `(controller, dependent)`.
    dependences: &'a [(&'a str, &'a str)],
}

/// Compute the post-dominators of the example and check every row of it.
fn check_example(names: &[&str], edges: &[(&str, &str)], expected: &Expected<'_>) {
    let list = graph_of(names, edges);
    let graph = list.graph();
    let post = PostDominatorTree::compute(&graph);
    let ascending = |wanted: &[&str]| -> Vec<u32> {
        let mut blocks: Vec<u32> = wanted.iter().map(|&name| index_of(names, name)).collect();
        blocks.sort_unstable();
        blocks
    };

    assert_eq!(post.exits(), ascending(expected.exits), "exits");
    assert_eq!(post.cut_roots(), ascending(expected.cut_roots), "cut roots");

    assert_eq!(expected.ipdoms.len(), names.len(), "a row for every block");
    for &(block, parent) in expected.ipdoms {
        let parent = if parent == "X*" {
            PostNode::Exit
        } else {
            PostNode::Node(index_of(names, parent))
        };
        assert_eq!(
            post.ipdom(index_of(names, block)),
            Some(parent),
            "ipdom({block})"
        );
    }
    let roots: Vec<&str> = expected
        .ipdoms
        .iter()
        .filter(|&&(_, parent)| parent == "X*")
        .map(|&(block, _)| block)
        .collect();
    assert_eq!(
        post.children(PostNode::Exit).collect::<Vec<_>>(),
        ascending(&roots),
        "the children of the virtual exit"
    );

    let dependences = control_dependence_graph(&graph, &post);
    let mut actual: Vec<(&str, &str)> = dependences
        .edges()
        .map(|edge| {
            (
                names[*dependences.node(edge.source()) as usize],
                names[*dependences.node(edge.target()) as usize],
            )
        })
        .collect();
    actual.sort_unstable();
    let mut wanted = expected.dependences.to_vec();
    wanted.sort_unstable();
    assert_eq!(actual, wanted, "control dependences");

    assert_matches(&post, &PostReference::new(&graph), &list);
}

#[test]
fn a_server_loop_without_an_exit_is_cut_at_its_latch() {
    check_example(
        &["entry", "head", "ok", "err", "latch"],
        &[
            ("entry", "head"),
            ("head", "ok"),
            ("head", "err"),
            ("ok", "latch"),
            ("err", "latch"),
            ("latch", "head"),
        ],
        &Expected {
            exits: &[],
            cut_roots: &["latch"],
            ipdoms: &[
                ("entry", "head"),
                ("head", "latch"),
                ("ok", "latch"),
                ("err", "latch"),
                ("latch", "X*"),
            ],
            dependences: &[
                ("head", "ok"),
                ("head", "err"),
                ("latch", "head"),
                ("latch", "latch"),
            ],
        },
    );
}

#[test]
fn a_branch_into_a_spin_does_not_let_the_exit_post_dominate_it() {
    check_example(
        &["entry", "spin", "out"],
        &[("entry", "spin"), ("entry", "out"), ("spin", "spin")],
        &Expected {
            exits: &["out"],
            cut_roots: &["spin"],
            ipdoms: &[("entry", "X*"), ("spin", "X*"), ("out", "X*")],
            dependences: &[("entry", "spin"), ("entry", "out")],
        },
    );
}

#[test]
fn a_flattened_function_with_a_tamper_loop_keeps_its_dispatcher_joins() {
    check_example(
        &["entry", "disp", "c1", "c2", "c3", "loop_end", "tamper"],
        &[
            ("entry", "disp"),
            ("disp", "c1"),
            ("disp", "c2"),
            ("disp", "c3"),
            ("disp", "tamper"),
            ("c1", "loop_end"),
            ("c2", "loop_end"),
            ("loop_end", "disp"),
            ("tamper", "tamper"),
        ],
        &Expected {
            exits: &["c3"],
            cut_roots: &["tamper"],
            ipdoms: &[
                ("entry", "disp"),
                ("disp", "X*"),
                ("c1", "loop_end"),
                ("c2", "loop_end"),
                ("c3", "X*"),
                ("loop_end", "disp"),
                ("tamper", "X*"),
            ],
            dependences: &[
                ("disp", "c1"),
                ("disp", "c2"),
                ("disp", "c3"),
                ("disp", "tamper"),
                ("disp", "loop_end"),
                ("disp", "disp"),
            ],
        },
    );
}

#[test]
fn an_irreducible_loop_entered_twice_is_cut_at_both_entries() {
    check_example(
        &["entry", "a", "b"],
        &[("entry", "a"), ("entry", "b"), ("a", "b"), ("b", "a")],
        &Expected {
            exits: &[],
            cut_roots: &["a", "b"],
            ipdoms: &[("entry", "X*"), ("a", "X*"), ("b", "X*")],
            dependences: &[("entry", "a"), ("entry", "b"), ("a", "b"), ("b", "a")],
        },
    );
}

/// An outer loop around a branch and an inner loop, with no exit.
const NESTED_BLOCKS: [&str; 7] = ["entry", "h1", "a", "b", "h2", "h2b", "l1"];
const NESTED_EDGES: [(&str, &str); 9] = [
    ("entry", "h1"),
    ("h1", "a"),
    ("h1", "b"),
    ("a", "h2"),
    ("b", "h2"),
    ("h2", "h2b"),
    ("h2", "l1"),
    ("h2b", "h2"),
    ("l1", "h1"),
];

#[test]
fn nested_loops_without_an_exit_keep_both_follow_nodes() {
    check_example(
        &NESTED_BLOCKS,
        &NESTED_EDGES,
        &Expected {
            exits: &[],
            cut_roots: &["l1"],
            ipdoms: &[
                ("entry", "h1"),
                ("h1", "h2"),
                ("a", "h2"),
                ("b", "h2"),
                ("h2", "l1"),
                ("h2b", "h2"),
                ("l1", "X*"),
            ],
            dependences: &[
                ("h1", "a"),
                ("h1", "b"),
                ("h2", "h2b"),
                ("h2", "h2"),
                ("l1", "h1"),
                ("l1", "h2"),
                ("l1", "l1"),
            ],
        },
    );
}

#[test]
fn the_frontier_of_a_block_lists_its_control_dependence_sources() {
    let list = graph_of(&NESTED_BLOCKS, &NESTED_EDGES);
    let graph = list.graph();
    let post = PostDominatorTree::compute(&graph);
    let dependences = control_dependence_graph(&graph, &post);

    let mut sources = Vec::new();
    for (index, name) in NESTED_BLOCKS.iter().enumerate() {
        post.frontier(&graph, node(index), &mut sources);
        sources.sort_unstable();
        let mut controllers: Vec<u32> = dependences
            .edges()
            .filter(|edge| *dependences.node(edge.target()) == node(index))
            .map(|edge| *dependences.node(edge.source()))
            .collect();
        controllers.sort_unstable();
        assert_eq!(sources, controllers, "sources of {name}");
    }

    // `h2` is control dependent on itself, through its inner loop, and on the
    // latch; the frontier lists them in the tree's preorder, root first.
    post.frontier(&graph, index_of(&NESTED_BLOCKS, "h2"), &mut sources);
    assert_eq!(
        sources,
        [
            index_of(&NESTED_BLOCKS, "l1"),
            index_of(&NESTED_BLOCKS, "h2")
        ]
    );
}

#[test]
fn a_branch_between_two_kinds_of_exit_is_post_dominated_by_the_virtual_exit() {
    // `unreachable` and `tail_call`, then `exceptional_exit` and a terminator
    // without records: every kind of exit is a block without successors.
    for arms in [
        ["unreachable", "tail_call"],
        ["exceptional_exit", "terminator"],
    ] {
        check_example(
            &["branch", arms[0], arms[1]],
            &[("branch", arms[0]), ("branch", arms[1])],
            &Expected {
                exits: &arms,
                cut_roots: &[],
                ipdoms: &[("branch", "X*"), (arms[0], "X*"), (arms[1], "X*")],
                dependences: &[("branch", arms[0]), ("branch", arms[1])],
            },
        );
    }
}

#[test]
fn a_component_of_exit_free_blocks_that_leads_to_another_is_not_cut() {
    // {a, b} loops and then falls into the spin at `c`. Only the sink
    // component {c} is cut, and it is cut once.
    check_example(
        &["entry", "a", "b", "c"],
        &[
            ("entry", "a"),
            ("a", "b"),
            ("b", "a"),
            ("b", "c"),
            ("c", "c"),
        ],
        &Expected {
            exits: &[],
            cut_roots: &["c"],
            ipdoms: &[("entry", "a"), ("a", "b"), ("b", "c"), ("c", "X*")],
            dependences: &[("b", "a"), ("b", "b")],
        },
    );
}

#[test]
fn a_loop_through_the_entry_is_cut_at_its_latch() {
    check_example(
        &["entry", "body"],
        &[("entry", "body"), ("body", "entry")],
        &Expected {
            exits: &[],
            cut_roots: &["body"],
            ipdoms: &[("entry", "body"), ("body", "X*")],
            dependences: &[("body", "entry"), ("body", "body")],
        },
    );
}

#[test]
fn an_entry_that_only_spins_is_its_own_cut() {
    check_example(
        &["entry"],
        &[("entry", "entry")],
        &Expected {
            exits: &[],
            cut_roots: &["entry"],
            ipdoms: &[("entry", "X*")],
            dependences: &[],
        },
    );
    let post = PostDominatorTree::compute(&EdgeList::new(1, vec![(0, 0)]).graph());
    assert!(!post.reaches_exit(0));
}

#[test]
fn an_entry_without_successors_is_the_only_exit() {
    let post = PostDominatorTree::compute(&EdgeList::new(1, Vec::new()).graph());
    assert_eq!(post.exits(), [0]);
    assert_eq!(post.cut_roots(), [] as [u32; 0]);
    assert_eq!(post.ipdom(0), Some(PostNode::Exit));
    assert!(post.reaches_exit(0));
    assert_eq!(post.preorder(), [0]);
}

#[test]
fn blocks_the_entry_cannot_reach_are_not_in_the_tree() {
    // `island` branches to blocks the entry reaches, and `island_exit` is an
    // exit nothing reaches: neither is an exit of the graph or a fact of it.
    let names = ["entry", "a", "island", "island_exit"];
    let list = graph_of(
        &names,
        &[("entry", "a"), ("island", "a"), ("island", "island_exit")],
    );
    let post = PostDominatorTree::compute(&list.graph());
    let (entry, a, island, island_exit) = (0, 1, 2, 3);

    assert_eq!(post.exits(), [a]);
    assert_eq!(post.preorder(), [a, entry]);
    for outside in [island, island_exit] {
        assert!(!post.contains(outside));
        assert_eq!(post.ipdom(outside), None);
        assert!(!post.reaches_exit(outside));
        assert_eq!(post.subtree(outside), [] as [u32; 0]);
        assert_eq!(post.children(PostNode::Node(outside)).count(), 0);
        assert!(!post.post_dominates(a, outside));
        assert!(!post.post_dominates(outside, a));
        assert!(post.post_dominates(outside, outside));
        assert!(!post.strictly_post_dominates(outside, outside));
        assert_eq!(post.nearest_common_post_dominator(a, outside), None);
        assert_eq!(post.nearest_common_post_dominator(outside, outside), None);
    }
    assert_eq!(
        post.nearest_common_post_dominator_of([island, entry, island_exit]),
        Some(PostNode::Node(entry))
    );
    assert_eq!(
        post.nearest_common_post_dominator_of([island, island_exit]),
        None
    );
    assert_eq!(post.nearest_common_post_dominator_of([]), None);
}

#[test]
fn the_nearest_common_post_dominator_of_separate_trees_is_the_virtual_exit() {
    let names = ["entry", "spin", "out"];
    let list = graph_of(
        &names,
        &[("entry", "spin"), ("entry", "out"), ("spin", "spin")],
    );
    let post = PostDominatorTree::compute(&list.graph());
    let (entry, spin, out) = (0, 1, 2);

    assert!(!post.post_dominates(spin, out));
    assert!(!post.post_dominates(out, spin));
    assert_eq!(
        post.nearest_common_post_dominator(spin, out),
        Some(PostNode::Exit)
    );
    assert_eq!(
        post.nearest_common_post_dominator(entry, spin),
        Some(PostNode::Exit)
    );
    assert_eq!(
        post.nearest_common_post_dominator_of([spin, out]),
        Some(PostNode::Exit)
    );
    assert_eq!(
        post.nearest_common_post_dominator_of([spin, spin]),
        Some(PostNode::Node(spin))
    );
    assert_eq!(
        post.nearest_common_post_dominator_of([out, spin, entry]),
        Some(PostNode::Exit),
        "the virtual exit stays the answer however many blocks follow"
    );
}

#[test]
fn a_consumer_identity_that_cannot_name_the_virtual_exit_is_enough() {
    let post = PostDominatorTree::compute(&Rooted::new(&BoundedDiamond, BoundedNode(0)));
    for block in 0..3 {
        assert_eq!(
            post.ipdom(BoundedNode(block)),
            Some(PostNode::Node(BoundedNode(3)))
        );
    }
    assert_eq!(post.ipdom(BoundedNode(3)), Some(PostNode::Exit));
    assert!(post.post_dominates(BoundedNode(3), BoundedNode(0)));
    assert_eq!(post.exits(), [BoundedNode(3)]);
}

#[test]
fn a_diamond_in_consumer_storage_is_post_dominated_by_its_merge() {
    let mut graph = Graph::<&str, ()>::new();
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    let c = graph.add_node("c");
    let d = graph.add_node("d");
    graph.add_edge(a, b, ());
    graph.add_edge(a, c, ());
    graph.add_edge(b, d, ());
    graph.add_edge(c, d, ());

    let post = PostDominatorTree::compute(&Rooted::new(&graph, a));
    assert_eq!(post.ipdom(a), Some(PostNode::Node(d)));
    assert_eq!(post.ipdom(b), Some(PostNode::Node(d)));
    assert_eq!(post.ipdom(c), Some(PostNode::Node(d)));
    assert_eq!(post.ipdom(d), Some(PostNode::Exit));
    assert!(post.post_dominates(d, a), "d post-dominates the entry");
    assert_eq!(
        post.children(PostNode::Node(d)).collect::<Vec<_>>(),
        [a, b, c]
    );
}

#[test]
fn a_root_other_than_the_first_block_is_honored() {
    // Block 0 is the entry of nothing here: rooting at 1 leaves it outside.
    let list = EdgeList::new(3, vec![(0, 1), (1, 2)]);
    let post = PostDominatorTree::compute(&Rooted::new(&list.graph(), 1));
    assert!(!post.contains(0));
    assert_eq!(post.exits(), [2]);
    assert_eq!(post.ipdom(1), Some(PostNode::Node(2)));
    assert_eq!(post.preorder(), [2, 1]);
}

#[test]
#[should_panic(expected = "post-dominator root is not a node of the graph")]
fn a_root_outside_the_graph_is_rejected() {
    let mut graph = Graph::<(), ()>::new();
    graph.add_node(());
    let _ = PostDominatorTree::compute(&Rooted::new(&graph, NodeId::from_raw(1)));
}

/// A view that claims `bound` nodes and has no edges.
struct Claimed(usize);

impl GraphView for Claimed {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        self.0
    }

    fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
        core::iter::once(0)
    }

    fn successors(&self, _node: usize) -> impl Iterator<Item = usize> + '_ {
        core::iter::empty()
    }

    fn predecessors(&self, _node: usize) -> impl Iterator<Item = usize> + '_ {
        core::iter::empty()
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "node bound plus the virtual exit exceeds the dense u32 range")]
fn a_node_bound_that_leaves_no_index_for_the_virtual_exit_is_rejected() {
    let bound = usize::try_from(u32::MAX).expect("u32 fits a 64-bit usize");
    let _ = PostDominatorTree::compute(&Rooted::new(&Claimed(bound), 0));
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "node bound plus the virtual exit exceeds the dense u32 range")]
fn a_node_bound_beyond_u32_is_rejected() {
    let bound = usize::try_from(u32::MAX).expect("u32 fits a 64-bit usize") + 1;
    let _ = PostDominatorTree::compute(&Rooted::new(&Claimed(bound), 0));
}

#[test]
fn a_chain_of_a_hundred_thousand_blocks_needs_no_recursion() {
    // Every traversal on the way, the cut's searches, the component
    // decomposition of the exit-free chain, the construction, and the
    // numbering, keeps its state on the heap.
    const LENGTH: usize = 100_000;
    let mut edges: Vec<(u32, u32)> = (0..LENGTH - 1)
        .map(|block| (node(block), node(block + 1)))
        .collect();

    let ending_in_an_exit = EdgeList::new(LENGTH, edges.clone()).graph();
    let post = PostDominatorTree::compute(&ending_in_an_exit);
    assert_eq!(post.exits(), [node(LENGTH - 1)]);
    assert_eq!(post.cut_roots(), [] as [u32; 0]);
    assert_eq!(post.ipdom(0), Some(PostNode::Node(1)));
    assert_eq!(post.subtree(node(LENGTH - 1)).len(), LENGTH);

    edges.push((node(LENGTH - 1), node(LENGTH - 1)));
    let ending_in_a_spin = EdgeList::new(LENGTH, edges).graph();
    let post = PostDominatorTree::compute(&ending_in_a_spin);
    assert_eq!(post.exits(), [] as [u32; 0]);
    assert_eq!(post.cut_roots(), [node(LENGTH - 1)]);
    assert_eq!(post.ipdom(node(LENGTH - 1)), Some(PostNode::Exit));
    assert_eq!(post.ipdom(0), Some(PostNode::Node(1)));
    assert!(!post.reaches_exit(0));
}

#[test]
fn one_scratch_reused_down_a_sequence_computes_what_a_fresh_one_does() {
    // Large first, so a stale entry left beyond a later graph's node bound
    // must not reach its answer.
    let sequence = [
        EdgeList::new(
            7,
            vec![
                (0, 1),
                (1, 2),
                (2, 1),
                (2, 3),
                (3, 4),
                (4, 3),
                (4, 5),
                (0, 6),
            ],
        ),
        EdgeList::new(2, vec![(0, 1), (1, 1)]),
        EdgeList::new(2, vec![(0, 1)]),
    ];
    let mut scratch = PostDominatorScratch::new();
    for _ in 0..2 {
        for edges in &sequence {
            let graph = edges.graph();
            assert_eq!(
                PostDominatorTree::compute_in(&mut scratch, &graph),
                PostDominatorTree::compute(&graph),
                "{edges:?}"
            );
        }
    }
}

#[test]
fn a_scratch_crosses_node_id_types_and_threads() {
    fn assert_send<T: Send>() {}
    assert_send::<PostDominatorScratch>();

    // One scratch, two graphs whose node identities are unrelated types.
    let mut cfg: Cfg<()> = Cfg::new();
    let exit = cfg.new_block();
    cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);
    let mut graph = Graph::<(), ()>::new();
    let first = graph.add_node(());
    let second = graph.add_node(());
    graph.add_edge(first, second, ());
    let rooted = Rooted::new(&graph, first);

    let mut scratch = PostDominatorScratch::new();
    assert_eq!(
        PostDominatorTree::compute_in(&mut scratch, &cfg),
        PostDominatorTree::compute(&cfg)
    );
    assert_eq!(
        PostDominatorTree::compute_in(&mut scratch, &rooted),
        PostDominatorTree::compute(&rooted)
    );
    assert_eq!(
        PostDominatorTree::compute_in(&mut scratch, &cfg),
        PostDominatorTree::compute(&cfg)
    );
}
