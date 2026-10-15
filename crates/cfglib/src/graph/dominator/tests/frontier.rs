//! Frontier queries against the definition of a dominance frontier.
//!
//! The definition is evaluated edge by edge over a tree whose dominance comes
//! from somewhere else: the brute-force reference on small graphs, the
//! iterative oracle on large ones. The iterated frontier is compared with the
//! fixpoint of that definition, in which a node the live-in test rejects is
//! neither added nor explored.

extern crate alloc;
extern crate std;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::{OnceCell, RefCell};
use core::fmt::Debug;

use super::support::{
    EdgeList, Family, Recipe, Reference, Rng, assert_forest_invariants, for_each_multigraph, node,
    random_graph,
};
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::dominator::{DominatorScratch, DominatorTree, FrontierScratch};
use crate::graph::view::{GraphView, Reversed};

/// Where the oracle reads dominance from.
trait Dominance {
    /// Whether `member` lies in a tree.
    fn in_forest(&self, member: u32) -> bool;
    /// Whether `dominator` dominates `member`, a node dominating itself.
    fn is_dominator(&self, dominator: u32, member: u32) -> bool;
}

impl Dominance for Reference {
    fn in_forest(&self, member: u32) -> bool {
        self.is_reachable(member as usize)
    }

    fn is_dominator(&self, dominator: u32, member: u32) -> bool {
        self.dominates(dominator as usize, member as usize)
    }
}

impl Dominance for DominatorTree<u32> {
    fn in_forest(&self, member: u32) -> bool {
        self.is_reachable(member)
    }

    fn is_dominator(&self, dominator: u32, member: u32) -> bool {
        self.dominates(dominator, member)
    }
}

/// The frontiers of a graph by definition: `y` is in the frontier of `x` when
/// `x` dominates a predecessor of `y` that lies in a tree and does not
/// strictly dominate `y`.
struct Oracle<'a, D> {
    dominance: &'a D,
    edges: &'a [(u32, u32)],
    in_forest: Vec<bool>,
    /// The frontier of each node, ascending, once somebody asks for it.
    frontiers: Vec<OnceCell<Vec<u32>>>,
}

impl<'a, D: Dominance> Oracle<'a, D> {
    fn new(dominance: &'a D, edges: &'a [(u32, u32)], node_count: usize) -> Self {
        Self {
            dominance,
            edges,
            in_forest: (0..node(node_count))
                .map(|member| dominance.in_forest(member))
                .collect(),
            frontiers: (0..node_count).map(|_| OnceCell::new()).collect(),
        }
    }

    /// The frontier of `x`, ascending.
    fn frontier(&self, x: u32) -> &[u32] {
        self.frontiers[x as usize].get_or_init(|| {
            if !self.in_forest[x as usize] {
                return Vec::new();
            }
            let mut found: Vec<u32> = self
                .edges
                .iter()
                .filter(|&&(from, to)| {
                    self.in_forest[from as usize]
                        && self.in_forest[to as usize]
                        && self.dominance.is_dominator(x, from)
                        && !(x != to && self.dominance.is_dominator(x, to))
                })
                .map(|&(_, to)| to)
                .collect();
            found.sort_unstable();
            found.dedup();
            found
        })
    }

    /// The least set that holds the frontier of every definition and of every
    /// member it admits, ascending. `live_in` admits a node or turns it away,
    /// and a definition is explored whichever it does.
    fn iterated(&self, defs: &[u32], live_in: Option<&dyn Fn(u32) -> bool>) -> Vec<u32> {
        let mut admitted = vec![false; self.in_forest.len()];
        let mut explored = vec![false; self.in_forest.len()];
        let mut pending: Vec<u32> = defs
            .iter()
            .copied()
            .filter(|&def| self.in_forest[def as usize])
            .collect();
        while let Some(x) = pending.pop() {
            if core::mem::replace(&mut explored[x as usize], true) {
                continue;
            }
            for &y in self.frontier(x) {
                if live_in.is_some_and(|accepts| !accepts(y)) {
                    continue;
                }
                admitted[y as usize] = true;
                pending.push(y);
            }
        }
        (0..node(admitted.len()))
            .filter(|&member| admitted[member as usize])
            .collect()
    }
}

/// A tree, the view it was computed over, and what the definition says.
struct Subject<'a, G, D> {
    view: &'a G,
    tree: &'a DominatorTree<u32>,
    oracle: Oracle<'a, D>,
    /// Preorder position of each node of the tree.
    position: Vec<usize>,
    /// What a failure names.
    context: &'a dyn Debug,
}

impl<'a, G: GraphView<NodeId = u32>, D: Dominance> Subject<'a, G, D> {
    fn new(
        view: &'a G,
        tree: &'a DominatorTree<u32>,
        dominance: &'a D,
        edges: &'a [(u32, u32)],
        context: &'a dyn Debug,
    ) -> Self {
        let node_count = view.node_bound();
        let mut position = vec![usize::MAX; node_count];
        for (index, &member) in tree.preorder().iter().enumerate() {
            position[member as usize] = index;
        }
        Self {
            view,
            tree,
            oracle: Oracle::new(dominance, edges, node_count),
            position,
            context,
        }
    }

    fn node_count(&self) -> usize {
        self.position.len()
    }

    /// Nodes of the tree in the order the tree numbers them.
    fn in_preorder(&self, mut members: Vec<u32>) -> Vec<u32> {
        members.sort_unstable_by_key(|&member| self.position[member as usize]);
        members
    }

    fn check_frontier(&self, member: u32, out: &mut Vec<u32>) {
        self.tree.frontier(self.view, member, out);
        let expected = self.in_preorder(self.oracle.frontier(member).to_vec());
        assert_eq!(*out, expected, "frontier of {member} in {:?}", self.context);
    }

    /// Check one iterated query, and that no node is offered to the live-in
    /// test twice.
    fn check_iterated(
        &self,
        scratch: &mut FrontierScratch,
        defs: &[u32],
        live_in: Option<&dyn Fn(u32) -> bool>,
        out: &mut Vec<u32>,
    ) {
        let offers = RefCell::new(vec![0_u32; self.node_count()]);
        let counted = |member: u32| {
            offers.borrow_mut()[member as usize] += 1;
            live_in.is_none_or(|accepts| accepts(member))
        };
        let counting: &dyn Fn(u32) -> bool = &counted;
        self.tree.iterated_frontier(
            self.view,
            scratch,
            defs.iter().copied(),
            live_in.map(|_| counting),
            out,
        );
        let expected = self.in_preorder(self.oracle.iterated(defs, live_in));
        let pruned = live_in.is_some();
        assert_eq!(
            *out, expected,
            "iterated frontier of {defs:?}, pruned {pruned}, in {:?}",
            self.context
        );
        assert!(
            offers.borrow().iter().all(|&offered| offered <= 1),
            "a node was offered to live-in twice: {defs:?} in {:?}",
            self.context
        );
    }

    /// Check the frontier of every node and the iterated frontier of every set
    /// of nodes, plain and pruned. Up to three nodes the pruned queries use
    /// every live-in set, and beyond that one chosen at random.
    fn check_every_query(&self, scratch: &mut FrontierScratch, rng: &mut Rng) {
        let count = self.node_count();
        let mut out = Vec::new();
        for member in 0..node(count) {
            self.check_frontier(member, &mut out);
        }
        for set in 0..1_usize << count {
            let defs: Vec<u32> = (0..node(count))
                .filter(|&member| (set >> member) & 1 == 1)
                .collect();
            self.check_iterated(scratch, &defs, None, &mut out);
            let masks = if count <= 3 {
                0..1_usize << count
            } else {
                let mask = rng.below(1 << count);
                mask..mask + 1
            };
            for mask in masks {
                let live_in = |member: u32| (mask >> member) & 1 == 1;
                self.check_iterated(scratch, &defs, Some(&live_in), &mut out);
            }
        }
    }

    /// Check a spread of frontiers and random definition sets, which may name
    /// nodes outside the tree and the same node twice.
    fn check_sampled(&self, scratch: &mut FrontierScratch, rng: &mut Rng) {
        let count = self.node_count();
        let mut out = Vec::new();
        for member in (0..node(count)).step_by((count / 150).max(1)) {
            self.check_frontier(member, &mut out);
        }
        for _ in 0..12 {
            let defs: Vec<u32> = (0..=rng.below(6)).map(|_| node(rng.below(count))).collect();
            self.check_iterated(scratch, &defs, None, &mut out);
            let live: Vec<bool> = (0..count).map(|_| rng.below(4) != 0).collect();
            let live_in = |member: u32| live[member as usize];
            self.check_iterated(scratch, &defs, Some(&live_in), &mut out);
        }
    }
}

/// The expected answer of a query whose frontier is empty.
const NOTHING: [u32; 0] = [];

/// A view over `node_count` nodes with the given edges.
fn digraph(node_count: usize, edges: &[(u32, u32)]) -> EdgeList {
    EdgeList::new(node_count, edges.to_vec())
}

/// The frontier of `member` in `tree`, over `view`.
fn frontier_of<G: GraphView<NodeId = u32>>(
    tree: &DominatorTree<u32>,
    view: &G,
    member: u32,
) -> Vec<u32> {
    let mut out = Vec::new();
    tree.frontier(view, member, &mut out);
    out
}

/// The iterated frontier of `defs`, plain or pruned by `live_in`.
fn iterated_of<G: GraphView<NodeId = u32>>(
    tree: &DominatorTree<u32>,
    view: &G,
    defs: &[u32],
    live_in: Option<&dyn Fn(u32) -> bool>,
) -> Vec<u32> {
    let mut out = Vec::new();
    let mut scratch = FrontierScratch::new();
    tree.iterated_frontier(view, &mut scratch, defs.iter().copied(), live_in, &mut out);
    out
}

#[test]
fn every_digraph_of_up_to_four_nodes_matches_the_definition() {
    let mut scratch = FrontierScratch::new();
    let mut dominators = DominatorScratch::new();
    let mut rng = Rng::new(4);
    for node_count in 1..=4 {
        for_each_multigraph(node_count, 1, |edges| {
            let graph = edges.graph();
            let tree = DominatorTree::compute_from_in(&mut dominators, &graph, 0);
            let reference = Reference::new(&graph, 0);
            let subject = Subject::new(&graph, &tree, &reference, &edges.edges, edges);
            subject.check_every_query(&mut scratch, &mut rng);
        });
    }
}

#[test]
fn every_multigraph_of_up_to_three_nodes_matches_the_definition() {
    let mut scratch = FrontierScratch::new();
    let mut rng = Rng::new(3);
    for node_count in 1..=3 {
        for_each_multigraph(node_count, 2, |edges| {
            let graph = edges.graph();
            let tree = DominatorTree::compute_from(&graph, 0);
            let reference = Reference::new(&graph, 0);
            let subject = Subject::new(&graph, &tree, &reference, &edges.edges, edges);
            subject.check_every_query(&mut scratch, &mut rng);
        });
    }
}

#[test]
fn an_entry_with_a_back_edge_is_in_its_own_frontier() {
    let two_cycle = digraph(2, &[(0, 1), (1, 0)]).graph();
    let tree = DominatorTree::compute_from(&two_cycle, 0);
    assert_eq!(frontier_of(&tree, &two_cycle, 0), [0]);
    assert_eq!(frontier_of(&tree, &two_cycle, 1), [0]);
    assert_eq!(iterated_of(&tree, &two_cycle, &[1], None), [0]);
    assert_eq!(iterated_of(&tree, &two_cycle, &[0], None), [0]);

    let self_loop = digraph(1, &[(0, 0)]).graph();
    let tree = DominatorTree::compute_from(&self_loop, 0);
    assert_eq!(frontier_of(&tree, &self_loop, 0), [0]);

    let inner_loop = digraph(3, &[(0, 1), (1, 2), (2, 1)]).graph();
    let tree = DominatorTree::compute_from(&inner_loop, 0);
    assert_eq!(frontier_of(&tree, &inner_loop, 0), NOTHING);
    assert_eq!(frontier_of(&tree, &inner_loop, 1), [1]);
    assert_eq!(frontier_of(&tree, &inner_loop, 2), [1]);

    let chain = digraph(3, &[(0, 1), (1, 2)]).graph();
    let tree = DominatorTree::compute_from(&chain, 0);
    assert_eq!(frontier_of(&tree, &chain, 0), NOTHING);
}

#[test]
fn a_rejected_node_is_not_explored_and_a_definition_is_always_a_root() {
    // Two nested diamonds: the arms 2 and 3 merge at 4, and the branches 1 and
    // 5 merge at 6.
    let edges = [
        (0, 1),
        (0, 5),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 4),
        (4, 6),
        (5, 6),
    ];
    let graph = digraph(7, &edges).graph();
    let tree = DominatorTree::compute_from(&graph, 0);
    assert_eq!(iterated_of(&tree, &graph, &[2], None), [4, 6]);
    assert_eq!(iterated_of(&tree, &graph, &[3, 2], None), [4, 6]);

    let not_four = |member: u32| member != 4;
    assert_eq!(iterated_of(&tree, &graph, &[2], Some(&not_four)), NOTHING);
    let not_six = |member: u32| member != 6;
    assert_eq!(iterated_of(&tree, &graph, &[2], Some(&not_six)), [4]);
    assert_eq!(iterated_of(&tree, &graph, &[2, 4], Some(&not_four)), [6]);
}

#[test]
fn the_reversed_view_gives_control_dependence() {
    // entry 0 -> header 1 -> body 2 -> header, header 1 -> exit 3.
    let graph = digraph(4, &[(0, 1), (1, 2), (2, 1), (1, 3)]).graph();
    let view = Reversed::new(&graph);
    let tree = DominatorTree::compute_from(&view, 3);
    assert_eq!(
        frontier_of(&tree, &view, 2),
        [1],
        "the body follows the header"
    );
    assert_eq!(
        frontier_of(&tree, &view, 1),
        [1],
        "a loop header controls itself"
    );
    assert_eq!(frontier_of(&tree, &view, 0), NOTHING);
    assert_eq!(frontier_of(&tree, &view, 3), NOTHING);
    assert_eq!(iterated_of(&tree, &view, &[2], None), [1]);
    assert_eq!(iterated_of(&tree, &view, &[0], None), NOTHING);
}

#[test]
fn every_rooted_digraph_of_up_to_three_nodes_matches_the_definition_on_the_reversed_view() {
    let mut scratch = FrontierScratch::new();
    let mut dominators = DominatorScratch::new();
    let mut rng = Rng::new(5);
    for node_count in 1..=3 {
        for_each_multigraph(node_count, 1, |edges| {
            let graph = edges.graph();
            let view = Reversed::new(&graph);
            let reversed = EdgeList::new(
                node_count,
                edges.edges.iter().map(|&(from, to)| (to, from)).collect(),
            );
            let reversed_graph = reversed.graph();
            for root in 0..node_count {
                let tree = DominatorTree::compute_from_in(&mut dominators, &view, node(root));
                let reference = Reference::new(&reversed_graph, root);
                let context = (edges, root);
                let subject = Subject::new(&view, &tree, &reference, &reversed.edges, &context);
                subject.check_every_query(&mut scratch, &mut rng);
            }
        });
    }
}

/// Eight nodes in two trees. Node 2 roots a tree with the child 6, and node 5
/// roots another with the children 1 and 4, which has the child 3. Nodes 0 and
/// 7 are in no tree. Every edge between the trees ends at a root, as in any
/// forest numbered from a virtual root.
fn two_trees() -> (DominatorTree<u32>, EdgeList) {
    let parents = [None, Some(5), None, Some(4), Some(5), None, Some(2), None];
    let reachable = [false, true, true, true, true, true, true, false];
    let forest = DominatorTree::from_forest_parts(&parents, &reachable);
    let edges = digraph(
        8,
        &[
            (2, 6),
            (5, 1),
            (5, 4),
            (4, 3),
            (6, 5),
            (3, 1),
            (1, 4),
            (4, 2),
            (5, 7),
            (0, 5),
        ],
    );
    (forest, edges)
}

#[test]
fn a_forest_has_frontier_members_in_other_trees_and_ignores_nodes_outside() {
    let (forest, edges) = two_trees();
    let graph = edges.graph();
    assert_forest_invariants(&forest, 8);
    assert_eq!(forest.preorder(), [2, 6, 5, 1, 4, 3]);

    assert_eq!(frontier_of(&forest, &graph, 2), [5]);
    assert_eq!(frontier_of(&forest, &graph, 6), [5]);
    assert_eq!(frontier_of(&forest, &graph, 5), [2]);
    assert_eq!(
        frontier_of(&forest, &graph, 4),
        [2, 1],
        "preorder, not index"
    );
    assert_eq!(frontier_of(&forest, &graph, 1), [4]);
    assert_eq!(frontier_of(&forest, &graph, 3), [1]);
    assert_eq!(frontier_of(&forest, &graph, 0), NOTHING);
    assert_eq!(frontier_of(&forest, &graph, 7), NOTHING);

    assert_eq!(iterated_of(&forest, &graph, &[3], None), [2, 5, 1, 4]);
    assert_eq!(iterated_of(&forest, &graph, &[6], None), [2, 5]);
    assert_eq!(iterated_of(&forest, &graph, &[3, 0, 7], None), [2, 5, 1, 4]);
    let not_four = |member: u32| member != 4;
    assert_eq!(iterated_of(&forest, &graph, &[3], Some(&not_four)), [1]);
}

/// The forest of `edges` numbered from a virtual root joined to node 0 and to
/// a random choice of other nodes, with the brute-force dominance of the graph
/// that includes the virtual root.
fn augmented_forest(edges: &EdgeList, rng: &mut Rng) -> (Reference, DominatorTree<u32>) {
    let count = edges.node_count;
    let virtual_root = node(count);
    let mut joined = edges.edges.clone();
    joined.push((virtual_root, 0));
    for member in 1..virtual_root {
        if rng.one_in(6) {
            joined.push((virtual_root, member));
        }
    }
    let reference = Reference::new(&EdgeList::new(count + 1, joined).graph(), count);
    let parents: Vec<Option<u32>> = (0..count)
        .map(|member| {
            reference
                .idom(member)
                .map(node)
                .filter(|&parent| parent != virtual_root)
        })
        .collect();
    let reachable: Vec<bool> = (0..count)
        .map(|member| reference.is_reachable(member))
        .collect();
    (
        reference,
        DominatorTree::from_forest_parts(&parents, &reachable),
    )
}

#[test]
fn random_forests_match_the_frontiers_of_the_graph_with_a_virtual_root() {
    let mut scratch = FrontierScratch::new();
    for seed in 0..150 {
        let mut rng = Rng::new(seed);
        let recipe = Recipe {
            family: Family::ALL[usize::try_from(seed % 3).expect("a residue fits in usize")],
            node_count: 6 + usize::try_from(seed % 19).expect("a residue fits in usize"),
            parallel: seed % 2 == 1,
            detached: usize::from(seed % 4 >= 2) * 5,
        };
        let edges = random_graph(&mut rng, recipe);
        let (reference, forest) = augmented_forest(&edges, &mut rng);
        assert_forest_invariants(&forest, edges.node_count);
        let graph = edges.graph();
        let context = (recipe, seed);
        let subject = Subject::new(&graph, &forest, &reference, &edges.edges, &context);
        subject.check_sampled(&mut scratch, &mut rng);
    }
}

/// Graph sizes and how many seeds to try at each.
const SWEEP: [(usize, u64); 5] = [(10, 120), (40, 40), (150, 20), (600, 4), (2_500, 2)];

#[test]
fn random_graphs_match_the_definition() {
    let mut scratch = FrontierScratch::new();
    for (node_count, seeds) in SWEEP {
        let size = u64::try_from(node_count).expect("a graph size fits in u64");
        for seed in 0..seeds {
            let mut rng = Rng::new(seed * 1_000_033 + size);
            let recipe = Recipe {
                family: Family::ALL[usize::try_from(seed % 3).expect("a residue fits in usize")],
                node_count,
                parallel: seed % 2 == 1,
                detached: if seed % 4 >= 2 { node_count / 4 } else { 0 },
            };
            let edges = random_graph(&mut rng, recipe);
            let graph = edges.graph();
            let tree = DominatorTree::compute_from(&graph, 0);
            let oracle = DominatorTree::compute_iterative(&graph, 0);
            let context = (recipe, seed);
            let subject = Subject::new(&graph, &tree, &oracle, &edges.edges, &context);
            subject.check_sampled(&mut scratch, &mut rng);
        }
    }
}

#[test]
fn a_reused_scratch_answers_what_a_fresh_one_does() {
    let mut shared = FrontierScratch::new();
    let sizes = [8, 400, 12, 90, 4, 150];
    for seed in 0..48_u64 {
        let mut rng = Rng::new(seed);
        let recipe = Recipe {
            family: Family::ALL[usize::try_from(seed % 3).expect("a residue fits in usize")],
            node_count: sizes[usize::try_from(seed % 6).expect("a residue fits in usize")],
            parallel: seed % 2 == 0,
            detached: 3,
        };
        let edges = random_graph(&mut rng, recipe);
        let graph = edges.graph();
        let tree = DominatorTree::compute_from(&graph, 0);
        let mut reused = Vec::new();
        let mut fresh = Vec::new();
        for _ in 0..6 {
            let defs: Vec<u32> = (0..=rng.below(5))
                .map(|_| node(rng.below(edges.node_count)))
                .collect();
            let live: Vec<bool> = (0..edges.node_count).map(|_| rng.below(3) != 0).collect();
            let live_in = |member: u32| live[member as usize];
            for pruned in [None, Some(&live_in as &dyn Fn(u32) -> bool)] {
                tree.iterated_frontier(
                    &graph,
                    &mut shared,
                    defs.iter().copied(),
                    pruned,
                    &mut reused,
                );
                tree.iterated_frontier(
                    &graph,
                    &mut FrontierScratch::new(),
                    defs.iter().copied(),
                    pruned,
                    &mut fresh,
                );
                assert_eq!(reused, fresh, "{recipe:?} seed {seed} defs {defs:?}");
            }
        }
    }
}

#[test]
fn one_scratch_serves_every_node_type() {
    let mut scratch = FrontierScratch::new();
    let mut out = Vec::new();

    let loop_graph = digraph(4, &[(0, 1), (1, 2), (1, 3), (2, 1)]).graph();
    let loop_tree = DominatorTree::compute_from(&loop_graph, 0);
    loop_tree.iterated_frontier(&loop_graph, &mut scratch, [2], None, &mut out);
    assert_eq!(out, [1]);

    let mut cfg = Cfg::<u32>::new();
    let entry = cfg.entry();
    let left = cfg.new_block();
    let right = cfg.new_block();
    let merge = cfg.new_block();
    cfg.add_edge(entry, left, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, right, EdgeKind::ConditionalFalse);
    cfg.add_edge(left, merge, EdgeKind::Fallthrough);
    cfg.add_edge(right, merge, EdgeKind::Fallthrough);
    let cfg_tree = DominatorTree::compute(&cfg);
    let mut blocks = Vec::new();
    cfg_tree.iterated_frontier(&cfg, &mut scratch, [left, right], None, &mut blocks);
    assert_eq!(blocks, [merge]);

    loop_tree.iterated_frontier(&loop_graph, &mut scratch, [2], None, &mut out);
    assert_eq!(out, [1], "the scratch still answers the first graph");
}

#[test]
fn a_long_chain_with_a_back_edge_is_searched_without_recursion() {
    const BLOCKS: usize = 100_000;
    let last = node(BLOCKS - 1);
    let mut edges: Vec<(u32, u32)> = (0..last).map(|member| (member, member + 1)).collect();
    edges.push((last, 1));
    let graph = EdgeList::new(BLOCKS, edges).graph();
    let tree = DominatorTree::compute_from(&graph, 0);

    assert_eq!(frontier_of(&tree, &graph, 0), NOTHING);
    for member in [1, 2, node(BLOCKS / 2), last] {
        assert_eq!(frontier_of(&tree, &graph, member), [1], "node {member}");
    }
    assert_eq!(iterated_of(&tree, &graph, &[last], None), [1]);
    assert_eq!(
        iterated_of(&tree, &graph, &[node(BLOCKS / 2), last], None),
        [1]
    );
    assert_eq!(iterated_of(&tree, &graph, &[0], None), NOTHING);
}

#[test]
fn a_query_replaces_whatever_the_output_held() {
    let graph = digraph(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]).graph();
    let tree = DominatorTree::compute_from(&graph, 0);
    let mut scratch = FrontierScratch::new();

    let mut out = vec![9, 3, 3];
    tree.frontier(&graph, 1, &mut out);
    assert_eq!(out, [3]);
    tree.frontier(&graph, 0, &mut out);
    assert_eq!(out, NOTHING);
    out.extend([8, 8]);
    tree.iterated_frontier(&graph, &mut scratch, [1], None, &mut out);
    assert_eq!(out, [3]);
    out.extend([8, 8]);
    tree.iterated_frontier(&graph, &mut scratch, [], None, &mut out);
    assert_eq!(out, NOTHING);

    let unreachable = digraph(3, &[(0, 1), (2, 1)]).graph();
    let tree = DominatorTree::compute_from(&unreachable, 0);
    let mut out = vec![1, 1];
    tree.frontier(&unreachable, 2, &mut out);
    assert_eq!(out, NOTHING, "an unreachable node has no frontier");
    out.push(1);
    tree.iterated_frontier(&unreachable, &mut scratch, [2], None, &mut out);
    assert_eq!(out, NOTHING, "an unreachable definition is ignored");
    assert_eq!(frontier_of(&tree, &unreachable, 0), NOTHING);
}

#[test]
fn a_query_that_panics_leaves_the_scratch_usable() {
    let graph = digraph(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]).graph();
    let tree = DominatorTree::compute_from(&graph, 0);
    let mut scratch = FrontierScratch::new();

    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let refuse = |_: u32| -> bool { panic!("the live-in test cannot answer") };
        let mut out = Vec::new();
        tree.iterated_frontier(&graph, &mut scratch, [1, 2], Some(&refuse), &mut out);
    }));
    assert!(panicked.is_err(), "the live-in test was never asked");

    let mut out = Vec::new();
    tree.iterated_frontier(&graph, &mut scratch, [], None, &mut out);
    assert_eq!(
        out, NOTHING,
        "a root queued before the panic must not survive it"
    );
    tree.iterated_frontier(&graph, &mut scratch, [1, 2], None, &mut out);
    assert_eq!(out, [3]);
}

#[test]
fn a_scratch_can_move_to_another_thread() {
    const fn assert_send<T: Send>() {}
    assert_send::<FrontierScratch>();
}
