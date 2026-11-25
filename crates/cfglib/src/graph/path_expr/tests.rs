extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use super::{
    GenKill, GenKillAlgebra, PathAlgebra, PathExprArena, PathExprId, PathExprNode, PathExpressions,
    interpret, path_expressions,
};
use crate::dataflow::bits::DenseBits;
use crate::dataflow::fixpoint::Direction;
use crate::dataflow::node_fixpoint::{NodeProblem, solve_node_problem};
use crate::graph::dominator::DominatorTree;
use crate::graph::store::{EdgeId, Graph, NodeId};
use crate::graph::structure::is_reducible;
use crate::graph::view::Rooted;
use crate::test_util::rng::Rng;

type Plain = Graph<(), ()>;

fn graph_of(nodes: usize, edges: &[(usize, usize)]) -> Plain {
    let mut graph = Graph::new();
    let ids: Vec<NodeId> = (0..nodes).map(|_| graph.add_node(())).collect();
    for &(from, to) in edges {
        graph.add_edge(ids[from], ids[to], ());
    }
    graph
}

fn expressions_of(graph: &Plain) -> PathExpressions<NodeId, EdgeId> {
    path_expressions(&Rooted::new(graph, NodeId::from_index(0)))
}

/// Every walk from node 0 of at most `bound` edges, as edge-index sequences,
/// grouped by the node it ends at.
fn walks(graph: &Plain, bound: usize) -> Vec<BTreeSet<Vec<usize>>> {
    let mut ends = vec![BTreeSet::new(); graph.node_bound()];
    let mut stack = vec![(NodeId::from_index(0), Vec::new())];
    while let Some((node, walk)) = stack.pop() {
        ends[node.index()].insert(walk.clone());
        if walk.len() == bound {
            continue;
        }
        for edge in graph.outgoing(node) {
            let mut longer = walk.clone();
            longer.push(edge.index());
            stack.push((graph.edge(edge).target(), longer));
        }
    }
    ends
}

/// The words of at most `bound` edges that each arena expression denotes.
fn bounded_languages(arena: &PathExprArena<EdgeId>, bound: usize) -> Vec<BTreeSet<Vec<usize>>> {
    let concat = |left: &BTreeSet<Vec<usize>>, right: &BTreeSet<Vec<usize>>| {
        let mut words = BTreeSet::new();
        for first in left {
            for second in right {
                if first.len() + second.len() <= bound {
                    words.insert([first.as_slice(), second].concat());
                }
            }
        }
        words
    };
    let mut languages: Vec<BTreeSet<Vec<usize>>> = Vec::new();
    for index in 0..arena.node_count() {
        let language = match *arena.node(PathExprId::from_index(index)) {
            PathExprNode::Zero => BTreeSet::new(),
            PathExprNode::One => BTreeSet::from([Vec::new()]),
            PathExprNode::Edge(edge) => BTreeSet::from([vec![edge.index()]]),
            PathExprNode::Concat(first, second) => {
                concat(&languages[first.index()], &languages[second.index()])
            }
            PathExprNode::Union(left, right) => languages[left.index()]
                .union(&languages[right.index()])
                .cloned()
                .collect(),
            PathExprNode::Star(body) => {
                let mut all = BTreeSet::from([Vec::new()]);
                let mut frontier = all.clone();
                while !frontier.is_empty() {
                    let next = concat(&frontier, &languages[body.index()]);
                    frontier = next.difference(&all).cloned().collect();
                    all.extend(frontier.iter().cloned());
                }
                all
            }
        };
        languages.push(language);
    }
    languages
}

fn assert_languages_match(graph: &Plain, bound: usize, context: &str) {
    let expressions = expressions_of(graph);
    let languages = bounded_languages(expressions.arena(), bound);
    let expected = walks(graph, bound);
    for node in graph.node_ids() {
        let denoted = expressions
            .to(node)
            .map_or_else(BTreeSet::new, |expr| languages[expr.index()].clone());
        assert_eq!(denoted, expected[node.index()], "{context}: node {node:?}");
    }
}

#[test]
fn every_digraph_of_up_to_four_nodes_denotes_exactly_its_walks() {
    for nodes in 1..=4_usize {
        // Every subset of the ordered pairs, self-loops included.
        let pairs: Vec<(usize, usize)> = (0..nodes)
            .flat_map(|from| (0..nodes).map(move |to| (from, to)))
            .collect();
        let bound = if nodes < 4 { 5 } else { 4 };
        for mask in 0_u32..1 << pairs.len() {
            let edges: Vec<_> = (0..pairs.len())
                .filter(|&bit| mask & (1 << bit) != 0)
                .map(|bit| pairs[bit])
                .collect();
            let context = alloc::format!("{nodes} nodes, edges {edges:?}");
            assert_languages_match(&graph_of(nodes, &edges), bound, &context);
        }
    }
}

#[test]
fn random_five_node_multigraphs_denote_exactly_their_walks() {
    for seed in 0..300 {
        let mut rng = Rng::new(seed);
        let edges: Vec<_> = (0..rng.below(10))
            .map(|_| (rng.below(5), rng.below(5)))
            .collect();
        assert_languages_match(&graph_of(5, &edges), 5, &alloc::format!("seed {seed}"));
    }
}

/// Walk counts by length, up to `LONGEST` edges.
const LONGEST: usize = 6;
type Counts = [u64; LONGEST + 1];

/// Counting walks by length is a path algebra whose sum is not idempotent:
/// it matches enumeration only when every walk is denoted exactly once.
struct WalkCounts;

impl<E> PathAlgebra<E> for WalkCounts {
    type Value = Counts;

    fn zero(&self) -> Counts {
        [0; LONGEST + 1]
    }

    fn one(&self) -> Counts {
        let mut counts = [0; LONGEST + 1];
        counts[0] = 1;
        counts
    }

    fn edge(&self, _: E) -> Counts {
        let mut counts = [0; LONGEST + 1];
        counts[1] = 1;
        counts
    }

    fn add(&self, a: &Counts, b: &Counts) -> Counts {
        core::array::from_fn(|length| a[length] + b[length])
    }

    fn mul(&self, a: &Counts, b: &Counts) -> Counts {
        core::array::from_fn(|length| (0..=length).map(|head| a[head] * b[length - head]).sum())
    }

    fn star(&self, a: &Counts) -> Counts {
        assert_eq!(a[0], 0, "a starred expression denotes no empty walk");
        let mut total = <Self as PathAlgebra<E>>::one(self);
        let mut power = total;
        for _ in 0..LONGEST {
            power = <Self as PathAlgebra<E>>::mul(self, &power, a);
            total = <Self as PathAlgebra<E>>::add(self, &total, &power);
        }
        total
    }
}

/// A random graph in which node 0 reaches every node: a random tree from
/// node 0, plus random edges that may run backward, loop or repeat.
fn random_rooted_graph(rng: &mut Rng, nodes: usize, extra: usize) -> Plain {
    let mut edges: Vec<_> = (1..nodes).map(|node| (rng.below(node), node)).collect();
    edges.extend((0..extra).map(|_| (rng.below(nodes), rng.below(nodes))));
    graph_of(nodes, &edges)
}

#[test]
fn every_walk_is_denoted_exactly_once() {
    let (mut reducible, mut irreducible) = (0, 0);
    for seed in 0..300 {
        let mut rng = Rng::new(seed + 10_000);
        let nodes = 1 + rng.below(7);
        let extra = rng.below(2 * nodes + 1);
        let graph = random_rooted_graph(&mut rng, nodes, extra);
        let rooted = Rooted::new(&graph, NodeId::from_index(0));
        if is_reducible(&rooted, &DominatorTree::compute(&rooted)) {
            reducible += 1;
        } else {
            irreducible += 1;
        }

        let counts = interpret(&expressions_of(&graph), &WalkCounts);
        let expected = walks(&graph, LONGEST);
        for node in graph.node_ids() {
            let mut by_length = [0; LONGEST + 1];
            for walk in &expected[node.index()] {
                by_length[walk.len()] += 1;
            }
            assert_eq!(
                counts[node.index()],
                Some(by_length),
                "seed {seed}, node {node:?}"
            );
        }
    }
    assert!(
        reducible > 50 && irreducible > 50,
        "{reducible} reducible, {irreducible} irreducible"
    );
}

/// A forward union-meet gen/kill problem with one transfer per node.
struct Transfers(Vec<GenKill>);

impl NodeProblem<Plain> for Transfers {
    type Fact = DenseBits;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _: &Plain) -> DenseBits {
        DenseBits::new(self.0[0].generate().len())
    }

    fn boundary(&self, graph: &Plain) -> DenseBits {
        self.bottom(graph)
    }

    fn meet(&self, a: &DenseBits, b: &DenseBits) -> DenseBits {
        let mut met = a.clone();
        met.union_with(b);
        met
    }

    fn transfer(&self, _: &Plain, node: NodeId, input: &DenseBits) -> DenseBits {
        self.0[node.index()].apply(input)
    }
}

fn random_bits(rng: &mut Rng, universe: usize, percent: usize) -> DenseBits {
    let mut bits = DenseBits::new(universe);
    for index in 0..universe {
        if rng.chance(percent) {
            bits.insert(index);
        }
    }
    bits
}

#[test]
fn gen_kill_interpretation_equals_the_node_fixpoint() {
    const UNIVERSE: usize = 70;
    let (mut reducible, mut irreducible) = (0, 0);
    for seed in 0..200 {
        let mut rng = Rng::new(seed + 20_000);
        let nodes = 1 + rng.below(12);
        let extra = rng.below(2 * nodes + 1);
        let graph = random_rooted_graph(&mut rng, nodes, extra);
        let rooted = Rooted::new(&graph, NodeId::from_index(0));
        if is_reducible(&rooted, &DominatorTree::compute(&rooted)) {
            reducible += 1;
        } else {
            irreducible += 1;
        }
        let transfers = Transfers(
            (0..nodes)
                .map(|_| {
                    let generate = random_bits(&mut rng, UNIVERSE, 10);
                    let kill = random_bits(&mut rng, UNIVERSE, 30);
                    GenKill::new(generate, kill)
                })
                .collect(),
        );
        let fixpoint = solve_node_problem(&graph, &transfers).expect("an unbounded solve");

        // An edge carries the transfer of the node it enters; the root's own
        // transfer comes first.
        let algebra = GenKillAlgebra::new(UNIVERSE, |edge: EdgeId| {
            transfers.0[graph.edge(edge).target().index()].clone()
        });
        let values = interpret(&expressions_of(&graph), &algebra);
        let empty = DenseBits::new(UNIVERSE);
        for node in graph.node_ids() {
            let paths = values[node.index()]
                .clone()
                .flatten()
                .expect("node 0 reaches every node");
            let out = transfers.0[0].then(&paths).apply(&empty);
            assert_eq!(&out, fixpoint.fact_out(node), "seed {seed}, node {node:?}");
        }
    }
    assert!(
        reducible > 30 && irreducible > 30,
        "{reducible} reducible, {irreducible} irreducible"
    );
}

/// A structured, reducible CFG of about `blocks` blocks: a sequence of
/// plain blocks, if/else diamonds and while loops nested up to eight deep.
fn structured_cfg(rng: &mut Rng, blocks: usize) -> Plain {
    let mut graph = Graph::new();
    let mut current = graph.add_node(());
    let mut open = Vec::new();
    while graph.node_count() < blocks {
        match rng.below(10) {
            0..=2 => {
                let [then, other, join] = [(); 3].map(|()| graph.add_node(()));
                graph.add_edge(current, then, ());
                graph.add_edge(current, other, ());
                graph.add_edge(then, join, ());
                graph.add_edge(other, join, ());
                current = join;
            }
            3 | 4 if open.len() < 8 => {
                let header = graph.add_node(());
                graph.add_edge(current, header, ());
                open.push(header);
                current = header;
            }
            5 | 6 if !open.is_empty() => {
                let header = open.pop().expect("a loop is open");
                graph.add_edge(current, header, ());
                let exit = graph.add_node(());
                graph.add_edge(header, exit, ());
                current = exit;
            }
            _ => {
                let next = graph.add_node(());
                graph.add_edge(current, next, ());
                current = next;
            }
        }
    }
    while let Some(header) = open.pop() {
        graph.add_edge(current, header, ());
        let exit = graph.add_node(());
        graph.add_edge(header, exit, ());
        current = exit;
    }
    graph
}

#[test]
fn a_large_reducible_cfg_has_an_arena_linear_in_its_edges() {
    for (seed, blocks) in [(1, 10_000), (2, 100_000)] {
        let graph = structured_cfg(&mut Rng::new(seed), blocks);
        let rooted = Rooted::new(&graph, NodeId::from_index(0));
        assert!(is_reducible(&rooted, &DominatorTree::compute(&rooted)));
        let expressions = path_expressions(&rooted);
        let arena = expressions.arena().node_count();
        let edges = graph.edge_count();
        assert!(
            arena <= 8 * edges,
            "{arena} arena nodes for {edges} edges ({blocks} blocks)"
        );
        assert!(graph.node_ids().all(|node| expressions.to(node).is_some()));
    }
}

#[test]
fn unreachable_and_removed_nodes_have_no_expression() {
    let mut graph = graph_of(4, &[(0, 1), (2, 1), (3, 0)]);
    graph.remove_node(NodeId::from_index(3));
    let expressions = expressions_of(&graph);
    assert_eq!(expressions.node_bound(), 4);
    assert_eq!(expressions.root(), NodeId::from_index(0));
    let reached: Vec<bool> = (0..4)
        .map(|node| expressions.to(NodeId::from_index(node)).is_some())
        .collect();
    assert_eq!(reached, [true, true, false, false]);
    assert_eq!(
        interpret(&expressions, &WalkCounts)[2],
        None,
        "an unreachable node has no value"
    );
}

#[test]
fn a_lone_root_denotes_only_the_empty_path() {
    let graph = graph_of(1, &[]);
    let expressions = expressions_of(&graph);
    let root = expressions
        .to(NodeId::from_index(0))
        .expect("the root reaches itself");
    assert_eq!(root, PathExprArena::<EdgeId>::ONE);
    assert_eq!(*expressions.arena().node(root), PathExprNode::One);
    assert_eq!(expressions.arena().node_count(), 2);
}

#[test]
fn the_classic_irreducible_loop_counts_its_walks_by_elimination() {
    // root -> a, root -> b, a <-> b: neither a nor b dominates the other.
    let graph = graph_of(3, &[(0, 1), (0, 2), (1, 2), (2, 1)]);
    let counts = interpret(&expressions_of(&graph), &WalkCounts);
    // Walks to `a` of length n: one for each n >= 1.
    assert_eq!(counts[1], Some([0, 1, 1, 1, 1, 1, 1]));
    assert_eq!(counts[2], Some([0, 1, 1, 1, 1, 1, 1]));
    assert_eq!(counts[0], Some([1, 0, 0, 0, 0, 0, 0]));
}

#[test]
fn parallel_edges_and_root_cycles_are_separate_walks() {
    // Two parallel edges root -> a and a cycle a -> root.
    let graph = graph_of(2, &[(0, 1), (0, 1), (1, 0)]);
    let counts = interpret(&expressions_of(&graph), &WalkCounts);
    assert_eq!(counts[0], Some([1, 0, 2, 0, 4, 0, 8]));
    assert_eq!(counts[1], Some([0, 2, 0, 4, 0, 8, 0]));
}

#[test]
fn the_arena_is_the_same_on_every_run() {
    let graph = random_rooted_graph(&mut Rng::new(5), 40, 60);
    let first = expressions_of(&graph);
    let second = expressions_of(&graph);
    assert_eq!(first.arena().node_count(), second.arena().node_count());
    for index in 0..first.arena().node_count() {
        let id = PathExprId::from_index(index);
        assert_eq!(first.arena().node(id), second.arena().node(id));
    }
    for node in graph.node_ids() {
        assert_eq!(first.to(node), second.to(node));
    }
}

#[test]
fn shared_subexpressions_are_stored_once() {
    let graph = random_rooted_graph(&mut Rng::new(9), 30, 40);
    let expressions = expressions_of(&graph);
    let arena = expressions.arena();
    let distinct: BTreeMap<_, _> = (0..arena.node_count())
        .map(|index| (*arena.node(PathExprId::from_index(index)), index))
        .collect();
    assert_eq!(distinct.len(), arena.node_count());
}

#[test]
fn gen_kill_operations_follow_their_definitions() {
    let bits = |indices: &[usize]| {
        let mut set = DenseBits::new(4);
        for &index in indices {
            set.insert(index);
        }
        set
    };
    let first = GenKill::new(bits(&[0]), bits(&[1, 2]));
    let second = GenKill::new(bits(&[2]), bits(&[0, 3]));
    let input = bits(&[1, 3]);
    assert_eq!(
        first.then(&second).apply(&input),
        second.apply(&first.apply(&input))
    );
    assert_eq!(GenKill::identity(4).apply(&input), input);

    let algebra = GenKillAlgebra::new(4, |(): ()| GenKill::identity(4));
    let sum = algebra.add(&Some(first.clone()), &Some(second.clone()));
    let mut either = first.apply(&input);
    either.union_with(&second.apply(&input));
    assert_eq!(
        sum.clone().expect("a sum of transfers").apply(&input),
        either
    );
    assert_eq!(algebra.add(&None, &sum), sum);
    assert_eq!(algebra.mul(&None, &sum), None);
    assert_eq!(algebra.star(&None), algebra.one());
    let star = algebra.star(&Some(first)).expect("a transfer");
    assert_eq!(star.apply(&input), bits(&[0, 1, 3]));
}
