//! Weak topological orders against Bourdoncle's recursive construction, the
//! defining properties on every small digraph, and the examples of Kim, Venet
//! and Thakur (*Memory-efficient fixpoint computation*, FMSD 2025, Fig. 1).

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::WeakTopologicalOrder;
use crate::graph::store::Graph;
use crate::graph::view::{Direction, GraphView, Reversed, Rooted};
use crate::identity::{NO_INDEX, slot};
use crate::test_util::rng::Rng;

/// Adjacency lists over `u32` nodes, successors in the order given.
struct Digraph {
    successors: Vec<Vec<u32>>,
    predecessors: Vec<Vec<u32>>,
}

impl Digraph {
    fn new(count: usize, edges: &[(u32, u32)]) -> Self {
        let mut graph = Self {
            successors: vec![Vec::new(); count],
            predecessors: vec![Vec::new(); count],
        };
        for &(source, target) in edges {
            graph.successors[source as usize].push(target);
            graph.predecessors[target as usize].push(source);
        }
        graph
    }

    /// The same edges, walked against their direction.
    fn reversed(&self) -> Self {
        Self {
            successors: self.predecessors.clone(),
            predecessors: self.successors.clone(),
        }
    }
}

impl GraphView for Digraph {
    type NodeId = u32;

    fn node_bound(&self) -> usize {
        self.successors.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = u32> + '_ {
        0..u32::try_from(self.successors.len()).expect("a toy graph has few nodes")
    }

    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.successors[node as usize].iter().copied()
    }

    fn predecessors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.predecessors[node as usize].iter().copied()
    }
}

/// One element of Bourdoncle's hierarchical order.
#[derive(Debug)]
enum Element {
    Node(u32),
    Component(u32, Vec<Element>),
}

/// Bourdoncle's recursive construction, as published: Tarjan's walk that
/// prepends each finished element, and re-walks a loop's body below its head.
struct Bourdoncle<'g> {
    graph: &'g Digraph,
    number: Vec<usize>,
    count: usize,
    stack: Vec<u32>,
}

impl Bourdoncle<'_> {
    const DONE: usize = usize::MAX;

    fn order(graph: &Digraph, root: u32) -> (Vec<u32>, Vec<u32>) {
        let mut walk = Bourdoncle {
            graph,
            number: vec![0; graph.successors.len()],
            count: 0,
            stack: Vec::new(),
        };
        let mut partition = Vec::new();
        walk.visit(root, &mut partition);
        let (mut order, mut heads) = (Vec::new(), Vec::new());
        flatten(&partition, &mut order, &mut heads);
        (order, heads)
    }

    fn visit(&mut self, node: u32, partition: &mut Vec<Element>) -> usize {
        self.stack.push(node);
        self.count += 1;
        self.number[node as usize] = self.count;
        let mut head = self.count;
        let mut is_loop = false;
        for &next in &self.graph.successors[node as usize] {
            let low = if self.number[next as usize] == 0 {
                self.visit(next, partition)
            } else {
                self.number[next as usize]
            };
            if low <= head {
                head = low;
                is_loop = true;
            }
        }
        if head == self.number[node as usize] {
            self.number[node as usize] = Self::DONE;
            let mut element = self.stack.pop().expect("the node is on the stack");
            if is_loop {
                while element != node {
                    self.number[element as usize] = 0;
                    element = self.stack.pop().expect("the head is below its body");
                }
                let component = self.component(node);
                partition.insert(0, component);
            } else {
                partition.insert(0, Element::Node(node));
            }
        }
        head
    }

    fn component(&mut self, head: u32) -> Element {
        let mut partition = Vec::new();
        for &next in &self.graph.successors[head as usize] {
            if self.number[next as usize] == 0 {
                self.visit(next, &mut partition);
            }
        }
        Element::Component(head, partition)
    }
}

fn flatten(partition: &[Element], order: &mut Vec<u32>, heads: &mut Vec<u32>) {
    for element in partition {
        match element {
            Element::Node(node) => order.push(*node),
            Element::Component(head, body) => {
                order.push(*head);
                heads.push(*head);
                flatten(body, order, heads);
            }
        }
    }
}

/// The nodes `root` reaches, by a plain search.
fn reached(graph: &Digraph, root: u32) -> Vec<bool> {
    let mut seen = vec![false; graph.successors.len()];
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if !core::mem::replace(&mut seen[node as usize], true) {
            stack.extend(&graph.successors[node as usize]);
        }
    }
    seen
}

/// Whether the component headed at `head` contains `node`.
fn contains(order: &WeakTopologicalOrder<u32>, head: u32, node: u32) -> bool {
    let (Some(head), Some(node)) = (order.rank(head), order.rank(node)) else {
        return false;
    };
    let (head, node) = (slot(head), slot(node));
    order.heads_at(head) && head <= node && node < slot(order.component_end[head])
}

/// Checks the definition of a weak topological order (Kim, Venet and Thakur,
/// FMSD 2025, Definitions 1 and 2) and the accessors against each other.
fn assert_weak_topological(graph: &Digraph, root: u32, order: &WeakTopologicalOrder<u32>) {
    let seen = reached(graph, root);
    let mut covered: Vec<u32> = order.order().to_vec();
    covered.sort_unstable();
    let expected: Vec<u32> = graph.node_ids().filter(|&n| seen[n as usize]).collect();
    assert_eq!(covered, expected, "the order covers the reached nodes");

    for (position, &node) in order.order().iter().enumerate() {
        assert_eq!(order.rank(node).map(slot), Some(position));
        let end = slot(order.component_end[position]);
        assert!(end >= position, "a component ends after its head");
        let parent = order.parent[position];
        if parent != NO_INDEX {
            let parent = slot(parent);
            assert!(order.heads_at(parent), "a parent is a head");
            assert!(parent < position && position < slot(order.component_end[parent]));
            assert!(end <= slot(order.component_end[parent]), "components nest");
        }
        // Contiguity: everything between a head and its end lies in its
        // component, and the innermost head of an element is its parent.
        for inner in position + 1..end {
            let mut ancestor = order.parent[inner];
            while ancestor != NO_INDEX && slot(ancestor) > position {
                ancestor = order.parent[slot(ancestor)];
            }
            assert_eq!(slot(ancestor), position, "{position} encloses {inner}");
        }
        let expected_head = if order.heads_at(position) {
            Some(node)
        } else {
            (parent != NO_INDEX).then(|| order.order()[slot(parent)])
        };
        assert_eq!(order.head_of(node), expected_head);
        let outermost = order.outermost(position);
        assert!(outermost.contains(&position));
        assert_eq!(
            order.parent[outermost.start], NO_INDEX,
            "an outermost element"
        );
        if order.heads_at(outermost.start) {
            assert_eq!(outermost.end, slot(order.component_end[outermost.start]));
        } else {
            assert_eq!(outermost, position..position + 1);
        }
    }
    for node in graph.node_ids().filter(|&n| !seen[n as usize]) {
        assert_eq!(order.rank(node), None);
        assert!(!order.is_head(node));
        assert_eq!(order.head_of(node), None);
    }

    for source in graph.node_ids().filter(|&n| seen[n as usize]) {
        for target in graph.successors(source) {
            assert!(
                order.rank(source) < order.rank(target) || contains(order, target, source),
                "edge {source} -> {target} breaks the order {:?}",
                order.order()
            );
        }
    }

    // Every cycle has a head: the reached nodes without heads are acyclic.
    let mut indegree = vec![0_usize; graph.successors.len()];
    let loose = |node: u32| seen[node as usize] && !order.is_head(node);
    for source in graph.node_ids().filter(|&n| loose(n)) {
        for target in graph.successors(source).filter(|&n| loose(n)) {
            indegree[target as usize] += 1;
        }
    }
    let mut ready: Vec<u32> = graph
        .node_ids()
        .filter(|&n| loose(n) && indegree[n as usize] == 0)
        .collect();
    let mut removed = 0;
    while let Some(node) = ready.pop() {
        removed += 1;
        for target in graph.successors(node).filter(|&n| loose(n)) {
            indegree[target as usize] -= 1;
            if indegree[target as usize] == 0 {
                ready.push(target);
            }
        }
    }
    let loose_count = graph.node_ids().filter(|&n| loose(n)).count();
    assert_eq!(removed, loose_count, "a cycle avoids every head");
}

/// The weak topological order of what `root` reaches in `graph`.
fn wto<G: GraphView>(graph: &G, root: G::NodeId) -> WeakTopologicalOrder<G::NodeId> {
    WeakTopologicalOrder::compute(&Rooted::new(graph, root))
}

fn check(graph: &Digraph, root: u32) {
    let order = wto(graph, root);
    assert_weak_topological(graph, root, &order);
    let (expected, heads) = Bourdoncle::order(graph, root);
    assert_eq!(order.order(), expected, "Bourdoncle's order");
    assert_eq!(
        order.heads().collect::<Vec<_>>(),
        heads,
        "Bourdoncle's heads"
    );
}

/// Calls `visit` with every digraph on `count` nodes whose edge set is drawn
/// from `pairs`.
fn for_each_digraph(count: usize, pairs: &[(u32, u32)], mut visit: impl FnMut(&Digraph)) {
    let mut edges = Vec::with_capacity(pairs.len());
    for mask in 0..1_u32 << pairs.len() {
        edges.clear();
        edges.extend(
            pairs
                .iter()
                .enumerate()
                .filter(|&(bit, _)| mask & (1 << bit) != 0)
                .map(|(_, &edge)| edge),
        );
        visit(&Digraph::new(count, &edges));
    }
}

fn pairs(count: u32, self_loops: bool) -> Vec<(u32, u32)> {
    (0..count)
        .flat_map(|source| (0..count).map(move |target| (source, target)))
        .filter(|&(source, target)| self_loops || source != target)
        .collect()
}

#[test]
fn every_digraph_of_up_to_four_nodes_with_self_loops_is_ordered_as_bourdoncle_orders_it() {
    for count in 1..=4 {
        for_each_digraph(count as usize, &pairs(count, true), |graph| {
            check(graph, 0);
        });
    }
}

#[test]
fn every_digraph_of_five_nodes_is_ordered_as_bourdoncle_orders_it() {
    for_each_digraph(5, &pairs(5, false), |graph| check(graph, 0));
}

#[test]
fn random_digraphs_match_bourdoncle_and_the_definition_both_ways() {
    let mut rng = Rng::new(0x5eed_0057);
    for _ in 0..2_000 {
        let count = 1 + rng.below(12);
        let edges: Vec<(u32, u32)> = (0..rng.below(3 * count + 1))
            .map(|_| {
                let source = u32::try_from(rng.below(count)).expect("few nodes");
                let target = u32::try_from(rng.below(count)).expect("few nodes");
                (source, target)
            })
            .collect();
        let graph = Digraph::new(count, &edges);
        check(&graph, 0);

        let reversed = graph.reversed();
        let root = u32::try_from(count - 1).expect("few nodes");
        let backward = WeakTopologicalOrder::compute(&Rooted::new(&Reversed::new(&graph), root));
        assert_eq!(
            backward,
            wto(&reversed, root),
            "walking incoming edges is walking the reversed graph"
        );
        assert_weak_topological(&reversed, root, &backward);
    }
}

/// The three control-flow graphs of Kim, Venet and Thakur (FMSD 2025,
/// Fig. 1), nodes `1..=9` at indices `1..=9` and index 0 unused. Successors
/// are listed so that the walk produces the orders the paper gives.
fn figure_one(loops: &[(u32, u32)]) -> Digraph {
    let mut edges = vec![
        (1, 2),
        (2, 3),
        (3, 7),
        (3, 4),
        (4, 6),
        (4, 5),
        (7, 9),
        (7, 8),
    ];
    edges.extend_from_slice(loops);
    Digraph::new(10, &edges)
}

/// The order with `(` before each head and `)` after each component, as the
/// paper writes it.
fn parenthesized(order: &WeakTopologicalOrder<u32>) -> alloc::string::String {
    use core::fmt::Write;
    let mut text = alloc::string::String::new();
    let mut open = Vec::new();
    for (position, &node) in order.order().iter().enumerate() {
        while open.last().is_some_and(|&end| end == position) {
            open.pop();
            text.push(')');
        }
        if !text.is_empty() && !text.ends_with('(') {
            text.push(' ');
        }
        if order.heads_at(position) {
            text.push('(');
            open.push(slot(order.component_end[position]));
        }
        write!(text, "{node}").expect("writing to a string succeeds");
    }
    text.extend(open.iter().map(|_| ')'));
    text
}

#[test]
fn the_three_graphs_of_figure_one_get_the_orders_of_the_paper() {
    let g1 = figure_one(&[]);
    let w1 = wto(&g1, 1);
    assert_eq!(parenthesized(&w1), "1 2 3 4 5 6 7 8 9");

    // G2 adds the loops 4 <-> 5 and 7 <-> 8 and the edge 2 -> 8.
    let g2 = figure_one(&[(2, 8), (5, 4), (8, 7)]);
    let w2 = wto(&g2, 1);
    assert_eq!(parenthesized(&w2), "1 2 3 (4 5) 6 (7 8) 9");
    assert_eq!(w2.head_of(5), Some(4));
    assert_eq!(w2.head_of(6), None);

    // G3 also closes the outer loop 3 ... 6 with 5 -> 3 and 6 -> 3.
    let g3 = figure_one(&[(2, 8), (5, 3), (5, 4), (6, 3), (8, 7)]);
    let w3 = wto(&g3, 1);
    assert_eq!(parenthesized(&w3), "1 2 (3 (4 5) 6) (7 8) 9");
    assert_eq!(w3.heads().collect::<Vec<_>>(), [3, 4, 7]);
    assert_eq!(w3.head_of(5), Some(4));
    assert_eq!(w3.head_of(6), Some(3));
    assert_eq!(w3.head_of(4), Some(4));
    for graph in [&g1, &g2, &g3] {
        check(graph, 1);
    }
}

#[test]
fn a_self_loop_heads_a_component_of_one_node() {
    let graph = Digraph::new(3, &[(0, 1), (1, 1), (1, 2)]);
    let order = wto(&graph, 0);
    assert_eq!(parenthesized(&order), "0 (1) 2");
    assert!(order.is_head(1));
    assert_eq!(order.head_of(2), None);
}

#[test]
fn an_irreducible_loop_is_headed_by_the_entry_the_walk_takes_first() {
    // 0 enters the cycle 1 <-> 2 at both nodes; the walk enters at 1.
    let graph = Digraph::new(4, &[(0, 1), (0, 2), (1, 2), (2, 1), (2, 3)]);
    let order = wto(&graph, 0);
    assert_eq!(parenthesized(&order), "0 (1 2) 3");
    check(&graph, 0);
}

#[test]
fn nodes_the_root_misses_and_removed_slots_have_no_rank() {
    let mut graph: Graph<(), ()> = Graph::new();
    let nodes: Vec<_> = (0..5).map(|_| graph.add_node(())).collect();
    graph.add_edge(nodes[0], nodes[2], ());
    graph.add_edge(nodes[2], nodes[0], ());
    graph.add_edge(nodes[3], nodes[0], ());
    graph.add_edge(nodes[2], nodes[4], ());
    assert!(graph.remove_node(nodes[4]));
    let order = wto(&graph, nodes[0]);
    assert_eq!(order.order(), [nodes[0], nodes[2]]);
    assert_eq!(order.heads().collect::<Vec<_>>(), [nodes[0]]);
    for missed in [nodes[1], nodes[3], nodes[4]] {
        assert_eq!(order.rank(missed), None);
        assert_eq!(order.head_of(missed), None);
    }
}

#[test]
fn later_roots_come_first_because_they_can_reach_earlier_ones() {
    // 3 -> 0 and 1 -> 2, walked from 0, then 1, then 3; the repeated 0 is
    // skipped.
    let graph = Digraph::new(4, &[(3, 0), (1, 2)]);
    let order = WeakTopologicalOrder::from_roots(&graph, [0, 1, 3, 0], Direction::Forward);
    assert_eq!(order.order(), [3, 1, 2, 0]);
    assert_eq!(order.heads().count(), 0);
}

#[test]
fn a_long_chain_of_nested_loops_needs_no_recursion() {
    // 0 -> 1 -> ... -> n - 1, and every node but 0 loops back to the one
    // before it: n - 1 heads, each nested in the one before.
    let count = 20_000_u32;
    let mut edges: Vec<(u32, u32)> = (1..count).map(|node| (node - 1, node)).collect();
    edges.extend((1..count).map(|node| (node, node - 1)));
    let graph = Digraph::new(count as usize, &edges);
    let order = wto(&graph, 0);
    assert_eq!(order.order(), (0..count).collect::<Vec<_>>());
    assert_eq!(order.heads().count(), count as usize - 1);
    assert_eq!(order.head_of(count - 1), Some(count - 2));
}

#[test]
#[should_panic(expected = "root is out of range")]
fn a_root_beyond_the_bound_panics() {
    let graph = Digraph::new(1, &[]);
    let _ = wto(&graph, 1);
}
