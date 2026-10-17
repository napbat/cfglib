//! Graphs and a brute-force reference shared by the dominator tests, with the
//! seeded generator they draw their graphs from.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::graph::dominator::DominatorTree;
use crate::graph::view::{GraphView, RootedView};

pub(super) use crate::test_util::rng::Rng;

/// The node with dense index `index`.
pub(super) fn node(index: usize) -> u32 {
    u32::try_from(index).expect("test node index fits in u32")
}

/// A directed multigraph, as a node count and an edge list.
///
/// A repeated edge is a parallel edge. Adjacency follows the order of the
/// list, so permuting the list permutes every successor and predecessor
/// sequence.
#[derive(Clone, Debug)]
pub(super) struct EdgeList {
    pub(super) node_count: usize,
    pub(super) edges: Vec<(u32, u32)>,
}

impl EdgeList {
    /// An edge list over `node_count` nodes.
    pub(super) fn new(node_count: usize, edges: Vec<(u32, u32)>) -> Self {
        Self { node_count, edges }
    }

    /// The view over these edges, rooted at node 0.
    pub(super) fn graph(&self) -> Digraph {
        let mut successors = vec![Vec::new(); self.node_count];
        let mut predecessors = vec![Vec::new(); self.node_count];
        for &(from, to) in &self.edges {
            successors[from as usize].push(to);
            predecessors[to as usize].push(from);
        }
        Digraph {
            successors,
            predecessors,
        }
    }

    /// The same graph with every edge written `copies` times in a row.
    pub(super) fn repeated(&self, copies: usize) -> Self {
        let edges = self
            .edges
            .iter()
            .flat_map(|&edge| core::iter::repeat_n(edge, copies))
            .collect();
        Self::new(self.node_count, edges)
    }
}

/// Visit every graph over `node_count` nodes in which each ordered pair of
/// nodes, self-loops included, is an edge `0..=copies` times.
pub(super) fn for_each_multigraph(
    node_count: usize,
    copies: usize,
    mut visit: impl FnMut(&EdgeList),
) {
    let pairs = node_count * node_count;
    let mut counts = vec![0; pairs];
    loop {
        let edges = counts
            .iter()
            .enumerate()
            .flat_map(|(pair, &count)| {
                let edge = (node(pair / node_count), node(pair % node_count));
                core::iter::repeat_n(edge, count)
            })
            .collect();
        visit(&EdgeList::new(node_count, edges));

        let Some(digit) = counts.iter().position(|&count| count < copies) else {
            return;
        };
        counts[..digit].fill(0);
        counts[digit] += 1;
    }
}

/// A [`GraphView`] over `u32` nodes whose root is node 0.
pub(super) struct Digraph {
    successors: Vec<Vec<u32>>,
    predecessors: Vec<Vec<u32>>,
}

impl GraphView for Digraph {
    type NodeId = u32;

    fn node_bound(&self) -> usize {
        self.successors.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = u32> + '_ {
        0..node(self.successors.len())
    }

    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.successors[node as usize].iter().copied()
    }

    fn predecessors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.predecessors[node as usize].iter().copied()
    }
}

impl RootedView for Digraph {
    fn root(&self) -> u32 {
        0
    }
}

/// What a graph's dominators are by definition, found by brute force.
///
/// A node `a` dominates `b` exactly when `b` is reachable from the root and
/// stops being reachable once `a` is removed. Everything else here follows
/// from that one rule with no cleverness, so a disagreement convicts the
/// implementation.
pub(super) struct Reference {
    root: usize,
    reachable: Vec<bool>,
    /// `removal[a][b]`: `a` and `b` are reachable and `b` is cut off from the
    /// root when `a` is removed.
    removal: Vec<Vec<bool>>,
}

impl Reference {
    /// Reference answers for `graph` rooted at `root`.
    pub(super) fn new(graph: &Digraph, root: usize) -> Self {
        let node_count = graph.node_bound();
        let reachable = Self::reach(graph, root, None);
        let removal = (0..node_count)
            .map(|removed| {
                if !reachable[removed] {
                    return vec![false; node_count];
                }
                let remaining = Self::reach(graph, root, Some(removed));
                (0..node_count)
                    .map(|node| reachable[node] && !remaining[node])
                    .collect()
            })
            .collect();
        Self {
            root,
            reachable,
            removal,
        }
    }

    /// The nodes reachable from `root` along paths that avoid `removed`.
    fn reach(graph: &Digraph, root: usize, removed: Option<usize>) -> Vec<bool> {
        let mut seen = vec![false; graph.node_bound()];
        if removed == Some(root) {
            return seen;
        }
        let mut stack = vec![root];
        seen[root] = true;
        while let Some(current) = stack.pop() {
            for next in graph.successors(node(current)) {
                let next = next as usize;
                if !seen[next] && removed != Some(next) {
                    seen[next] = true;
                    stack.push(next);
                }
            }
        }
        seen
    }

    pub(super) fn node_count(&self) -> usize {
        self.reachable.len()
    }

    pub(super) fn is_reachable(&self, node: usize) -> bool {
        self.reachable[node]
    }

    /// Cfglib's answer to "does `dominator` dominate `node`": every node
    /// dominates itself, and a pair with an unreachable member does not
    /// dominate otherwise.
    pub(super) fn dominates(&self, dominator: usize, node: usize) -> bool {
        dominator == node || self.removal[dominator][node]
    }

    fn strict_dominators(&self, node: usize) -> Vec<usize> {
        (0..self.node_count())
            .filter(|&other| other != node && self.removal[other][node])
            .collect()
    }

    /// How many nodes dominate `node` other than itself, for a reachable node.
    pub(super) fn level(&self, node: usize) -> Option<usize> {
        self.reachable[node].then(|| self.strict_dominators(node).len())
    }

    /// The strict dominator of `node` that every other strict dominator of
    /// `node` dominates.
    pub(super) fn idom(&self, node: usize) -> Option<usize> {
        let strict = self.strict_dominators(node);
        strict.iter().copied().find(|&candidate| {
            strict
                .iter()
                .all(|&other| other == candidate || self.removal[other][candidate])
        })
    }

    /// The deepest node that dominates both, for two reachable nodes.
    pub(super) fn nearest_common_dominator(&self, first: usize, second: usize) -> Option<usize> {
        if !self.reachable[first] || !self.reachable[second] {
            return None;
        }
        (0..self.node_count())
            .filter(|&candidate| {
                self.dominates(candidate, first) && self.dominates(candidate, second)
            })
            .max_by_key(|&candidate| self.level(candidate))
    }

    /// The nodes `node` dominates, in ascending index.
    pub(super) fn dominated_by(&self, node: usize) -> Vec<usize> {
        (0..self.node_count())
            .filter(|&other| self.reachable[other] && self.dominates(node, other))
            .collect()
    }

    /// The tree's preorder with children in ascending index.
    pub(super) fn preorder(&self) -> Vec<usize> {
        let mut children = vec![Vec::new(); self.node_count()];
        for node in 0..self.node_count() {
            if let Some(parent) = self.idom(node) {
                children[parent].push(node);
            }
        }
        let mut order = Vec::new();
        let mut stack = vec![self.root];
        while let Some(current) = stack.pop() {
            order.push(current);
            stack.extend(children[current].iter().rev());
        }
        order
    }
}

/// Check every property a numbered forest has whatever graph produced it.
///
/// This needs no reference tree, so it also runs on graphs too large for
/// brute force.
pub(super) fn assert_forest_invariants(tree: &DominatorTree<u32>, node_count: usize) {
    let order = tree.preorder();
    let mut position = vec![None; node_count];
    for (index, &member) in order.iter().enumerate() {
        assert!(
            position[member as usize].replace(index).is_none(),
            "{member} appears twice in the preorder"
        );
    }

    let mut children = vec![Vec::new(); node_count];
    for member in 0..node(node_count) {
        let Some(place) = position[member as usize] else {
            assert!(!tree.is_reachable(member));
            assert_eq!(tree.idom(member), None);
            assert_eq!(tree.level(member), None);
            assert_eq!(tree.subtree(member).len(), 0);
            assert_eq!(tree.children(member).count(), 0);
            continue;
        };
        assert!(tree.is_reachable(member));
        assert_eq!(tree.subtree(member).first(), Some(&member));
        match tree.idom(member) {
            None => assert_eq!(tree.level(member), Some(0), "root {member}"),
            Some(parent) => {
                let parent_level = tree.level(parent).expect("a parent is in the forest");
                assert_eq!(tree.level(member), Some(parent_level + 1), "node {member}");
                assert!(position[parent as usize].is_some_and(|before| before < place));
                assert!(tree.strictly_dominates(parent, member));
                children[parent as usize].push(member);
            }
        }
    }

    let roots: Vec<u32> = order
        .iter()
        .copied()
        .filter(|&member| tree.idom(member).is_none())
        .collect();
    assert!(
        roots.is_sorted(),
        "roots are not in ascending index: {roots:?}"
    );

    for &member in order {
        assert_eq!(
            tree.children(member).collect::<Vec<_>>(),
            children[member as usize],
            "children of {member}"
        );
        let size = 1 + children[member as usize]
            .iter()
            .map(|&child| tree.subtree(child).len())
            .sum::<usize>();
        let subtree = tree.subtree(member);
        assert_eq!(subtree.len(), size, "subtree of {member}");
        let after = position[member as usize].expect("a member has a position") + size;
        let last = order[after - 1];
        assert!(tree.dominates(member, last));
        assert!(
            order
                .get(after)
                .is_none_or(|&next| !tree.dominates(member, next)),
            "the subtree of {member} stops early"
        );
    }
}

/// The shape of a random graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Family {
    /// Two uniformly random edges per node.
    Sparse,
    /// A path through every node plus one random edge per node, forward and
    /// back, which makes loops with several entries common.
    Spine,
    /// A spine plus many loops of two nodes that are entered from two places,
    /// so the graph is irreducible by construction.
    Irreducible,
}

impl Family {
    /// Every family, for a sweep.
    pub(super) const ALL: [Self; 3] = [Self::Sparse, Self::Spine, Self::Irreducible];
}

/// What to put in a random graph besides the family's own edges.
#[derive(Clone, Copy, Debug)]
pub(super) struct Recipe {
    pub(super) family: Family,
    /// Nodes the family spans.
    pub(super) node_count: usize,
    /// Whether to duplicate a quarter of the edges.
    pub(super) parallel: bool,
    /// Extra nodes appended after the family's, whose edges leave them and
    /// are never entered from the family.
    pub(super) detached: usize,
}

/// A random graph following `recipe`, with its edges in random order.
pub(super) fn random_graph(rng: &mut Rng, recipe: Recipe) -> EdgeList {
    let Recipe {
        family,
        node_count: span,
        parallel,
        detached,
    } = recipe;
    assert!(span >= 4, "a family needs room for its gadgets");
    let mut edges = Vec::new();
    match family {
        Family::Sparse => {
            for _ in 0..2 * span {
                edges.push((node(rng.below(span)), node(rng.below(span))));
            }
        }
        Family::Spine | Family::Irreducible => {
            for index in 0..span - 1 {
                edges.push((node(index), node(index + 1)));
            }
            for _ in 0..span {
                edges.push((node(rng.below(span)), node(rng.below(span))));
            }
        }
    }
    if family == Family::Irreducible {
        for _ in 0..=span / 10 {
            // A loop between the neighbors `header` and `header + 1` that an
            // earlier node enters at the second of them.
            let header = 1 + rng.below(span - 2);
            let entry = rng.below(header);
            edges.push((node(header + 1), node(header)));
            edges.push((node(entry), node(header + 1)));
        }
    }
    if parallel {
        let copies: Vec<_> = edges.iter().filter(|_| rng.one_in(4)).copied().collect();
        edges.extend(copies);
    }
    let total = span + detached;
    for source in span..total {
        for _ in 0..2 {
            edges.push((node(source), node(rng.below(total))));
        }
    }
    rng.shuffle(&mut edges);
    EdgeList::new(total, edges)
}
