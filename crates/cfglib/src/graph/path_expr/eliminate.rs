//! Tarjan's elimination of path expressions over the dominator tree.
//!
//! For a node `c` with immediate dominator `h`, three expressions are built
//! bottom-up:
//!
//! - `loops(c)`: every closed path at `c` inside the subtree of `c` — the
//!   star of the edges back into `c` from that subtree;
//! - `entry(c)`: every path from `h` to `c` that does not return to `h`; it
//!   ends with the last step into the subtree of `c` (from `h` or a sibling
//!   subtree) followed by `loops(c)`;
//! - the summary from a child of `h` down to any node `y` of its subtree,
//!   which is the product of `entry` along the tree path below the child,
//!   read from a forest with path compression.
//!
//! Every path from the root to `v` splits uniquely at its last visit to
//! `idom(v)`, so the expression of `v` is that of `idom(v)` followed by
//! `entry(v)`, and the root's is `loops(root)`. Each split is unique, which is
//! what makes the expressions unambiguous.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::{PathExprArena, PathExprId, PathExpressions};
use crate::graph::dominator::DominatorTree;
use crate::graph::edge_view::EdgeView;
use crate::graph::scc::tarjan_scc;
use crate::graph::store::{Graph, NodeId};
use crate::graph::traverse::reverse_postorder;
use crate::graph::view::Direction;
use crate::graph::view::{DenseId, RootedView};

/// An unset slot of a dense side table.
const NONE: usize = usize::MAX;

const ZERO: PathExprId = PathExprArena::<()>::ZERO;
const ONE: PathExprId = PathExprArena::<()>::ONE;

pub(super) fn path_expressions<G: RootedView + EdgeView>(
    graph: &G,
) -> PathExpressions<G::NodeId, G::EdgeId> {
    let bound = graph.node_bound();
    let root = graph.root();
    let tree = DominatorTree::compute(graph);
    let mut rank = vec![NONE; bound];
    for (position, node) in reverse_postorder(graph, root, Direction::Forward)
        .into_iter()
        .enumerate()
    {
        rank[node.index()] = position;
    }

    let mut elimination = Elimination {
        graph,
        tree: &tree,
        rank,
        arena: PathExprArena::new(),
        ancestor: vec![NONE; bound],
        summary: vec![ONE; bound],
        loops: vec![ONE; bound],
        entry: vec![ZERO; bound],
        local: vec![NONE; bound],
    };
    for &node in tree.preorder().iter().rev() {
        elimination.enter_children(node);
        elimination.close_loops(node);
    }

    let Elimination {
        mut arena,
        loops,
        entry,
        ..
    } = elimination;
    let mut to = vec![ZERO; bound];
    for &node in tree.preorder() {
        to[node.index()] = match tree.idom(node) {
            None => loops[node.index()],
            Some(parent) => arena.concat(to[parent.index()], entry[node.index()]),
        };
    }
    PathExpressions { arena, to, root }
}

/// The tables of one elimination, indexed by dense node index.
struct Elimination<'g, G: EdgeView> {
    graph: &'g G,
    tree: &'g DominatorTree<G::NodeId>,
    /// Reverse-postorder position of each reachable node.
    rank: Vec<usize>,
    arena: PathExprArena<G::EdgeId>,
    /// The forest parent of a node whose subtree is solved, until the node's
    /// own tree root is linked; `NONE` for a forest root.
    ancestor: Vec<usize>,
    /// The paths from just below the forest parent down to the node: the
    /// product of `entry` along that stretch of the dominator tree.
    summary: Vec<PathExprId>,
    loops: Vec<PathExprId>,
    entry: Vec<PathExprId>,
    /// Scratch: the position of each child among its siblings while their
    /// parent is processed.
    local: Vec<usize>,
}

/// An edge of the derived graph of one sibling group: from a sibling, the
/// paths from it down to the edge's source and then the edge.
struct Derived {
    from: usize,
    to: usize,
    path: PathExprId,
}

impl<G: RootedView + EdgeView> Elimination<'_, G> {
    /// Solve `entry` for every child of `parent` and link the children under
    /// it. The children's subtrees are solved already.
    fn enter_children(&mut self, parent: G::NodeId) {
        let mut children: Vec<G::NodeId> = self.tree.children(parent).collect();
        if children.is_empty() {
            return;
        }
        children.sort_unstable_by_key(|child| self.rank[child.index()]);
        for (position, child) in children.iter().enumerate() {
            self.local[child.index()] = position;
        }

        // The direct steps from the parent into each child, and the derived
        // edges between siblings, in the order of the children and of their
        // incoming edges.
        let mut direct = vec![ZERO; children.len()];
        let mut derived = Vec::new();
        let mut retreating = false;
        for (to, &child) in children.iter().enumerate() {
            for edge in self.graph.incoming(child) {
                let source = self.graph.edge(edge).source();
                // An edge from inside the child's own subtree is one of its
                // loops, already closed.
                if !self.tree.is_reachable(source) || self.tree.dominates(child, source) {
                    continue;
                }
                let step = self.arena.edge(edge);
                if source == parent {
                    direct[to] = self.arena.union(direct[to], step);
                } else {
                    let (sibling, below) = self.eval(source.index());
                    let from = self.local[sibling];
                    let path = self.arena.concat(below, step);
                    retreating |= from > to;
                    derived.push(Derived { from, to, path });
                }
            }
        }

        if retreating {
            self.solve_cyclic(&children, &direct, &derived);
        } else {
            // Every derived edge runs forward in reverse postorder, so the
            // children are solved in that order.
            let mut edges = derived.iter().peekable();
            for (to, &child) in children.iter().enumerate() {
                let mut paths = direct[to];
                while let Some(edge) = edges.next_if(|edge| edge.to == to) {
                    let via = self
                        .arena
                        .concat(self.entry[children[edge.from].index()], edge.path);
                    paths = self.arena.union(paths, via);
                }
                self.entry[child.index()] = self.arena.concat(paths, self.loops[child.index()]);
            }
        }

        for &child in &children {
            self.ancestor[child.index()] = parent.index();
            self.summary[child.index()] = self.entry[child.index()];
            self.local[child.index()] = NONE;
        }
    }

    /// Solve the children of one parent whose derived graph may have cycles:
    /// strongly connected groups in topological order, each by Gaussian
    /// elimination of `Y_j = b_j + Σ_i Y_i · A[i][j]`, where `Y_j` is the
    /// paths that reach child `j` from outside its subtree.
    fn solve_cyclic(&mut self, children: &[G::NodeId], direct: &[PathExprId], derived: &[Derived]) {
        let mut shape: Graph<(), ()> = Graph::with_capacity(children.len(), derived.len());
        let nodes: Vec<NodeId> = children.iter().map(|_| shape.add_node(())).collect();
        for edge in derived {
            shape.add_edge(nodes[edge.from], nodes[edge.to], ());
        }
        let decomposition = tarjan_scc(&shape);
        // Leaves come first, so the reversed order is topological.
        let order: Vec<&[NodeId]> = decomposition.components().rev().collect();
        let mut component = vec![0; children.len()];
        let mut position = vec![0; children.len()];
        for (index, members) in order.iter().enumerate() {
            for (at, member) in members.iter().enumerate() {
                component[member.index()] = index;
                position[member.index()] = at;
            }
        }
        let mut incoming: Vec<Vec<&Derived>> = vec![Vec::new(); order.len()];
        for edge in derived {
            incoming[component[edge.to]].push(edge);
        }

        for (index, members) in order.iter().enumerate() {
            let size = members.len();
            let mut outside: Vec<PathExprId> = members
                .iter()
                .map(|member| direct[member.index()])
                .collect();
            let mut within = vec![vec![ZERO; size]; size];
            for edge in &incoming[index] {
                let source = children[edge.from].index();
                let target = position[edge.to];
                if component[edge.from] == index {
                    let step = self.arena.concat(self.loops[source], edge.path);
                    let cell = &mut within[position[edge.from]][target];
                    *cell = self.arena.union(*cell, step);
                } else {
                    // An earlier component, solved already.
                    let via = self.arena.concat(self.entry[source], edge.path);
                    outside[target] = self.arena.union(outside[target], via);
                }
            }
            let reached = self.eliminate(&mut outside, &mut within);
            for (member, paths) in members.iter().zip(reached) {
                let child = children[member.index()].index();
                self.entry[child] = self.arena.concat(paths, self.loops[child]);
            }
        }
    }

    /// Solve `Y_j = b_j + Σ_i Y_i · A[i][j]` for every `j`: forward
    /// elimination of `Y_k` from the later equations, then back substitution.
    fn eliminate(&mut self, b: &mut [PathExprId], a: &mut [Vec<PathExprId>]) -> Vec<PathExprId> {
        let size = b.len();
        let mut loops = vec![ONE; size];
        for k in 0..size {
            loops[k] = self.arena.star(a[k][k]);
            for j in k + 1..size {
                if a[k][j] == ZERO {
                    continue;
                }
                let via = self.arena.concat(loops[k], a[k][j]);
                let through = self.arena.concat(b[k], via);
                b[j] = self.arena.union(b[j], through);
                for row in a.iter_mut().skip(k + 1) {
                    if row[k] != ZERO {
                        let through = self.arena.concat(row[k], via);
                        row[j] = self.arena.union(row[j], through);
                    }
                }
            }
        }
        let mut solved = vec![ZERO; size];
        for k in (0..size).rev() {
            let mut paths = b[k];
            for i in k + 1..size {
                let through = self.arena.concat(solved[i], a[i][k]);
                paths = self.arena.union(paths, through);
            }
            solved[k] = self.arena.concat(paths, loops[k]);
        }
        solved
    }

    /// Close the loops of `node`: the star of every edge back into it from
    /// its own subtree. Its children are linked under it already.
    fn close_loops(&mut self, node: G::NodeId) {
        let mut body = ZERO;
        for edge in self.graph.incoming(node) {
            let source = self.graph.edge(edge).source();
            if !self.tree.is_reachable(source) || !self.tree.dominates(node, source) {
                continue;
            }
            let step = self.arena.edge(edge);
            let path = if source == node {
                step
            } else {
                let (_, below) = self.eval(source.index());
                self.arena.concat(below, step)
            };
            body = self.arena.union(body, path);
        }
        self.loops[node.index()] = self.arena.star(body);
    }

    /// The forest root above `node` and the paths from just below that root
    /// down to `node`, compressing the forest path on the way.
    fn eval(&mut self, node: usize) -> (usize, PathExprId) {
        if self.ancestor[node] == NONE {
            return (node, ONE);
        }
        let mut path = Vec::new();
        let mut top = node;
        while self.ancestor[self.ancestor[top]] != NONE {
            path.push(top);
            top = self.ancestor[top];
        }
        // `top` hangs directly under the root; fold the summaries down from
        // it so that every node on the path hangs there too.
        while let Some(below) = path.pop() {
            let above = self.ancestor[below];
            self.summary[below] = self.arena.concat(self.summary[above], self.summary[below]);
            self.ancestor[below] = self.ancestor[above];
        }
        (self.ancestor[node], self.summary[node])
    }
}
