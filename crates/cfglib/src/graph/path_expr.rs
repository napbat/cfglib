//! Single-source path expressions, and their interpretation in a path
//! algebra.
//!
//! A *path expression* for a node `v` is a regular expression over edge ids
//! whose language is exactly the set of paths from the root to `v`.
//! [`path_expressions`] computes one for every node that the root of a
//! [`RootedView`] reaches, with the dominator-tree elimination of
//! R. E. Tarjan, *Fast Algorithms for Solving Path Problems* (JACM 1981), and
//! keeps every subexpression once in a hash-consed [`PathExprArena`].
//! [`interpret`] then evaluates the expressions in a [`PathAlgebra`] chosen by
//! the consumer — a semiring with a star — once per arena node. That is
//! algebraic program analysis (Kincaid, Reps and Cyphert, CAV 2021): one set
//! of expressions answers many dataflow questions, and the star of a loop can
//! be a closed form rather than an iteration.
//!
//! The expressions are *unambiguous*: each path is denoted exactly once. An
//! algebra whose sum is not idempotent, such as path counting, therefore gets
//! the right answer too, not only one whose sum is idempotent, such as a bit
//! vector. [`GenKillAlgebra`] is the algebra of a forward, union-meet
//! bit-vector problem, whose meet over all paths equals the fixpoint that
//! [`solve_node_problem`](crate::solve_node_problem) computes.
//!
//! # Algorithm and cost
//!
//! The nodes are processed bottom-up over the dominator tree. The children of
//! a node `h` form a small derived graph whose edges summarize the paths that
//! leave one child's subtree and enter a sibling; a path from `h` to a child
//! is a path through that derived graph, and the loops of a node are the
//! edges back into it from its own subtree, closed under a star. The summary
//! of the path from a child down to an edge's source is read from a forest
//! with path compression, which creates one arena node per compressed step.
//!
//! - On a reducible view each derived graph is acyclic, and its children are
//!   solved in reverse postorder. Without the balanced linking of Tarjan's
//!   paper, path compression bounds the work by O(m log n) for `m` edges and
//!   `n` nodes; it is near-linear in practice.
//! - On an irreducible view, each strongly connected group of `r` siblings in
//!   the derived graph ([`tarjan_scc`](crate::tarjan_scc)) is solved by
//!   Gaussian elimination at an extra O(r³).
//!   [`make_reducible`](crate::make_reducible) removes that cost by splitting
//!   nodes first.
//!
//! The order in which arena nodes are created is fixed by the dominator tree,
//! the reverse postorder and the adjacency order of the view, so the ids are
//! deterministic.
//!
//! Expressions mention only edge ids. Transition formulas, closed-form loop
//! summaries and every other semantic algebra stay with the consumer.

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::graph::edge_view::EdgeView;
use crate::graph::view::{DenseId, RootedView};
use crate::identity::define_dense_id;

mod eliminate;
mod gen_kill;

pub use gen_kill::{GenKill, GenKillAlgebra};

define_dense_id! {
    /// One expression of a [`PathExprArena`].
    ///
    /// Ids are dense in creation order, and an expression's operands are
    /// always older than the expression itself, so evaluating the ids in
    /// ascending order evaluates every operand first.
    pub struct PathExprId(u32);
    /// Create an id from a dense zero-based index.
    ///
    /// # Panics
    ///
    /// Panics when `index` exceeds `u32::MAX`.
    from_index = "path expression index exceeds u32::MAX";
    dense_id;
}

/// One node of a path expression over edges `E`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PathExprNode<E> {
    /// No path at all.
    Zero,
    /// The empty path.
    One,
    /// The path of one edge.
    Edge(E),
    /// A path of the first expression followed by a path of the second.
    Concat(PathExprId, PathExprId),
    /// A path of either expression.
    Union(PathExprId, PathExprId),
    /// Zero or more paths of the expression, one after another.
    Star(PathExprId),
}

/// A hash-consed DAG of path expressions over edges `E`.
///
/// Every distinct node is stored once, so expressions that share a
/// subexpression share its id, and an interpretation evaluates it once.
/// [`ZERO`](Self::ZERO) and [`ONE`](Self::ONE) are always present.
#[derive(Debug, Clone)]
pub struct PathExprArena<E> {
    nodes: Vec<PathExprNode<E>>,
    /// The id of each node, for hash-consing.
    dedup: BTreeMap<PathExprNode<E>, PathExprId>,
}

impl<E> PathExprArena<E> {
    /// The expression that denotes no path.
    pub const ZERO: PathExprId = PathExprId::from_raw(0);
    /// The expression that denotes the empty path.
    pub const ONE: PathExprId = PathExprId::from_raw(1);

    /// The node behind `id`.
    ///
    /// # Panics
    ///
    /// Panics when `id` does not belong to this arena.
    #[must_use]
    pub fn node(&self, id: PathExprId) -> &PathExprNode<E> {
        &self.nodes[id.index()]
    }

    /// The number of distinct expression nodes, [`ZERO`](Self::ZERO) and
    /// [`ONE`](Self::ONE) included; every id is below it.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

impl<E: Copy + Ord> PathExprArena<E> {
    /// An arena that holds only [`ZERO`](Self::ZERO) and [`ONE`](Self::ONE).
    fn new() -> Self {
        let mut arena = Self {
            nodes: Vec::new(),
            dedup: BTreeMap::new(),
        };
        arena.intern(PathExprNode::Zero);
        arena.intern(PathExprNode::One);
        arena
    }

    /// The id of `node`, adding it when it is new.
    fn intern(&mut self, node: PathExprNode<E>) -> PathExprId {
        let next = PathExprId::from_index(self.nodes.len());
        let id = *self.dedup.entry(node).or_insert(next);
        if id == next {
            self.nodes.push(node);
        }
        id
    }

    fn edge(&mut self, edge: E) -> PathExprId {
        self.intern(PathExprNode::Edge(edge))
    }

    /// `first` then `second`, with the identities of zero and one applied.
    fn concat(&mut self, first: PathExprId, second: PathExprId) -> PathExprId {
        if first == Self::ZERO || second == Self::ZERO {
            Self::ZERO
        } else if first == Self::ONE {
            second
        } else if second == Self::ONE {
            first
        } else {
            self.intern(PathExprNode::Concat(first, second))
        }
    }

    /// Either expression, with the identity of zero applied. A sum is not
    /// assumed idempotent, so `a + a` stays two terms.
    fn union(&mut self, left: PathExprId, right: PathExprId) -> PathExprId {
        if left == Self::ZERO {
            right
        } else if right == Self::ZERO {
            left
        } else {
            self.intern(PathExprNode::Union(left, right))
        }
    }

    /// Zero or more repetitions; the star of zero is one.
    fn star(&mut self, body: PathExprId) -> PathExprId {
        if body == Self::ZERO {
            Self::ONE
        } else {
            self.intern(PathExprNode::Star(body))
        }
    }
}

/// The path expressions from a root to every node of a graph.
///
/// Made by [`path_expressions`]; evaluated by [`interpret`].
#[derive(Debug, Clone)]
pub struct PathExpressions<N, E> {
    arena: PathExprArena<E>,
    /// Per dense node index, its expression; zero for a node the root does
    /// not reach.
    to: Vec<PathExprId>,
    root: N,
}

impl<N: DenseId, E> PathExpressions<N, E> {
    /// The arena every expression lives in.
    #[must_use]
    pub fn arena(&self) -> &PathExprArena<E> {
        &self.arena
    }

    /// The node every path starts at.
    #[must_use]
    pub fn root(&self) -> N {
        self.root
    }

    /// The node bound of the graph the expressions were computed from.
    #[must_use]
    pub fn node_bound(&self) -> usize {
        self.to.len()
    }

    /// The expression whose language is every path from the root to `node`,
    /// or `None` when the root does not reach `node`.
    ///
    /// The root's own expression is the star of its loops: the empty path
    /// and every cycle through the root.
    ///
    /// # Panics
    ///
    /// Panics when `node` is not below [`node_bound`](Self::node_bound).
    #[must_use]
    pub fn to(&self, node: N) -> Option<PathExprId> {
        let expr = self.to[node.index()];
        (expr != PathExprArena::<E>::ZERO).then_some(expr)
    }
}

/// The single-source path expressions of `graph` from its root.
///
/// See the [module docs](self) for the algorithm and its cost. A node the
/// root does not reach, and a removed node slot, have no expression.
///
/// # Panics
///
/// Panics when the node bound of `graph` exceeds `u32::MAX`.
///
/// # Examples
///
/// ```
/// use cfglib::{Graph, PathAlgebra, Rooted, interpret, path_expressions};
///
/// // A loop: entry -> header <-> body, header -> exit.
/// let mut graph = Graph::new();
/// let [entry, header, body, exit] = [(); 4].map(|()| graph.add_node(()));
/// graph.add_edge(entry, header, ());
/// graph.add_edge(header, body, ());
/// graph.add_edge(body, header, ());
/// graph.add_edge(header, exit, ());
/// let expressions = path_expressions(&Rooted::new(&graph, entry));
///
/// // The shortest path to each node: a tropical path algebra.
/// struct Hops;
/// impl<E> PathAlgebra<E> for Hops {
///     type Value = u32;
///     fn zero(&self) -> u32 { u32::MAX }
///     fn one(&self) -> u32 { 0 }
///     fn edge(&self, _: E) -> u32 { 1 }
///     fn add(&self, a: &u32, b: &u32) -> u32 { *a.min(b) }
///     fn mul(&self, a: &u32, b: &u32) -> u32 { a.saturating_add(*b) }
///     fn star(&self, _: &u32) -> u32 { 0 }
/// }
/// let hops = interpret(&expressions, &Hops);
/// assert_eq!(hops[exit.index()], Some(2));
/// assert_eq!(hops[body.index()], Some(2));
/// ```
#[must_use]
pub fn path_expressions<G: RootedView + EdgeView>(
    graph: &G,
) -> PathExpressions<G::NodeId, G::EdgeId> {
    eliminate::path_expressions(graph)
}

/// A path algebra chosen by the consumer: a semiring with a star.
///
/// [`interpret`] maps each expression node to a value: `Zero` to
/// [`zero`](Self::zero), `One` to [`one`](Self::one), an edge to
/// [`edge`](Self::edge), a concatenation to [`mul`](Self::mul) (first
/// operand first), a union to [`add`](Self::add) and a star to
/// [`star`](Self::star). For the answer to be the value of the set of paths,
/// `add` must be associative and commutative with `zero` as its identity,
/// `mul` associative with `one` as its identity and `zero` as its
/// annihilator, `mul` must distribute over `add`, and `star(a)` must be
/// `one + a + a·a + …`. The expressions are unambiguous, so `add` need not
/// be idempotent.
pub trait PathAlgebra<E> {
    /// The value of a set of paths.
    type Value: Clone;

    /// The value of no path.
    fn zero(&self) -> Self::Value;

    /// The value of the empty path.
    fn one(&self) -> Self::Value;

    /// The value of the path of one edge.
    fn edge(&self, edge: E) -> Self::Value;

    /// The value of the paths of either operand.
    fn add(&self, a: &Self::Value, b: &Self::Value) -> Self::Value;

    /// The value of a path of `a` followed by a path of `b`.
    fn mul(&self, a: &Self::Value, b: &Self::Value) -> Self::Value;

    /// The value of zero or more paths of `a`, one after another.
    fn star(&self, a: &Self::Value) -> Self::Value;
}

/// The value of every node's path expression in `algebra`, indexed by dense
/// node index; `None` for a node the root does not reach.
///
/// Every arena node that some node's expression uses is evaluated exactly
/// once, in id order, so shared subexpressions cost one algebra operation.
#[must_use]
pub fn interpret<N: DenseId, E: Copy, A: PathAlgebra<E>>(
    exprs: &PathExpressions<N, E>,
    algebra: &A,
) -> Vec<Option<A::Value>> {
    let nodes = &exprs.arena.nodes;
    // Operands are older than the expressions that use them, so one
    // descending sweep marks every subexpression a result needs.
    let mut needed = vec![false; nodes.len()];
    for &expr in &exprs.to {
        needed[expr.index()] = true;
    }
    for index in (0..nodes.len()).rev() {
        if needed[index] {
            match nodes[index] {
                PathExprNode::Concat(first, second) | PathExprNode::Union(first, second) => {
                    needed[first.index()] = true;
                    needed[second.index()] = true;
                }
                PathExprNode::Star(body) => needed[body.index()] = true,
                PathExprNode::Zero | PathExprNode::One | PathExprNode::Edge(_) => {}
            }
        }
    }

    let mut values: Vec<Option<A::Value>> = vec![None; nodes.len()];
    for (index, node) in nodes.iter().enumerate() {
        if !needed[index] {
            continue;
        }
        let value = match *node {
            PathExprNode::Zero => algebra.zero(),
            PathExprNode::One => algebra.one(),
            PathExprNode::Edge(edge) => algebra.edge(edge),
            PathExprNode::Concat(first, second) => {
                algebra.mul(operand(&values, first), operand(&values, second))
            }
            PathExprNode::Union(left, right) => {
                algebra.add(operand(&values, left), operand(&values, right))
            }
            PathExprNode::Star(body) => algebra.star(operand(&values, body)),
        };
        values[index] = Some(value);
    }
    exprs
        .to
        .iter()
        .map(|&expr| (expr != PathExprArena::<E>::ZERO).then(|| operand(&values, expr).clone()))
        .collect()
}

/// The value of an evaluated expression.
fn operand<V>(values: &[Option<V>], id: PathExprId) -> &V {
    values[id.index()]
        .as_ref()
        .expect("an operand is older than its expression, so it is evaluated first")
}

#[cfg(test)]
mod tests;
