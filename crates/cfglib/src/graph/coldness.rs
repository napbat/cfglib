//! Coldness classes: which nodes of a rooted graph never run, rarely run, or
//! run normally.
//!
//! The input is a rooted edge view with a seed class for every node, a cap
//! for every edge, and a flag for the nodes that can end the execution
//! themselves. [`ColdnessClasses::compute`] finds the greatest assignment of
//! [`ColdClass`]es that obeys three upper bounds. The operator `∧` is `min`,
//! the colder class.
//!
//! - (S) `c(n) ≤ seed(n)`.
//! - (B) When `n` has out-edges,
//!   `c(n) ≤ lift_n(max over out-edges e of n: cap(e) ∧ c(target(e)))`, where
//!   `lift_n(Never)` is `Rare` when `ends(n)` holds and `Never` otherwise, and
//!   `lift_n(x) = x` for every other class. A node runs only as often as the
//!   hottest edge it can take, unless it can end the execution before its
//!   edges matter.
//! - (F) When `n` is not the root,
//!   `c(n) ≤ max over edges e into n from reachable nodes: cap(e) ∧ c(source(e))`.
//!   A node that only cold nodes reach is cold.
//!
//! Every bound is monotone in the classes, and the lattice is finite, so the
//! greatest assignment exists and is unique. It depends on the root, the
//! reachable edges with their caps, and the seeds and `ends` of the reachable
//! nodes. It does not depend on the order of the nodes or of the edges, or on
//! how many parallel edges join two nodes. A node that the root does not
//! reach is [`ColdClass::Never`].
//!
//! Some consequences of the bounds:
//!
//! - A cycle with no other bound stays `Normal`: the only constraint on a loop
//!   `t → t` is `c(t) ≤ c(t)`. A loop whose only exit leads to a node seeded
//!   `Rare` also stays `Normal`; the exit node is `Rare`.
//! - A capped edge does not make its target cold when an uncapped edge from a
//!   `Normal` node also reaches it: a handler that normal flow also enters
//!   stays `Normal`, as far as its own edges allow.
//! - When every path from the root ends in a node seeded `Rare`, the root is
//!   `Rare` too.
//!
//! # Cost
//!
//! The propagation starts every reachable node at its seed, in reverse
//! postorder from the root, and lowers classes until every bound holds. A
//! class falls at most two times, and each fall reads the edges that leave and
//! enter the node once. Each node keeps two counters for the values of its
//! out-edges and two for its in-edges, so a bound costs constant time, and a
//! node with many edges is not read again for each fall of a neighbor. Time
//! is O(n + m) for `n` node slots and `m` reachable edges, and the work list
//! is an explicit queue, so a chain of 10⁵ nodes needs no more call stack
//! than a diamond.

extern crate alloc;
use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::block::BlockId;
use crate::graph::edge_view::EdgeView;
use crate::graph::traverse::reverse_postorder;
use crate::graph::view::{DenseId, Direction, RootedView};

/// How often a node runs, in three classes. A colder class is a smaller
/// class: `Never < Rare < Normal`.
///
/// The classes describe the structure of the graph and the facts that the
/// caller states. A consumer gives them a meaning through its seeds, caps,
/// and `ends`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum ColdClass {
    /// No execution reaches the node: the root does not reach it, or every
    /// path to it passes a node that the bounds make `Never`.
    Never,
    /// The node runs only in executions that also take a capped edge, reach
    /// a node seeded `Rare`, or end in a node that can end the execution.
    Rare,
    /// The bounds give no evidence that the node is cold.
    Normal,
}

/// The coldness class of every node of a rooted graph.
///
/// See the [module documentation](self) for the bounds that define the
/// classes.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, ColdClass, ColdnessClasses, EdgeKind};
///
/// // The entry branches to a normal arm and, through an exceptional edge, to
/// // a handler that aborts: the consumer seeds the abort `Rare`.
/// let mut cfg = Cfg::<&str>::new();
/// let entry = cfg.entry();
/// let [arm, handler, abort] = [(); 3].map(|()| cfg.new_block());
/// cfg.add_edge(entry, arm, EdgeKind::Fallthrough);
/// cfg.add_edge(entry, handler, EdgeKind::ExceptionUnwind);
/// cfg.add_edge(handler, abort, EdgeKind::Fallthrough);
///
/// let classes = ColdnessClasses::compute(
///     &cfg,
///     |block| if block == abort { ColdClass::Rare } else { ColdClass::Normal },
///     |edge| {
///         if cfg.edge(edge).kind().is_exceptional() {
///             ColdClass::Rare
///         } else {
///             ColdClass::Normal
///         }
///     },
///     |_| false,
/// );
/// assert_eq!(classes.class(entry), ColdClass::Normal);
/// assert_eq!(classes.class(arm), ColdClass::Normal);
/// assert_eq!(classes.class(handler), ColdClass::Rare);
/// assert_eq!(classes.class(abort), ColdClass::Rare);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColdnessClasses<N = BlockId> {
    /// The class of each node slot; `Never` for a node the root does not
    /// reach.
    class: Vec<ColdClass>,
    node: PhantomData<N>,
}

impl<N: DenseId> ColdnessClasses<N> {
    /// Computes the greatest assignment of classes that obeys the bounds (S),
    /// (B), and (F) on `graph`.
    ///
    /// `seed` is the hottest class that a node can have: bound (S). `cap` is
    /// the hottest class that an edge can pass on: [`ColdClass::Normal`] puts
    /// no limit on the edge, and [`ColdClass::Rare`] marks an edge that only
    /// rare executions take, such as an exceptional edge. `ends` names the
    /// nodes that can end the execution before they take an edge; bound (B)
    /// then keeps such a node at least `Rare` when all its edges lead to
    /// `Never` nodes. The closures are asked about the reachable part of the
    /// graph only, and `cap` also about the edges into it from unreachable
    /// nodes.
    ///
    /// Time is O(n + m) for `n` node slots and `m` reachable edges.
    ///
    /// # Panics
    ///
    /// Panics when the root of `graph` is not a node in it.
    #[must_use]
    pub fn compute<G>(
        graph: &G,
        seed: impl Fn(N) -> ColdClass,
        cap: impl Fn(G::EdgeId) -> ColdClass,
        ends: impl Fn(N) -> bool,
    ) -> Self
    where
        G: RootedView<NodeId = N> + EdgeView,
    {
        let mut propagation = Propagation::new(graph, seed, cap, ends);
        propagation.run();
        Self {
            class: propagation.slots.iter().map(|slot| slot.class).collect(),
            node: PhantomData,
        }
    }

    /// Returns the class of `node`.
    ///
    /// A node that the root does not reach is `Never`, and so is a node at or
    /// above the node bound of the graph at compute time.
    #[must_use]
    pub fn class(&self, node: N) -> ColdClass {
        self.class
            .get(node.index())
            .copied()
            .unwrap_or(ColdClass::Never)
    }

    /// Returns the class of an edge from `source` to `target` with the cap
    /// `cap`: `c(source) ∧ cap ∧ c(target)`.
    ///
    /// An edge runs only as often as both of its ends and its cap allow.
    #[must_use]
    pub fn edge(&self, source: N, cap: ColdClass, target: N) -> ColdClass {
        self.class(source).min(cap).min(self.class(target))
    }
}

/// How many edges of one node have at least a given value.
#[derive(Debug, Clone, Copy, Default)]
struct Tally {
    /// The edges of value `Rare` or `Normal`.
    at_least_rare: u32,
    /// The edges of value `Normal`.
    normal: u32,
}

impl Tally {
    /// Counts one more edge of value `value`.
    fn add(&mut self, value: ColdClass) {
        self.at_least_rare += u32::from(value >= ColdClass::Rare);
        self.normal += u32::from(value >= ColdClass::Normal);
    }

    /// Counts one edge less of value `value`.
    fn remove(&mut self, value: ColdClass) {
        self.at_least_rare -= u32::from(value >= ColdClass::Rare);
        self.normal -= u32::from(value >= ColdClass::Normal);
    }

    /// Replaces one edge of value `old` with one of value `new`.
    fn replace(&mut self, old: ColdClass, new: ColdClass) {
        self.remove(old);
        self.add(new);
    }

    /// Returns the hottest counted value, or `Never` when no edge is
    /// counted.
    const fn hottest(self) -> ColdClass {
        if self.normal > 0 {
            ColdClass::Normal
        } else if self.at_least_rare > 0 {
            ColdClass::Rare
        } else {
            ColdClass::Never
        }
    }
}

/// Where a node stands in the propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mark {
    /// The root does not reach the node.
    Unreached,
    /// The node is reachable and not in the queue.
    Idle,
    /// The node waits in the queue.
    Queued,
}

/// The propagation state of one node slot.
#[derive(Debug, Clone, Copy)]
struct Slot {
    class: ColdClass,
    mark: Mark,
    /// Whether the node has an out-edge, so that bound (B) applies.
    has_edges: bool,
    /// The values `cap ∧ c(target)` of the out-edges of the node.
    leaving: Tally,
    /// The values `cap ∧ c(source)` of the edges from reachable nodes to the
    /// node.
    entering: Tally,
}

impl Slot {
    const UNREACHED: Self = Self {
        class: ColdClass::Never,
        mark: Mark::Unreached,
        has_edges: false,
        leaving: Tally {
            at_least_rare: 0,
            normal: 0,
        },
        entering: Tally {
            at_least_rare: 0,
            normal: 0,
        },
    };
}

/// One run of the propagation over one graph.
struct Propagation<'g, G: EdgeView, Cap, Ends> {
    graph: &'g G,
    root: G::NodeId,
    cap: Cap,
    ends: Ends,
    slots: Vec<Slot>,
    queue: VecDeque<G::NodeId>,
}

impl<'g, G, Cap, Ends> Propagation<'g, G, Cap, Ends>
where
    G: RootedView + EdgeView,
    Cap: Fn(G::EdgeId) -> ColdClass,
    Ends: Fn(G::NodeId) -> bool,
{
    /// Starts every reachable node at its seed, queues it in reverse
    /// postorder, and counts the values of the edges.
    fn new(graph: &'g G, seed: impl Fn(G::NodeId) -> ColdClass, cap: Cap, ends: Ends) -> Self {
        let root = graph.root();
        let mut slots = vec![Slot::UNREACHED; graph.node_bound()];
        let queue: VecDeque<G::NodeId> = reverse_postorder(graph, root, Direction::Forward).into();
        for &node in &queue {
            let slot = &mut slots[node.index()];
            slot.class = seed(node);
            slot.mark = Mark::Queued;
        }
        for &source in &queue {
            let from = slots[source.index()].class;
            for edge in graph.outgoing(source) {
                let target = graph.edge(edge).target();
                let cap = cap(edge);
                let to = slots[target.index()].class;
                let slot = &mut slots[source.index()];
                slot.has_edges = true;
                slot.leaving.add(cap.min(to));
                slots[target.index()].entering.add(cap.min(from));
            }
        }
        Self {
            graph,
            root,
            cap,
            ends,
            slots,
            queue,
        }
    }

    /// Queues `node` unless it waits already.
    fn push(&mut self, node: G::NodeId) {
        let slot = &mut self.slots[node.index()];
        if slot.mark == Mark::Idle {
            slot.mark = Mark::Queued;
            self.queue.push_back(node);
        }
    }

    /// Returns the hottest class that the bounds (B) and (F) allow `node`.
    fn bound(&self, node: G::NodeId) -> ColdClass {
        let slot = &self.slots[node.index()];
        let mut bound = ColdClass::Normal;
        if slot.has_edges {
            bound = match slot.leaving.hottest() {
                ColdClass::Never if (self.ends)(node) => ColdClass::Rare,
                hottest => hottest,
            };
        }
        if node != self.root {
            bound = bound.min(slot.entering.hottest());
        }
        bound
    }

    /// Lowers the class of `node` to `to`, updates the counters of the nodes
    /// that its edges join, and queues those nodes.
    fn lower(&mut self, node: G::NodeId, to: ColdClass) {
        let from = core::mem::replace(&mut self.slots[node.index()].class, to);
        let graph = self.graph;
        for edge in graph.outgoing(node) {
            let cap = (self.cap)(edge);
            let (old, new) = (cap.min(from), cap.min(to));
            if old != new {
                let target = graph.edge(edge).target();
                self.slots[target.index()].entering.replace(old, new);
                self.push(target);
            }
        }
        for edge in graph.incoming(node) {
            let source = graph.edge(edge).source();
            if self.slots[source.index()].mark == Mark::Unreached {
                continue;
            }
            let cap = (self.cap)(edge);
            let (old, new) = (cap.min(from), cap.min(to));
            if old != new {
                self.slots[source.index()].leaving.replace(old, new);
                self.push(source);
            }
        }
    }

    /// Lowers classes until every bound holds.
    fn run(&mut self) {
        while let Some(node) = self.queue.pop_front() {
            self.slots[node.index()].mark = Mark::Idle;
            let bound = self.bound(node);
            if bound < self.slots[node.index()].class {
                self.lower(node, bound);
            }
        }
    }
}
