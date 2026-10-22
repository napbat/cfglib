//! Coldness classes: which nodes of a rooted graph never run, rarely run, or
//! run normally.
//!
//! The input is a [`ColdnessGraph`]: a rooted graph whose nodes hold ordered
//! successor records, with a seed class for every node, a cap for every
//! record, and a flag for the nodes that can end the execution themselves.
//! [`ColdnessClasses::compute`] finds the greatest assignment of
//! [`ColdClass`]es that obeys three upper bounds. The operator `∧` is `min`,
//! the colder class.
//!
//! - (S) `c(n) ≤ seed(n)`.
//! - (B) When `n` has records,
//!   `c(n) ≤ lift_n(max over records r of n: cap(r) ∧ c(target(r)))`, where
//!   `lift_n(Never)` is `Rare` when `ends(n)` holds and `Never` otherwise, and
//!   `lift_n(x) = x` for every other class. A node runs only as often as the
//!   hottest record it can take, unless it can end the execution before its
//!   records matter.
//! - (F) When `n` is not the entry,
//!   `c(n) ≤ max over records r to n from reachable nodes: cap(r) ∧ c(source(r))`.
//!   A node that only cold nodes reach is cold.
//!
//! Every bound is monotone in the classes, and the lattice is finite, so the
//! greatest assignment exists and is unique. It depends on the entry, the
//! reachable records with their caps, and the seeds and `ends` of the reachable
//! nodes. It does not depend on the order of the nodes or of the records, or
//! on how many parallel records join two nodes. A node that the entry does not
//! reach is [`ColdClass::Never`].
//!
//! Some consequences of the bounds:
//!
//! - A cycle with no other bound stays `Normal`: the only constraint on a loop
//!   `t → t` is `c(t) ≤ c(t)`. A loop whose only exit leads to a node seeded
//!   `Rare` also stays `Normal`; the exit node is `Rare`.
//! - A capped record does not make its target cold when an uncapped record
//!   from a `Normal` node also reaches it: a handler that normal flow also
//!   enters stays `Normal`, as far as its own records allow.
//! - When every path from the entry ends in a node seeded `Rare`, the entry is
//!   `Rare` too.
//!
//! # Cost
//!
//! The propagation starts every reachable node at its seed and lowers classes
//! until every bound holds. A class falls at most two times, and each fall
//! reads the records that leave and enter the node once. Each node keeps two
//! counters for the values of its leaving records and two for its entering
//! records, so a bound costs constant time, and a node with many records is
//! not read again for each fall of a neighbor. Time is O(n + m) for `n` node
//! slots and `m` reachable records, and the work list is an explicit queue,
//! so a chain of 10⁵ nodes needs no more call stack than a diamond.

extern crate alloc;
use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::block::BlockId;
use crate::graph::view::DenseId;

/// How often a node runs, in three classes. A colder class is a smaller
/// class: `Never < Rare < Normal`.
///
/// The classes describe the structure of the graph and the facts that a
/// [`ColdnessGraph`] reports. A consumer gives them a meaning through its
/// seeds, caps, and `ends`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum ColdClass {
    /// No execution reaches the node: the entry does not reach it, or every
    /// path to it passes a node that the bounds make `Never`.
    Never,
    /// The node runs only in executions that also take a capped record, reach
    /// a node seeded `Rare`, or end in a node that can end the execution.
    Rare,
    /// The bounds give no evidence that the node is cold.
    Normal,
}

/// A rooted graph with ordered successor records and the inputs of the
/// coldness propagation.
///
/// A record is one edge out of its source node; two records of one node can
/// name the same target. The trait asks for both directions of every record,
/// with its cap, so that a consumer can serve them from its own adjacency
/// tables without building a copy.
///
/// # Contract
///
/// - Every node that a method yields or receives has an index below
///   [`node_bound`](Self::node_bound).
/// - [`reachable`](Self::reachable) yields exactly the nodes that the entry
///   reaches through records, the entry included.
/// - [`incoming`](Self::incoming) mirrors [`records`](Self::records): for each
///   reachable node, it lists every record of a reachable node that names it,
///   once, with the same cap. It can also list records of nodes that the entry
///   does not reach; the propagation ignores them.
///
/// A graph that breaks the contract gets an unspecified result, and the
/// computation can panic.
///
/// # Examples
///
/// ```
/// use cfglib::{ColdClass, ColdnessClasses, ColdnessGraph};
///
/// /// Node 0 branches to a normal arm 1 and an exceptional arm 2. Node 3
/// /// aborts: it has no records and the seed `Rare`.
/// struct Branch;
///
/// const RECORDS: [&[(usize, ColdClass)]; 4] = [
///     &[(1, ColdClass::Normal), (2, ColdClass::Rare)],
///     &[],
///     &[(3, ColdClass::Normal)],
///     &[],
/// ];
///
/// impl ColdnessGraph for Branch {
///     type NodeId = usize;
///     fn node_bound(&self) -> usize {
///         RECORDS.len()
///     }
///     fn entry(&self) -> usize {
///         0
///     }
///     fn reachable(&self) -> impl Iterator<Item = usize> + '_ {
///         0..RECORDS.len()
///     }
///     fn records(&self, node: usize) -> impl Iterator<Item = (usize, ColdClass)> + '_ {
///         RECORDS[node].iter().copied()
///     }
///     fn incoming(&self, node: usize) -> impl Iterator<Item = (usize, ColdClass)> + '_ {
///         (0..RECORDS.len()).flat_map(move |source| {
///             RECORDS[source]
///                 .iter()
///                 .filter(move |&&(target, _)| target == node)
///                 .map(move |&(_, cap)| (source, cap))
///         })
///     }
///     fn seed(&self, node: usize) -> ColdClass {
///         if node == 3 { ColdClass::Rare } else { ColdClass::Normal }
///     }
///     fn ends(&self, _node: usize) -> bool {
///         false
///     }
/// }
///
/// let classes = ColdnessClasses::compute(&Branch);
/// assert_eq!(classes.class(0), ColdClass::Normal);
/// assert_eq!(classes.class(1), ColdClass::Normal);
/// assert_eq!(classes.class(2), ColdClass::Rare);
/// assert_eq!(classes.class(3), ColdClass::Rare);
/// ```
pub trait ColdnessGraph {
    /// Node identity used by this graph.
    type NodeId: DenseId;

    /// An exclusive upper bound on every node index this graph yields.
    fn node_bound(&self) -> usize;

    /// The entry node. Bound (F) does not apply to it.
    fn entry(&self) -> Self::NodeId;

    /// Iterate over every node that the entry reaches, the entry included.
    ///
    /// A node that appears again is skipped. The order does not change the
    /// result; the propagation visits the nodes first in this order, so a
    /// reverse postorder lowers most classes in one pass.
    fn reachable(&self) -> impl Iterator<Item = Self::NodeId> + '_;

    /// Iterate over the successor records of `node` as `(target, cap)` pairs.
    ///
    /// The cap is the hottest class that the record can pass on.
    /// [`ColdClass::Normal`] puts no limit on the record, and
    /// [`ColdClass::Rare`] marks a record that only rare executions take,
    /// such as an exceptional edge.
    fn records(&self, node: Self::NodeId) -> impl Iterator<Item = (Self::NodeId, ColdClass)> + '_;

    /// Iterate over the records that name `node` as `(source, cap)` pairs.
    fn incoming(&self, node: Self::NodeId) -> impl Iterator<Item = (Self::NodeId, ColdClass)> + '_;

    /// The hottest class that `node` can have: bound (S).
    fn seed(&self, node: Self::NodeId) -> ColdClass;

    /// Whether `node` can end the execution before it takes a record. Bound
    /// (B) then keeps the node at least `Rare` when all its records lead to
    /// `Never` nodes.
    fn ends(&self, node: Self::NodeId) -> bool;
}

/// The coldness class of every node of a [`ColdnessGraph`].
///
/// See the [module documentation](self) for the bounds that define the
/// classes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColdnessClasses<N = BlockId> {
    /// The class of each node slot; `Never` for a node the entry does not
    /// reach.
    class: Vec<ColdClass>,
    node: PhantomData<N>,
}

impl<N: DenseId> ColdnessClasses<N> {
    /// Computes the greatest assignment of classes that obeys the bounds (S),
    /// (B), and (F) on `graph`.
    ///
    /// Time is O(n + m) for `n` node slots and `m` reachable records.
    ///
    /// # Panics
    ///
    /// Panics when `graph` yields a node whose index is not below its
    /// [`node_bound`](ColdnessGraph::node_bound). A graph that breaks another
    /// rule of the [`ColdnessGraph`] contract can also make it panic.
    #[must_use]
    pub fn compute<G>(graph: &G) -> Self
    where
        G: ColdnessGraph<NodeId = N>,
    {
        let mut propagation = Propagation::new(graph);
        propagation.run();
        Self {
            class: propagation.slots.iter().map(|slot| slot.class).collect(),
            node: PhantomData,
        }
    }

    /// Returns the class of `node`.
    ///
    /// A node that the entry does not reach is `Never`, and so is a node at or
    /// above the node bound of the graph at compute time.
    #[must_use]
    pub fn class(&self, node: N) -> ColdClass {
        self.class
            .get(node.index())
            .copied()
            .unwrap_or(ColdClass::Never)
    }

    /// Returns the class of a record from `source` to `target` with the cap
    /// `cap`: `c(source) ∧ cap ∧ c(target)`.
    ///
    /// A record runs only as often as both of its ends and its cap allow.
    #[must_use]
    pub fn record(&self, source: N, cap: ColdClass, target: N) -> ColdClass {
        self.class(source).min(cap).min(self.class(target))
    }
}

/// How many records of one node have at least a given value.
#[derive(Debug, Clone, Copy, Default)]
struct Tally {
    /// The records of value `Rare` or `Normal`.
    at_least_rare: u32,
    /// The records of value `Normal`.
    normal: u32,
}

impl Tally {
    /// Counts one more record of value `value`.
    fn add(&mut self, value: ColdClass) {
        self.at_least_rare += u32::from(value >= ColdClass::Rare);
        self.normal += u32::from(value >= ColdClass::Normal);
    }

    /// Counts one record less of value `value`.
    fn remove(&mut self, value: ColdClass) {
        self.at_least_rare -= u32::from(value >= ColdClass::Rare);
        self.normal -= u32::from(value >= ColdClass::Normal);
    }

    /// Replaces one record of value `old` with one of value `new`.
    fn replace(&mut self, old: ColdClass, new: ColdClass) {
        self.remove(old);
        self.add(new);
    }

    /// Returns the hottest counted value, or `Never` when no record is
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
    /// The entry does not reach the node.
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
    /// Whether the node has a record, so that bound (B) applies.
    has_records: bool,
    /// The values `cap ∧ c(target)` of the records of the node.
    leaving: Tally,
    /// The values `cap ∧ c(source)` of the records from reachable nodes to
    /// the node.
    entering: Tally,
}

impl Slot {
    const UNREACHED: Self = Self {
        class: ColdClass::Never,
        mark: Mark::Unreached,
        has_records: false,
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
struct Propagation<'g, G: ColdnessGraph> {
    graph: &'g G,
    entry: G::NodeId,
    slots: Vec<Slot>,
    queue: VecDeque<G::NodeId>,
}

impl<'g, G: ColdnessGraph> Propagation<'g, G> {
    /// Starts every reachable node at its seed, queues it, and counts the
    /// values of the records.
    fn new(graph: &'g G) -> Self {
        let mut slots = vec![Slot::UNREACHED; graph.node_bound()];
        let mut queue = VecDeque::new();
        for node in graph.reachable() {
            let slot = &mut slots[node.index()];
            if slot.mark == Mark::Unreached {
                slot.class = graph.seed(node);
                slot.mark = Mark::Queued;
                queue.push_back(node);
            }
        }
        for &source in &queue {
            let from = slots[source.index()].class;
            for (target, cap) in graph.records(source) {
                let to = slots[target.index()].class;
                let slot = &mut slots[source.index()];
                slot.has_records = true;
                slot.leaving.add(cap.min(to));
                slots[target.index()].entering.add(cap.min(from));
            }
        }
        Self {
            graph,
            entry: graph.entry(),
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
        if slot.has_records {
            bound = match slot.leaving.hottest() {
                ColdClass::Never if self.graph.ends(node) => ColdClass::Rare,
                hottest => hottest,
            };
        }
        if node != self.entry {
            bound = bound.min(slot.entering.hottest());
        }
        bound
    }

    /// Lowers the class of `node` to `to`, updates the counters of the nodes
    /// that its records join, and queues those nodes.
    fn lower(&mut self, node: G::NodeId, to: ColdClass) {
        let from = core::mem::replace(&mut self.slots[node.index()].class, to);
        for (target, cap) in self.graph.records(node) {
            let (old, new) = (cap.min(from), cap.min(to));
            if old != new {
                self.slots[target.index()].entering.replace(old, new);
                self.push(target);
            }
        }
        for (source, cap) in self.graph.incoming(node) {
            if self.slots[source.index()].mark == Mark::Unreached {
                continue;
            }
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
