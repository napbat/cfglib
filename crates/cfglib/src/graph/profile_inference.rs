//! Profile inference: block and edge counts that conserve flow, repaired from
//! noisy and partial estimates.
//!
//! Sampled hardware profiles, and counts carried over from an older binary, are
//! noisy, partial, and inconsistent. Every count-driven consumer assumes flow
//! conservation: a block runs as often as the edges into it are taken, and as
//! often as the edges out of it are. [`InferredCounts::compute`] repairs the
//! estimates into such counts as a minimum-cost flow, after W. He, J. Mestre,
//! S. Pupyrev, L. Wang and H. Yu, *Profile Inference Revisited* (POPL 2022),
//! with estimates on edges as well as blocks, after A. Ayupov, M. Panchenko
//! and S. Pupyrev, *Stale Profile Matching* (CC 2024).
//!
//! # Model
//!
//! - Each block is split into an `in` and an `out` node, and the arcs between
//!   them carry its count. Each edge is an arc from the `out` node of its
//!   source to the `in` node of its target.
//! - A super-source feeds the `in` node of the root, the `out` node of every
//!   exit drains to a super-sink, and the super-sink returns to the
//!   super-source, so the counts are a circulation.
//! - A block or an edge with an estimate `c` costs
//!   [`decrease`](InferenceCosts::decrease) for each unit below `c` and
//!   [`increase`](InferenceCosts::increase) for each unit above it: an arc of
//!   capacity `c` and cost `-decrease` beside an unbounded arc of cost
//!   `increase`. Everything else costs nothing. Putting the cost of an edge
//!   estimate on the arc of the edge is the subdivision of the CC 2024
//!   reduction, without the extra node.
//! - The counts are a minimum-cost circulation of that network
//!   ([`FlowNetwork::min_cost_flow`] with a limit of zero).
//!
//! # Branch priors
//!
//! The estimates rarely pin every count, so many circulations cost the least.
//! Among them, the counts are the ones closest to the branch priors, as He et
//! al. prefer. The node potentials of the first solve describe every
//! circulation of least cost: an arc of positive reduced cost carries nothing
//! in any of them, an arc of negative reduced cost is full in all of them, and
//! the other arcs are free. A second solve over the free arcs alone minimizes
//! the sum over the edges of the distance between the count and the expected
//! count. The result is lexicographically best: the least deviation from the
//! estimates first, then the least deviation from the priors, with no weight
//! that trades one for the other.
//!
//! The expected count of an edge is its [`BlockFrequencies`] under the priors
//! and the loops of [`LoopForest::compute`], times the count of the root in the
//! first solve. A missing prior splits evenly, as
//! [`BlockFrequencies::compute`] describes, so blocks without estimates share
//! the flow evenly among their successors.
//!
//! # Caveats
//!
//! - A block that is not an exit can only pass its flow on: when an estimated
//!   block has no successor and is not an exit, its count is zero.
//! - Conservation is all the network asks for, so a cycle can carry flow that
//!   the root does not send when estimates on the cycle make that cheaper.
//!
//! # Cost
//!
//! Two minimum-cost flows over `2n + 2` nodes and `n + m` to `2(n + m)` arcs
//! plus one for each exit (see [`FlowNetwork`]), and the dominator tree, loop
//! forest, and frequencies of the graph.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use super::dominator::DominatorTree;
use super::edge_view::EdgeView;
use super::frequency::{BlockFrequencies, FREQUENCY_SHIFT, FrequencyConfig, Probability};
use super::loop_forest::LoopForest;
use super::network_flow::{ArcId, FlowError, FlowNetwork};
use super::view::{DenseId, RootedView};
use crate::block::BlockId;
use crate::edge::EdgeId;
use crate::identity::{NO_INDEX, slot, word};

/// The capacity of an arc that any amount may take.
const UNBOUNDED: i64 = i64::MAX;

/// The price of one unit of deviation from an estimate. Neither price may be
/// negative: a negative price would reward running a count without bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InferenceCosts {
    /// The cost of each unit that a count runs above its estimate.
    pub increase: i64,
    /// The cost of each unit that a count runs below its estimate.
    pub decrease: i64,
}

/// Block and edge counts that conserve flow, inferred from estimates.
///
/// See the [module documentation](self) for the model and the tie-breaking by
/// branch priors.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, Edge, EdgeKind, InferenceCosts, InferredCounts, Probability};
///
/// // entry -> left (30%) or right (70%), both -> join. Only the entry and the
/// // join were sampled.
/// let mut cfg = Cfg::<u32>::new();
/// let (left, right, join) = (cfg.new_block(), cfg.new_block(), cfg.new_block());
/// let to_left = cfg.add_edge(cfg.entry(), left, Edge::new(EdgeKind::ConditionalTrue).with_weight(0.3));
/// cfg.add_edge(cfg.entry(), right, Edge::new(EdgeKind::ConditionalFalse).with_weight(0.7));
/// cfg.add_edge(left, join, EdgeKind::Fallthrough);
/// cfg.add_edge(right, join, EdgeKind::Fallthrough);
///
/// let counts = InferredCounts::compute(
///     &cfg,
///     |block| block == join,
///     |block| (block == cfg.entry() || block == join).then_some(1_000),
///     |_| None,
///     |edge| cfg.edge(edge).weight().and_then(Probability::from_f64),
///     &InferenceCosts { increase: 1, decrease: 1 },
/// )?;
/// assert_eq!(counts.block(left), 300);
/// assert_eq!(counts.edge(to_left), 300);
/// assert_eq!(counts.block(right), 700);
/// # Ok::<(), cfglib::FlowError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferredCounts<N = BlockId, E = EdgeId> {
    /// The count of each node slot.
    block: Vec<u64>,
    /// The count of each edge slot.
    edge: Vec<u64>,
    ids: PhantomData<(N, E)>,
}

impl<N: DenseId, E: DenseId> InferredCounts<N, E> {
    /// Infer the counts of the blocks and edges of `graph`.
    ///
    /// `is_exit` names the blocks where an execution can end.
    /// `block_estimate` and `edge_estimate` give the measured counts, or
    /// `None` where nothing was measured. `branch_prior` gives the branch
    /// probability of an edge relative to the other edges leaving its source,
    /// or `None`. `costs` prices a deviation from an estimate.
    ///
    /// # Errors
    ///
    /// [`FlowError::Overflow`] when an estimate exceeds `i64::MAX`, or when a
    /// total cost or count exceeds the range of [`FlowNetwork`].
    ///
    /// # Panics
    ///
    /// Panics when [`InferenceCosts::increase`] or
    /// [`InferenceCosts::decrease`] is negative.
    pub fn compute<G>(
        graph: &G,
        is_exit: impl Fn(N) -> bool,
        block_estimate: impl Fn(N) -> Option<u64>,
        edge_estimate: impl Fn(E) -> Option<u64>,
        branch_prior: impl Fn(E) -> Option<Probability>,
        costs: &InferenceCosts,
    ) -> Result<Self, FlowError>
    where
        G: RootedView<NodeId = N> + EdgeView<EdgeId = E>,
    {
        assert!(
            costs.increase >= 0 && costs.decrease >= 0,
            "an inference cost is negative"
        );
        let mut model = Model::new(graph.node_bound(), graph.edge_bound());
        for node in graph.node_ids() {
            let index = node.index();
            model.block[index] =
                word(model.add(2 * index, 2 * index + 1, block_estimate(node), costs)?);
        }
        for node in graph.node_ids() {
            for edge in graph.outgoing(node) {
                let target = graph.edge(edge).target().index();
                model.edge[edge.index()] = word(model.add(
                    2 * node.index() + 1,
                    2 * target,
                    edge_estimate(edge),
                    costs,
                )?);
            }
        }
        let entry = model.add(model.source, 2 * graph.root().index(), None, costs)?;
        for node in graph.node_ids().filter(|&node| is_exit(node)) {
            model.add(2 * node.index() + 1, model.sink, None, costs)?;
        }
        model.add(model.sink, model.source, None, costs)?;
        model.network.min_cost_flow(model.source, model.sink, 0)?;

        let runs = model
            .network
            .flow(ArcId::from_position(model.counts[entry].first));
        let expected = expected_counts(graph, branch_prior, runs);
        let mut target = vec![None; model.counts.len()];
        for (edge, &count) in model.edge.iter().enumerate() {
            if count != NO_INDEX {
                target[slot(count)] = Some(expected[edge]);
            }
        }
        let counts = model.closest_to(&target)?;
        let read = |count: u32| counts.get(slot(count)).copied().unwrap_or(0);
        Ok(Self {
            block: model.block.iter().map(|&count| read(count)).collect(),
            edge: model.edge.iter().map(|&count| read(count)).collect(),
            ids: PhantomData,
        })
    }

    /// The count of `node`: `0` for a node at or above the node bound of the
    /// graph at compute time.
    #[must_use]
    pub fn block(&self, node: N) -> u64 {
        self.block.get(node.index()).copied().unwrap_or(0)
    }

    /// The count of `edge`: `0` for an edge at or above the edge bound of the
    /// graph at compute time.
    #[must_use]
    pub fn edge(&self, edge: E) -> u64 {
        self.edge.get(edge.index()).copied().unwrap_or(0)
    }
}

/// The expected count of every edge slot: its frequency under the priors,
/// times `runs` executions of the root.
fn expected_counts<G>(
    graph: &G,
    branch_prior: impl Fn(G::EdgeId) -> Option<Probability>,
    runs: i64,
) -> Vec<i64>
where
    G: RootedView + EdgeView,
{
    let dominators = DominatorTree::compute(graph);
    let loops = LoopForest::compute(graph, &dominators);
    let frequencies =
        BlockFrequencies::compute(graph, &loops, branch_prior, FrequencyConfig::new());
    let runs = u128::try_from(runs).expect("a count is not negative");
    let mut expected = vec![0; graph.edge_bound()];
    for edge in graph.edge_ids() {
        let product = runs.saturating_mul(u128::from(frequencies.edge(edge)));
        let rounded = product.saturating_add(1 << (FREQUENCY_SHIFT - 1)) >> FREQUENCY_SHIFT;
        expected[edge.index()] = i64::try_from(rounded).unwrap_or(i64::MAX);
    }
    expected
}

/// One count of the model, a block, an edge, or a link between the super
/// nodes, as the consecutive arcs between two nodes whose flows add up to it.
struct Count {
    tail: usize,
    head: usize,
    first: usize,
    len: usize,
}

/// The network of the counts, and where each block and edge is in it.
struct Model {
    network: FlowNetwork,
    source: usize,
    sink: usize,
    counts: Vec<Count>,
    /// The count of each node slot, [`NO_INDEX`] for a removed slot.
    block: Vec<u32>,
    /// The count of each edge slot, [`NO_INDEX`] for a removed slot.
    edge: Vec<u32>,
}

impl Model {
    fn new(node_bound: usize, edge_bound: usize) -> Self {
        let nodes = 2 * node_bound + 2;
        let mut network = FlowNetwork::with_capacity(nodes, 2 * (node_bound + edge_bound) + 2);
        for _ in 0..nodes {
            network.add_node();
        }
        Self {
            network,
            source: nodes - 2,
            sink: nodes - 1,
            counts: Vec::new(),
            block: vec![NO_INDEX; node_bound],
            edge: vec![NO_INDEX; edge_bound],
        }
    }

    /// Add a count from `tail` to `head` with its estimate, and return its
    /// index.
    fn add(
        &mut self,
        tail: usize,
        head: usize,
        estimate: Option<u64>,
        costs: &InferenceCosts,
    ) -> Result<usize, FlowError> {
        let first = self.network.arc_count();
        match estimate {
            None => {
                self.network.add_arc(tail, head, UNBOUNDED, 0);
            }
            Some(estimate) => {
                let estimate = i64::try_from(estimate).map_err(|_| FlowError::Overflow)?;
                if estimate > 0 {
                    self.network.add_arc(tail, head, estimate, -costs.decrease);
                }
                self.network.add_arc(tail, head, UNBOUNDED, costs.increase);
            }
        }
        self.counts.push(Count {
            tail,
            head,
            first,
            len: self.network.arc_count() - first,
        });
        Ok(self.counts.len() - 1)
    }

    fn arcs(count: &Count) -> impl Iterator<Item = ArcId> {
        (count.first..count.first + count.len).map(ArcId::from_position)
    }

    /// Among the circulations of least cost, the one whose counts with a
    /// target are closest to it in total, as the value of every count.
    fn closest_to(&self, target: &[Option<i64>]) -> Result<Vec<u64>, FlowError> {
        let nodes = self.network.node_count();
        let mut second = FlowNetwork::with_capacity(nodes, self.network.arc_count());
        for _ in 0..nodes {
            second.add_node();
        }
        // A fixed arc keeps its flow in every circulation of least cost: the
        // free arcs must bring that flow to its tail and take it from its
        // head.
        let mut supply = vec![0_i64; nodes];
        let mut lower = Vec::with_capacity(self.counts.len());
        let mut pieces = Vec::with_capacity(self.counts.len());
        for (count, &target) in self.counts.iter().zip(target) {
            let (mut fixed, mut free) = (0_i64, 0_i64);
            for arc in Self::arcs(count) {
                if self.network.reduced_cost(arc)? == 0 {
                    free = free.saturating_add(self.network.capacity(arc));
                } else {
                    fixed = fixed
                        .checked_add(self.network.flow(arc))
                        .ok_or(FlowError::Overflow)?;
                }
            }
            supply[count.tail] = supply[count.tail]
                .checked_sub(fixed)
                .ok_or(FlowError::Overflow)?;
            supply[count.head] = supply[count.head]
                .checked_add(fixed)
                .ok_or(FlowError::Overflow)?;
            let first = second.arc_count();
            if free > 0 {
                if let Some(target) = target {
                    let below = target.saturating_sub(fixed).clamp(0, free);
                    if below > 0 {
                        second.add_arc(count.tail, count.head, below, -1);
                    }
                    if free > below {
                        second.add_arc(count.tail, count.head, free - below, 1);
                    }
                } else {
                    second.add_arc(count.tail, count.head, free, 0);
                }
            }
            lower.push(fixed);
            pieces.push(first..second.arc_count());
        }
        second.min_cost_supply(&supply)?;
        lower
            .iter()
            .zip(pieces)
            .map(|(&fixed, arcs)| {
                let count = arcs
                    .map(ArcId::from_position)
                    .try_fold(fixed, |sum, arc| sum.checked_add(second.flow(arc)));
                count
                    .and_then(|count| u64::try_from(count).ok())
                    .ok_or(FlowError::Overflow)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
