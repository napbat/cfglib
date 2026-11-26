//! Integer flow networks: maximum flow and minimum-cost flow.
//!
//! A [`FlowNetwork`] is built by the caller arc by arc, with an `i64`
//! capacity and an `i64` cost on each arc, so the solvers depend on no graph
//! view. [`FlowNetwork::max_flow`] is Dinic's algorithm: breadth-first levels
//! and blocking flows found by a depth-first walk with an explicit stack.
//! [`FlowNetwork::min_cost_flow`] is capacity-scaled successive shortest paths
//! with Dijkstra's algorithm over reduced costs (Johnson potentials).
//!
//! # Representation
//!
//! Arc `a` is stored as two half arcs: `2a` in its direction and `2a + 1`
//! against it, each with a residual capacity. The flow of the arc is the
//! residual capacity of the half arc against it, so a solve reads and writes
//! the residual network in place and copies no graph. The half arcs out of each
//! node are grouped once per change of the network, in insertion order.
//!
//! # Determinism
//!
//! Every scan follows arc insertion order, and Dijkstra's algorithm breaks
//! ties between equal distances by the smaller node index, so a network gives
//! the same flows on every platform and in every run. All arithmetic is
//! integer: distances and potentials are `i128`, and an overflow is an error,
//! not a wrap.
//!
//! # Cost
//!
//! [`FlowNetwork::max_flow`] costs O(V²·E) in general and O(E·min(√E, V^(2/3)))
//! when every capacity is one. [`FlowNetwork::min_cost_flow`] runs one scale
//! for each bit of the largest amount it may move, and each scale saturates
//! the arcs whose reduced cost is negative and then routes the excess along
//! shortest paths; each search stops at the first node that can absorb flow.

extern crate alloc;
use alloc::collections::{BinaryHeap, VecDeque};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::fmt;

use crate::identity::{define_dense_id, word};

define_dense_id! {
    /// An arc of a [`FlowNetwork`], numbered in insertion order from zero.
    pub struct ArcId(u32);
}

impl ArcId {
    /// The arc at `position` in insertion order.
    pub(crate) fn from_position(position: usize) -> Self {
        Self(word(position))
    }
}

/// The value and total cost of a flow from [`FlowNetwork::min_cost_flow`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlowSolution {
    /// The amount that leaves the source, net of what returns to it.
    pub value: i64,
    /// The sum over the arcs of flow times cost.
    pub cost: i64,
}

/// A flow computation that exceeded the integer range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlowError {
    /// A total cost, a count, or a shortest-path distance does not fit its
    /// integer type.
    Overflow,
}

impl fmt::Display for FlowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => {
                formatter.write_str("a flow cost, count, or distance exceeds the integer range")
            }
        }
    }
}

impl core::error::Error for FlowError {}

/// A directed network with integer capacities and costs, and the flow that
/// the last solve left on it.
///
/// Nodes are dense indices from zero. Parallel arcs and self-loops are
/// allowed, and a cost may be negative. Each solve starts from the zero flow
/// and leaves its flow readable through [`flow`](Self::flow) until the next
/// solve.
///
/// # Examples
///
/// ```
/// use cfglib::FlowNetwork;
///
/// // Two routes from 0 to 3: a cheap one of capacity 2, a dear one of 5.
/// let mut network = FlowNetwork::new();
/// let [source, cheap, dear, sink] = [(); 4].map(|()| network.add_node());
/// let into_cheap = network.add_arc(source, cheap, 2, 1);
/// network.add_arc(cheap, sink, 2, 1);
/// let into_dear = network.add_arc(source, dear, 5, 3);
/// network.add_arc(dear, sink, 5, 3);
///
/// assert_eq!(network.max_flow(source, sink), 7);
///
/// let solution = network.min_cost_flow(source, sink, 3).expect("costs fit i64");
/// assert_eq!((solution.value, solution.cost), (3, 2 * 2 + 6));
/// assert_eq!(network.flow(into_cheap), 2);
/// assert_eq!(network.flow(into_dear), 1);
/// ```
#[derive(Debug, Clone, Default)]
pub struct FlowNetwork {
    nodes: usize,
    /// The node each half arc leads to.
    head: Vec<usize>,
    /// The residual capacity of each half arc.
    residual: Vec<i64>,
    /// The capacity of each arc.
    capacity: Vec<i64>,
    /// The cost of each arc, per unit of flow.
    cost: Vec<i64>,
    /// Where the half arcs out of each node start in `out`, with one more
    /// entry for the end.
    first: Vec<usize>,
    /// The half arcs grouped by the node they leave, in insertion order.
    out: Vec<usize>,
    /// The node potentials that the last minimum-cost solve ended with.
    potential: Vec<i128>,
}

impl FlowNetwork {
    /// An empty network.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty network with room for `nodes` nodes and `arcs` arcs.
    #[must_use]
    pub fn with_capacity(nodes: usize, arcs: usize) -> Self {
        Self {
            head: Vec::with_capacity(2 * arcs),
            residual: Vec::with_capacity(2 * arcs),
            capacity: Vec::with_capacity(arcs),
            cost: Vec::with_capacity(arcs),
            first: Vec::with_capacity(nodes + 1),
            out: Vec::with_capacity(2 * arcs),
            ..Self::default()
        }
    }

    /// Add a node and return its index.
    pub fn add_node(&mut self) -> usize {
        self.nodes += 1;
        self.nodes - 1
    }

    /// The number of nodes.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.nodes
    }

    /// The number of arcs.
    #[must_use]
    pub fn arc_count(&self) -> usize {
        self.capacity.len()
    }

    /// Add an arc from `from` to `to` that carries up to `capacity` units at
    /// `cost` each, and return its identity.
    ///
    /// # Panics
    ///
    /// Panics when `from` or `to` is not a node of the network, when
    /// `capacity` is negative, or when the network already holds `2^32` arcs.
    pub fn add_arc(&mut self, from: usize, to: usize, capacity: i64, cost: i64) -> ArcId {
        assert!(
            from < self.nodes && to < self.nodes,
            "an arc endpoint is not a node of the network"
        );
        assert!(capacity >= 0, "an arc capacity is negative");
        let id = ArcId::from_position(self.capacity.len());
        self.head.extend([to, from]);
        self.residual.extend([capacity, 0]);
        self.capacity.push(capacity);
        self.cost.push(cost);
        id
    }

    /// The flow that the last solve put on `arc`: `0` before any solve.
    ///
    /// # Panics
    ///
    /// Panics when `arc` is not an arc of this network.
    #[must_use]
    pub fn flow(&self, arc: ArcId) -> i64 {
        self.residual[2 * arc.index() + 1]
    }

    /// Find a maximum flow from `source` to `sink` and return its value.
    ///
    /// The value saturates: a network that could carry more than `i64::MAX`
    /// gets a valid flow of value `i64::MAX`. The arc flows are readable
    /// afterwards through [`flow`](Self::flow).
    ///
    /// # Panics
    ///
    /// Panics when `source` or `sink` is not a node of the network, or when
    /// they are the same node.
    pub fn max_flow(&mut self, source: usize, sink: usize) -> i64 {
        self.max_flow_up_to(source, sink, i64::MAX)
    }

    /// Find a flow of value `min(limit, maximum)` from `source` to `sink` with
    /// the least total cost.
    ///
    /// The flow may run around cycles: a cycle of negative cost runs at its
    /// capacity, and with `limit` zero the result is a minimum-cost
    /// circulation. The arc flows are readable afterwards through
    /// [`flow`](Self::flow).
    ///
    /// # Errors
    ///
    /// [`FlowError::Overflow`] when the total cost does not fit `i64`, or a
    /// shortest-path distance does not fit `i128`. The arc flows are then
    /// unspecified.
    ///
    /// # Panics
    ///
    /// Panics when `source` or `sink` is not a node of the network, when they
    /// are the same node, or when `limit` is negative.
    pub fn min_cost_flow(
        &mut self,
        source: usize,
        sink: usize,
        limit: i64,
    ) -> Result<FlowSolution, FlowError> {
        assert!(limit >= 0, "the flow limit is negative");
        let value = self.max_flow_up_to(source, sink, limit);
        let mut excess = vec![0_i128; self.nodes];
        excess[source] = i128::from(value);
        excess[sink] = -i128::from(value);
        Ok(FlowSolution {
            value,
            cost: self.balance(excess)?,
        })
    }

    /// Find the flow of least total cost that leaves `supply[v]` more at each
    /// node `v` than enters it, and return its cost. The supplies must add up
    /// to zero, and some flow must meet them.
    pub(crate) fn min_cost_supply(&mut self, supply: &[i64]) -> Result<i64, FlowError> {
        debug_assert_eq!(supply.len(), self.nodes, "one supply for each node");
        self.build_adjacency();
        self.balance(supply.iter().copied().map(i128::from).collect())
    }

    /// Route `excess`, from the zero flow, along shortest paths at
    /// decreasing scales, and return the total cost.
    fn balance(&mut self, mut excess: Vec<i128>) -> Result<i64, FlowError> {
        self.reset();
        self.potential.clear();
        self.potential.resize(self.nodes, 0);
        let mut search = Search::new(self.nodes);
        let mut scale = self.initial_scale(&excess);
        while scale > 0 {
            self.saturate_negative_arcs(scale, &mut excess)?;
            // Routing only lowers excess, so one pass finds every node that
            // still has at least `scale` to send.
            for start in 0..self.nodes {
                while excess[start] >= i128::from(scale) {
                    let Some(end) = self.shortest_path(start, scale, &excess, &mut search)? else {
                        break;
                    };
                    self.augment(start, end, &mut excess, &search);
                }
            }
            scale /= 2;
        }
        debug_assert!(
            excess.iter().all(|&amount| amount == 0),
            "the supplies are feasible"
        );
        self.total_cost()
    }

    /// The reduced cost of `arc` under the potentials of the last
    /// minimum-cost solve: zero when the arc can carry more or less flow in
    /// some flow of least cost, positive when no such flow uses it, and
    /// negative when every such flow saturates it.
    pub(crate) fn reduced_cost(&self, arc: ArcId) -> Result<i128, FlowError> {
        self.half_reduced_cost(2 * arc.index())
    }

    /// The capacity of `arc`.
    pub(crate) fn capacity(&self, arc: ArcId) -> i64 {
        self.capacity[arc.index()]
    }

    fn check_terminals(&self, source: usize, sink: usize) {
        assert!(
            source < self.nodes && sink < self.nodes,
            "a terminal is not a node of the network"
        );
        assert_ne!(source, sink, "the source is the sink");
    }

    /// Start from the zero flow.
    fn reset(&mut self) {
        for (arc, &capacity) in self.capacity.iter().enumerate() {
            self.residual[2 * arc] = capacity;
            self.residual[2 * arc + 1] = 0;
        }
    }

    /// Group the half arcs by the node they leave, unless that is done.
    fn build_adjacency(&mut self) {
        if self.first.len() == self.nodes + 1 && self.out.len() == self.head.len() {
            return;
        }
        self.first.clear();
        self.first.resize(self.nodes + 1, 0);
        for half in 0..self.head.len() {
            self.first[self.head[half ^ 1] + 1] += 1;
        }
        for node in 0..self.nodes {
            self.first[node + 1] += self.first[node];
        }
        let mut fill = self.first.clone();
        self.out.clear();
        self.out.resize(self.head.len(), 0);
        for half in 0..self.head.len() {
            let tail = self.head[half ^ 1];
            self.out[fill[tail]] = half;
            fill[tail] += 1;
        }
    }

    fn arcs_out(&self, node: usize) -> &[usize] {
        &self.out[self.first[node]..self.first[node + 1]]
    }

    /// Dinic's algorithm, stopped once the flow reaches `limit`: every phase
    /// adds at least one unit, so a limit of `k` costs at most `k` phases.
    pub(crate) fn max_flow_up_to(&mut self, source: usize, sink: usize, limit: i64) -> i64 {
        self.check_terminals(source, sink);
        self.reset();
        self.build_adjacency();
        let mut level = vec![usize::MAX; self.nodes];
        let mut current = vec![0; self.nodes];
        let mut queue = VecDeque::new();
        let mut path = Vec::new();
        let mut total = 0;
        while total < limit && self.levels(source, sink, &mut level, &mut queue) {
            current.copy_from_slice(&self.first[..self.nodes]);
            while total < limit {
                let pushed = self.blocking_step(
                    source,
                    sink,
                    limit - total,
                    &level,
                    &mut current,
                    &mut path,
                );
                if pushed == 0 {
                    break;
                }
                total += pushed;
            }
        }
        total
    }

    /// The nodes that `source` reaches over arcs with residual capacity in
    /// the flow of the last solve. After a maximum flow they are the source
    /// side of the minimum cut closest to the source, which every maximum flow
    /// gives alike.
    pub(crate) fn residual_reachable(&self, source: usize) -> Vec<bool> {
        let mut level = vec![usize::MAX; self.nodes];
        self.levels(source, source, &mut level, &mut VecDeque::new());
        level.into_iter().map(|level| level != usize::MAX).collect()
    }

    /// Number the nodes by their distance from `source` over arcs with
    /// residual capacity, and report whether `sink` is reached.
    fn levels(
        &self,
        source: usize,
        sink: usize,
        level: &mut [usize],
        queue: &mut VecDeque<usize>,
    ) -> bool {
        level.fill(usize::MAX);
        level[source] = 0;
        queue.clear();
        queue.push_back(source);
        while let Some(node) = queue.pop_front() {
            for &half in self.arcs_out(node) {
                let next = self.head[half];
                if self.residual[half] > 0 && level[next] == usize::MAX {
                    level[next] = level[node] + 1;
                    queue.push_back(next);
                }
            }
        }
        level[sink] != usize::MAX
    }

    /// Push flow along one path that climbs the levels from `source` to
    /// `sink`, at most `limit`, and return the amount; `0` once no such path
    /// is left. `current` is the next arc to try at each node, so an arc that
    /// led nowhere is not tried again within the phase.
    fn blocking_step(
        &mut self,
        source: usize,
        sink: usize,
        limit: i64,
        level: &[usize],
        current: &mut [usize],
        path: &mut Vec<usize>,
    ) -> i64 {
        path.clear();
        let mut node = source;
        loop {
            if node == sink {
                let amount = path
                    .iter()
                    .map(|&half| self.residual[half])
                    .fold(limit, i64::min);
                for &half in path.iter() {
                    self.residual[half] -= amount;
                    self.residual[half ^ 1] += amount;
                }
                return amount;
            }
            let end = self.first[node + 1];
            let mut advanced = false;
            while current[node] < end {
                let half = self.out[current[node]];
                let next = self.head[half];
                if self.residual[half] > 0 && level[next] == level[node] + 1 {
                    path.push(half);
                    node = next;
                    advanced = true;
                    break;
                }
                current[node] += 1;
            }
            if !advanced {
                let Some(half) = path.pop() else {
                    return 0;
                };
                node = self.head[half ^ 1];
                current[node] += 1;
            }
        }
    }

    /// The largest power of two that is at most the largest amount a solve
    /// can move: the largest excess, or all the capacity of negative cost.
    fn initial_scale(&self, excess: &[i128]) -> i64 {
        let negative: i128 = self
            .capacity
            .iter()
            .zip(&self.cost)
            .filter(|&(_, &cost)| cost < 0)
            .map(|(&capacity, _)| i128::from(capacity))
            .sum();
        let excess = excess.iter().map(|amount| amount.abs()).max().unwrap_or(0);
        let largest = negative.max(excess).min(i128::from(i64::MAX));
        let largest = u64::try_from(largest).expect("the amount is between zero and i64::MAX");
        if largest == 0 {
            0
        } else {
            1 << largest.ilog2()
        }
    }

    /// The cost of half arc `half` reduced by the potentials of its ends.
    fn half_reduced_cost(&self, half: usize) -> Result<i128, FlowError> {
        let cost = i128::from(self.cost[half / 2]);
        let cost = if half.is_multiple_of(2) { cost } else { -cost };
        cost.checked_add(self.potential[self.head[half ^ 1]])
            .and_then(|cost| cost.checked_sub(self.potential[self.head[half]]))
            .ok_or(FlowError::Overflow)
    }

    /// Saturate every half arc with at least `scale` residual capacity and a
    /// negative reduced cost, moving the excess to its head.
    fn saturate_negative_arcs(&mut self, scale: i64, excess: &mut [i128]) -> Result<(), FlowError> {
        for half in 0..self.head.len() {
            let amount = self.residual[half];
            if amount >= scale && self.half_reduced_cost(half)? < 0 {
                self.residual[half] = 0;
                self.residual[half ^ 1] += amount;
                excess[self.head[half ^ 1]] -= i128::from(amount);
                excess[self.head[half]] += i128::from(amount);
            }
        }
        Ok(())
    }

    /// Find the nearest node that can absorb at least `scale` from `start`,
    /// over half arcs with at least `scale` residual capacity, and update the
    /// potentials so that every such arc keeps a nonnegative reduced cost.
    fn shortest_path(
        &mut self,
        start: usize,
        scale: i64,
        excess: &[i128],
        search: &mut Search,
    ) -> Result<Option<usize>, FlowError> {
        search.reset();
        search.reach(start, 0, usize::MAX);
        let mut end = None;
        while let Some(Reverse((distance, node))) = search.heap.pop() {
            if search.done[node] {
                continue;
            }
            search.done[node] = true;
            search.finished.push(node);
            if excess[node] <= -i128::from(scale) {
                end = Some(node);
                break;
            }
            for &half in self.arcs_out(node) {
                let next = self.head[half];
                if self.residual[half] < scale || search.done[next] {
                    continue;
                }
                let candidate = distance
                    .checked_add(self.half_reduced_cost(half)?)
                    .ok_or(FlowError::Overflow)?;
                if candidate < search.distance[next] {
                    search.reach(next, candidate, half);
                }
            }
        }
        let Some(end) = end else {
            return Ok(None);
        };
        // A node settled at distance `d` moves by `d - bound`, and the others
        // keep their potential: every arc that the search relaxed keeps a
        // nonnegative reduced cost, and the arcs of the path reduce to zero.
        let bound = search.distance[end];
        for &node in &search.finished {
            self.potential[node] = self.potential[node]
                .checked_add(search.distance[node] - bound)
                .ok_or(FlowError::Overflow)?;
        }
        Ok(Some(end))
    }

    /// Push as much as the path from `start` to `end` that the last search
    /// found carries, `start` holds, and `end` absorbs.
    fn augment(&mut self, start: usize, end: usize, excess: &mut [i128], search: &Search) {
        let mut amount = excess[start].min(-excess[end]);
        let mut node = end;
        while node != start {
            let half = search.parent[node];
            amount = amount.min(i128::from(self.residual[half]));
            node = self.head[half ^ 1];
        }
        let amount = i64::try_from(amount).expect("the amount is at most one residual capacity");
        let mut node = end;
        while node != start {
            let half = search.parent[node];
            self.residual[half] -= amount;
            self.residual[half ^ 1] += amount;
            node = self.head[half ^ 1];
        }
        excess[start] -= i128::from(amount);
        excess[end] += i128::from(amount);
    }

    fn total_cost(&self) -> Result<i64, FlowError> {
        let total = self
            .cost
            .iter()
            .enumerate()
            .try_fold(0_i128, |total, (arc, &cost)| {
                total.checked_add(i128::from(self.residual[2 * arc + 1]) * i128::from(cost))
            })
            .ok_or(FlowError::Overflow)?;
        i64::try_from(total).map_err(|_| FlowError::Overflow)
    }
}

/// The state of one shortest-path search, reset in time proportional to what
/// the last search touched.
struct Search {
    distance: Vec<i128>,
    /// The half arc each reached node was reached by.
    parent: Vec<usize>,
    done: Vec<bool>,
    touched: Vec<usize>,
    /// The settled nodes, in settling order.
    finished: Vec<usize>,
    heap: BinaryHeap<Reverse<(i128, usize)>>,
}

impl Search {
    fn new(nodes: usize) -> Self {
        Self {
            distance: vec![i128::MAX; nodes],
            parent: vec![usize::MAX; nodes],
            done: vec![false; nodes],
            touched: Vec::new(),
            finished: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    fn reset(&mut self) {
        for &node in &self.touched {
            self.distance[node] = i128::MAX;
            self.done[node] = false;
        }
        self.touched.clear();
        self.finished.clear();
        self.heap.clear();
    }

    fn reach(&mut self, node: usize, distance: i128, half: usize) {
        if self.distance[node] == i128::MAX {
            self.touched.push(node);
        }
        self.distance[node] = distance;
        self.parent[node] = half;
        self.heap.push(Reverse((distance, node)));
    }
}

#[cfg(test)]
mod tests;
