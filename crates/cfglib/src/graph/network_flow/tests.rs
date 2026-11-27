//! The solvers against textbook networks and against simple oracles:
//! Edmonds–Karp for the maximum flow, and negative-cycle cancelling followed
//! by Bellman–Ford successive shortest paths for the minimum-cost flow.

use alloc::collections::VecDeque;

use super::*;
use crate::test_util::rng::Rng;

/// An arc as `(from, to, capacity, cost)`.
type Arc = (usize, usize, i64, i64);

fn network(nodes: usize, arcs: &[Arc]) -> (FlowNetwork, Vec<ArcId>) {
    let mut network = FlowNetwork::with_capacity(nodes, arcs.len());
    for _ in 0..nodes {
        network.add_node();
    }
    let ids = arcs
        .iter()
        .map(|&(from, to, capacity, cost)| network.add_arc(from, to, capacity, cost))
        .collect();
    (network, ids)
}

/// Every arc within its capacity, and every node but the terminals balanced;
/// returns the net amount that leaves `source`.
fn check_flow(nodes: usize, arcs: &[Arc], flows: &[i64], source: usize, sink: usize) -> i64 {
    let mut balance = vec![0_i128; nodes];
    for (&(from, to, capacity, _), &flow) in arcs.iter().zip(flows) {
        assert!(
            (0..=capacity).contains(&flow),
            "flow {flow} on {from} -> {to}"
        );
        balance[from] -= i128::from(flow);
        balance[to] += i128::from(flow);
    }
    for (node, &amount) in balance.iter().enumerate() {
        if node != source && node != sink {
            assert_eq!(amount, 0, "node {node} is unbalanced");
        }
    }
    assert_eq!(balance[source], -balance[sink]);
    i64::try_from(-balance[source]).expect("a test flow fits i64")
}

fn edmonds_karp(nodes: usize, arcs: &[Arc], source: usize, sink: usize) -> i64 {
    let mut residual = vec![vec![0_i64; nodes]; nodes];
    for &(from, to, capacity, _) in arcs {
        if from != to {
            residual[from][to] += capacity;
        }
    }
    let mut total = 0;
    loop {
        let mut parent = vec![usize::MAX; nodes];
        parent[source] = source;
        let mut queue = VecDeque::from([source]);
        while let Some(node) = queue.pop_front() {
            for next in 0..nodes {
                if parent[next] == usize::MAX && residual[node][next] > 0 {
                    parent[next] = node;
                    queue.push_back(next);
                }
            }
        }
        if parent[sink] == usize::MAX {
            return total;
        }
        let mut amount = i64::MAX;
        let mut node = sink;
        while node != source {
            amount = amount.min(residual[parent[node]][node]);
            node = parent[node];
        }
        let mut node = sink;
        while node != source {
            residual[parent[node]][node] -= amount;
            residual[node][parent[node]] += amount;
            node = parent[node];
        }
        total += amount;
    }
}

/// The oracle's residual network: half arc `2a` along arc `a`, `2a + 1`
/// against it.
struct Oracle {
    nodes: usize,
    head: Vec<usize>,
    residual: Vec<i64>,
    cost: Vec<i64>,
}

impl Oracle {
    fn new(nodes: usize, arcs: &[Arc]) -> Self {
        let mut residual = Self {
            nodes,
            head: Vec::new(),
            residual: Vec::new(),
            cost: Vec::new(),
        };
        for &(from, to, capacity, cost) in arcs {
            residual.head.extend([to, from]);
            residual.residual.extend([capacity, 0]);
            residual.cost.extend([cost, -cost]);
        }
        residual
    }

    fn tail(&self, half: usize) -> usize {
        self.head[half ^ 1]
    }

    fn push(&mut self, half: usize, amount: i64) {
        self.residual[half] -= amount;
        self.residual[half ^ 1] += amount;
    }

    /// Bellman–Ford from `sources` at distance zero: the distances and the
    /// half arc each node was last improved by, and a node whose distance
    /// still fell in the last round, which a negative cycle leads to.
    fn bellman_ford(&self, sources: &[usize]) -> (Vec<i64>, Vec<usize>, Option<usize>) {
        let mut distance = vec![i64::MAX; self.nodes];
        let mut parent = vec![usize::MAX; self.nodes];
        for &source in sources {
            distance[source] = 0;
        }
        let mut last = None;
        for _ in 0..=self.nodes {
            last = None;
            for half in 0..self.head.len() {
                let (from, to) = (self.tail(half), self.head[half]);
                if self.residual[half] > 0
                    && distance[from] != i64::MAX
                    && distance[from] + self.cost[half] < distance[to]
                {
                    distance[to] = distance[from] + self.cost[half];
                    parent[to] = half;
                    last = Some(to);
                }
            }
            if last.is_none() {
                break;
            }
        }
        (distance, parent, last)
    }

    fn cancel_negative_cycles(&mut self) {
        let everyone: Vec<usize> = (0..self.nodes).collect();
        loop {
            let (_, parent, Some(mut node)) = self.bellman_ford(&everyone) else {
                return;
            };
            for _ in 0..self.nodes {
                node = self.tail(parent[node]);
            }
            let mut cycle = Vec::new();
            let mut at = node;
            loop {
                let half = parent[at];
                cycle.push(half);
                at = self.tail(half);
                if at == node {
                    break;
                }
            }
            let amount = cycle.iter().map(|&half| self.residual[half]).min();
            let amount = amount.expect("a cycle has an arc");
            for half in cycle {
                self.push(half, amount);
            }
        }
    }

    fn total_cost(&self) -> i64 {
        (0..self.head.len() / 2)
            .map(|arc| self.residual[2 * arc + 1] * self.cost[2 * arc])
            .sum()
    }
}

/// The value and cost of a minimum-cost flow of value `min(limit, maximum)`.
fn min_cost_oracle(
    nodes: usize,
    arcs: &[Arc],
    source: usize,
    sink: usize,
    limit: i64,
) -> (i64, i64) {
    let mut residual = Oracle::new(nodes, arcs);
    residual.cancel_negative_cycles();
    let mut value = 0;
    while value < limit {
        let (distance, parent, cycle) = residual.bellman_ford(&[source]);
        assert!(
            cycle.is_none(),
            "successive shortest paths keep no negative cycle"
        );
        if distance[sink] == i64::MAX {
            break;
        }
        let mut path = Vec::new();
        let mut node = sink;
        while node != source {
            path.push(parent[node]);
            node = residual.tail(parent[node]);
        }
        let amount = path
            .iter()
            .map(|&half| residual.residual[half])
            .fold(limit - value, i64::min);
        for half in path {
            residual.push(half, amount);
        }
        value += amount;
    }
    (value, residual.total_cost())
}

fn random_arcs(rng: &mut Rng, nodes: usize, negative: bool) -> Vec<Arc> {
    let count = rng.below(3 * nodes + 1);
    (0..count)
        .map(|_| {
            let capacity = i64::try_from(rng.below(10)).expect("small");
            let cost = i64::try_from(rng.below(16)).expect("small") - if negative { 6 } else { 0 };
            (rng.below(nodes), rng.below(nodes), capacity, cost)
        })
        .collect()
}

#[test]
fn the_textbook_network_carries_twenty_three() {
    // Cormen et al., figure 26.1: s = 0, v1..v4 = 1..4, t = 5.
    let arcs = [
        (0, 1, 16, 0),
        (0, 2, 13, 0),
        (2, 1, 4, 0),
        (1, 3, 12, 0),
        (3, 2, 9, 0),
        (2, 4, 14, 0),
        (4, 3, 7, 0),
        (3, 5, 20, 0),
        (4, 5, 4, 0),
    ];
    let (mut network, ids) = network(6, &arcs);
    assert_eq!(network.max_flow(0, 5), 23);
    let flows: Vec<i64> = ids.iter().map(|&arc| network.flow(arc)).collect();
    assert_eq!(check_flow(6, &arcs, &flows, 0, 5), 23);
    assert_eq!(
        network.max_flow(5, 0),
        0,
        "nothing leads back to the source"
    );
}

#[test]
fn a_crossing_arc_is_undone_when_a_later_path_needs_it() {
    // Unit diamond with the cross arc 1 -> 2: the first path takes it, and the
    // second cancels it.
    let arcs = [
        (0, 1, 1, 0),
        (0, 2, 1, 0),
        (1, 2, 1, 0),
        (1, 3, 1, 0),
        (2, 3, 1, 0),
    ];
    let (mut network, ids) = network(4, &arcs);
    assert_eq!(network.max_flow(0, 3), 2);
    assert_eq!(network.flow(ids[2]), 0);
}

#[test]
fn max_flow_matches_edmonds_karp_on_random_networks() {
    let mut rng = Rng::new(65);
    for _ in 0..1_000 {
        let nodes = 2 + rng.below(7);
        let arcs = random_arcs(&mut rng, nodes, false);
        let (source, sink) = (0, nodes - 1);
        let (mut network, ids) = network(nodes, &arcs);
        let value = network.max_flow(source, sink);
        assert_eq!(value, edmonds_karp(nodes, &arcs, source, sink), "{arcs:?}");
        let flows: Vec<i64> = ids.iter().map(|&arc| network.flow(arc)).collect();
        assert_eq!(check_flow(nodes, &arcs, &flows, source, sink), value);
    }
}

#[test]
fn min_cost_flow_matches_the_oracle_on_random_networks_with_negative_costs() {
    let mut rng = Rng::new(165);
    for _ in 0..1_000 {
        let nodes = 2 + rng.below(6);
        let arcs = random_arcs(&mut rng, nodes, true);
        let (source, sink) = (0, nodes - 1);
        let limit = match rng.below(3) {
            0 => 0,
            1 => i64::try_from(rng.below(20)).expect("small"),
            _ => i64::MAX,
        };
        let (mut network, ids) = network(nodes, &arcs);
        let solution = network
            .min_cost_flow(source, sink, limit)
            .expect("small networks do not overflow");
        let expected = min_cost_oracle(nodes, &arcs, source, sink, limit);
        assert_eq!(
            (solution.value, solution.cost),
            expected,
            "{arcs:?} with limit {limit}"
        );
        let flows: Vec<i64> = ids.iter().map(|&arc| network.flow(arc)).collect();
        assert_eq!(
            check_flow(nodes, &arcs, &flows, source, sink),
            solution.value
        );
        let cost: i64 = arcs
            .iter()
            .zip(&flows)
            .map(|(arc, flow)| arc.3 * flow)
            .sum();
        assert_eq!(cost, solution.cost);
    }
}

#[test]
fn reduced_costs_certify_the_minimum_cost_flow() {
    let mut rng = Rng::new(265);
    for _ in 0..300 {
        let nodes = 2 + rng.below(6);
        let arcs = random_arcs(&mut rng, nodes, true);
        let (mut network, ids) = network(nodes, &arcs);
        network
            .min_cost_flow(0, nodes - 1, i64::MAX)
            .expect("small networks do not overflow");
        for (&arc, &(_, _, capacity, _)) in ids.iter().zip(&arcs) {
            let reduced = network.reduced_cost(arc).expect("small potentials");
            let flow = network.flow(arc);
            if reduced > 0 {
                assert_eq!(flow, 0, "a dear arc carries flow");
            }
            if reduced < 0 {
                assert_eq!(flow, capacity, "a cheap arc has room");
            }
        }
    }
}

#[test]
fn a_negative_cycle_runs_at_its_capacity_in_a_circulation() {
    // No path from 0 to 1; the cycle 2 -> 3 -> 2 costs -1 per unit, up to 3.
    let arcs = [(2, 3, 3, -2), (3, 2, 5, 1), (2, 2, 4, -1)];
    let (mut network, ids) = network(4, &arcs);
    let solution = network.min_cost_flow(0, 1, 0).expect("small");
    assert_eq!(solution, FlowSolution { value: 0, cost: -7 });
    assert_eq!(
        ids.iter().map(|&arc| network.flow(arc)).collect::<Vec<_>>(),
        [3, 3, 4]
    );
}

#[test]
fn the_cheapest_path_is_filled_first_and_the_limit_is_honored() {
    let arcs = [(0, 1, 4, 5), (0, 1, 4, 1), (1, 2, 10, 0)];
    let (mut network, ids) = network(3, &arcs);
    let solution = network.min_cost_flow(0, 2, 6).expect("small");
    assert_eq!(
        solution,
        FlowSolution {
            value: 6,
            cost: 4 + 2 * 5
        }
    );
    assert_eq!(network.flow(ids[0]), 2);
    assert_eq!(network.flow(ids[1]), 4);
    let solution = network.min_cost_flow(0, 2, i64::MAX).expect("small");
    assert_eq!(
        solution,
        FlowSolution {
            value: 8,
            cost: 4 + 4 * 5
        }
    );
}

#[test]
fn supplies_are_met_at_the_least_cost() {
    // Node 0 supplies 5 and node 3 another 2; node 2 takes 7. The direct arc
    // from 0 is dear, the detour through 1 cheap but narrow.
    let arcs = [(0, 2, 10, 4), (0, 1, 3, 1), (1, 2, 10, 1), (3, 1, 10, 0)];
    let (mut network, ids) = network(4, &arcs);
    let cost = network.min_cost_supply(&[5, 0, -7, 2]).expect("small");
    let flows: Vec<i64> = ids.iter().map(|&arc| network.flow(arc)).collect();
    assert_eq!(flows, [2, 3, 5, 2]);
    assert_eq!(cost, 2 * 4 + 3 + 5);
}

#[test]
fn a_huge_negative_cycle_reports_overflow() {
    let arcs = [(0, 1, i64::MAX, -2), (1, 0, i64::MAX, 0)];
    let (mut network, _) = network(3, &arcs);
    assert_eq!(network.min_cost_flow(0, 2, 0), Err(FlowError::Overflow));
}

#[test]
fn max_flow_saturates_at_the_largest_value() {
    let arcs = [(0, 1, i64::MAX, 0), (0, 1, i64::MAX, 0)];
    let (mut network, ids) = network(2, &arcs);
    assert_eq!(network.max_flow(0, 1), i64::MAX);
    assert_eq!(network.flow(ids[0]) + network.flow(ids[1]), i64::MAX);
}

#[test]
fn every_solve_starts_from_the_zero_flow() {
    let arcs = [(0, 1, 3, 1), (1, 2, 3, 1)];
    let (mut network, ids) = network(3, &arcs);
    assert_eq!(network.flow(ids[0]), 0, "no solve yet");
    assert_eq!(network.max_flow(0, 2), 3);
    assert_eq!(network.max_flow(0, 2), 3);
    let solution = network.min_cost_flow(0, 2, 1).expect("small");
    assert_eq!(solution, FlowSolution { value: 1, cost: 2 });
    assert_eq!(network.flow(ids[0]), 1);

    // A new arc after a solve joins the next one.
    let bypass = network.add_arc(0, 2, 5, 0);
    assert_eq!(network.max_flow(0, 2), 8);
    assert_eq!(network.flow(bypass), 5);
    assert_eq!((network.node_count(), network.arc_count()), (3, 3));
}

#[test]
#[should_panic(expected = "the source is the sink")]
fn the_source_must_differ_from_the_sink() {
    let (mut network, _) = network(2, &[]);
    network.max_flow(1, 1);
}

#[test]
#[should_panic(expected = "an arc endpoint is not a node of the network")]
fn an_arc_needs_both_endpoints() {
    let (mut network, _) = network(2, &[]);
    network.add_arc(0, 2, 1, 0);
}

#[test]
#[should_panic(expected = "an arc capacity is negative")]
fn a_capacity_cannot_be_negative() {
    let (mut network, _) = network(2, &[]);
    network.add_arc(0, 1, -1, 0);
}

#[test]
#[should_panic(expected = "the flow limit is negative")]
fn a_limit_cannot_be_negative() {
    let (mut network, _) = network(2, &[]);
    let _ = network.min_cost_flow(0, 1, -1);
}

#[test]
fn errors_describe_themselves() {
    use alloc::string::ToString;
    assert_eq!(
        FlowError::Overflow.to_string(),
        "a flow cost, count, or distance exceeds the integer range"
    );
}
