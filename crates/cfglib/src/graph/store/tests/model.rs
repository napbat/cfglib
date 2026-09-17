//! A randomized comparison of the store against a naive reference model.
//!
//! The store's correctness claims — insertion order across base and delta,
//! liveness, mirrored adjacency, renumbering after compaction — are all
//! properties of *sequences of operations*, so the test applies a long
//! deterministic sequence to both the store and a `Vec`-backed model that
//! implements the same contract in the obvious way, and compares everything
//! observable after every single operation.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::graph::store::{Graph, Id, NodeTag};

/// A xorshift generator, so the sequence is reproducible without a
/// dependency and without a table of pre-baked operations.
struct Rng(u64);

impl Rng {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut state = self.0;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.0 = state;
        state
    }

    /// A value in `0..bound`, which must be nonzero.
    fn below(&mut self, bound: usize) -> usize {
        let bound = u64::try_from(bound).expect("bound fits in u64");
        usize::try_from(self.next_u64() % bound).expect("a value below bound fits in usize")
    }
}

#[derive(Clone)]
struct ModelNode {
    payload: u64,
    live: bool,
}

#[derive(Clone)]
struct ModelEdge {
    source: usize,
    target: usize,
    payload: u64,
    live: bool,
}

/// The obvious implementation: dense slot vectors plus per-node adjacency
/// lists that keep every edge ever appended, filtered by liveness on read.
#[derive(Default)]
struct Model {
    nodes: Vec<ModelNode>,
    edges: Vec<ModelEdge>,
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
}

impl Model {
    fn add_node(&mut self, payload: u64) -> usize {
        self.nodes.push(ModelNode {
            payload,
            live: true,
        });
        self.outgoing.push(Vec::new());
        self.incoming.push(Vec::new());
        self.nodes.len() - 1
    }

    fn add_edge(&mut self, source: usize, target: usize, payload: u64) -> usize {
        let slot = self.edges.len();
        self.edges.push(ModelEdge {
            source,
            target,
            payload,
            live: true,
        });
        self.outgoing[source].push(slot);
        self.incoming[target].push(slot);
        slot
    }

    /// Move one endpoint, keeping the edge's identity and appending it to
    /// the new endpoint's adjacency — the contract
    /// [`redirect_edge_source`](Graph::redirect_edge_source) states.
    fn redirect_edge(&mut self, edge: usize, endpoint: usize, outgoing: bool) -> usize {
        let record = &mut self.edges[edge];
        let moved = if outgoing {
            &mut record.source
        } else {
            &mut record.target
        };
        let previous = *moved;
        if previous == endpoint {
            return previous;
        }
        *moved = endpoint;
        let adjacency = if outgoing {
            &mut self.outgoing
        } else {
            &mut self.incoming
        };
        adjacency[previous].retain(|&candidate| candidate != edge);
        adjacency[endpoint].push(edge);
        previous
    }

    fn remove_edge(&mut self, edge: usize) -> bool {
        match self.edges.get_mut(edge) {
            Some(record) if record.live => {
                record.live = false;
                true
            }
            _ => false,
        }
    }

    fn remove_node(&mut self, node: usize) -> bool {
        if !self.nodes.get(node).is_some_and(|entry| entry.live) {
            return false;
        }
        let attached: Vec<usize> = self.outgoing[node]
            .iter()
            .chain(&self.incoming[node])
            .copied()
            .collect();
        for edge in attached {
            self.remove_edge(edge);
        }
        self.nodes[node].live = false;
        true
    }

    fn live_edges_of<'m>(&'m self, adjacency: &'m [usize]) -> impl Iterator<Item = usize> + 'm {
        adjacency
            .iter()
            .copied()
            .filter(|&edge| self.edges[edge].live)
    }

    fn live_nodes(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.nodes.len()).filter(|&node| self.nodes[node].live)
    }

    fn live_edge_slots(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.edges.len()).filter(|&edge| self.edges[edge].live)
    }

    /// Renumber exactly as the store does: live entries keep their relative
    /// order, and everything else disappears.
    fn compact(&mut self) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
        let mut node_map = vec![None; self.nodes.len()];
        let mut nodes = Vec::new();
        for (slot, entry) in self.nodes.iter().enumerate() {
            if entry.live {
                node_map[slot] = Some(nodes.len());
                nodes.push(entry.clone());
            }
        }

        let mut edge_map = vec![None; self.edges.len()];
        let mut edges = Vec::new();
        let mut outgoing = vec![Vec::new(); nodes.len()];
        let mut incoming = vec![Vec::new(); nodes.len()];
        for (slot, entry) in self.edges.iter().enumerate() {
            if !entry.live {
                continue;
            }
            let source = node_map[entry.source].expect("a live edge has live endpoints");
            let target = node_map[entry.target].expect("a live edge has live endpoints");
            edge_map[slot] = Some(edges.len());
            outgoing[source].push(edges.len());
            incoming[target].push(edges.len());
            edges.push(ModelEdge {
                source,
                target,
                payload: entry.payload,
                live: true,
            });
        }

        self.nodes = nodes;
        self.edges = edges;
        self.outgoing = outgoing;
        self.incoming = incoming;
        (node_map, edge_map)
    }
}

type Store = Graph<u64, u64>;

fn node_id(slot: usize) -> Id<NodeTag> {
    Id::from_index(slot)
}

/// Compare two identity sequences element by element, which keeps the
/// per-operation comparison allocation-free.
fn assert_sequence(
    what: &str,
    step: usize,
    mut actual: impl Iterator<Item = usize>,
    mut expected: impl Iterator<Item = usize>,
) {
    let mut position = 0;
    loop {
        let actual = actual.next();
        let expected = expected.next();
        assert_eq!(
            actual, expected,
            "{what} at position {position}, step {step}"
        );
        if actual.is_none() {
            return;
        }
        position += 1;
    }
}

fn assert_agrees(graph: &Store, model: &Model, step: usize) {
    assert_eq!(
        graph.node_bound(),
        model.nodes.len(),
        "node slot count at step {step}"
    );
    assert_eq!(
        graph.edge_bound(),
        model.edges.len(),
        "edge slot count at step {step}"
    );
    assert_eq!(
        graph.node_count(),
        model.live_nodes().count(),
        "live node count at step {step}"
    );
    assert_eq!(
        graph.edge_count(),
        model.live_edge_slots().count(),
        "live edge count at step {step}"
    );
    assert_sequence(
        "live node identity",
        step,
        graph.node_ids().map(Id::index),
        model.live_nodes(),
    );
    assert_sequence(
        "live edge identity",
        step,
        graph.edge_ids().map(Id::index),
        model.live_edge_slots(),
    );

    for (slot, entry) in model.nodes.iter().enumerate() {
        let node = node_id(slot);
        assert_eq!(
            graph.contains_node(node),
            entry.live,
            "node {slot} liveness at step {step}"
        );
        assert_eq!(
            *graph.node(node),
            entry.payload,
            "node {slot} payload at step {step}"
        );
        assert_sequence(
            "outgoing edge",
            step,
            graph.outgoing(node).map(Id::index),
            model.live_edges_of(&model.outgoing[slot]),
        );
        assert_sequence(
            "incoming edge",
            step,
            graph.incoming(node).map(Id::index),
            model.live_edges_of(&model.incoming[slot]),
        );
        assert_sequence(
            "successor",
            step,
            graph.successors(node).map(Id::index),
            model
                .live_edges_of(&model.outgoing[slot])
                .map(|edge| model.edges[edge].target),
        );
        assert_sequence(
            "predecessor",
            step,
            graph.predecessors(node).map(Id::index),
            model
                .live_edges_of(&model.incoming[slot])
                .map(|edge| model.edges[edge].source),
        );
    }

    for (slot, entry) in model.edges.iter().enumerate() {
        let edge = Id::from_index(slot);
        assert_eq!(
            graph.contains_edge(edge),
            entry.live,
            "edge {slot} liveness at step {step}"
        );
        let record = graph.edge(edge);
        assert_eq!(
            record.source().index(),
            entry.source,
            "edge {slot} source at step {step}"
        );
        assert_eq!(
            record.target().index(),
            entry.target,
            "edge {slot} target at step {step}"
        );
        assert_eq!(
            *record.payload(),
            entry.payload,
            "edge {slot} payload at step {step}"
        );
    }
}

/// Slot ceilings that keep one comparison pass cheap enough to run after
/// every one of the ten thousand operations.
const MAX_NODE_SLOTS: usize = 128;
const MAX_EDGE_SLOTS: usize = 512;
const OPERATIONS: usize = 10_000;

/// Counts proving the sequence reached the states the comparison is for.
#[derive(Default)]
struct Exercised {
    compactions: usize,
    removed_nodes: usize,
    removed_edges: usize,
    redirected_edges: usize,
    self_edges: usize,
    parallel_edges: usize,
}

/// One store, one model, and the generator driving both of them.
struct Session {
    rng: Rng,
    graph: Store,
    model: Model,
    exercised: Exercised,
    payloads: u64,
}

impl Session {
    fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            graph: Store::new(),
            model: Model::default(),
            exercised: Exercised::default(),
            payloads: 0,
        }
    }

    fn next_payload(&mut self) -> u64 {
        self.payloads += 1;
        self.payloads
    }

    /// Apply one operation chosen by the generator, favoring growth so the
    /// sequence keeps reaching interesting sizes.
    fn apply(&mut self, step: usize) {
        let live_nodes: Vec<usize> = self.model.live_nodes().collect();
        let choice = self.rng.below(100);
        match choice {
            _ if live_nodes.is_empty()
                || (choice < 22 && self.model.nodes.len() < MAX_NODE_SLOTS) =>
            {
                self.add_node(step);
            }
            _ if choice < 68 && self.model.edges.len() < MAX_EDGE_SLOTS => {
                self.add_edge(&live_nodes, step);
            }
            _ if choice < 78 && !self.model.edges.is_empty() => self.remove_edge(step),
            _ if choice < 86 && self.model.live_edge_slots().next().is_some() => {
                self.redirect_edge(&live_nodes, step);
            }
            _ if choice < 94 && !self.model.nodes.is_empty() => self.remove_node(step),
            _ => self.compact(step),
        }
    }

    fn add_node(&mut self, step: usize) {
        let payload = self.next_payload();
        let expected = self.model.add_node(payload);
        let actual = self.graph.add_node(payload);
        assert_eq!(actual.index(), expected, "add_node identity at step {step}");
    }

    fn add_edge(&mut self, live_nodes: &[usize], step: usize) {
        let payload = self.next_payload();
        let source = live_nodes[self.rng.below(live_nodes.len())];
        let target = live_nodes[self.rng.below(live_nodes.len())];
        if source == target {
            self.exercised.self_edges += 1;
        }
        if self
            .model
            .live_edges_of(&self.model.outgoing[source])
            .any(|edge| self.model.edges[edge].target == target)
        {
            self.exercised.parallel_edges += 1;
        }
        let expected = self.model.add_edge(source, target, payload);
        let actual = self
            .graph
            .add_edge(node_id(source), node_id(target), payload);
        assert_eq!(actual.index(), expected, "add_edge identity at step {step}");
    }

    fn redirect_edge(&mut self, live_nodes: &[usize], step: usize) {
        let live: Vec<usize> = self.model.live_edge_slots().collect();
        let edge = live[self.rng.below(live.len())];
        let endpoint = live_nodes[self.rng.below(live_nodes.len())];
        let outgoing = self.rng.below(2) == 0;

        let expected = self.model.redirect_edge(edge, endpoint, outgoing);
        let actual = if outgoing {
            self.graph
                .redirect_edge_source(Id::from_index(edge), node_id(endpoint))
        } else {
            self.graph
                .redirect_edge_target(Id::from_index(edge), node_id(endpoint))
        };
        assert_eq!(
            actual.index(),
            expected,
            "redirect reports the previous endpoint at step {step}"
        );
        self.exercised.redirected_edges += usize::from(expected != endpoint);
    }

    fn remove_edge(&mut self, step: usize) {
        let edge = self.rng.below(self.model.edges.len());
        let expected = self.model.remove_edge(edge);
        let actual = self.graph.remove_edge(Id::from_index(edge));
        assert_eq!(actual, expected, "remove_edge report at step {step}");
        self.exercised.removed_edges += usize::from(expected);
    }

    fn remove_node(&mut self, step: usize) {
        let node = self.rng.below(self.model.nodes.len());
        let expected = self.model.remove_node(node);
        let actual = self.graph.remove_node(node_id(node));
        assert_eq!(actual, expected, "remove_node report at step {step}");
        self.exercised.removed_nodes += usize::from(expected);
    }

    fn compact(&mut self, step: usize) {
        let survivors: Vec<u64> = self
            .model
            .live_nodes()
            .map(|node| self.model.nodes[node].payload)
            .collect();
        let (expected_nodes, expected_edges) = self.model.compact();
        let renumbering = self.graph.compact();
        self.exercised.compactions += 1;

        assert!(self.graph.is_compact(), "compacted store at step {step}");
        for (old, new) in expected_nodes.into_iter().enumerate() {
            assert_eq!(
                renumbering.node(node_id(old)).map(Id::index),
                new,
                "node {old} renumbering at step {step}"
            );
        }
        for (old, new) in expected_edges.into_iter().enumerate() {
            assert_eq!(
                renumbering.edge(Id::from_index(old)).map(Id::index),
                new,
                "edge {old} renumbering at step {step}"
            );
        }
        // A renumbered identity must carry the payload its old identity
        // carried: survivors keep their relative order, so the nth survivor
        // is the nth node of the compacted store.
        for (new, payload) in survivors.into_iter().enumerate() {
            assert_eq!(
                *self.graph.node(node_id(new)),
                payload,
                "renumbered payload at step {step}"
            );
        }
    }
}

#[test]
fn random_operation_sequences_match_the_reference_model() {
    let mut session = Session::new(0x2545_F491_4F6C_DD1D);
    for step in 0..OPERATIONS {
        session.apply(step);
        assert_agrees(&session.graph, &session.model, step);
    }

    // The comparison above proves little unless the sequence actually
    // reached the states it is meant to cover.
    let exercised = &session.exercised;
    assert!(
        exercised.compactions > 100,
        "compactions: {}",
        exercised.compactions
    );
    assert!(
        exercised.removed_nodes > 100,
        "node removals: {}",
        exercised.removed_nodes
    );
    assert!(
        exercised.removed_edges > 100,
        "edge removals: {}",
        exercised.removed_edges
    );
    assert!(
        exercised.redirected_edges > 100,
        "edge redirections: {}",
        exercised.redirected_edges
    );
    assert!(
        exercised.self_edges > 10,
        "self edges: {}",
        exercised.self_edges
    );
    assert!(
        exercised.parallel_edges > 10,
        "parallel edges: {}",
        exercised.parallel_edges
    );
    assert!(
        !session.graph.is_empty(),
        "the sequence must not end with an empty store"
    );
}
