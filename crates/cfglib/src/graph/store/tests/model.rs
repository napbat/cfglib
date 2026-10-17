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
use crate::graph::traverse::TraversalDirection;
use crate::test_util::rng::Rng;

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

    /// Move every live edge of `old` to `new` along one direction, one
    /// redirect at a time: the contract the store's bulk move states.
    /// Returns how many edges moved.
    fn move_adjacency(&mut self, old: usize, new: usize, outgoing: bool) -> usize {
        let adjacency = if outgoing {
            &self.outgoing
        } else {
            &self.incoming
        };
        let moving: Vec<usize> = self.live_edges_of(&adjacency[old]).collect();
        for &edge in &moving {
            self.redirect_edge(edge, new, outgoing);
        }
        moving.len()
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
    /// order, everything else disappears, and each node's adjacency keeps the
    /// order it read in — which a redirect makes differ from identity order.
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
        for (slot, entry) in self.edges.iter().enumerate() {
            if !entry.live {
                continue;
            }
            edge_map[slot] = Some(edges.len());
            edges.push(ModelEdge {
                source: node_map[entry.source].expect("a live edge has live endpoints"),
                target: node_map[entry.target].expect("a live edge has live endpoints"),
                payload: entry.payload,
                live: true,
            });
        }

        let mut outgoing = vec![Vec::new(); nodes.len()];
        let mut incoming = vec![Vec::new(); nodes.len()];
        for (slot, node) in node_map.iter().enumerate() {
            if let Some(node) = *node {
                outgoing[node] = self.outgoing[slot]
                    .iter()
                    .filter_map(|&edge| edge_map[edge])
                    .collect();
                incoming[node] = self.incoming[slot]
                    .iter()
                    .filter_map(|&edge| edge_map[edge])
                    .collect();
            }
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
const MAX_EDGE_SLOTS: usize = 512;
const OPERATIONS: usize = 10_000;

/// The operation mix of one session: the cumulative percentage below which a
/// draw picks each operation, with compaction taking whatever is left.
#[derive(Clone, Copy)]
struct Mix {
    max_nodes: usize,
    add_node: usize,
    add_edge: usize,
    remove_edge: usize,
    redirect_edge: usize,
    remove_node: usize,
}

/// Growth first, with a compaction every sixteen operations or so: the store
/// is rebuilt before an overlay can grow old.
const BALANCED: Mix = Mix {
    max_nodes: 128,
    add_node: 22,
    add_edge: 68,
    remove_edge: 78,
    redirect_edge: 86,
    remove_node: 94,
};

/// A few nodes, constant redirection, and a compaction every hundred
/// operations: the same nodes take overlay runs, then appends into their
/// delta chains, then more relocations, which is the interleaving a balanced
/// mix reaches only by luck.
const CHURN: Mix = Mix {
    max_nodes: 12,
    add_node: 6,
    add_edge: 40,
    remove_edge: 44,
    redirect_edge: 98,
    remove_node: 99,
};

/// Counts proving the sequence reached the states the comparison is for.
#[derive(Default)]
struct Exercised {
    compactions: usize,
    removed_nodes: usize,
    removed_edges: usize,
    redirected_edges: usize,
    moved_adjacencies: usize,
    self_edges: usize,
    parallel_edges: usize,
    /// Edges appended to a node whose adjacency was already served from an
    /// overlay run, so the edge lives in the node's delta chain beside it.
    appended_beside_overlay: usize,
    /// Relocations that edited the run of a node holding such an appended
    /// edge, which the run has to take over.
    relocated_beside_appended: usize,
    /// Compactions of a store in which some node with an overlay read its
    /// edges out of identity order.
    compactions_keeping_overlay_order: usize,
    /// Compactions of a store in which some node *without* an overlay read
    /// its edges out of identity order, so the order was carried by a base an
    /// earlier compaction built.
    compactions_keeping_base_order: usize,
}

/// One store, one model, and the generator driving both of them.
struct Session {
    rng: Rng,
    mix: Mix,
    graph: Store,
    model: Model,
    exercised: Exercised,
    payloads: u64,
}

/// The direction a source (`true`) or target (`false`) endpoint is read along.
const fn direction(outgoing: bool) -> TraversalDirection {
    if outgoing {
        TraversalDirection::Outgoing
    } else {
        TraversalDirection::Incoming
    }
}

impl Session {
    /// Whether `node` is served from an overlay run along `direction` and also
    /// lists an edge that is not in that run: one appended since.
    fn appended_beside_overlay(&self, node: usize, direction: TraversalDirection) -> bool {
        let Some(run) = self.graph.overlay(node, direction) else {
            return false;
        };
        self.graph
            .adjacent(node_id(node), direction)
            .any(|edge| !run.iter().any(|&slot| Id::from_raw(slot) == edge))
    }

    /// Whether `node` reads its live edges along `direction` in anything but
    /// ascending identity order, which only a redirect produces.
    fn reads_out_of_identity_order(&self, node: usize, direction: TraversalDirection) -> bool {
        let edges: Vec<usize> = self
            .graph
            .adjacent(node_id(node), direction)
            .map(Id::index)
            .collect();
        edges.windows(2).any(|pair| pair[0] > pair[1])
    }

    fn new(seed: u64, mix: Mix) -> Self {
        Self {
            rng: Rng::new(seed),
            mix,
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
        let mix = self.mix;
        match choice {
            _ if live_nodes.is_empty()
                || (choice < mix.add_node && self.model.nodes.len() < mix.max_nodes) =>
            {
                self.add_node(step);
            }
            _ if choice < mix.add_edge && self.model.edges.len() < MAX_EDGE_SLOTS => {
                self.add_edge(&live_nodes, step);
            }
            _ if choice < mix.remove_edge && !self.model.edges.is_empty() => {
                self.remove_edge(step);
            }
            _ if choice < mix.redirect_edge && self.model.live_edge_slots().next().is_some() => {
                if self.rng.one_in(4) {
                    self.move_adjacency(&live_nodes);
                } else {
                    self.redirect_edge(&live_nodes, step);
                }
            }
            _ if choice < mix.remove_node && !self.model.nodes.is_empty() => {
                self.remove_node(step);
            }
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
        self.exercised.appended_beside_overlay += usize::from(
            self.graph
                .overlay(source, TraversalDirection::Outgoing)
                .is_some()
                || self
                    .graph
                    .overlay(target, TraversalDirection::Incoming)
                    .is_some(),
        );
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
        let outgoing = self.rng.one_in(2);

        let record = &self.model.edges[edge];
        let previous = if outgoing {
            record.source
        } else {
            record.target
        };
        self.exercised.relocated_beside_appended += usize::from(
            previous != endpoint
                && (self.appended_beside_overlay(previous, direction(outgoing))
                    || self.appended_beside_overlay(endpoint, direction(outgoing))),
        );
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

    /// Move a whole adjacency run between two distinct live nodes, which the
    /// model spells as redirecting its live edges one by one.
    fn move_adjacency(&mut self, live_nodes: &[usize]) {
        let old = live_nodes[self.rng.below(live_nodes.len())];
        let new = live_nodes[self.rng.below(live_nodes.len())];
        let outgoing = self.rng.one_in(2);
        if old == new {
            return;
        }

        let beside_appended = self.appended_beside_overlay(old, direction(outgoing))
            || self.appended_beside_overlay(new, direction(outgoing));
        let moved = self.model.move_adjacency(old, new, outgoing);
        self.graph
            .move_adjacency(node_id(old), node_id(new), direction(outgoing));
        self.exercised.moved_adjacencies += usize::from(moved > 0);
        self.exercised.relocated_beside_appended += usize::from(moved > 0 && beside_appended);
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
        let (mut overlay_order, mut base_order) = (false, false);
        for node in self.model.live_nodes() {
            for outgoing in [true, false] {
                if self.reads_out_of_identity_order(node, direction(outgoing)) {
                    if self.graph.overlay(node, direction(outgoing)).is_some() {
                        overlay_order = true;
                    } else {
                        base_order = true;
                    }
                }
            }
        }
        self.exercised.compactions_keeping_overlay_order += usize::from(overlay_order);
        self.exercised.compactions_keeping_base_order += usize::from(base_order);
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

/// Apply `OPERATIONS` operations drawn with `mix`, comparing the store with
/// the model after every one.
fn run_session(seed: u64, mix: Mix) -> Session {
    let mut session = Session::new(seed, mix);
    for step in 0..OPERATIONS {
        session.apply(step);
        assert_agrees(&session.graph, &session.model, step);
    }
    session
}

/// Fail unless a session reached `what` more than `above` times.
///
/// Each bound is a fifth of the least count measured over the sessions below,
/// rounded down: loose enough that another seed passes, tight enough that a
/// mix which no longer produces the state fails. The balanced mix reaches an
/// edge appended beside an overlay twice in ten thousand operations, which is
/// the difference the churn bounds are there to keep.
fn assert_reached(what: &str, count: usize, above: usize) {
    assert!(count > above, "{what}: {count}, wanted more than {above}");
}

#[test]
fn random_operation_sequences_match_the_reference_model() {
    let session = run_session(0x2545_F491_4F6C_DD1D, BALANCED);

    // The comparison above proves little unless the sequence actually
    // reached the states it is meant to cover.
    let exercised = &session.exercised;
    assert_reached("compactions", exercised.compactions, 100);
    assert_reached("node removals", exercised.removed_nodes, 100);
    assert_reached("edge removals", exercised.removed_edges, 100);
    assert_reached("edge redirections", exercised.redirected_edges, 100);
    assert_reached("bulk moves", exercised.moved_adjacencies, 100);
    assert_reached("self edges", exercised.self_edges, 10);
    assert_reached("parallel edges", exercised.parallel_edges, 10);
    assert_reached(
        "edges appended beside an overlay",
        exercised.appended_beside_overlay,
        30,
    );
    assert_reached(
        "compactions that kept an overlay's order",
        exercised.compactions_keeping_overlay_order,
        50,
    );
    assert_reached(
        "compactions that kept an order the base carried",
        exercised.compactions_keeping_base_order,
        100,
    );
    assert!(
        !session.graph.is_empty(),
        "the sequence must not end with an empty store"
    );
}

/// A relocation gives a node an overlay run, and an edge appended afterwards
/// lives in the node's delta chain beside it. Compacting every few operations
/// rebuilds the store before the two can meet, so this session keeps a dozen
/// nodes and rarely compacts, across several seeds.
#[test]
fn relocated_nodes_keep_the_edges_appended_after_their_overlay() {
    for seed in [
        0x2545_F491_4F6C_DD1D,
        0x9E37_79B9_7F4A_7C15,
        0xD1B5_4A32_D192_ED03,
        0x8CB9_2BA7_2F3D_8DD7,
    ] {
        let session = run_session(seed, CHURN);
        let exercised = &session.exercised;
        assert_reached("edge redirections", exercised.redirected_edges, 1_000);
        assert_reached("bulk moves", exercised.moved_adjacencies, 200);
        assert_reached("compactions", exercised.compactions, 20);
        assert_reached(
            "edges appended beside an overlay",
            exercised.appended_beside_overlay,
            600,
        );
        assert_reached(
            "relocations of a node holding an appended edge",
            exercised.relocated_beside_appended,
            400,
        );
        assert_reached(
            "compactions that kept an overlay's order",
            exercised.compactions_keeping_overlay_order,
            15,
        );
        assert_reached(
            "compactions that kept an order the base carried",
            exercised.compactions_keeping_base_order,
            7,
        );
    }
}
