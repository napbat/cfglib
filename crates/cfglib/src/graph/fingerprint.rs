//! Deterministic structural fingerprints: Weisfeiler–Leman colors, multiset
//! fingerprints, and a canonical order and hash of a rooted graph.
//!
//! [`wl_colors`] refines a label per node by the labels around it, so two
//! nodes with equal colors have neighborhoods of equal shape out to the
//! number of rounds run. [`multiset_fingerprint`] folds any multiset of
//! colors into one value that does not depend on node numbering.
//! [`canonical_order`] numbers the nodes a root reaches by a walk the graph
//! itself determines, and [`canonical_hash`] hashes the graph in that order.
//!
//! The caller supplies every label. cfglib reads the structure (nodes, edges,
//! and the keys the caller attaches to them) and never an instruction: a
//! consumer that wants content in a fingerprint hashes its instructions into
//! the node labels itself.
//!
//! Every value comes from one fixed 64-bit [`Mixer`], the `SplitMix64`
//! finalizer applied to a running state, and never from a `RandomState`. A
//! fingerprint is therefore the same across runs, platforms, and versions,
//! and can be stored.
//!
//! A fingerprint has 64 bits, so two different graphs can collide: equal
//! fingerprints are evidence of isomorphism, not proof. One-dimensional WL
//! also cannot tell some different graphs apart at all, whatever the number
//! of rounds. In a directed 6-cycle and in two directed 3-cycles, every node
//! has one predecessor and one successor of its own color, so the two get
//! the same colors.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::edge_view::EdgeView;
use super::view::{DenseId, RootedView};
use crate::identity::{NO_INDEX, check_bound, word};

/// The rounds of [`wl_colors`] that [`canonical_hash`] runs to fingerprint
/// the nodes the root does not reach.
const UNREACHED_ROUNDS: usize = 3;

/// A running 64-bit hash: each value is folded into the state with the
/// `SplitMix64` finalizer, so the result depends on the values and on their
/// order.
///
/// It is the one mixer behind every fingerprint of this module, and its
/// values are part of the contract: the same values in the same order give
/// the same hash on every run, platform, and version. A consumer that hashes
/// its own block content, such as the [`BlockHashes`](crate::BlockHashes) of
/// a [`CascadeDiff`](crate::CascadeDiff), can use it for the same guarantee.
///
/// # Examples
///
/// ```
/// use cfglib::Mixer;
///
/// let hash = |values: &[u64]| {
///     values
///         .iter()
///         .fold(Mixer::new(), |mixer, &value| mixer.mix(value))
///         .finish()
/// };
/// assert_eq!(hash(&[1, 2]), hash(&[1, 2]));
/// assert_ne!(hash(&[1, 2]), hash(&[2, 1]));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mixer(u64);

impl Mixer {
    /// The state before any value is mixed in.
    #[must_use]
    pub const fn new() -> Self {
        Self(0)
    }

    /// The state after `value`.
    #[must_use]
    pub const fn mix(self, value: u64) -> Self {
        Self(splitmix64(self.0 ^ value))
    }

    /// The state after a count, which delimits the list that follows it.
    #[must_use]
    pub(crate) const fn mix_count(self, count: usize) -> Self {
        self.mix(count as u64)
    }

    /// The state after a sorted copy of `pairs`, preceded by their count.
    /// Sorting makes the result independent of the order of `pairs`.
    #[must_use]
    pub(crate) fn mix_pairs(self, pairs: &mut [(u64, u64)]) -> Self {
        pairs.sort_unstable();
        pairs
            .iter()
            .fold(self.mix_count(pairs.len()), |state, &(first, second)| {
                state.mix(first).mix(second)
            })
    }

    /// The hash of everything mixed in so far.
    #[must_use]
    pub const fn finish(self) -> u64 {
        self.0
    }
}

/// The `SplitMix64` output function: a bijection on `u64` that spreads every
/// input bit over the whole word.
const fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Weisfeiler–Leman (1-WL) colors over directed, labeled edges.
///
/// Round 0 is `seed`. In each round, a node's color mixes its previous
/// color, the sorted multiset of `(edge_label, color)` over its out-edges,
/// and the sorted multiset of `(edge_label, color)` over its in-edges, where
/// `color` is the previous color of the node at the other end. After `r`
/// rounds, two nodes with different colors are known to have different
/// labeled neighborhoods within `r` edges; equal colors are evidence, not
/// proof, that the neighborhoods match (see the [module docs](self)).
///
/// The result is indexed by [`DenseId::index`] and has
/// [`node_bound`](crate::GraphView::node_bound) slots. A removed slot holds
/// 0, so fingerprint a graph with removed nodes from the colors of its
/// [`node_ids`](crate::GraphView::node_ids), not from the whole vector.
///
/// `seed` is called once per live node and `edge_label` once per live edge.
/// A round takes O(n + m log d) time for `n` nodes, `m` edges and maximum
/// degree `d`, because each node sorts its neighbors.
///
/// # Examples
///
/// A diamond and a chain of four nodes look alike before the first round:
/// every node has the same seed. One round sees that the diamond branches.
///
/// ```
/// use cfglib::{Cfg, EdgeKind, multiset_fingerprint, wl_colors};
///
/// let mut diamond = Cfg::<u32>::new();
/// let [left, right, join] = [(); 3].map(|()| diamond.new_block());
/// diamond.add_edge(diamond.entry(), left, EdgeKind::ConditionalTrue);
/// diamond.add_edge(diamond.entry(), right, EdgeKind::ConditionalFalse);
/// diamond.add_edge(left, join, EdgeKind::Fallthrough);
/// diamond.add_edge(right, join, EdgeKind::Fallthrough);
///
/// let mut chain = Cfg::<u32>::new();
/// let [first, second, third] = [(); 3].map(|()| chain.new_block());
/// chain.add_edge(chain.entry(), first, EdgeKind::Fallthrough);
/// chain.add_edge(first, second, EdgeKind::Fallthrough);
/// chain.add_edge(second, third, EdgeKind::Fallthrough);
///
/// // A constant label, so only the wiring tells the nodes apart.
/// let fingerprint = |cfg: &Cfg<u32>, rounds| {
///     multiset_fingerprint(wl_colors(cfg, rounds, |_| 7, |_| 0))
/// };
/// assert_eq!(fingerprint(&diamond, 0), fingerprint(&chain, 0));
/// assert_ne!(fingerprint(&diamond, 1), fingerprint(&chain, 1));
///
/// // The values are fixed: the mixer is part of the contract.
/// assert_eq!(fingerprint(&chain, 1), 0x4b0a_40af_4dfc_1c53);
/// ```
#[must_use]
pub fn wl_colors<G: EdgeView>(
    graph: &G,
    rounds: usize,
    mut seed: impl FnMut(G::NodeId) -> u64,
    edge_label: impl FnMut(G::EdgeId) -> u64,
) -> Vec<u64> {
    let mut colors = vec![0; graph.node_bound()];
    for node in graph.node_ids() {
        colors[node.index()] = seed(node);
    }
    if rounds == 0 {
        return colors;
    }
    refine(graph, colors, rounds, &edge_table(graph, edge_label))
}

/// Runs `rounds` rounds of [`wl_colors`] from `colors`, with each edge's
/// label read from `edge_labels` by its index.
fn refine<G: EdgeView>(
    graph: &G,
    mut colors: Vec<u64>,
    rounds: usize,
    edge_labels: &[u64],
) -> Vec<u64> {
    // Both buffers keep 0 in the removed slots, which no round writes.
    let mut next = vec![0; colors.len()];
    let mut neighbors: Vec<(u64, u64)> = Vec::new();
    for _ in 0..rounds {
        for node in graph.node_ids() {
            let state = Mixer::new().mix(colors[node.index()]);
            neighbors.clear();
            neighbors.extend(graph.outgoing(node).map(|edge| {
                let target = graph.edge(edge).target();
                (edge_labels[edge.index()], colors[target.index()])
            }));
            let state = state.mix_pairs(&mut neighbors);
            neighbors.clear();
            neighbors.extend(graph.incoming(node).map(|edge| {
                let source = graph.edge(edge).source();
                (edge_labels[edge.index()], colors[source.index()])
            }));
            next[node.index()] = state.mix_pairs(&mut neighbors).finish();
        }
        core::mem::swap(&mut colors, &mut next);
    }
    colors
}

/// `label` of every live edge, indexed by the edge's index.
fn edge_table<G: EdgeView>(graph: &G, mut label: impl FnMut(G::EdgeId) -> u64) -> Vec<u64> {
    let mut table = vec![0; graph.edge_bound()];
    for edge in graph.edge_ids() {
        table[edge.index()] = label(edge);
    }
    table
}

/// An order-independent fingerprint of a multiset of colors: the colors are
/// sorted, then mixed after their count.
///
/// Two equal multisets give the same value, however they are ordered; two
/// different multisets give different values unless the 64-bit hash
/// collides.
///
/// # Examples
///
/// ```
/// use cfglib::multiset_fingerprint;
///
/// assert_eq!(multiset_fingerprint([3, 1, 2]), multiset_fingerprint([2, 3, 1]));
/// assert_ne!(multiset_fingerprint([1, 2]), multiset_fingerprint([1, 2, 2]));
/// assert_eq!(multiset_fingerprint([]), 0xe220_a839_7b1d_cdaf);
/// ```
#[must_use]
pub fn multiset_fingerprint(colors: impl IntoIterator<Item = u64>) -> u64 {
    let mut colors: Vec<u64> = colors.into_iter().collect();
    colors.sort_unstable();
    colors
        .iter()
        .fold(Mixer::new().mix_count(colors.len()), |state, &color| {
            state.mix(color)
        })
        .finish()
}

/// The canonical numbering of the nodes the root reaches: a depth-first
/// preorder that visits each node's out-edges sorted by `edge_key`, and
/// edges with equal keys in adjacency order.
///
/// The walk is fixed by the graph, so the order is canonical up to ties in
/// `edge_key`: two isomorphic graphs whose isomorphism keeps edge keys get
/// orders that the isomorphism maps onto each other whenever no node has two
/// out-edges with one key. For a [`Cfg`](crate::Cfg), keying each edge by
/// its [`EdgeKind`](crate::EdgeKind) is canonical as long as no block has
/// two out-edges of one kind. The cases of a switch share a kind, so put
/// the case value into the key; otherwise their adjacency order decides,
/// which is the caveat [`CfgDiff`](crate::CfgDiff) documents too.
///
/// Nodes the root does not reach are not in the order. Renumbering the
/// nodes in this order before [`write_text`](crate::write_text) gives text
/// that two isomorphic graphs share.
///
/// `edge_key` is called once per live edge. The walk takes O(n + m log d)
/// time for `n` nodes, `m` edges and maximum out-degree `d`.
///
/// # Panics
///
/// Panics when the node bound of `graph` exceeds `u32::MAX`.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, canonical_order};
///
/// // `entry -> {then, else}`, with the blocks and edges made in an order
/// // that does not match the walk.
/// let mut cfg = Cfg::<u32>::new();
/// let (else_arm, then_arm) = (cfg.new_block(), cfg.new_block());
/// cfg.add_edge(cfg.entry(), else_arm, EdgeKind::ConditionalFalse);
/// cfg.add_edge(cfg.entry(), then_arm, EdgeKind::ConditionalTrue);
///
/// let order = canonical_order(&cfg, |edge| cfg[edge].kind() as u64);
/// assert_eq!(order, [cfg.entry(), then_arm, else_arm]);
/// ```
#[must_use]
pub fn canonical_order<G: RootedView + EdgeView>(
    graph: &G,
    edge_key: impl FnMut(G::EdgeId) -> u64,
) -> Vec<G::NodeId> {
    preorder(graph, &edge_table(graph, edge_key)).0
}

/// The [`canonical_order`] walk with the edge keys read from `keys`: the
/// order, and each node's position in it by node index ([`NO_INDEX`] for a
/// node outside it).
///
/// The walk keeps its own loop instead of running the shared depth-first
/// core over a successor table sorted up front: sorting each node's
/// out-edges as the walk enters it, on one stack, and marking positions on
/// entry. On 10,000 blocks the core form measured 1.14x the time of the
/// previous release (`api_canonical_order`), and this loop 0.99x to 1.02x.
fn preorder<G: RootedView + EdgeView>(graph: &G, keys: &[u64]) -> (Vec<G::NodeId>, Vec<u32>) {
    check_bound(graph.node_bound(), "node bound of a canonical order");
    let mut position = vec![NO_INDEX; graph.node_bound()];
    let mut order = Vec::new();
    // The out-edges of every node on the walk's stack, as `(key, adjacency
    // position, target)` sorted by key and then position, each node's run
    // above the runs of the nodes below it. A frame holds where its run
    // starts and the next edge to take.
    let mut edges: Vec<(u64, usize, G::NodeId)> = Vec::new();
    let mut frames: Vec<(usize, usize)> = Vec::new();
    let mut entered = Some(graph.root());
    loop {
        if let Some(node) = entered.take() {
            position[node.index()] = word(order.len());
            order.push(node);
            let start = edges.len();
            edges.extend(graph.outgoing(node).enumerate().map(|(adjacency, edge)| {
                (keys[edge.index()], adjacency, graph.edge(edge).target())
            }));
            edges[start..].sort_unstable_by_key(|&(key, adjacency, _)| (key, adjacency));
            frames.push((start, start));
        }
        let Some(frame) = frames.last_mut() else {
            break;
        };
        if let Some(&(_, _, target)) = edges.get(frame.1) {
            frame.1 += 1;
            if position[target.index()] == NO_INDEX {
                entered = Some(target);
            }
        } else {
            edges.truncate(frame.0);
            frames.pop();
        }
    }
    (order, position)
}

/// A hash of the graph in [`canonical_order`]: the number of nodes the root
/// reaches, then for each of them in that order its label and its out-edges
/// as `(edge_key, position of the target in the order)`, sorted.
///
/// The nodes the root does not reach have no canonical position. The hash
/// folds in the [`multiset_fingerprint`] of their [`wl_colors`] after three
/// rounds seeded with `node_label`, so they count without breaking the
/// invariance under renumbering.
///
/// The hash is canonical exactly when [`canonical_order`] is: two graphs
/// that an isomorphism maps onto each other, labels and keys included, hash
/// alike when no node has two out-edges with one key. A change of one label,
/// one key, or one edge target changes the hash unless the 64-bit hash
/// collides.
///
/// `node_label` is called once per live node and `edge_key` once per live
/// edge. The hash takes O(n + m log d) time for `n` nodes, `m` edges and
/// maximum degree `d`, and three rounds of [`wl_colors`] more when the root
/// does not reach every node.
///
/// # Panics
///
/// Panics when the node bound of `graph` exceeds `u32::MAX`.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, canonical_hash};
///
/// // The same loop, with its blocks made in two orders.
/// let build = |swap: bool| {
///     let mut cfg = Cfg::<u32>::new();
///     let (mut header, mut body) = (cfg.new_block(), cfg.new_block());
///     if swap {
///         (header, body) = (body, header);
///     }
///     cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
///     cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
///     cfg.add_edge(body, header, EdgeKind::Back);
///     cfg
/// };
/// let hash = |cfg: &Cfg<u32>| canonical_hash(cfg, |_| 0, |edge| cfg[edge].kind() as u64);
/// assert_eq!(hash(&build(false)), hash(&build(true)));
/// assert_eq!(hash(&build(false)), 0x9a5d_7363_3ca3_070d);
/// ```
#[must_use]
pub fn canonical_hash<G: RootedView + EdgeView>(
    graph: &G,
    mut node_label: impl FnMut(G::NodeId) -> u64,
    edge_key: impl FnMut(G::EdgeId) -> u64,
) -> u64 {
    let keys = edge_table(graph, edge_key);
    let (order, position) = preorder(graph, &keys);
    let mut labels = vec![0; graph.node_bound()];
    for node in graph.node_ids() {
        labels[node.index()] = node_label(node);
    }
    let mut out_edges: Vec<(u64, u64)> = Vec::new();
    let mut state = Mixer::new().mix_count(order.len());
    for &node in &order {
        out_edges.clear();
        out_edges.extend(graph.outgoing(node).map(|edge| {
            let target = graph.edge(edge).target();
            (keys[edge.index()], u64::from(position[target.index()]))
        }));
        state = state.mix(labels[node.index()]).mix_pairs(&mut out_edges);
    }
    let unreached = |node: &G::NodeId| position[node.index()] == NO_INDEX;
    let unreached_count = graph.node_ids().filter(unreached).count();
    state = state.mix_count(unreached_count);
    if unreached_count > 0 {
        let colors = refine(graph, labels, UNREACHED_ROUNDS, &keys);
        state = state.mix(multiset_fingerprint(
            graph
                .node_ids()
                .filter(unreached)
                .map(|node| colors[node.index()]),
        ));
    }
    state.finish()
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::block::BlockId;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::graph::store::{Graph, NodeId};
    use crate::graph::traverse::reachable;
    use crate::graph::view::Direction;
    use crate::graph::view::Rooted;
    use crate::test_util::rng::Rng;

    fn kind_key(cfg: &Cfg<u32>) -> impl Fn(crate::edge::EdgeId) -> u64 + '_ {
        |edge| cfg[edge].kind() as u64
    }

    /// A graph whose node payloads are labels and whose edge payloads are
    /// keys, with self-loops, parallel edges and unreachable nodes.
    fn random_graph(rng: &mut Rng, distinct_keys: bool) -> Graph<u64, u64> {
        let mut graph = Graph::new();
        let node_count = 1 + rng.below(24);
        let nodes: Vec<NodeId> = (0..node_count)
            .map(|_| graph.add_node(rng.below(3) as u64))
            .collect();
        for key in 0..rng.below(3 * node_count) {
            let (source, target) = (nodes[rng.below(node_count)], nodes[rng.below(node_count)]);
            let key = if distinct_keys {
                key as u64
            } else {
                rng.below(2) as u64
            };
            graph.add_edge(source, target, key);
        }
        graph
    }

    /// `graph` with node `i` renumbered to `permutation[i]`. The edges keep
    /// their insertion order unless `shuffle_edges`, which also shuffles the
    /// adjacency order of every node.
    fn permuted(
        rng: &mut Rng,
        graph: &Graph<u64, u64>,
        permutation: &[usize],
        shuffle_edges: bool,
    ) -> Graph<u64, u64> {
        let mut originals: Vec<usize> = (0..permutation.len()).collect();
        originals.sort_by_key(|&original| permutation[original]);
        let mut copy = Graph::new();
        for original in originals {
            copy.add_node(*graph.node(NodeId::from_index(original)));
        }
        let mut edges: Vec<_> = graph.edge_ids().collect();
        if shuffle_edges {
            rng.shuffle(&mut edges);
        }
        for edge in edges {
            let record = graph.edge(edge);
            copy.add_edge(
                NodeId::from_index(permutation[record.source().index()]),
                NodeId::from_index(permutation[record.target().index()]),
                *record.payload(),
            );
        }
        copy
    }

    fn label(graph: &Graph<u64, u64>) -> impl Fn(NodeId) -> u64 + '_ {
        |node| *graph.node(node)
    }

    fn key(graph: &Graph<u64, u64>) -> impl Fn(crate::graph::store::EdgeId) -> u64 + '_ {
        |edge| *graph.edge(edge).payload()
    }

    #[test]
    fn renumbering_maps_colors_orders_and_hashes_onto_each_other() {
        let mut rng = Rng::new(79);
        for _ in 0..300 {
            let distinct_keys = rng.chance(50);
            let graph = random_graph(&mut rng, distinct_keys);
            let mut permutation: Vec<usize> = (0..graph.node_count()).collect();
            rng.shuffle(&mut permutation);
            // With distinct keys the adjacency order must not matter; with
            // tied keys it decides, so the copy keeps it.
            let copy = permuted(&mut rng, &graph, &permutation, distinct_keys);

            let colors = wl_colors(&graph, 3, label(&graph), key(&graph));
            let copied_colors = wl_colors(&copy, 3, label(&copy), key(&copy));
            for (original, &color) in colors.iter().enumerate() {
                assert_eq!(copied_colors[permutation[original]], color);
            }
            assert_eq!(
                multiset_fingerprint(colors),
                multiset_fingerprint(copied_colors)
            );

            let root = rng.below(graph.node_count());
            let rooted = Rooted::new(&graph, NodeId::from_index(root));
            let copied_root = NodeId::from_index(permutation[root]);
            let copied_rooted = Rooted::new(&copy, copied_root);
            let order = canonical_order(&rooted, key(&graph));
            let copied_order = canonical_order(&copied_rooted, key(&copy));
            let mapped: Vec<NodeId> = order
                .iter()
                .map(|node| NodeId::from_index(permutation[node.index()]))
                .collect();
            assert_eq!(mapped, copied_order);
            assert_eq!(
                canonical_hash(&rooted, label(&graph), key(&graph)),
                canonical_hash(&copied_rooted, label(&copy), key(&copy)),
            );
        }
    }

    #[test]
    fn the_order_is_a_preorder_of_exactly_the_reachable_nodes() {
        let mut rng = Rng::new(80);
        for _ in 0..200 {
            let distinct_keys = rng.chance(50);
            let graph = random_graph(&mut rng, distinct_keys);
            let rooted = Rooted::new(&graph, NodeId::from_index(0));
            let order = canonical_order(&rooted, key(&graph));
            let root = NodeId::from_index(0);
            let reached = reachable(&graph, [root], Direction::Forward);
            let mut sorted = order.clone();
            sorted.sort_unstable();
            let expected: Vec<NodeId> = graph
                .node_ids()
                .filter(|node| reached[node.index()])
                .collect();
            assert_eq!(sorted, expected);
            assert_eq!(order[0], NodeId::from_index(0));
            // In a preorder, every node after the root has a predecessor
            // that comes before it.
            for (position, &node) in order.iter().enumerate().skip(1) {
                assert!(
                    crate::graph::view::GraphView::predecessors(&graph, node)
                        .any(|predecessor| order[..position].contains(&predecessor))
                );
            }
        }
    }

    #[test]
    fn a_diamond_and_a_chain_differ_after_one_round() {
        let mut diamond = Cfg::<u32>::new();
        let [left, right, join] = [(); 3].map(|()| diamond.new_block());
        diamond.add_edge(diamond.entry(), left, EdgeKind::Fallthrough);
        diamond.add_edge(diamond.entry(), right, EdgeKind::Fallthrough);
        diamond.add_edge(left, join, EdgeKind::Fallthrough);
        diamond.add_edge(right, join, EdgeKind::Fallthrough);
        let mut chain = Cfg::<u32>::new();
        let [first, second, third] = [(); 3].map(|()| chain.new_block());
        chain.add_edge(chain.entry(), first, EdgeKind::Fallthrough);
        chain.add_edge(first, second, EdgeKind::Fallthrough);
        chain.add_edge(second, third, EdgeKind::Fallthrough);

        let fingerprint =
            |cfg: &Cfg<u32>, rounds| multiset_fingerprint(wl_colors(cfg, rounds, |_| 1, |_| 0));
        assert_eq!(fingerprint(&diamond, 0), fingerprint(&chain, 0));
        for rounds in 1..4 {
            assert_ne!(fingerprint(&diamond, rounds), fingerprint(&chain, rounds));
        }
    }

    /// One-dimensional WL is blind to the difference between one directed
    /// 6-cycle and two directed 3-cycles: every node of both has one
    /// predecessor and one successor of its own color in every round. The
    /// fingerprints agree; the limitation is the algorithm's.
    #[test]
    fn one_wl_cannot_tell_a_six_cycle_from_two_three_cycles() {
        let cycles = |lengths: &[usize]| {
            let mut graph = Graph::<u64, u64>::new();
            for &length in lengths {
                let nodes: Vec<NodeId> = (0..length).map(|_| graph.add_node(0)).collect();
                for (index, &node) in nodes.iter().enumerate() {
                    graph.add_edge(node, nodes[(index + 1) % length], 0);
                }
            }
            graph
        };
        let (one, two) = (cycles(&[6]), cycles(&[3, 3]));
        for rounds in 0..6 {
            assert_eq!(
                multiset_fingerprint(wl_colors(&one, rounds, label(&one), key(&one))),
                multiset_fingerprint(wl_colors(&two, rounds, label(&two), key(&two))),
            );
        }
    }

    #[test]
    fn zero_rounds_return_the_seeds_and_removed_slots_hold_zero() {
        let mut graph = Graph::<u64, u64>::new();
        let [a, b, c] = [5, 6, 7].map(|label| graph.add_node(label));
        graph.add_edge(a, b, 0);
        graph.add_edge(b, c, 0);
        assert_eq!(wl_colors(&graph, 0, label(&graph), key(&graph)), [5, 6, 7]);
        graph.remove_node(b);
        let colors = wl_colors(&graph, 2, label(&graph), key(&graph));
        assert_eq!(colors[b.index()], 0);
        assert_ne!(colors[a.index()], 0);
        let empty = Graph::<u64, u64>::new();
        assert_eq!(wl_colors(&empty, 3, |_| 1, |_| 1), Vec::<u64>::new());
    }

    /// `entry -> {then, else} -> join`, with `join` looping to itself.
    fn branch() -> (Cfg<u32>, [BlockId; 3]) {
        let mut cfg = Cfg::<u32>::new();
        let blocks = [(); 3].map(|()| cfg.new_block());
        let [then_arm, else_arm, join] = blocks;
        cfg.add_edge(cfg.entry(), then_arm, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), else_arm, EdgeKind::ConditionalFalse);
        cfg.add_edge(then_arm, join, EdgeKind::Fallthrough);
        cfg.add_edge(else_arm, join, EdgeKind::Unconditional);
        cfg.add_edge(join, join, EdgeKind::Back);
        (cfg, blocks)
    }

    #[test]
    fn the_hash_sees_a_changed_kind_target_label_or_unreachable_block() {
        let (cfg, [then_arm, else_arm, join]) = branch();
        let hash = |cfg: &Cfg<u32>| canonical_hash(cfg, |_| 0, kind_key(cfg));
        let original = hash(&cfg);

        let mut rekinded = cfg.clone();
        let arm = rekinded.outgoing(else_arm).next().expect("the arm leaves");
        rekinded.edge_mut(arm).set_kind(EdgeKind::Fallthrough);
        assert_ne!(hash(&rekinded), original, "one edge kind changed");

        let mut retargeted = cfg.clone();
        let latch = retargeted.outgoing(join).next().expect("the join loops");
        retargeted.redirect_edge_target(latch, then_arm);
        assert_ne!(hash(&retargeted), original, "one edge target changed");

        assert_ne!(
            canonical_hash(&cfg, |block| u64::from(block == join), kind_key(&cfg)),
            original,
            "one label changed"
        );

        let mut orphaned = cfg.clone();
        let orphan = orphaned.new_block();
        assert_ne!(hash(&orphaned), original, "an unreachable block joined");
        assert_eq!(
            canonical_order(&orphaned, kind_key(&orphaned)),
            canonical_order(&cfg, kind_key(&cfg)),
        );
        orphaned.add_edge(orphan, join, EdgeKind::Fallthrough);
        let wired = hash(&orphaned);
        orphaned.add_edge(orphan, then_arm, EdgeKind::Jump);
        assert_ne!(
            hash(&orphaned),
            wired,
            "the unreachable block's edges count"
        );
    }

    #[test]
    fn the_walk_takes_edges_by_key_and_ties_in_adjacency_order() {
        let (cfg, [then_arm, else_arm, join]) = branch();
        assert_eq!(
            canonical_order(&cfg, kind_key(&cfg)),
            [cfg.entry(), then_arm, join, else_arm]
        );
        // Reversing the keys visits the false arm first.
        let reversed = |edge: crate::edge::EdgeId| u64::MAX - cfg[edge].kind() as u64;
        assert_eq!(
            canonical_order(&cfg, reversed),
            [cfg.entry(), else_arm, join, then_arm]
        );
        // Equal keys leave the adjacency order to decide.
        assert_eq!(
            canonical_order(&cfg, |_| 0),
            [cfg.entry(), then_arm, join, else_arm]
        );
    }

    /// Fixed graphs give fixed values, so a change of the mixer, or of what
    /// it mixes, shows here before it reaches a stored fingerprint.
    #[test]
    fn golden_values() {
        assert_eq!(Mixer::new().finish(), 0);
        assert_eq!(splitmix64(0), 0xe220_a839_7b1d_cdaf);
        assert_eq!(multiset_fingerprint([]), 0xe220_a839_7b1d_cdaf);
        assert_eq!(multiset_fingerprint([3, 1, 2]), 0x4e28_89b2_1ba4_3ff3);

        let (cfg, _) = branch();
        assert_eq!(
            wl_colors(&cfg, 2, |block| block.index() as u64 % 2, kind_key(&cfg)),
            [
                0x40ae_e0aa_70f0_3021,
                0x0122_d7d3_4391_2ed1,
                0x6e5c_c116_9390_69ce,
                0xfb1e_b1ab_7836_11f0,
            ]
        );
        assert_eq!(
            canonical_hash(&cfg, |_| 0, kind_key(&cfg)),
            0x3970_e886_ef00_3ff0
        );
        let mut orphaned = cfg.clone();
        orphaned.new_block();
        assert_eq!(
            canonical_hash(&orphaned, |_| 0, kind_key(&orphaned)),
            0xf6b8_55fc_6f7e_04e6
        );
    }
}
