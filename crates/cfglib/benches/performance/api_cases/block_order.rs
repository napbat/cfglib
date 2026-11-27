//! Block orders for linearization: the Ext-TSP order of a profiled loop
//! nest, and the hot/cold partition of a base order.

use cfglib::{
    BlockId, Cfg, ColdClass, ColdnessClasses, Direction, EdgeId, EdgeKind, Emitter, ExtTspConfig,
    Graph, NodeId, Rooted, ext_tsp_order, ext_tsp_score, falls_through, hot_cold_order,
    reverse_postorder,
};

use super::BenchmarkSuite;
use crate::harness::benchmark_case;

/// Diamonds in the chain. Every fourth false arm aborts, seeded `Rare`.
const DIAMONDS: usize = 1_024;

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    register_ext_tsp(suite);
    register_hot_cold(suite);
}

fn register_ext_tsp(suite: &mut BenchmarkSuite<'_>) {
    let config = ExtTspConfig::default();
    let small = ProfiledLoops::new(1_000);
    benchmark_case!(
        suite,
        "api_ext_tsp_order",
        covers[ext_tsp_order],
        || small.order(&config),
        |order: &Vec<BlockId>| small.check(order, &config)
    );
    let large = ProfiledLoops::new(10_000);
    benchmark_case!(
        suite,
        "api_ext_tsp_order_10k",
        || large.order(&config),
        |order: &Vec<BlockId>| large.check(order, &config)
    );
    let reverse_postorder = reverse_postorder(&small.cfg, small.cfg.entry(), Direction::Forward);
    benchmark_case!(
        suite,
        "api_ext_tsp_score",
        covers[ext_tsp_score],
        || small.score(&reverse_postorder, &config),
        |score: &u128| assert!(*score > 0)
    );
    benchmark_case!(
        suite,
        "api_ext_tsp_order_falls_through",
        covers[falls_through],
        || {
            ext_tsp_order(
                &small.cfg,
                |block| small.sizes[block.index()],
                |edge| small.counts[edge.index()],
                |edge| falls_through(&small.cfg, edge, &Inverting),
                &config,
            )
        },
        |order: &Vec<BlockId>| {
            assert_eq!(order.len(), small.cfg.block_ids().count());
            assert_eq!(order[0], small.cfg.entry());
        }
    );
}

fn register_hot_cold(suite: &mut BenchmarkSuite<'_>) {
    let chain = DiamondChain::new(DIAMONDS);
    let rooted = Rooted::new(&chain.graph, NodeId::from_index(0));
    let rare = DIAMONDS.div_ceil(4);
    benchmark_case!(
        suite,
        "api_coldness_classes",
        || chain.classes(),
        |classes: &ColdnessClasses<NodeId>| {
            let cold = chain
                .nodes()
                .filter(|&node| classes.class(node) == ColdClass::Rare)
                .count();
            assert_eq!(cold, rare);
        }
    );
    let classes = chain.classes();
    let base: Vec<NodeId> = chain.nodes().collect();
    benchmark_case!(
        suite,
        "api_hot_cold_order",
        covers[hot_cold_order],
        || hot_cold_order(&rooted, &base, &classes),
        |[hot, cold, never]: &[Vec<NodeId>; 3]| {
            assert_eq!(hot[0], NodeId::from_index(0));
            assert_eq!(hot.len(), base.len() - rare);
            assert_eq!(cold.len(), rare);
            assert_eq!(never.as_slice(), []);
        }
    );
}

/// An emitter whose target form inverts every branch.
struct Inverting;

impl Emitter<u8> for Inverting {
    fn emit_jump(&self, _target: BlockId) -> u8 {
        0
    }

    fn emit_conditional_branch(&self, _condition: &u8, _target: BlockId) -> u8 {
        1
    }

    fn emit_block_start(&self, _block: BlockId) -> Option<u8> {
        None
    }

    fn emit_inverted_conditional_branch(&self, _condition: &u8, _target: BlockId) -> Option<u8> {
        Some(2)
    }
}

/// A chain of profiled loops, five blocks each: a header that enters a hot
/// body or leaves to the exit, a body that rarely branches to a side block,
/// and a latch back to the header. The blocks are allocated in an order that
/// puts the cold side block between the body and the latch.
struct ProfiledLoops {
    cfg: Cfg<u8>,
    sizes: Vec<u64>,
    counts: Vec<u64>,
}

impl ProfiledLoops {
    fn new(block_count: usize) -> Self {
        let mut cfg = Cfg::new();
        let mut counts = Vec::new();
        let mut add = |cfg: &mut Cfg<u8>, source, target, kind, count| {
            let edge: EdgeId = cfg.add_edge(source, target, kind);
            debug_assert_eq!(edge.index(), counts.len());
            counts.push(count);
        };
        let mut header = cfg.entry();
        while cfg.block_bound() + 5 <= block_count {
            let [body, side, latch, exit] = [(); 4].map(|()| cfg.new_block());
            for block in [header, body, side, latch, exit] {
                cfg.block_mut(block).push(0);
            }
            add(&mut cfg, header, exit, EdgeKind::ConditionalFalse, 10);
            add(&mut cfg, header, body, EdgeKind::ConditionalTrue, 1_000);
            add(&mut cfg, body, side, EdgeKind::ConditionalTrue, 10);
            add(&mut cfg, body, latch, EdgeKind::ConditionalFalse, 990);
            add(&mut cfg, side, latch, EdgeKind::Unconditional, 10);
            add(&mut cfg, latch, header, EdgeKind::Back, 990);
            header = exit;
        }
        let sizes = (0..cfg.block_bound())
            .map(|index| {
                let index = u64::try_from(index).expect("a block index fits in u64");
                4 + index.wrapping_mul(0x9E37_79B9_7F4A_7C15) % 61
            })
            .collect();
        Self { cfg, sizes, counts }
    }

    fn order(&self, config: &ExtTspConfig) -> Vec<BlockId> {
        ext_tsp_order(
            &self.cfg,
            |block| self.sizes[block.index()],
            |edge| self.counts[edge.index()],
            |edge| self.falls(edge),
            config,
        )
    }

    fn score(&self, order: &[BlockId], config: &ExtTspConfig) -> u128 {
        ext_tsp_score(
            &self.cfg,
            order,
            |block| self.sizes[block.index()],
            |edge| self.counts[edge.index()],
            |edge| self.falls(edge),
            config,
        )
    }

    /// Every edge but a jump can fall through.
    fn falls(&self, edge: EdgeId) -> bool {
        !matches!(
            self.cfg.edge(edge).kind(),
            EdgeKind::Unconditional | EdgeKind::Back
        )
    }

    fn check(&self, order: &[BlockId], config: &ExtTspConfig) {
        assert_eq!(order.len(), self.cfg.block_ids().count());
        assert_eq!(order[0], self.cfg.entry());
        let base = reverse_postorder(&self.cfg, self.cfg.entry(), Direction::Forward);
        assert!(self.score(order, config) > self.score(&base, config));
    }
}

/// A chain of if/else diamonds over numbered nodes: diamond `i` branches
/// from node `3i` to the arms `3i + 1` and `3i + 2`, which join at
/// `3i + 3`. The false arm of every fourth diamond aborts instead.
struct DiamondChain {
    graph: Graph<(), ()>,
}

impl DiamondChain {
    fn new(diamonds: usize) -> Self {
        let count = 3 * diamonds + 1;
        let mut graph = Graph::with_capacity(count, 4 * diamonds);
        let nodes: Vec<NodeId> = (0..count).map(|_| graph.add_node(())).collect();
        for diamond in 0..diamonds {
            let (branch, join) = (3 * diamond, 3 * diamond + 3);
            graph.add_edge(nodes[branch], nodes[branch + 1], ());
            graph.add_edge(nodes[branch], nodes[branch + 2], ());
            graph.add_edge(nodes[branch + 1], nodes[join], ());
            if !Self::aborts(branch + 2) {
                graph.add_edge(nodes[branch + 2], nodes[join], ());
            }
        }
        Self { graph }
    }

    fn nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        (0..self.graph.node_count()).map(NodeId::from_index)
    }

    /// The classes with the aborting arms seeded `Rare` and every edge
    /// uncapped.
    fn classes(&self) -> ColdnessClasses<NodeId> {
        ColdnessClasses::compute(
            &Rooted::new(&self.graph, NodeId::from_index(0)),
            |node| {
                if Self::aborts(node.index()) {
                    ColdClass::Rare
                } else {
                    ColdClass::Normal
                }
            },
            |_| ColdClass::Normal,
            |_| false,
        )
    }

    /// Whether `node` is the false arm of every fourth diamond.
    fn aborts(node: usize) -> bool {
        node % 12 == 2
    }
}
