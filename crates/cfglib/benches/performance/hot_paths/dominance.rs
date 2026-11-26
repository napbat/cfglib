use std::cell::OnceCell;

use cfglib::{
    BlockId, CertificateError, Cfg, DominanceFrontiers, DominatorScratch, DominatorTree,
    DominatorUpdate, Graph, GraphView, LowHighOrder, NodeId, PostDominatorTree, PostNode, Rooted,
    control_dependence_graph,
};

use super::NODE_COUNT;
use crate::analysis_oracles::{
    assert_branchy_dominators, assert_branchy_post_dominators, assert_control_dependence_graph,
    assert_dominance_frontiers,
};
use crate::fixtures::{branchy_graph, linear_cfg, many_exit_cfg};
use crate::harness::{BenchmarkSuite, benchmark_case};
use crate::rng::Rng;

/// Nodes of the large sparse graph of the update cases.
const SPARSE_NODES: usize = 1 << 20;

/// A path through every node plus one random edge out of each, forward or
/// back: about two edges per node, every node reachable from node 0.
fn sparse_graph(nodes: usize) -> Graph<(), ()> {
    let mut graph = Graph::with_capacity(nodes, 2 * nodes);
    let ids: Vec<NodeId> = (0..nodes).map(|_| graph.add_node(())).collect();
    for pair in ids.windows(2) {
        graph.add_edge(pair[0], pair[1], ());
    }
    let mut stream = Rng::splitmix(nodes as u64);
    for &id in &ids {
        graph.add_edge(id, ids[stream.below(nodes)], ());
    }
    graph
}

/// A graph, the dominator tree it had, and a batch of edits already applied
/// to it.
struct Edited {
    tree: DominatorTree<NodeId>,
    graph: Graph<(), ()>,
    updates: Vec<DominatorUpdate<NodeId>>,
}

impl Edited {
    /// `inserts` insertions that each move a node, and `deletes` deletions of
    /// edges into a node that does not dominate their tail, applied to
    /// `graph` from node 0.
    fn new(graph: &Graph<(), ()>, inserts: usize, deletes: usize, seed: u64) -> Self {
        let root = NodeId::from_index(0);
        let tree = DominatorTree::compute(&Rooted::new(graph, root));
        let mut edited = graph.clone();
        let mut stream = Rng::splitmix(seed);
        let nodes = graph.node_count();
        let mut updates = Vec::new();
        while updates.len() < inserts {
            let (from, to) = (
                NodeId::from_index(stream.below(nodes)),
                NodeId::from_index(stream.below(nodes)),
            );
            let common = tree.nearest_common_dominator(from, to);
            if common != Some(to) && common != tree.idom(to) {
                edited.add_edge(from, to, ());
                updates.push(DominatorUpdate::Insert { from, to });
            }
        }
        let mut removed = 0;
        while removed < deletes {
            let from = NodeId::from_index(stream.below(nodes));
            let edges: Vec<_> = edited.outgoing(from).collect();
            let Some(&edge) = edges.get(stream.below(edges.len().max(1))) else {
                continue;
            };
            let to = edited.edge(edge).target();
            if tree.dominates(to, from)
                || edges
                    .iter()
                    .filter(|&&other| edited.edge(other).target() == to)
                    .count()
                    > 1
            {
                continue;
            }
            edited.remove_edge(edge);
            updates.push(DominatorUpdate::Delete { from, to });
            removed += 1;
        }
        Self {
            tree,
            graph: edited,
            updates,
        }
    }

    /// The edited graph from node 0.
    fn rooted(&self) -> Rooted<'_, Graph<(), ()>> {
        Rooted::new(&self.graph, NodeId::from_index(0))
    }

    /// Update a copy of the old tree with the batch, as a consumer that keeps
    /// its tree would, and return it with the nodes that moved.
    fn update(
        &self,
        scratch: &mut DominatorScratch,
        changed: &mut Vec<NodeId>,
    ) -> (DominatorTree<NodeId>, Vec<NodeId>) {
        let mut tree = self.tree.clone();
        tree.apply_updates_in(scratch, &self.rooted(), &self.updates, changed);
        (tree, changed.clone())
    }

    /// Recompute the tree and diff it against the old one: a full
    /// construction plus an O(n) scan for the nodes whose parent changed.
    fn recompute(&self) -> (DominatorTree<NodeId>, Vec<NodeId>) {
        let graph = self.rooted();
        let next = DominatorTree::compute(&graph);
        let changed = (0..graph.node_bound())
            .map(NodeId::from_index)
            .filter(|&node| self.tree.idom(node) != next.idom(node))
            .collect();
        (next, changed)
    }

    /// Check an answer against a fresh construction and its diff.
    fn check(&self, (tree, changed): &(DominatorTree<NodeId>, Vec<NodeId>)) {
        let (expected, moved) = self.recompute();
        assert!(
            tree == &expected,
            "the updated tree differs from a fresh one"
        );
        assert_eq!(changed, &moved);
    }
}

/// Register the update of `edited` against its recomputation, under the two
/// names given.
macro_rules! update_cases {
    ($suite:expr, $edited:expr, $incremental:literal, $recomputed:literal) => {{
        let mut scratch = DominatorScratch::default();
        let mut changed = Vec::new();
        benchmark_case!(
            $suite,
            $incremental,
            || $edited().update(&mut scratch, &mut changed),
            |result| $edited().check(result)
        );
        benchmark_case!($suite, $recomputed, || $edited().recompute(), |result| {
            $edited().check(result)
        });
    }};
}

/// One insertion that moves nodes, one deletion, and a batch of eight of
/// each, on the branchy fixture and on a sparse graph of 2²⁰ nodes, each
/// against a full recomputation and its diff. The large graph and its
/// batches are built only when one of its cases runs.
fn register_dominator_updates(suite: &mut BenchmarkSuite<'_>) {
    let branchy = branchy_graph(NODE_COUNT);
    let insert = Edited::new(&branchy, 1, 0, 1);
    let delete = Edited::new(&branchy, 0, 1, 2);
    let batch = Edited::new(&branchy, 8, 8, 3);
    update_cases!(
        suite,
        || &insert,
        "graph_dominators_update_insert",
        "graph_dominators_diff_insert"
    );
    update_cases!(
        suite,
        || &delete,
        "graph_dominators_update_delete",
        "graph_dominators_diff_delete"
    );
    update_cases!(
        suite,
        || &batch,
        "graph_dominators_update_batch16",
        "graph_dominators_diff_batch16"
    );

    let sparse = OnceCell::new();
    let large = || sparse.get_or_init(|| sparse_graph(SPARSE_NODES));
    let (insert, delete, batch) = (OnceCell::new(), OnceCell::new(), OnceCell::new());
    update_cases!(
        suite,
        || insert.get_or_init(|| Edited::new(large(), 1, 0, 4)),
        "sparse_dominators_update_insert",
        "sparse_dominators_diff_insert"
    );
    update_cases!(
        suite,
        || delete.get_or_init(|| Edited::new(large(), 0, 1, 5)),
        "sparse_dominators_update_delete",
        "sparse_dominators_diff_delete"
    );
    update_cases!(
        suite,
        || batch.get_or_init(|| Edited::new(large(), 8, 8, 6)),
        "sparse_dominators_update_batch16",
        "sparse_dominators_diff_batch16"
    );
}

fn register_dominators(suite: &mut BenchmarkSuite<'_>, cfg: &Cfg<u32>) {
    let cfg_dominators = DominatorTree::compute(cfg);
    let linear = linear_cfg(NODE_COUNT / 2);
    let linear_dominators = DominatorTree::compute(&linear);
    benchmark_case!(
        suite,
        "cfg_dominators",
        || DominatorTree::compute(cfg),
        |result: &DominatorTree| assert_branchy_dominators(result, NODE_COUNT)
    );
    // The tree and its low-high order, from a fresh scratch as `compute`
    // starts from, and the linear check of that pair on its own.
    benchmark_case!(
        suite,
        "cfg_dominators_certified",
        || DominatorTree::compute_certified_in(&mut DominatorScratch::default(), cfg),
        |(tree, certificate): &(DominatorTree, LowHighOrder)| {
            assert_branchy_dominators(tree, NODE_COUNT);
            assert_eq!(tree.verify_certificate(cfg, certificate), Ok(()));
        }
    );
    let (_, certificate) =
        DominatorTree::compute_certified_in(&mut DominatorScratch::default(), cfg);
    benchmark_case!(
        suite,
        "cfg_dominators_verify",
        || cfg_dominators.verify_certificate(cfg, &certificate),
        |result: &Result<(), CertificateError<BlockId>>| assert_eq!(*result, Ok(()))
    );
    benchmark_case!(
        suite,
        "cfg_dominance_frontiers",
        || { DominanceFrontiers::compute(cfg, &cfg_dominators) },
        |result: &DominanceFrontiers| assert_dominance_frontiers(result, cfg, &cfg_dominators,)
    );
    benchmark_case!(
        suite,
        "cfg_dominator_queries_linear",
        || {
            let root = BlockId::from_index(0);
            (0..NODE_COUNT / 2)
                .filter(|&index| linear_dominators.dominates(root, BlockId::from_index(index)))
                .count()
        },
        |result: &usize| assert_eq!(*result, NODE_COUNT / 2)
    );
}

fn register_post_dominators(suite: &mut BenchmarkSuite<'_>, cfg: &Cfg<u32>) {
    let cfg_post_dominators = PostDominatorTree::compute(cfg);
    let many_exits = many_exit_cfg(NODE_COUNT - 1);
    benchmark_case!(
        suite,
        "cfg_post_dominators",
        || PostDominatorTree::compute(cfg),
        |result: &PostDominatorTree| assert_branchy_post_dominators(result, NODE_COUNT)
    );
    benchmark_case!(
        suite,
        "cfg_post_dominators_many_exits",
        || { PostDominatorTree::compute(&many_exits) },
        |result: &PostDominatorTree| {
            for index in 0..NODE_COUNT {
                let block = BlockId::from_index(index);
                assert!(result.contains(block));
                assert_eq!(result.ipdom(block), Some(PostNode::Exit));
            }
        }
    );
    benchmark_case!(
        suite,
        "cfg_control_dependence_graph",
        || { control_dependence_graph(cfg, &cfg_post_dominators) },
        |result: &Graph<BlockId, ()>| assert_control_dependence_graph(
            result,
            cfg,
            &cfg_post_dominators,
        )
    );
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>, cfg: &Cfg<u32>) {
    register_dominators(suite, cfg);
    register_post_dominators(suite, cfg);
    register_dominator_updates(suite);
}
