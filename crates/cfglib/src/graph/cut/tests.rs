//! Cuts and paths against brute force on small graphs, the canonical cut
//! across adjacency orders, single-node cuts against dominators, and the
//! limit.

use core::cell::Cell;

use super::*;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::{Edge, EdgeId, EdgeKind};
use crate::graph::dominator::DominatorTree;
use crate::graph::edge_view::EdgeRef;
use crate::test_util::rng::Rng;

fn block(index: usize) -> BlockId {
    BlockId::from_index(index)
}

fn cfg_of(count: usize, edges: &[(usize, usize)]) -> Cfg<u32> {
    let mut cfg = Cfg::<u32>::new();
    for _ in 1..count {
        cfg.new_block();
    }
    for &(from, to) in edges {
        cfg.add_edge(block(from), block(to), EdgeKind::Jump);
    }
    cfg
}

/// Whether no source reaches a sink once the marked edges and nodes are gone.
fn separated(
    cfg: &Cfg<u32>,
    sources: &[BlockId],
    sinks: &[BlockId],
    cut_edge: &[bool],
    cut_node: &[bool],
) -> bool {
    let mut seen = vec![false; cfg.block_bound()];
    let mut stack: Vec<BlockId> = sources.to_vec();
    for source in sources {
        seen[source.index()] = true;
    }
    while let Some(node) = stack.pop() {
        if sinks.contains(&node) {
            return false;
        }
        for edge in cfg.outgoing(node) {
            let next = cfg.edge(edge).target();
            if !cut_edge[edge.index()] && !cut_node[next.index()] && !seen[next.index()] {
                seen[next.index()] = true;
                stack.push(next);
            }
        }
    }
    true
}

/// The size of the smallest set of items, out of `items`, whose removal
/// separates the terminals.
fn brute_force(items: &[usize], bound: usize, separates: impl Fn(&[bool]) -> bool) -> usize {
    let mut best = usize::MAX;
    for mask in 0_u32..1 << items.len() {
        let size = usize::try_from(mask.count_ones()).expect("small");
        if size >= best {
            continue;
        }
        let mut removed = vec![false; bound];
        for (bit, &item) in items.iter().enumerate() {
            removed[item] = mask & (1 << bit) != 0;
        }
        if separates(&removed) {
            best = size;
        }
    }
    best
}

/// A random graph of up to eight nodes and twelve edges, self-loops and
/// parallel edges included, with disjoint nonempty sources and sinks.
fn random_case(rng: &mut Rng) -> (Cfg<u32>, Vec<BlockId>, Vec<BlockId>) {
    let count = 2 + rng.below(7);
    let edges: Vec<(usize, usize)> = (0..rng.below(13))
        .map(|_| (rng.below(count), rng.below(count)))
        .collect();
    let mut nodes: Vec<usize> = (0..count).collect();
    rng.shuffle(&mut nodes);
    let sources = 1 + rng.below(count - 1);
    let sinks = 1 + rng.below(count - sources);
    let pick = |slots: &[usize]| slots.iter().copied().map(block).collect();
    (
        cfg_of(count, &edges),
        pick(&nodes[..sources]),
        pick(&nodes[sources..sources + sinks]),
    )
}

/// Each path runs from a source to a sink along consecutive edges, and no
/// two share an edge.
fn assert_disjoint_paths(
    cfg: &Cfg<u32>,
    paths: &[Vec<EdgeId>],
    sources: &[BlockId],
    sinks: &[BlockId],
) {
    let mut used = vec![false; cfg.edge_bound()];
    for path in paths {
        let first = cfg
            .edge(*path.first().expect("a path has an edge"))
            .source();
        let last = cfg.edge(*path.last().expect("a path has an edge")).target();
        assert!(sources.contains(&first) && sinks.contains(&last));
        for pair in path.windows(2) {
            assert_eq!(cfg.edge(pair[0]).target(), cfg.edge(pair[1]).source());
        }
        for edge in path {
            assert!(
                !core::mem::replace(&mut used[edge.index()], true),
                "shared edge"
            );
        }
    }
}

#[test]
fn cuts_and_paths_match_brute_force_on_small_graphs() {
    let mut rng = Rng::new(82);
    for _ in 0..1_000 {
        let (cfg, sources, sinks) = random_case(&mut rng);
        let (nodes, edges) = (cfg.block_bound(), cfg.edge_bound());
        let none = vec![false; nodes.max(edges)];

        let cut = min_edge_cut(&cfg, sources.clone(), sinks.clone(), None).expect("disjoint");
        let all_edges: Vec<usize> = (0..edges).collect();
        let smallest = brute_force(&all_edges, edges, |removed| {
            separated(&cfg, &sources, &sinks, removed, &none)
        });
        assert_eq!(cut.value, smallest);
        assert_eq!(cut.edges.len(), cut.value);
        let mut removed = vec![false; edges];
        for edge in &cut.edges {
            removed[edge.index()] = true;
            assert!(cut.source_side[cfg.edge(*edge).source().index()]);
            assert!(!cut.source_side[cfg.edge(*edge).target().index()]);
        }
        assert!(separated(&cfg, &sources, &sinks, &removed, &none));
        assert!(sources.iter().all(|node| cut.source_side[node.index()]));
        assert!(sinks.iter().all(|node| !cut.source_side[node.index()]));

        let paths =
            edge_disjoint_paths(&cfg, sources.clone(), sinks.clone(), None).expect("disjoint");
        assert_eq!(paths.len(), cut.value);
        assert_disjoint_paths(&cfg, &paths, &sources, &sinks);

        let adjacent = cfg.edge_ids().any(|edge| {
            sources.contains(&cfg.edge(edge).source()) && sinks.contains(&cfg.edge(edge).target())
        });
        let Some(cut) = min_vertex_cut(&cfg, sources.clone(), sinks.clone(), None) else {
            assert!(
                adjacent,
                "only an edge between the terminals leaves no vertex cut"
            );
            continue;
        };
        assert!(!adjacent);
        let inner: Vec<usize> = (0..nodes)
            .filter(|&slot| !sources.contains(&block(slot)) && !sinks.contains(&block(slot)))
            .collect();
        let smallest = brute_force(&inner, nodes, |removed| {
            separated(&cfg, &sources, &sinks, &none, removed)
        });
        assert_eq!(cut.value, smallest);
        assert_eq!(cut.vertices.len(), cut.value);
        let mut removed = vec![false; nodes];
        for node in &cut.vertices {
            assert!(inner.contains(&node.index()));
            assert!(!cut.source_side[node.index()]);
            removed[node.index()] = true;
        }
        assert!(separated(&cfg, &sources, &sinks, &none, &removed));
    }
}

#[test]
fn the_cut_is_the_same_in_every_adjacency_order() {
    let mut rng = Rng::new(182);
    for _ in 0..300 {
        let count = 2 + rng.below(7);
        let mut edges: Vec<(usize, usize)> = (0..rng.below(13))
            .map(|_| (rng.below(count), rng.below(count)))
            .collect();
        let (sources, sinks) = ([block(0)], [block(count - 1)]);
        let first = cfg_of(count, &edges);
        let edge_side = min_edge_cut(&first, sources, sinks, None).map(|cut| cut.source_side);
        let vertex_side = min_vertex_cut(&first, sources, sinks, None).map(|cut| cut.source_side);
        for _ in 0..10 {
            rng.shuffle(&mut edges);
            let shuffled = cfg_of(count, &edges);
            assert_eq!(
                min_edge_cut(&shuffled, sources, sinks, None).map(|cut| cut.source_side),
                edge_side
            );
            assert_eq!(
                min_vertex_cut(&shuffled, sources, sinks, None).map(|cut| cut.source_side),
                vertex_side
            );
        }
    }
}

#[test]
fn one_node_separates_the_root_from_a_block_exactly_when_the_root_is_not_its_idom() {
    let mut rng = Rng::new(282);
    let mut both = [0; 2];
    for _ in 0..1_000 {
        let count = 2 + rng.below(9);
        let edges: Vec<(usize, usize)> = (0..rng.below(3 * count))
            .map(|_| (rng.below(count), rng.below(count)))
            .collect();
        let cfg = cfg_of(count, &edges);
        let root = cfg.entry();
        let dominators = DominatorTree::compute(&cfg);
        for target in cfg.block_ids() {
            let direct = cfg.successors(root).any(|next| next == target);
            if target == root || direct || !dominators.is_reachable(target) {
                continue;
            }
            let cut = min_vertex_cut(&cfg, [root], [target], None).expect("not adjacent");
            let dominated = dominators.idom(target) != Some(root);
            assert_eq!(cut.value == 1, dominated, "{edges:?} to {target:?}");
            both[usize::from(dominated)] += 1;
        }
    }
    assert!(both.iter().all(|&cases| cases > 100), "{both:?}");
}

/// A view of a CFG that counts every adjacency item and edge it hands out.
struct Counting<'g> {
    cfg: &'g Cfg<u32>,
    touched: Cell<usize>,
}

impl Counting<'_> {
    fn touch(&self) {
        self.touched.set(self.touched.get() + 1);
    }
}

impl GraphView for Counting<'_> {
    type NodeId = BlockId;

    fn node_bound(&self) -> usize {
        self.cfg.block_bound()
    }

    fn node_ids(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.cfg.block_ids()
    }

    fn successors(&self, node: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        GraphView::successors(self.cfg, node).inspect(|_| self.touch())
    }

    fn predecessors(&self, node: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        GraphView::predecessors(self.cfg, node).inspect(|_| self.touch())
    }
}

impl EdgeView for Counting<'_> {
    type EdgeId = EdgeId;
    type EdgeData = Edge;

    fn edge_bound(&self) -> usize {
        self.cfg.edge_bound()
    }

    fn edge_ids(&self) -> impl Iterator<Item = EdgeId> + '_ {
        self.cfg.edge_ids().inspect(|_| self.touch())
    }

    fn outgoing(&self, node: BlockId) -> impl Iterator<Item = EdgeId> + '_ {
        self.cfg.outgoing(node).inspect(|_| self.touch())
    }

    fn incoming(&self, node: BlockId) -> impl Iterator<Item = EdgeId> + '_ {
        self.cfg.incoming(node).inspect(|_| self.touch())
    }

    fn edge(&self, edge: EdgeId) -> EdgeRef<'_, BlockId, EdgeId, Edge> {
        self.touch();
        self.cfg.edge(edge)
    }
}

#[test]
fn a_limit_stops_one_past_it_after_reading_the_view_once() {
    // Ten parallel routes 0 -> i -> 11: ten edges or ten nodes separate them.
    let edges: Vec<(usize, usize)> = (1..=10)
        .flat_map(|route| [(0, route), (route, 11)])
        .collect();
    let cfg = cfg_of(12, &edges);
    let (sources, sinks) = ([block(0)], [block(11)]);
    let edge_count = edges.len();

    let view = Counting {
        cfg: &cfg,
        touched: Cell::new(0),
    };
    let cut = min_edge_cut(&view, sources, sinks, Some(3)).expect("disjoint");
    assert_eq!(
        cut,
        EdgeCut {
            value: 4,
            edges: Vec::new(),
            source_side: Vec::new()
        }
    );
    assert!(
        view.touched.get() <= 4 * edge_count,
        "{} touches",
        view.touched.get()
    );

    view.touched.set(0);
    let cut = min_vertex_cut(&view, sources, sinks, Some(3)).expect("not adjacent");
    assert_eq!(
        cut,
        VertexCut {
            value: 4,
            vertices: Vec::new(),
            source_side: Vec::new()
        }
    );
    assert!(
        view.touched.get() <= 4 * edge_count,
        "{} touches",
        view.touched.get()
    );

    let paths = edge_disjoint_paths(&cfg, sources, sinks, Some(3)).expect("disjoint");
    assert_eq!(paths.len(), 4);
    assert_disjoint_paths(&cfg, &paths, &sources, &sinks);

    assert_eq!(
        min_edge_cut(&cfg, sources, sinks, Some(9))
            .expect("disjoint")
            .value,
        10
    );
    let exact = min_edge_cut(&cfg, sources, sinks, Some(10)).expect("disjoint");
    assert_eq!((exact.value, exact.edges.len()), (10, 10));
    let exact = min_vertex_cut(&cfg, sources, sinks, Some(usize::MAX)).expect("not adjacent");
    assert_eq!(exact.vertices, (1..=10).map(block).collect::<Vec<_>>());
}

#[test]
fn terminals_that_meet_have_no_cut() {
    let cfg = cfg_of(3, &[(0, 1), (1, 2)]);
    assert_eq!(
        min_edge_cut(&cfg, [block(0), block(1)], [block(1)], None),
        None
    );
    assert_eq!(min_vertex_cut(&cfg, [block(1)], [block(1)], None), None);
    assert_eq!(
        edge_disjoint_paths(&cfg, [block(1)], [block(1)], None),
        None
    );

    // An edge between a source and a sink cannot be cut by removing nodes.
    assert_eq!(min_vertex_cut(&cfg, [block(0)], [block(1)], None), None);
    let cut = min_edge_cut(&cfg, [block(0)], [block(1)], None).expect("disjoint");
    assert_eq!(cut.value, 1);
}

#[test]
fn parallel_edges_count_and_self_loops_do_not() {
    let cfg = cfg_of(2, &[(0, 0), (0, 1), (0, 1), (1, 1)]);
    let cut = min_edge_cut(&cfg, [block(0)], [block(1)], None).expect("disjoint");
    assert_eq!(cut.value, 2);
    assert_eq!(cut.edges, [EdgeId::from_index(1), EdgeId::from_index(2)]);
    let paths = edge_disjoint_paths(&cfg, [block(0)], [block(1)], None).expect("disjoint");
    assert_eq!(
        paths,
        [vec![EdgeId::from_index(1)], vec![EdgeId::from_index(2)]]
    );
}

#[test]
fn nothing_to_separate_costs_nothing() {
    let mut cfg = cfg_of(4, &[(0, 1), (2, 3)]);
    cfg.remove_block(block(2));
    let cut = min_edge_cut(&cfg, [block(0)], [block(3)], None).expect("disjoint");
    assert_eq!((cut.value, cut.edges), (0, Vec::new()));
    assert_eq!(cut.source_side, [true, true, false, false]);
    let cut = min_vertex_cut(&cfg, [], [block(3)], None).expect("disjoint");
    assert_eq!((cut.value, cut.source_side), (0, vec![false; 4]));
    assert_eq!(
        edge_disjoint_paths(&cfg, [block(0)], [], None),
        Some(Vec::new())
    );
}

#[test]
#[should_panic(expected = "a sink is out of range")]
fn a_terminal_must_be_below_the_bound() {
    let cfg = cfg_of(2, &[(0, 1)]);
    let _ = min_edge_cut(&cfg, [block(0)], [block(5)], None);
}
