//! Deterministic pseudo-random graph shapes and gen/kill effects, written
//! down as plain data so the reference solvers read adjacency from the data
//! and never from the library's graph store.

use std::collections::BTreeSet;
use std::fmt::Debug;

use cfglib::{
    BlockId, Cfg, DenseId, Direction, EdgeKind, Facts, IterationOrder, Seeds, Solve, SolveConfig,
    SolveStats,
};

/// A xorshift generator: the same seed gives the same graphs everywhere.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut state = self.0;
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        self.0 = state;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A value below `bound`.
    pub fn below(&mut self, bound: usize) -> usize {
        let bound = u64::try_from(bound).expect("a small bound");
        usize::try_from((self.next_u64() >> 16) % bound).expect("below the bound")
    }
}

/// The edge kinds a random graph draws from.
const KINDS: [EdgeKind; 4] = [
    EdgeKind::Fallthrough,
    EdgeKind::ConditionalTrue,
    EdgeKind::ConditionalFalse,
    EdgeKind::ExceptionUnwind,
];

/// A directed graph of numbered blocks; block `0` is the entry.
#[derive(Debug, Clone)]
pub struct Shape {
    /// How many blocks the store allocates, the removed one included.
    pub blocks: usize,
    /// Every edge in insertion order, with its kind.
    pub edges: Vec<(usize, usize, EdgeKind)>,
    /// A block removed after construction, leaving a hole in the numbering.
    pub hole: Option<usize>,
}

impl Shape {
    /// A random shape. With `entry_has_no_predecessors` no edge enters block
    /// `0`; with `unwinds` some edges are unwind edges.
    pub fn random(rng: &mut Rng, entry_has_no_predecessors: bool, unwinds: bool) -> Self {
        let blocks = 2 + rng.below(7);
        let mut edges = Vec::new();
        for _ in 0..blocks + rng.below(blocks + 1) {
            let source = rng.below(blocks);
            let target = rng.below(blocks);
            if entry_has_no_predecessors && target == 0 {
                continue;
            }
            let kinds = if unwinds {
                KINDS.len()
            } else {
                KINDS.len() - 1
            };
            edges.push((source, target, KINDS[rng.below(kinds)]));
        }
        let hole = (rng.below(3) == 0).then(|| 1 + rng.below(blocks - 1));
        Self {
            blocks,
            edges,
            hole,
        }
    }

    /// A shape with no hole.
    pub fn new(blocks: usize, edges: &[(usize, usize)]) -> Self {
        Self {
            blocks,
            edges: edges
                .iter()
                .map(|&(source, target)| (source, target, EdgeKind::Fallthrough))
                .collect(),
            hole: None,
        }
    }

    /// Build the control-flow graph with `code[b]` as the instructions of
    /// block `b`: every block allocated, every edge added, then the hole
    /// removed, which also removes the edges that touch it.
    pub fn build<I>(&self, code: Vec<Vec<I>>) -> Cfg<I> {
        let mut cfg = Cfg::<I>::new();
        let mut ids = vec![cfg.entry()];
        ids.extend((1..self.blocks).map(|_| cfg.new_block()));
        for (id, instructions) in ids.iter().zip(code) {
            cfg.block_mut(*id).instructions_mut().extend(instructions);
        }
        for &(source, target, kind) in &self.edges {
            cfg.add_edge(ids[source], ids[target], kind);
        }
        if let Some(hole) = self.hole {
            assert!(cfg.remove_block(ids[hole]));
        }
        cfg
    }

    /// The control-flow graph without instructions.
    pub fn cfg(&self) -> Cfg<()> {
        self.build((0..self.blocks).map(|_| Vec::new()).collect())
    }

    fn touches_hole(&self, source: usize, target: usize) -> bool {
        self.hole == Some(source) || self.hole == Some(target)
    }

    /// Every live block, ascending.
    pub fn live(&self) -> Vec<usize> {
        (0..self.blocks).filter(|b| self.hole != Some(*b)).collect()
    }

    /// The live edges with their identity (insertion index), in order.
    pub fn live_edges(&self) -> Vec<(usize, (usize, usize, EdgeKind))> {
        self.edges
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, (source, target, _))| !self.touches_hole(*source, *target))
            .collect()
    }

    /// Successors of `block` over live edges, with multiplicity.
    pub fn successors(&self, block: usize) -> Vec<usize> {
        self.live_edges()
            .into_iter()
            .filter(|(_, (source, _, _))| *source == block)
            .map(|(_, (_, target, _))| target)
            .collect()
    }

    /// Predecessors of `block` over live edges, with multiplicity.
    pub fn predecessors(&self, block: usize) -> Vec<usize> {
        self.live_edges()
            .into_iter()
            .filter(|(_, (_, target, _))| *target == block)
            .map(|(_, (source, _, _))| source)
            .collect()
    }

    /// The blocks reachable from the entry over live edges.
    pub fn reachable(&self) -> BTreeSet<usize> {
        let mut seen = BTreeSet::new();
        let mut stack = vec![0];
        while let Some(block) = stack.pop() {
            if seen.insert(block) {
                stack.extend(self.successors(block));
            }
        }
        seen
    }
}

/// Per-block gen and kill sets over a small universe.
pub struct GenKill {
    generated: Vec<BTreeSet<u8>>,
    killed: Vec<BTreeSet<u8>>,
}

impl GenKill {
    pub fn random(rng: &mut Rng, blocks: usize) -> Self {
        let draw = |rng: &mut Rng| {
            (0..blocks)
                .map(|_| {
                    (0..6_u8)
                        .filter(|_| rng.below(4) == 0)
                        .collect::<BTreeSet<u8>>()
                })
                .collect::<Vec<_>>()
        };
        let generated = draw(rng);
        let killed = draw(rng);
        Self { generated, killed }
    }

    pub fn with_gen(blocks: usize, generated: &[(usize, u8)]) -> Self {
        let mut effects = Self {
            generated: vec![BTreeSet::new(); blocks],
            killed: vec![BTreeSet::new(); blocks],
        };
        for &(block, fact) in generated {
            effects.generated[block].insert(fact);
        }
        effects
    }

    /// `gen ∪ (input − kill)`.
    pub fn apply(&self, block: usize, input: &BTreeSet<u8>) -> BTreeSet<u8> {
        input
            .difference(&self.killed[block])
            .chain(&self.generated[block])
            .copied()
            .collect()
    }

    /// Draws new gen and kill sets for `block`, as an edit of its code.
    pub fn redraw(&mut self, rng: &mut Rng, block: usize) {
        let mut draw = || (0..6_u8).filter(|_| rng.below(4) == 0).collect();
        self.generated[block] = draw();
        self.killed[block] = draw();
    }
}

/// One random edit of a control-flow graph, as an editor makes it.
#[derive(Debug, Clone, Copy)]
pub enum Edit {
    /// An edge added from the first block to the second.
    Added(usize, usize),
    /// The edge from the first block to the second removed.
    Removed(usize, usize),
    /// New effects for one block.
    Redrawn(usize),
}

impl Edit {
    /// Applies a random edit to `cfg` and `effects`: an added edge of a
    /// random kind among `kinds`, or, unless `only_grow`, also a removed edge
    /// or new effects for a block.
    pub fn random(
        rng: &mut Rng,
        cfg: &mut Cfg<()>,
        effects: &mut GenKill,
        shape: &Shape,
        kinds: &[EdgeKind],
        only_grow: bool,
    ) -> Self {
        let live = shape.live();
        let edges: Vec<_> = cfg
            .edges()
            .map(|edge| (edge.id(), edge.source().index(), edge.target().index()))
            .collect();
        let choice = if only_grow { 0 } else { rng.below(3) };
        if choice == 1 && !edges.is_empty() {
            let (edge, source, target) = edges[rng.below(edges.len())];
            assert!(cfg.remove_edge(edge));
            return Self::Removed(source, target);
        }
        if choice == 2 {
            let block = live[rng.below(live.len())];
            effects.redraw(rng, block);
            return Self::Redrawn(block);
        }
        let source = live[rng.below(live.len())];
        let target = live[rng.below(live.len())];
        let kind = kinds[rng.below(kinds.len())];
        cfg.add_edge(
            BlockId::from_index(source),
            BlockId::from_index(target),
            kind,
        );
        Self::Added(source, target)
    }

    /// The block whose transfer, boundary or upstream edges along
    /// `direction` the edit changed.
    pub fn changed(self, direction: Direction) -> BlockId {
        let block = match (self, direction) {
            (Self::Added(_, target) | Self::Removed(_, target), Direction::Forward) => target,
            (Self::Added(source, _) | Self::Removed(source, _), Direction::Backward) => source,
            (Self::Redrawn(block), _) => block,
        };
        BlockId::from_index(block)
    }
}

/// The union of two sets.
pub fn union(left: &BTreeSet<u8>, right: &BTreeSet<u8>) -> BTreeSet<u8> {
    left.union(right).copied().collect()
}

/// The intersection of two sets.
pub fn intersection(left: &BTreeSet<u8>, right: &BTreeSet<u8>) -> BTreeSet<u8> {
    left.intersection(right).copied().collect()
}

/// Every iteration order a solve can run in.
pub const ORDERS: [IterationOrder; 3] = [
    IterationOrder::Index,
    IterationOrder::ReversePostorder,
    IterationOrder::WeakTopological,
];

/// An index-order solve config that stops after `max_steps` steps.
pub fn step_limit(max_steps: usize) -> SolveConfig {
    SolveConfig {
        max_steps: Some(max_steps),
        ..SolveConfig::default()
    }
}

/// A solve from every live node, within `config`.
pub fn live<'a, N>(config: SolveConfig) -> Solve<'a, N> {
    Solve {
        seeds: Seeds::Live,
        config,
    }
}

/// A solve from `seeds` alone, within `config`.
pub fn seeded<N>(seeds: &[N], config: SolveConfig) -> Solve<'_, N> {
    Solve {
        seeds: Seeds::Of(seeds),
        config,
    }
}

/// The solve of a per-block analysis of a control-flow graph, within
/// `config`: forward from the entry, which starts from the boundary fact
/// even where it is a branch target, so that only what it reaches is
/// solved; backward from every live block.
pub fn rooted<'a>(cfg: &Cfg<()>, direction: Direction, config: SolveConfig) -> Solve<'a, BlockId> {
    let seeds = match direction {
        Direction::Forward => Seeds::Reachable(cfg.entry()),
        Direction::Backward => Seeds::Live,
    };
    Solve { seeds, config }
}

/// Asserts that two solves agree on the IN and OUT facts of every node in
/// `nodes`, whatever number of steps each took.
pub fn assert_same_facts<N: DenseId, F: PartialEq + Debug>(
    actual: &Facts<F>,
    expected: &Facts<F>,
    nodes: impl IntoIterator<Item = N>,
    what: &str,
) {
    for node in nodes {
        assert_eq!(
            actual.fact_in(node),
            expected.fact_in(node),
            "IN of {node:?}: {what}"
        );
        assert_eq!(
            actual.fact_out(node),
            expected.fact_out(node),
            "OUT of {node:?}: {what}"
        );
    }
}

/// Runs `stream` with a visitor and asserts that it visits exactly `nodes`,
/// once each, with the facts of `expected`, in as many steps.
pub fn assert_streams_like<N: DenseId, F: PartialEq + Debug>(
    expected: &Facts<F>,
    nodes: impl IntoIterator<Item = N>,
    stream: impl FnOnce(&mut dyn FnMut(N, &F, &F)) -> SolveStats,
    what: &str,
) {
    let mut visited = Vec::new();
    let stats = stream(&mut |node, input, output| {
        assert_eq!(input, expected.fact_in(node), "IN of {node:?}: {what}");
        assert_eq!(output, expected.fact_out(node), "OUT of {node:?}: {what}");
        visited.push(node);
    });
    assert_eq!(
        stats.step_count(),
        expected.stats().step_count(),
        "steps: {what}"
    );
    visited.sort_unstable();
    let mut nodes: Vec<N> = nodes.into_iter().collect();
    nodes.sort_unstable();
    assert_eq!(visited, nodes, "visits: {what}");
}
