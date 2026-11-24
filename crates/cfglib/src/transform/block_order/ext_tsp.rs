//! Ext-TSP block order: the profile-guided placement of Newell and Pupyrev,
//! by greedy chain merging over integer scores.
//!
//! The Ext-TSP score rewards a layout for each edge whose target is close to
//! its source: a fallthrough earns the full count of the edge, and a short
//! jump earns a part of it that falls with the distance. Maximizing it is
//! APX-hard, so the order is the greedy of the reference implementation
//! (LLVM's `CodeLayout`, BOLT): every block starts as a chain, and the pair
//! of chains whose merge gains the most is merged until no merge gains.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::dataflow::cfg_view::CfgView;
use crate::edge::EdgeId;
use crate::graph::edge_view::EdgeView;
use crate::graph::traverse::{TraversalDirection, reverse_postorder};
use crate::graph::view::{DenseId, RootedView};
use crate::transform::linearize::{Emitter, falls_through};

mod chains;
#[cfg(test)]
mod tests;

/// The credit of a fallthrough, in the thousandths that the jump weights of
/// [`ExtTspConfig`] count in.
const FALLTHROUGH_WEIGHT: u32 = 1_000;

/// The parameters of the Ext-TSP score and of its greedy order.
///
/// The [`Default`] holds the values of the reference implementation (LLVM's
/// `CodeLayout`): a jump of up to 1024 bytes forward or 640 bytes backward
/// earns up to a tenth of a fallthrough, and chains of up to 128 blocks may
/// be split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtTspConfig {
    /// The longest forward jump that earns a credit, in bytes from the end
    /// of its source to the start of its target.
    pub forward_distance: u64,
    /// The longest backward jump that earns a credit, in bytes from the end
    /// of its source back to the start of its target.
    pub backward_distance: u64,
    /// The credit of a forward jump of distance zero, in thousandths of a
    /// fallthrough. The credit falls linearly to zero at
    /// [`forward_distance`](Self::forward_distance).
    pub forward_weight: u32,
    /// The credit of a backward jump of distance zero, in thousandths of a
    /// fallthrough. The credit falls linearly to zero at
    /// [`backward_distance`](Self::backward_distance).
    pub backward_weight: u32,
    /// The longest chain, in blocks, that a merge may split to insert another
    /// chain into it or rotate it. Longer chains are only concatenated.
    pub chain_split_threshold: usize,
}

impl Default for ExtTspConfig {
    fn default() -> Self {
        Self {
            forward_distance: 1_024,
            backward_distance: 640,
            forward_weight: 100,
            backward_weight: 100,
            chain_split_threshold: 128,
        }
    }
}

/// A block order that maximizes the Ext-TSP score, entry first.
///
/// The consumer supplies the profile: `block_size` in bytes, `edge_count`
/// as the number of times each edge runs, and `can_fall_through` for the
/// edges whose stream needs no transfer when the target follows the source
/// (see [`ext_tsp_score`]). The callbacks are asked about the reachable part
/// of the graph only, and `can_fall_through` only about the edges with a
/// nonzero count.
///
/// Nodes the root does not reach are omitted, as with reverse postorder, so
/// the result goes to [`linearize`](crate::linearize) as a
/// [`BlockOrder::Custom`](crate::BlockOrder::Custom). The result is the
/// better of the greedy order and the reverse postorder by [`ext_tsp_score`],
/// so it never scores below the default order, and a graph without counts
/// keeps its reverse postorder.
///
/// # Algorithm
///
/// Every reachable block starts as a chain. A merge of two chains places one
/// after the other, or, when the first is no longer than
/// [`chain_split_threshold`](ExtTspConfig::chain_split_threshold), splits it
/// and puts the second inside it, before it, or after it. Merge gains wait in
/// a priority queue keyed by the gain, then by the smaller chain head; after
/// a merge, only the pairs that touch the merged chain are evaluated again.
/// The entry's chain keeps the entry first. The chains that no merge joins
/// follow the entry's chain in the reverse postorder of their heads.
///
/// Scores are integers, so the order is deterministic. The greedy evaluates
/// a pair in time linear in the jumps between the two chains, times the
/// split positions it tries. Its worst case can be quadratic in the number
/// of blocks; the split threshold bounds the search.
///
/// # Panics
///
/// Panics when `graph` yields a node at or above its
/// [`node_bound`](crate::GraphView::node_bound).
///
/// # Examples
///
/// A branch whose true arm runs nine times out of ten. With a stream that
/// can fall through to either arm, the hot arm follows the branch.
///
/// ```
/// use cfglib::{Cfg, EdgeKind, ExtTspConfig, ext_tsp_order};
///
/// let mut cfg = Cfg::<&str>::new();
/// let entry = cfg.entry();
/// let [cold, hot, join] = [(); 3].map(|()| cfg.new_block());
/// let to_cold = cfg.add_edge(entry, cold, EdgeKind::ConditionalFalse);
/// let to_hot = cfg.add_edge(entry, hot, EdgeKind::ConditionalTrue);
/// cfg.add_edge(cold, join, EdgeKind::Fallthrough);
/// cfg.add_edge(hot, join, EdgeKind::Fallthrough);
///
/// let count = |edge| if edge == to_hot { 90 } else if edge == to_cold { 10 } else { 50 };
/// let order = ext_tsp_order(&cfg, |_| 16, count, |_| true, &ExtTspConfig::default());
/// assert_eq!(order[..2], [entry, hot]);
/// assert_eq!(order.len(), 4);
/// ```
#[must_use]
pub fn ext_tsp_order<G>(
    graph: &G,
    block_size: impl Fn(G::NodeId) -> u64,
    edge_count: impl Fn(G::EdgeId) -> u64,
    can_fall_through: impl Fn(G::EdgeId) -> bool,
    config: &ExtTspConfig,
) -> Vec<G::NodeId>
where
    G: RootedView + EdgeView,
{
    let nodes = reverse_postorder(graph, graph.root(), TraversalDirection::Outgoing);
    let problem = Problem::new(graph, &nodes, block_size, edge_count, can_fall_through);
    let greedy = chains::merge(&problem, config);
    let chosen =
        if problem.score(greedy.iter().copied(), config) > problem.score(0..nodes.len(), config) {
            greedy
        } else {
            (0..nodes.len()).collect()
        };
    chosen.into_iter().map(|local| nodes[local]).collect()
}

/// The score that [`ext_tsp_order`] maximizes, for tests and comparisons.
///
/// The blocks of `order` are laid out one after another, each `block_size`
/// bytes long. Every edge between two blocks of `order` adds its count times
/// a credit, in thousandths:
///
/// - 1000 when its target directly follows its source and
///   `can_fall_through` holds for it;
/// - otherwise, for a jump of `d` bytes from the end of the source to the
///   start of the target, `forward_weight · (forward_distance − d) /
///   forward_distance` when the target starts at or after that end and
///   `d <= forward_distance`, `backward_weight · (backward_distance − d) /
///   backward_distance` when it starts before it and
///   `d <= backward_distance`, and nothing for a longer jump.
///
/// An edge that cannot fall through therefore earns the credit of a forward
/// jump of distance zero when its target is next. Each term is rounded down,
/// and the sum saturates at `u128::MAX`. Edges into or out of a node that
/// `order` does not hold add nothing. `order` holds each node at most once;
/// a repeated node gives an unspecified score.
///
/// # Panics
///
/// Panics when `order` names a node at or above the graph's
/// [`node_bound`](crate::GraphView::node_bound).
#[must_use]
pub fn ext_tsp_score<G: EdgeView>(
    graph: &G,
    order: &[G::NodeId],
    block_size: impl Fn(G::NodeId) -> u64,
    edge_count: impl Fn(G::EdgeId) -> u64,
    can_fall_through: impl Fn(G::EdgeId) -> bool,
    config: &ExtTspConfig,
) -> u128 {
    Problem::new(graph, order, block_size, edge_count, can_fall_through)
        .score(0..order.len(), config)
}

/// [`ext_tsp_order`] over a CFG, with the fallthrough rule of
/// [`linearize`](crate::linearize) under `emitter`.
///
/// An edge can fall through when `linearize` emits no transfer for it while
/// its target is the next block: [`Fallthrough`](crate::EdgeKind::Fallthrough),
/// [`ConditionalFalse`](crate::EdgeKind::ConditionalFalse), and
/// [`CallReturn`](crate::EdgeKind::CallReturn) always;
/// [`ConditionalTrue`](crate::EdgeKind::ConditionalTrue) when the emitter
/// [inverts](Emitter::emit_inverted_conditional_branch) the condition of
/// its block toward the block's layout target; and the jump kinds when the
/// emitter [elides jumps to the next block](Emitter::elides_jump_to_next).
/// Dispatch, call, and exceptional edges never do. To learn whether a
/// condition inverts, the function asks the emitter for the inverted branch
/// of each counted true edge and drops the instruction. Pass the result to
/// `linearize` with the same emitter as a
/// [`BlockOrder::Custom`](crate::BlockOrder::Custom).
///
/// A consumer whose profile holds branch probabilities
/// ([`CfgView::edge_weight`]) converts them to counts first.
///
/// # Examples
///
/// The hot true arm follows the branch once the emitter can invert the
/// branch, and the branch block then emits one transfer instead of two.
///
/// ```
/// use cfglib::{
///     BlockId, BlockOrder, Cfg, EdgeKind, Emitter, ExtTspConfig, cfg_ext_tsp_order, linearize,
/// };
///
/// struct Asm;
///
/// impl Emitter<&'static str> for Asm {
///     fn emit_jump(&self, _target: BlockId) -> &'static str {
///         "jmp"
///     }
///     fn emit_conditional_branch(&self, _condition: &&'static str, _target: BlockId) -> &'static str {
///         "bnz"
///     }
///     fn emit_block_start(&self, _block: BlockId) -> Option<&'static str> {
///         None
///     }
///     fn emit_inverted_conditional_branch(
///         &self,
///         _condition: &&'static str,
///         _target: BlockId,
///     ) -> Option<&'static str> {
///         Some("bz")
///     }
/// }
///
/// let mut cfg = Cfg::new();
/// let entry = cfg.entry();
/// let [hot, cold] = [(); 2].map(|()| cfg.new_block());
/// cfg.block_mut(entry).push("test");
/// cfg.block_mut(hot).push("work");
/// cfg.block_mut(cold).push("report");
/// let to_hot = cfg.add_edge(entry, hot, EdgeKind::ConditionalTrue);
/// cfg.add_edge(entry, cold, EdgeKind::ConditionalFalse);
///
/// let count = |edge| if edge == to_hot { 90 } else { 10 };
/// let order = cfg_ext_tsp_order(&cfg, &Asm, |_| 16, count, &ExtTspConfig::default());
/// assert_eq!(order, [entry, hot, cold]);
///
/// let stream = linearize(&cfg, BlockOrder::Custom(order), &Asm)?;
/// let text: Vec<&str> = stream.iter().map(|line| line.inst).collect();
/// assert_eq!(text, ["test", "bz", "work", "report"]);
/// # Ok::<(), cfglib::LinearizeError>(())
/// ```
#[must_use]
pub fn cfg_ext_tsp_order<G: CfgView>(
    cfg: &G,
    emitter: &dyn Emitter<G::Inst>,
    block_size: impl Fn(BlockId) -> u64,
    edge_count: impl Fn(EdgeId) -> u64,
    config: &ExtTspConfig,
) -> Vec<BlockId> {
    ext_tsp_order(
        cfg,
        block_size,
        edge_count,
        |edge| falls_through(cfg, edge, emitter),
        config,
    )
}

/// One edge with a nonzero count, between local node numbers.
#[derive(Debug, Clone, Copy)]
struct Jump {
    source: usize,
    target: usize,
    count: u64,
    /// Whether the edge needs no transfer when its target is next.
    falls: bool,
}

/// A set of nodes in local numbers `0..n`, with their sizes and the jumps
/// between them.
struct Problem {
    sizes: Vec<u64>,
    jumps: Vec<Jump>,
}

impl Problem {
    /// Numbers `nodes` by position and collects the edges between them.
    fn new<G: EdgeView>(
        graph: &G,
        nodes: &[G::NodeId],
        block_size: impl Fn(G::NodeId) -> u64,
        edge_count: impl Fn(G::EdgeId) -> u64,
        can_fall_through: impl Fn(G::EdgeId) -> bool,
    ) -> Self {
        let mut local = vec![usize::MAX; graph.node_bound()];
        for (index, node) in nodes.iter().enumerate() {
            local[node.index()] = index;
        }
        let mut jumps = Vec::new();
        for (source, &node) in nodes.iter().enumerate() {
            for edge in graph.outgoing(node) {
                let target = local[graph.edge(edge).target().index()];
                if target == usize::MAX {
                    continue;
                }
                let count = edge_count(edge);
                if count > 0 {
                    jumps.push(Jump {
                        source,
                        target,
                        count,
                        falls: can_fall_through(edge),
                    });
                }
            }
        }
        Self {
            sizes: nodes.iter().map(|&node| block_size(node)).collect(),
            jumps,
        }
    }

    /// The score of laying the local nodes out in `order`, which holds each
    /// of them once.
    fn score(&self, order: impl IntoIterator<Item = usize>, config: &ExtTspConfig) -> u128 {
        let mut position = vec![0; self.sizes.len()];
        let mut address = vec![0_u64; self.sizes.len()];
        let mut next = 0_u64;
        for (index, node) in order.into_iter().enumerate() {
            position[node] = index;
            address[node] = next;
            next = next.saturating_add(self.sizes[node]);
        }
        self.jumps
            .iter()
            .map(|jump| {
                jump_score(
                    jump.count,
                    jump.falls && position[jump.target] == position[jump.source] + 1,
                    address[jump.source].saturating_add(self.sizes[jump.source]),
                    address[jump.target],
                    config,
                )
            })
            .fold(0, u128::saturating_add)
    }
}

/// The credit of one edge that runs `count` times, in thousandths: a
/// fallthrough when `falls`, else a jump from `source_end` to
/// `target_start`.
fn jump_score(
    count: u64,
    falls: bool,
    source_end: u64,
    target_start: u64,
    config: &ExtTspConfig,
) -> u128 {
    if falls {
        return u128::from(count) * u128::from(FALLTHROUGH_WEIGHT);
    }
    let (distance, limit, weight) = if target_start >= source_end {
        (
            target_start - source_end,
            config.forward_distance,
            config.forward_weight,
        )
    } else {
        (
            source_end - target_start,
            config.backward_distance,
            config.backward_weight,
        )
    };
    if distance > limit {
        return 0;
    }
    if limit == 0 {
        return u128::from(count) * u128::from(weight);
    }
    // `count · weight · (limit − distance) / limit`, rounded down. Profile
    // counts keep the product within 64 bits, where division is cheap.
    let remaining = limit - distance;
    if let Some(product) = count
        .checked_mul(u64::from(weight))
        .and_then(|full| full.checked_mul(remaining))
    {
        return u128::from(product / limit);
    }
    // `full` has at most 96 bits and both factors at most 64, so splitting
    // `full` by `limit` keeps every product below 2^128.
    let full = u128::from(count) * u128::from(weight);
    let (limit, remaining) = (u128::from(limit), u128::from(remaining));
    full / limit * remaining + full % limit * remaining / limit
}
