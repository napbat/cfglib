//! Block and edge execution frequencies from branch probabilities, by
//! loop-scaled propagation.
//!
//! [`BlockFrequencies::compute`] is the static profile of Y. Wu and
//! J. R. Larus, *Static branch frequency and program profile analysis*
//! (MICRO 1994). The entry runs once. A block runs as often as the edges into
//! it are taken, and an edge is taken as often as its source runs, times its
//! branch probability. A loop multiplies what enters its header by
//! `1 / (1 - cp)`, where the cyclic probability `cp` is the share of one
//! iteration that returns to the header along its back edges.
//!
//! # Algorithm
//!
//! The loops of the [`LoopForest`] are visited from the innermost out. In each
//! loop a unit mass starts at the header and flows through the body in reverse
//! postorder from the header, and the back edges collect `cp`. A finished inner
//! loop is one unit of the loop around it: the mass that enters it leaves along
//! its exit edges, multiplied by its scale. The scale is `1 / (1 - cp)`, capped
//! at [`FrequencyConfig::max_loop_scale`], so a loop that no edge leaves
//! (`cp = 1`) runs that many times instead of overflowing. Last, the whole graph
//! is propagated as a loop without back edges from a unit mass at the root, and
//! every loop gives its blocks the frequency of its header times their share of
//! one iteration.
//!
//! Every block and edge is propagated once in its innermost loop, and an exit
//! edge once more in each further loop it leaves, so the cost after the loop
//! forest is O(n + m) when every exit leaves one loop and O(n + m·d) when exits
//! leave up to `d` loops at once.
//!
//! # Probabilities
//!
//! The caller states the probability of each edge, or `None`. The edges of one
//! source without a probability share what the stated ones leave of `1`
//! evenly, and nothing when the stated ones already reach `1`. The
//! probabilities of one source are then divided by their sum, so they add up
//! to `1`. When they are all zero, every edge of the source gets the same
//! share. The view has no edge kind, so an exceptional edge that should be
//! rare gets its small probability from the caller.
//!
//! # Fixed point
//!
//! The computation uses integers only, so the result is the same on every
//! platform. Shares of one iteration have 62 fractional bits, products use
//! `u128` intermediates, and every product rounds to the nearest value, ties
//! upward. A reported frequency has [`FREQUENCY_SHIFT`] fractional bits and
//! saturates at `u64::MAX`.
//!
//! # Approximations
//!
//! For a reducible graph and the forest of [`LoopForest::compute`], the
//! result is exact up to rounding. Beyond that it is an approximation:
//!
//! - An edge that leads back to a block before its source in the reverse
//!   postorder of the loop being propagated, without being a back edge of that
//!   loop, carries its frequency but adds nothing to its target. Such an edge
//!   closes a cycle that the forest does not name, such as an irreducible
//!   cycle of [`LoopForest::compute`].
//! - A tagged irreducible loop of [`LoopForest::compute`] has several
//!   entries. The mass that enters it at any of them is propagated as mass
//!   entering at its header, with the same scale and the same cap.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::marker::PhantomData;
use core::ops::Range;

use super::edge_view::EdgeView;
use super::fanout::DenseFanout;
use super::loop_forest::{LoopForest, LoopId};
use super::view::{DenseId, RootedView};
use crate::block::BlockId;
use crate::edge::{EdgeId, Probability};
use crate::identity::{NO_INDEX, check_bound, slot, word};
use crate::union_find::DisjointSet;

/// The number of fractional bits of a frequency: one execution of the entry
/// is `1 << FREQUENCY_SHIFT`.
pub const FREQUENCY_SHIFT: u32 = 24;

/// The fractional bits of a share of one iteration.
const MASS_SHIFT: u32 = 62;

/// A whole iteration: the mass that starts at a header.
const ONE: u64 = 1 << MASS_SHIFT;

/// The limits of a [`BlockFrequencies::compute`] run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrequencyConfig {
    /// The cap on the scale of one loop, in iterations for each entry. A
    /// scale below `1` counts as `1`.
    pub max_loop_scale: u32,
}

impl Default for FrequencyConfig {
    /// A loop runs at most 4096 times for each entry.
    fn default() -> Self {
        Self {
            max_loop_scale: 4096,
        }
    }
}

/// Execution counts of the blocks and edges of a rooted graph.
///
/// Two analyses produce it, and its unit is the one of the constructor:
///
/// - [`compute`](Self::compute) estimates them statically from branch
///   probabilities, in fixed point with [`FREQUENCY_SHIFT`] fractional bits,
///   relative to one execution of the root. See the
///   [module documentation](self) for the algorithm, the treatment of
///   missing probabilities, and where the result is an approximation.
/// - [`infer`](Self::infer) infers whole counts that conserve flow from
///   measured estimates; see [`profile_inference`](super::profile_inference).
///
/// The counts hold no instruction data and do not borrow the graph.
///
/// # Examples
///
/// ```
/// use cfglib::{
///     BlockFrequencies, Cfg, DominatorTree, Edge, EdgeKind, FREQUENCY_SHIFT, FrequencyConfig,
///     LoopForest, Probability,
/// };
///
/// // entry -> header; header -> body (7/8) or exit (1/8); body -> header.
/// let probability = |value| Probability::from_f64(value).expect("a probability");
/// let mut cfg = Cfg::<u32>::new();
/// let header = cfg.new_block();
/// let body = cfg.new_block();
/// let exit = cfg.new_block();
/// cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
/// cfg.add_edge(
///     header,
///     body,
///     Edge::new(EdgeKind::ConditionalTrue).with_weight(probability(0.875)),
/// );
/// let leave = cfg.add_edge(
///     header,
///     exit,
///     Edge::new(EdgeKind::ConditionalFalse).with_weight(probability(0.125)),
/// );
/// cfg.add_edge(body, header, EdgeKind::Back);
///
/// let dominators = DominatorTree::compute(&cfg);
/// let loops = LoopForest::compute(&cfg, &dominators);
/// let frequencies = BlockFrequencies::compute(
///     &cfg,
///     &loops,
///     |edge| cfg.edge(edge).weight(),
///     FrequencyConfig::default(),
/// );
///
/// let once = 1 << FREQUENCY_SHIFT;
/// assert_eq!(frequencies.block(cfg.entry()), once);
/// assert_eq!(frequencies.block(header), 8 * once);
/// assert_eq!(frequencies.block(body), 7 * once);
/// assert_eq!(frequencies.edge(leave), once);
/// assert_eq!(frequencies.block(exit), once);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockFrequencies<N = BlockId, E = EdgeId> {
    /// The count of each node slot.
    block: Vec<u64>,
    /// The count of each edge slot.
    edge: Vec<u64>,
    ids: PhantomData<(N, E)>,
}

impl<N: DenseId, E: DenseId> BlockFrequencies<N, E> {
    /// Estimate how often each block and edge of `graph` runs for one
    /// execution of its root, with [`FREQUENCY_SHIFT`] fractional bits.
    ///
    /// `loops` must be the loop forest of `graph`. `probability` states the
    /// branch probability of an edge relative to the other edges that leave its
    /// source, or `None` when the caller has none; over a CFG it is
    /// `|edge| cfg.edge(edge).weight()`.
    ///
    /// # Panics
    ///
    /// Panics when the node bound of `graph` plus the number of loops exceeds
    /// `u32::MAX`.
    #[must_use]
    pub fn compute<G>(
        graph: &G,
        loops: &LoopForest<N>,
        probability: impl Fn(E) -> Option<Probability>,
        config: FrequencyConfig,
    ) -> Self
    where
        G: RootedView<NodeId = N> + EdgeView<EdgeId = E>,
    {
        let branch = branch_probabilities(graph, probability);
        let mut propagation = Propagation::new(graph, loops);
        let order = propagation.innermost_first(loops);
        for &x in &order {
            propagation.propagate_loop(
                graph,
                &branch,
                x,
                loops.header(x),
                config.max_loop_scale.max(1),
            );
        }
        propagation.propagate_level(graph, &branch, propagation.top, graph.root());
        let block = propagation.frequencies(&order);
        let mut edge = vec![0; graph.edge_bound()];
        for node in graph.node_ids() {
            let runs = block[node.index()];
            if runs == 0 {
                continue;
            }
            for id in graph.outgoing(node) {
                edge[id.index()] = saturate(mul_div(
                    u128::from(runs),
                    branch[id.index()],
                    u128::from(ONE),
                ));
            }
        }
        Self::from_tables(block, edge)
    }

    /// The counts of every node slot and every edge slot.
    pub(super) const fn from_tables(block: Vec<u64>, edge: Vec<u64>) -> Self {
        Self {
            block,
            edge,
            ids: PhantomData,
        }
    }

    /// How often `node` runs, in the unit of the constructor.
    ///
    /// From [`compute`](Self::compute), the root runs `1 << FREQUENCY_SHIFT`
    /// times, or that times the scale of the loop it heads, and a node that
    /// the root does not reach runs `0` times. A node at or above the node
    /// bound of the graph at compute time runs `0` times.
    #[must_use]
    pub fn block(&self, node: N) -> u64 {
        self.block.get(node.index()).copied().unwrap_or(0)
    }

    /// How often `edge` is taken, in the unit of the constructor.
    ///
    /// From [`compute`](Self::compute), it is the frequency of its source
    /// times its probability, and an edge from a node that the root does not
    /// reach is taken `0` times. An edge at or above the edge bound of the
    /// graph at compute time is taken `0` times.
    #[must_use]
    pub fn edge(&self, edge: E) -> u64 {
        self.edge.get(edge.index()).copied().unwrap_or(0)
    }
}

/// The normalized probability of every edge slot, as a share of [`ONE`].
fn branch_probabilities<G: EdgeView>(
    graph: &G,
    probability: impl Fn(G::EdgeId) -> Option<Probability>,
) -> Vec<u64> {
    let mut branch = vec![0; graph.edge_bound()];
    let mut stated = Vec::new();
    for node in graph.node_ids() {
        stated.clear();
        stated.extend(graph.outgoing(node).map(|edge| {
            (
                edge.index(),
                probability(edge).map(|p| u64::from(p.numerator())),
            )
        }));
        let given: u64 = stated.iter().filter_map(|&(_, p)| p).sum();
        let missing = count(stated.iter().filter(|(_, p)| p.is_none()).count());
        let remainder = u64::from(Probability::DENOMINATOR).saturating_sub(given);
        let missing_share = (remainder + missing / 2).checked_div(missing).unwrap_or(0);
        let total = given + missing_share * missing;
        let even = count(stated.len());
        for &(edge, p) in &stated {
            let (part, whole) = if total == 0 {
                (1, even)
            } else {
                (p.unwrap_or(missing_share), total)
            };
            branch[edge] = saturate(mul_div(u128::from(part), ONE, u128::from(whole)));
        }
    }
    branch
}

/// The scale of a finished loop.
#[derive(Clone, Copy)]
enum Scale {
    /// The capped scale: this many iterations for each entry.
    Times(u32),
    /// `1 / (1 - cp)`, stored as `1 - cp` over [`ONE`].
    Over(u64),
}

impl Scale {
    /// The scale of a loop whose iteration returns `cyclic` to its header.
    fn of(cyclic: u64, cap: u32) -> Self {
        let rest = ONE.saturating_sub(cyclic);
        if u128::from(rest) * u128::from(cap) <= u128::from(ONE) {
            Self::Times(cap)
        } else {
            Self::Over(rest)
        }
    }

    fn apply(self, value: u128) -> u128 {
        match self {
            Self::Times(cap) => value.saturating_mul(u128::from(cap)),
            Self::Over(rest) => mul_div(value, ONE, u128::from(rest)),
        }
    }
}

/// Where an arc of one level leads.
#[derive(Clone, Copy)]
enum To {
    /// A unit of the level, by its position in the level.
    Unit(usize),
    /// The header of the loop being propagated.
    Back,
    /// Out of the loop being propagated.
    Out,
}

/// One edge out of a unit of the level being propagated.
#[derive(Clone, Copy)]
struct Arc {
    edge: usize,
    target: usize,
    to: To,
    /// The share of the unit's mass that takes the edge.
    factor: u64,
}

/// An edge that leaves a finished loop, with the mass that takes it for one
/// entry into the loop.
#[derive(Clone, Copy)]
struct Exit {
    edge: usize,
    target: usize,
    weight: u64,
}

/// The loop nest as levels of units, and the state of the propagation.
///
/// An element is a node slot (below `bound`) or a loop (`bound` plus its
/// index). Every element is a unit of one level: a node of the innermost loop
/// that holds it, and a loop of its parent. Level `top`, one past the last
/// loop, is the whole graph.
struct Propagation {
    bound: usize,
    top: usize,
    loops: Vec<LoopId>,
    /// The level each element is a unit of.
    level: Vec<usize>,
    /// The units of each level, grouped by level.
    units: Vec<usize>,
    /// Where the units of each level start in `units`, with one more entry
    /// for the end.
    starts: Vec<u32>,
    /// The finished loops: each set is a finished loop with everything in it,
    /// or a node in no finished loop.
    sets: DisjointSet,
    /// The element that names each set, by its root.
    label: Vec<usize>,
    /// The share of one iteration of its level that reaches each element.
    mass: Vec<u64>,
    scale: Vec<Scale>,
    exits: Vec<Exit>,
    /// Where the exits of each loop are in `exits`.
    exit_range: Vec<(usize, usize)>,
    scratch: Scratch,
}

/// Buffers that each level reuses.
#[derive(Default)]
struct Scratch {
    /// The position of each element in the units of its level.
    local: Vec<usize>,
    arcs: Vec<Arc>,
    arc_starts: Vec<usize>,
    /// The units in reverse postorder from the start.
    order: Vec<usize>,
    /// The position of each unit in `order`, [`NO_INDEX`] when unreached.
    position: Vec<u32>,
    stack: Vec<(usize, usize)>,
    mass: Vec<u64>,
}

impl Propagation {
    fn new<G, N>(graph: &G, forest: &LoopForest<N>) -> Self
    where
        G: RootedView<NodeId = N>,
        N: DenseId,
    {
        let bound = graph.node_bound();
        let loops: Vec<LoopId> = forest.loops().collect();
        let top = loops.len();
        let elements = bound + top;
        check_bound(elements, "node and loop count of block frequencies");
        let mut level = vec![top; elements];
        let live: Vec<usize> = graph.node_ids().map(DenseId::index).collect();
        for &slot in &live {
            if let Some(lp) = forest.innermost(N::from_index(slot)) {
                level[slot] = lp.index();
            }
        }
        // A loop that is not properly nested keeps its own set of blocks, so a
        // block takes it when it is deeper than the loop the block has.
        for &lp in loops.iter().filter(|&&lp| !forest.is_nested(lp)) {
            for slot in forest.body(lp).map(DenseId::index) {
                if slot < bound
                    && (level[slot] == top || forest.depth(lp) > forest.depth(loops[level[slot]]))
                {
                    level[slot] = lp.index();
                }
            }
        }
        for (x, &lp) in loops.iter().enumerate() {
            level[bound + x] = forest.parent(lp).map_or(top, LoopId::index);
        }
        let members = || live.iter().copied().chain(bound..elements);
        let mut starts = Vec::new();
        let mut units = Vec::new();
        DenseFanout::group_into(
            top + 1,
            || members().map(|element| (level[element], element)),
            &mut starts,
            &mut units,
        );
        Self {
            bound,
            top,
            loops,
            level,
            units,
            starts,
            sets: DisjointSet::new(elements),
            label: (0..elements).collect(),
            mass: vec![0; elements],
            scale: vec![Scale::Times(1); top],
            exits: Vec::new(),
            exit_range: vec![(0, 0); top],
            scratch: Scratch {
                local: vec![0; elements],
                ..Scratch::default()
            },
        }
    }

    /// The loops in the order the propagation finishes them: deepest first,
    /// and by index within a depth.
    fn innermost_first<N: DenseId>(&self, forest: &LoopForest<N>) -> Vec<LoopId> {
        let mut order = self.loops.clone();
        order.sort_by_key(|&lp| Reverse(forest.depth(lp)));
        order
    }

    /// Where the units of `level` lie in `units`.
    fn level_units(&self, level: usize) -> Range<usize> {
        slot(self.starts[level])..slot(self.starts[level + 1])
    }

    /// The unit of `level` that holds node slot `node`, if any.
    fn unit_of(&mut self, level: usize, node: usize) -> Option<usize> {
        let unit = self.label[self.sets.find(node)];
        (self.level[unit] == level).then_some(unit)
    }

    /// Propagate one iteration of loop `x`, then make it a unit of its parent.
    fn propagate_loop<G>(
        &mut self,
        graph: &G,
        branch: &[u64],
        x: LoopId,
        header: G::NodeId,
        max_scale: u32,
    ) where
        G: EdgeView,
    {
        let level = x.index();
        let first = self.exits.len();
        let cyclic = self.propagate_level(graph, branch, level, header);
        let scale = Scale::of(cyclic, max_scale);
        for exit in &mut self.exits[first..] {
            exit.weight = saturate(scale.apply(u128::from(exit.weight)));
        }
        self.scale[level] = scale;
        self.exit_range[level] = (first, self.exits.len());
        let element = self.bound + level;
        for slot in self.level_units(level) {
            self.sets.union(self.units[slot], element);
        }
        let root = self.sets.find(element);
        self.label[root] = element;
    }

    /// Propagate a unit mass from `start` through the units of `level`, and
    /// return the mass that returns to `start` along back edges. The exits of a
    /// loop level are recorded with their unscaled mass.
    fn propagate_level<G>(
        &mut self,
        graph: &G,
        branch: &[u64],
        level: usize,
        start: G::NodeId,
    ) -> u64
    where
        G: EdgeView,
    {
        let header = start.index();
        let Some(start) = self.unit_of(level, header) else {
            return 0;
        };
        let Range { start: begin, end } = self.level_units(level);
        for slot in begin..end {
            self.scratch.local[self.units[slot]] = slot - begin;
        }
        self.collect_arcs(graph, branch, level, header);
        self.order_from(self.scratch.local[start]);
        let scratch = &mut self.scratch;
        scratch.mass.clear();
        scratch.mass.resize(end - begin, 0);
        scratch.mass[scratch.local[start]] = ONE;
        let mut cyclic = 0_u64;
        for (position, &unit) in scratch.order.iter().enumerate() {
            let mass = scratch.mass[unit];
            if mass == 0 {
                continue;
            }
            for arc in &scratch.arcs[scratch.arc_starts[unit]..scratch.arc_starts[unit + 1]] {
                let share = saturate(mul_div(u128::from(mass), arc.factor, u128::from(ONE)));
                match arc.to {
                    To::Back => cyclic = cyclic.saturating_add(share),
                    To::Out => self.exits.push(Exit {
                        edge: arc.edge,
                        target: arc.target,
                        weight: share,
                    }),
                    To::Unit(next) => {
                        if slot(scratch.position[next]) > position {
                            scratch.mass[next] = scratch.mass[next].saturating_add(share);
                        }
                    }
                }
            }
        }
        for slot in begin..end {
            self.mass[self.units[slot]] = scratch.mass[slot - begin];
        }
        cyclic
    }

    /// The arcs out of every unit of `level`, grouped by unit.
    fn collect_arcs<G: EdgeView>(
        &mut self,
        graph: &G,
        branch: &[u64],
        level: usize,
        header: usize,
    ) {
        let mut arcs = core::mem::take(&mut self.scratch.arcs);
        let mut arc_starts = core::mem::take(&mut self.scratch.arc_starts);
        arcs.clear();
        arc_starts.clear();
        for slot in self.level_units(level) {
            arc_starts.push(arcs.len());
            let unit = self.units[slot];
            if unit < self.bound {
                for id in graph.outgoing(G::NodeId::from_index(unit)) {
                    let target = graph.edge(id).target().index();
                    let edge = id.index();
                    arcs.push(Arc {
                        edge,
                        target,
                        to: self.destination(level, header, target),
                        factor: branch[edge],
                    });
                }
            } else {
                let (first, last) = self.exit_range[unit - self.bound];
                for index in first..last {
                    let exit = self.exits[index];
                    arcs.push(Arc {
                        edge: exit.edge,
                        target: exit.target,
                        to: self.destination(level, header, exit.target),
                        factor: exit.weight,
                    });
                }
            }
        }
        arc_starts.push(arcs.len());
        self.scratch.arcs = arcs;
        self.scratch.arc_starts = arc_starts;
    }

    fn destination(&mut self, level: usize, header: usize, target: usize) -> To {
        if level != self.top && target == header {
            return To::Back;
        }
        match self.unit_of(level, target) {
            Some(unit) => To::Unit(self.scratch.local[unit]),
            None => To::Out,
        }
    }

    /// Order the units that `start` reaches through the arcs in reverse
    /// postorder.
    ///
    /// This loop stays outside the shared depth-first core of
    /// [`graph::traverse`](super::traverse): a level holds a handful of
    /// units, and the core's per-walk setup over a view of the level's arcs
    /// made `api_block_frequencies_1k` and `_10k` about 1.5 times slower.
    fn order_from(&mut self, start: usize) {
        let scratch = &mut self.scratch;
        let units = scratch.arc_starts.len() - 1;
        scratch.order.clear();
        scratch.position.clear();
        scratch.position.resize(units, NO_INDEX);
        // `position` marks a unit as seen while the walk runs, and holds its
        // place in the order once the walk is done.
        scratch.position[start] = 0;
        scratch.stack.push((start, scratch.arc_starts[start]));
        while let Some((unit, next)) = scratch.stack.last_mut() {
            if *next == scratch.arc_starts[*unit + 1] {
                scratch.order.push(*unit);
                scratch.stack.pop();
                continue;
            }
            let arc = scratch.arcs[*next];
            *next += 1;
            if let To::Unit(successor) = arc.to
                && scratch.position[successor] == NO_INDEX
            {
                scratch.position[successor] = 0;
                scratch
                    .stack
                    .push((successor, scratch.arc_starts[successor]));
            }
        }
        scratch.order.reverse();
        for (position, &unit) in scratch.order.iter().enumerate() {
            scratch.position[unit] = word(position);
        }
    }

    /// The frequency of every node slot, from the levels outermost first.
    fn frequencies(&self, innermost_first: &[LoopId]) -> Vec<u64> {
        let mut block = vec![0; self.bound];
        let mut base = vec![0_u128; self.top + 1];
        base[self.top] = u128::from(ONE);
        let levels = core::iter::once(self.top)
            .chain(innermost_first.iter().rev().copied().map(LoopId::index));
        for level in levels {
            let runs = base[level];
            if runs == 0 {
                continue;
            }
            for &unit in &self.units[self.level_units(level)] {
                let entering = mul_div(runs, self.mass[unit], u128::from(ONE));
                if unit < self.bound {
                    block[unit] = saturate(round_shift(entering, MASS_SHIFT - FREQUENCY_SHIFT));
                } else {
                    let x = unit - self.bound;
                    base[x] = self.scale[x].apply(entering);
                }
            }
        }
        block
    }
}

/// `value * factor / divisor`, rounded to nearest with ties upward, and
/// saturating at `u128::MAX`. `divisor` must be nonzero and at most `2^64`.
fn mul_div(value: u128, factor: u64, divisor: u128) -> u128 {
    let factor = u128::from(factor);
    let quotient = value / divisor;
    // The remainder is below the divisor, so its product with a 64-bit factor
    // fits, and so does half the divisor added to it.
    let remainder = value % divisor;
    quotient
        .saturating_mul(factor)
        .saturating_add((remainder * factor + divisor / 2) / divisor)
}

/// `value / 2^shift`, rounded to nearest with ties upward.
fn round_shift(value: u128, shift: u32) -> u128 {
    (value >> shift) + ((value >> (shift - 1)) & 1)
}

fn saturate(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn count(items: usize) -> u64 {
    u64::try_from(items).expect("a count of in-memory items fits u64")
}

#[cfg(test)]
mod tests;
