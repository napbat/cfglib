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
//! - A tagged irreducible loop of [`LoopForest::compute_tagged`] has several
//!   entries. The mass that enters it at any of them is propagated as mass
//!   entering at its header, with the same scale and the same cap.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::marker::PhantomData;

use super::edge_view::EdgeView;
use super::loop_forest::{LoopForest, LoopId};
use super::view::{DenseId, RootedView};
use crate::block::BlockId;
use crate::edge::EdgeId;
use crate::identity::{NO_INDEX, check_bound, slot, word};
use crate::union_find::DisjointSet;

/// The number of fractional bits of a frequency: one execution of the entry
/// is `1 << FREQUENCY_SHIFT`.
pub const FREQUENCY_SHIFT: u32 = 24;

/// The fractional bits of a share of one iteration.
const MASS_SHIFT: u32 = 62;

/// A whole iteration: the mass that starts at a header.
const ONE: u64 = 1 << MASS_SHIFT;

/// A branch probability in fixed point: `numerator / 2^31`.
///
/// # Examples
///
/// ```
/// use cfglib::Probability;
///
/// let half = Probability::from_f64(0.5).expect("0.5 is a probability");
/// assert_eq!(half.numerator(), Probability::DENOMINATOR / 2);
/// assert_eq!(Probability::from_f64(1.5), None);
/// assert_eq!(Probability::from_f64(f64::NAN), None);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Probability(u32);

impl Probability {
    /// The denominator of every probability, `2^31`.
    pub const DENOMINATOR: u32 = 1 << 31;

    /// The probability of an edge that is never taken.
    pub const ZERO: Self = Self(0);

    /// The probability of an edge that is always taken.
    pub const ONE: Self = Self(Self::DENOMINATOR);

    /// The probability `numerator / 2^31`, or `None` when `numerator` is
    /// above [`DENOMINATOR`](Self::DENOMINATOR).
    #[must_use]
    pub const fn from_numerator(numerator: u32) -> Option<Self> {
        if numerator <= Self::DENOMINATOR {
            Some(Self(numerator))
        } else {
            None
        }
    }

    /// Round `probability` to the nearest fixed-point probability, ties
    /// upward, or `None` when it is NaN or outside `0.0..=1.0`.
    #[must_use]
    pub fn from_f64(probability: f64) -> Option<Self> {
        if !(0.0..=1.0).contains(&probability) {
            return None;
        }
        // Scaling by a power of two is exact, and so is the fraction of a
        // value below 2^52, so the rounding is exact too.
        let scaled = probability * f64::from(Self::DENOMINATOR);
        let floor = truncate(scaled);
        let round_up = scaled - f64::from(floor) >= 0.5;
        Some(Self(floor + u32::from(round_up)))
    }

    /// The numerator over [`DENOMINATOR`](Self::DENOMINATOR).
    #[must_use]
    pub const fn numerator(self) -> u32 {
        self.0
    }
}

/// The integer part of a value in `0.0..=2^31`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is a scaled probability in 0.0..=2^31, whose integer part fits u32"
)]
fn truncate(scaled: f64) -> u32 {
    scaled as u32
}

/// The limits of a [`BlockFrequencies`] computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrequencyConfig {
    max_loop_scale: u32,
}

impl FrequencyConfig {
    /// The default cap on the scale of one loop: 4096 iterations for each
    /// entry.
    pub const DEFAULT_MAX_LOOP_SCALE: u32 = 4096;

    /// The default limits.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_loop_scale: Self::DEFAULT_MAX_LOOP_SCALE,
        }
    }

    /// Cap the scale of one loop at `scale` iterations for each entry. A scale
    /// below `1` counts as `1`.
    #[must_use]
    pub const fn with_max_loop_scale(self, scale: u32) -> Self {
        Self {
            max_loop_scale: if scale == 0 { 1 } else { scale },
        }
    }

    /// The cap on the scale of one loop.
    #[must_use]
    pub const fn max_loop_scale(self) -> u32 {
        self.max_loop_scale
    }
}

impl Default for FrequencyConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Fixed-point execution frequencies of the blocks and edges of a rooted
/// graph, relative to one execution of its entry.
///
/// See the [module documentation](self) for the algorithm, the treatment of
/// missing probabilities, and where the result is an approximation. The
/// frequencies hold no instruction data and do not borrow the graph.
///
/// # Examples
///
/// ```
/// use cfglib::{
///     BlockFrequencies, Cfg, DominatorTree, EdgeKind, FREQUENCY_SHIFT, FrequencyConfig,
///     LoopForest, Probability,
/// };
///
/// // entry -> header; header -> body (7/8) or exit (1/8); body -> header.
/// let mut cfg = Cfg::<u32>::new();
/// let header = cfg.new_block();
/// let body = cfg.new_block();
/// let exit = cfg.new_block();
/// cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
/// cfg.add_weighted_edge(header, body, EdgeKind::ConditionalTrue, 0.875);
/// let leave = cfg.add_weighted_edge(header, exit, EdgeKind::ConditionalFalse, 0.125);
/// cfg.add_edge(body, header, EdgeKind::Back);
///
/// let dominators = DominatorTree::compute(&cfg);
/// let loops = LoopForest::compute(&cfg, &dominators);
/// let frequencies = BlockFrequencies::compute(
///     &cfg,
///     &loops,
///     |edge| cfg.edge(edge).weight().and_then(Probability::from_f64),
///     FrequencyConfig::new(),
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
    /// The frequency of each node slot.
    block: Vec<u64>,
    /// The frequency of each edge slot.
    edge: Vec<u64>,
    ids: PhantomData<(N, E)>,
}

impl<N: DenseId, E: DenseId> BlockFrequencies<N, E> {
    /// Estimate how often each block and edge of `graph` runs for one
    /// execution of its root.
    ///
    /// `loops` must be the loop forest of `graph`. `probability` states the
    /// branch probability of an edge relative to the other edges that leave its
    /// source, or `None` when the caller has none.
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
            propagation.propagate_loop(graph, &branch, x, loops.header(x), config.max_loop_scale);
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
        Self {
            block,
            edge,
            ids: PhantomData,
        }
    }

    /// How often `node` runs, with [`FREQUENCY_SHIFT`] fractional bits.
    ///
    /// The root runs `1 << FREQUENCY_SHIFT` times, or that times the scale of
    /// the loop it heads. A node that the root does not reach runs `0` times,
    /// and so does a node at or above the node bound of the graph at compute
    /// time.
    #[must_use]
    pub fn block(&self, node: N) -> u64 {
        self.block.get(node.index()).copied().unwrap_or(0)
    }

    /// How often `edge` is taken, with [`FREQUENCY_SHIFT`] fractional bits: the
    /// frequency of its source times its probability.
    ///
    /// An edge from a node that the root does not reach is taken `0` times,
    /// and so is an edge at or above the edge bound of the graph at compute
    /// time.
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
        stated.extend(
            graph
                .outgoing(node)
                .map(|edge| (edge.index(), probability(edge).map(|p| u64::from(p.0)))),
        );
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
    starts: Vec<usize>,
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
        let mut starts = vec![0; top + 2];
        for element in members() {
            starts[level[element] + 1] += 1;
        }
        for slot in 1..starts.len() {
            starts[slot] += starts[slot - 1];
        }
        let mut fill = starts.clone();
        let mut units = vec![0; starts[top + 1]];
        for element in members() {
            units[fill[level[element]]] = element;
            fill[level[element]] += 1;
        }
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
        for slot in self.starts[level]..self.starts[level + 1] {
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
        let (begin, end) = (self.starts[level], self.starts[level + 1]);
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
        for slot in self.starts[level]..self.starts[level + 1] {
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
            for &unit in &self.units[self.starts[level]..self.starts[level + 1]] {
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
