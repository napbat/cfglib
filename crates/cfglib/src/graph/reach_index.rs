//! A pairwise reachability index for repeated `reaches(source, target)`
//! questions about one static graph.
//!
//! [`reachable`](crate::reachable) answers "what does this seed set reach"
//! with one walk into a dense `Vec<bool>`, so asking "does `s` reach `t`" that
//! way costs a whole walk plus an O(node bound) fill per pair. A consumer that
//! asks many single pairs of one graph — "may function A transitively call B"
//! across a whole-program call graph, "does block X reach block Y" for a table
//! of block pairs — pays that per pair. [`ReachIndex`] is paid for once and
//! then answers each pair exactly, most of them in constant time.
//!
//! # How a pair is answered
//!
//! Everything works on the condensation, numbered by
//! [`SccNumbering::SourcesFirst`] (so a component only reaches
//! higher-numbered ones):
//!
//! - Two nodes of one component reach each other, and a source numbered
//!   after its target reaches nothing of it.
//! - Otherwise a set of constant-time *observations* is consulted, after
//!   O'Reach (Hanauer, Schulz and Trummer, *O'Reach: Even Faster Reachability
//!   in Large Graphs*, ACM JEA 27, 2022): weakly connected component ids,
//!   forward and backward topological levels, extended depth-first
//!   topological orders whose subtree intervals prove reachability and whose
//!   reach bounds refute it, and supportive components with explicit reach
//!   bits in both directions.
//! - A pair no observation decides falls back to a bidirectional search over
//!   the condensation, which applies the same observations to every component
//!   it meets: a component proven to reach the target ends the search, and one
//!   proven not to is not expanded.
//!
//! Every answer is exact. The observations only decide how much searching an
//! answer takes.
//!
//! # Exact chain mode
//!
//! [`ReachIndexConfig::exact_max_chains`] opts into a chain-cover index
//! (Jagadish's compression, as Kritikakis and Tollis parameterize it in *Fast
//! Reachability Using DAG Decomposition*, SEA 2023): the condensation is split
//! greedily into chains whose members reach one another in order, and every
//! component stores, per chain, the earliest member it reaches. A pair is then
//! one table read with no search at all. The table costs one word per
//! component per chain, which suits narrow condensations — a single
//! function's blocks — and not wide ones such as a whole-program call graph,
//! whose width approaches the number of leaf functions. The decomposition is
//! abandoned as soon as it needs more chains than the bound, and the index
//! falls back to the observations above.
//!
//! # Scope
//!
//! The index pays off only for repeated single-pair questions. The closure of
//! a seed set is one walk either way, so [`reachable`](crate::reachable)
//! stays the tool for it.
//!
//! The index is a snapshot: it describes the graph it was computed from, and
//! any edit to that graph — an added or removed node or edge — leaves it
//! stale. Compute it again after an edit.

extern crate alloc;
use alloc::vec::Vec;

use crate::graph::epoch::EpochMarks;
use crate::graph::fanout::DenseFanout;
use crate::graph::scc::{SccDecomposition, SccNumbering};
use crate::graph::view::{DenseId, GraphView};
use crate::identity::{slot, word};
use chains::ChainTable;
use observe::Observations;

mod chains;
mod observe;

/// The most supportive components an index keeps: one bit each in a `u64`
/// row per component and direction.
const MAX_SUPPORTIVE: usize = 64;

/// What a [`ReachIndex`] computes besides the components.
///
/// A plain struct with public fields: set the fields to change it.
/// [`Default`] is the same as [`new`](ReachIndexConfig::new), which is also
/// usable in a `const`.
///
/// # Examples
///
/// ```
/// use cfglib::ReachIndexConfig;
///
/// let config = ReachIndexConfig {
///     orders: 4,
///     exact_max_chains: Some(16),
///     ..ReachIndexConfig::default()
/// };
/// assert_eq!(config.supportive, 32);
/// assert_eq!(ReachIndexConfig::default(), ReachIndexConfig::new());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReachIndexConfig {
    /// How many extended depth-first topological orders to keep, each 16
    /// bytes per component.
    ///
    /// The orders alternate between a walk along the edges and one against
    /// them, each reading adjacency in stored order; the next two read it in
    /// reverse, and every further four start each adjacency list one place
    /// later. There is no randomness: the same graph gives the same orders.
    pub orders: usize,
    /// How many supportive components to keep, at most 64.
    ///
    /// Each one is a component whose reach is recorded explicitly as one bit
    /// per component and direction. The components are chosen by a fixed
    /// score favoring balanced reach into and out of the component: the
    /// larger product of its in- and out-degree in the condensation, then the
    /// larger of the smaller of its forward and backward topological levels,
    /// then the smaller component index.
    pub supportive: usize,
    /// Switch to the exact chain mode when the greedy chain decomposition of
    /// the condensation needs at most this many chains. `None` never tries.
    ///
    /// When the decomposition fits, the orders and supportive components are
    /// not computed: every query is answered by the chain table alone.
    pub exact_max_chains: Option<usize>,
}

impl ReachIndexConfig {
    /// Two orders, 32 supportive components, and no exact chain mode.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            orders: 2,
            supportive: 32,
            exact_max_chains: None,
        }
    }
}

impl Default for ReachIndexConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Answers "does `source` reach `target`" exactly, for many pairs of one
/// static graph.
///
/// [`compute`](Self::compute) runs the sources-first [`SccDecomposition`]
/// and builds the index over the condensation in O((t + 1)·(n + m)) for `t`
/// orders: the supportive components cost one pass of 64-bit unions per
/// direction however many are kept. An exact table of `c` chains costs
/// O(c·(n + m)) instead.
/// [`reaches`](Self::reaches) answers one pair, and
/// [`reaches_in`](Self::reaches_in) one of many over reused buffers. See the
/// [module documentation](self) for what is stored and how a pair is decided.
///
/// Besides the decomposition, the observations keep both adjacency lists of
/// the condensation and 12 bytes per component, 16 more per order, and 16
/// more when there are supportive components. An exact table keeps 16 bytes
/// per component plus one 4-byte entry for each chain that existed when the
/// component was placed, at most `c`.
///
/// The index holds no randomness: two computations over the same graph and
/// configuration compare equal and answer every pair the same way. It is a
/// snapshot, invalid once the graph is edited.
///
/// # Examples
///
/// ```
/// use cfglib::{Graph, ReachIndex, ReachIndexConfig, ReachScratch};
///
/// // main -> parse -> lex, parse <-> eval (mutual recursion), log alone.
/// let mut calls = Graph::<&str, ()>::new();
/// let main = calls.add_node("main");
/// let parse = calls.add_node("parse");
/// let lex = calls.add_node("lex");
/// let eval = calls.add_node("eval");
/// let log = calls.add_node("log");
/// calls.add_edge(main, parse, ());
/// calls.add_edge(parse, lex, ());
/// calls.add_edge(parse, eval, ());
/// calls.add_edge(eval, parse, ());
///
/// let index = ReachIndex::compute(&calls, ReachIndexConfig::default());
/// let mut scratch = ReachScratch::default();
/// assert!(index.reaches_in(&mut scratch, main, lex));
/// assert!(index.reaches_in(&mut scratch, eval, lex));
/// assert!(!index.reaches_in(&mut scratch, lex, main));
/// assert!(!index.reaches_in(&mut scratch, main, log));
/// assert!(index.reaches(log, log));
///
/// // The same pairs from a chain table, without any search.
/// let exact_mode = ReachIndexConfig { exact_max_chains: Some(8), ..ReachIndexConfig::default() };
/// let exact = ReachIndex::compute(&calls, exact_mode);
/// assert!(exact.chain_count().is_some());
/// assert!(exact.reaches(main, lex));
/// assert!(!exact.reaches(main, log));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReachIndex<N> {
    /// The strongly connected components, numbered sources first.
    components: SccDecomposition<N>,
    /// How a pair of distinct components, the source numbered first, is
    /// decided.
    answers: Answers,
}

/// The two ways an index decides a pair of distinct components.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Answers {
    /// A chain table that answers every pair with one read.
    Exact(ChainTable),
    /// Constant-time observations with a pruned search behind them.
    Observed(Observations),
}

impl<N: DenseId> ReachIndex<N> {
    /// Compute the index of `graph` under `config`.
    ///
    /// # Panics
    ///
    /// Panics when `config.supportive` is above 64, when the node bound
    /// exceeds the range of a dense `u32` table, and when the condensation
    /// has more than `u32::MAX` edges.
    #[must_use]
    pub fn compute<G: GraphView<NodeId = N>>(graph: &G, config: ReachIndexConfig) -> Self {
        assert!(
            config.supportive <= MAX_SUPPORTIVE,
            "a reach index keeps at most {MAX_SUPPORTIVE} supportive components, not {}",
            config.supportive
        );
        let components = SccDecomposition::compute(graph, SccNumbering::SourcesFirst);
        let count = components.len();
        // The condensation's edges, straight into the two fan-outs rather
        // than through a condensed graph. Every component index fits a `u32`
        // because the node bound does, with `u32::MAX` itself left free as a
        // sentinel.
        let mut edges: Vec<(usize, u32)> = Vec::new();
        components.for_each_component_edge(graph, |from, to| edges.push((from, word(to))));
        let successors = DenseFanout::from_pairs(count, edges.iter().copied());
        let chains = config
            .exact_max_chains
            .and_then(|max_chains| ChainTable::compute(&successors, max_chains));
        let answers = if let Some(chains) = chains {
            Answers::Exact(chains)
        } else {
            let predecessors = DenseFanout::from_pairs(
                count,
                edges.iter().map(|&(from, to)| (slot(to), word(from))),
            );
            Answers::Observed(Observations::compute(successors, predecessors, config))
        };
        Self {
            components,
            answers,
        }
    }

    /// Whether `source` reaches `target` along zero or more edges.
    ///
    /// A node always reaches itself, with or without a self-loop. The answer
    /// is exact. It costs O(1) when the components or an observation decide
    /// it, and always in the exact chain mode; otherwise it is a pruned
    /// bidirectional search over the condensation in buffers of its own.
    /// [`reaches_in`](Self::reaches_in) keeps those buffers across queries.
    ///
    /// # Panics
    ///
    /// Panics when `source` or `target` is not a node of the graph the index
    /// was computed from: a slot that holds no node, such as one the graph
    /// removed, or one at or beyond its node bound. This is the contract of
    /// [`SccDecomposition::component_index`].
    #[must_use]
    pub fn reaches(&self, source: N, target: N) -> bool {
        self.reaches_in(&mut ReachScratch::default(), source, target)
    }

    /// [`reaches`](Self::reaches) with the search in `scratch`, which grows
    /// to this index's component count and is reused by every later query.
    ///
    /// # Panics
    ///
    /// Panics under the conditions of [`reaches`](Self::reaches).
    #[must_use]
    pub fn reaches_in(&self, scratch: &mut ReachScratch, source: N, target: N) -> bool {
        let source = self.components.component_index(source);
        let target = self.components.component_index(target);
        #[cfg(test)]
        {
            scratch.coverage.queries += 1;
        }
        if source == target {
            return true;
        }
        // Sources first: every edge leads to a component numbered no lower.
        if source > target {
            return false;
        }
        match &self.answers {
            Answers::Exact(chains) => chains.reaches(source, target),
            Answers::Observed(observations) => observations.reaches(source, target, scratch),
        }
    }

    /// The number of chains of the exact chain mode, or `None` when the index
    /// answers through observations and search instead: the mode was not
    /// requested, or the decomposition needed more chains than allowed.
    #[must_use]
    pub fn chain_count(&self) -> Option<usize> {
        match &self.answers {
            Answers::Exact(chains) => Some(chains.chain_count()),
            Answers::Observed(_) => None,
        }
    }
}

/// The working buffers of [`ReachIndex::reaches_in`]: epoch marks and the two
/// frontiers of the fallback search.
///
/// It starts empty ([`Default`]) and nothing is released by a query. The
/// marks grow on entry to the component count of the index a query runs on
/// and are cleared in O(1) per search, so one scratch serves every query of a
/// pass, across indexes of any size, and allocates only while it grows.
///
/// # Examples
///
/// ```
/// use cfglib::{Graph, ReachIndex, ReachIndexConfig, ReachScratch};
///
/// let mut graph = Graph::<(), ()>::new();
/// let nodes: Vec<_> = (0..4).map(|_| graph.add_node(())).collect();
/// graph.add_edge(nodes[0], nodes[1], ());
/// graph.add_edge(nodes[1], nodes[2], ());
///
/// let index = ReachIndex::compute(&graph, ReachIndexConfig::default());
/// let mut scratch = ReachScratch::default();
/// let reached = nodes.iter().filter(|&&target| index.reaches_in(&mut scratch, nodes[0], target));
/// assert_eq!(reached.count(), 3);
/// ```
#[derive(Debug, Clone, Default)]
pub struct ReachScratch {
    /// Two marks per component: `2c` once the forward search met component
    /// `c`, `2c + 1` once the backward search did.
    marks: EpochMarks,
    /// The forward search's queue of components, read from a head cursor.
    forward: Vec<u32>,
    /// The backward search's queue of components, read from a head cursor.
    backward: Vec<u32>,
    /// How many queries ran and how many of them needed the search.
    #[cfg(test)]
    coverage: Coverage,
}

/// The share of queries an index answered without its fallback search.
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default)]
struct Coverage {
    /// Every query, however it was answered.
    queries: usize,
    /// The queries that ran the fallback search.
    searched: usize,
}

#[cfg(test)]
mod tests;
