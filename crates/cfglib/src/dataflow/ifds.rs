//! Interprocedural, context-sensitive dataflow over a consumer supergraph:
//! the IFDS tabulation of Reps, Horwitz and Sagiv.
//!
//! An IFDS problem (*interprocedural, finite, distributive, subset*) asks
//! which facts of a finite domain hold at each node along *realizable*
//! paths only: paths on which every return goes back to the call that
//! entered its procedure. T. Reps, S. Horwitz and M. Sagiv, *Precise
//! interprocedural dataflow analysis via graph reachability* (POPL 1995),
//! turn it into reachability over the *exploded supergraph*, whose nodes are
//! (node, fact) pairs, and solve it by tabulation: path edges from a
//! procedure's entry fact to the facts at each of its nodes, the facts that
//! reach each exit (end summaries), and the call sites that entered each
//! procedure with each fact. The worst case is O(E·D³) time and O(E·D²)
//! space for E supergraph edges and D facts; sparse flow functions, as in
//! taint analysis, cost far less.
//!
//! [`solve_ifds`] owns only that tabulation. The consumer supplies the shape
//! as a [`Supergraph`] (nodes may be blocks or instructions; call resolution
//! is the consumer's) and the flow functions as an [`IfdsProblem`] over dense
//! facts, so access paths, variables and every other fact domain stay in its
//! IR crate. [`CfgSupergraph`] adapts a set of keyed CFGs whose calls come
//! from the [`Flow`](crate::Flow) of their instructions, the input of
//! [`call_graph`](crate::call_graph), with one node per program point.
//!
//! # Determinism and limits
//!
//! The worklist pops path edges in ascending (node, fact) order, the facts
//! at a node come out ascending, and [`IfdsConfig::max_steps`] bounds the
//! number of path edges processed exactly as
//! [`SolveConfig::max_steps`](crate::SolveConfig::max_steps) bounds the
//! worklist entries of the other solvers.
//!
//! # Collecting path edges
//!
//! With [`IfdsConfig::collect_path_edges`], the solver drops the path edges
//! of a context (a procedure entered with one fact) once neither the context
//! nor any context it calls, transitively, has work left: none of them can
//! produce a path edge again. That is the fact-level collection of D. He,
//! Y. Gui, Y. Gao and J. Xue (ISSTA 2023). A collected context keeps its end
//! summaries and stays known, so a later call into it reuses them instead of
//! analyzing it again. The results and the processed path edges are the same
//! either way; only the peak number of path edges held at once
//! ([`SolveStats::peak_count`]) goes down.

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;

use super::solve::{SolveError, SolveStats};
use crate::graph::view::DenseId;

mod cfg_supergraph;
mod tabulate;

pub use cfg_supergraph::CfgSupergraph;

/// The interprocedural shape an IFDS problem is solved over.
///
/// Nodes and procedures are dense identities. Each node belongs to one
/// procedure, and each procedure has one start node. A call node lists its
/// return sites as its [`successors`](Self::successors) and the procedures it
/// may call as its [`callees`](Self::callees); a node without callees is not
/// a call. An exit is a node from which control returns to the caller.
///
/// # Contract
///
/// - Every node and procedure index is below [`node_bound`](Self::node_bound)
///   or [`proc_bound`](Self::proc_bound).
/// - [`successors`](Self::successors) stay in the node's procedure.
/// - Every method answers the same for the same input.
pub trait Supergraph {
    /// Node identity: a block, an instruction, or a program point.
    type Node: DenseId;
    /// Procedure identity.
    type Proc: DenseId;

    /// An exclusive upper bound on every node index.
    fn node_bound(&self) -> usize;

    /// An exclusive upper bound on every procedure index.
    fn proc_bound(&self) -> usize;

    /// The procedure `node` belongs to.
    fn proc_of(&self, node: Self::Node) -> Self::Proc;

    /// The node at which `proc` starts.
    fn start(&self, proc: Self::Proc) -> Self::Node;

    /// Whether control returns to the caller from `node`.
    fn is_exit(&self, node: Self::Node) -> bool;

    /// Push the intraprocedural successors of `node` onto `out`; a call node
    /// pushes its return sites.
    fn successors(&self, node: Self::Node, out: &mut Vec<Self::Node>);

    /// Push the procedures `node` may call onto `out`; nothing for a node
    /// that is not a call.
    fn callees(&self, node: Self::Node, out: &mut Vec<Self::Proc>);
}

/// The distributive flow functions of an IFDS problem over the dense facts
/// `0..fact_bound()`.
///
/// Fact `0` is the IFDS zero fact Λ, which holds wherever the analysis
/// reaches. Each function pushes the facts that `fact` maps to onto `out`.
/// The solver adds the zero fact's own identity, `0 ↦ 0`, to every function
/// itself, so a function called with fact `0` pushes only the facts it
/// generates. Facts may be pushed more than once.
///
/// Facts are `u32` for compact tables; every pushed fact must be below
/// [`fact_bound`](Self::fact_bound).
pub trait IfdsProblem<S: Supergraph> {
    /// The number of facts, zero included.
    fn fact_bound(&self) -> usize;

    /// The facts at `to` that `fact` at `from` produces along an
    /// intraprocedural edge from a node that is not a call.
    fn normal(&self, from: S::Node, to: S::Node, fact: u32, out: &mut Vec<u32>);

    /// The facts at the start of `callee` that `fact` at `call` produces.
    fn call(&self, call: S::Node, callee: S::Proc, fact: u32, out: &mut Vec<u32>);

    /// The facts at `return_site` that `fact` at `exit` of `callee`
    /// produces, for a return to `call`.
    fn ret(
        &self,
        call: S::Node,
        callee: S::Proc,
        exit: S::Node,
        return_site: S::Node,
        fact: u32,
        out: &mut Vec<u32>,
    );

    /// The facts at `return_site` that `fact` at `call` produces alongside
    /// the call: the facts the callee neither reads nor changes, such as the
    /// caller's locals.
    fn call_to_return(&self, call: S::Node, return_site: S::Node, fact: u32, out: &mut Vec<u32>);
}

/// Configuration of [`solve_ifds`]. The default is an unbounded solve that
/// keeps every path edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IfdsConfig {
    /// The most path edges the solve processes; with path edges left at the
    /// limit it stops with [`SolveError::StepLimitExceeded`]. `None` is
    /// unbounded.
    pub max_steps: Option<usize>,
    /// Whether the path edges of finished contexts are collected; see the
    /// [module docs](self).
    pub collect_path_edges: bool,
}

/// The answer of [`solve_ifds`]: the facts at each node and the end
/// summaries of each context.
pub struct IfdsResults<S: Supergraph> {
    /// Per node, where its facts start in `facts`; one extra entry at the end.
    offsets: Vec<usize>,
    /// The facts of every node, each node's ascending.
    facts: Vec<u32>,
    /// Each context, a procedure and its entry fact, and its end summaries:
    /// the (exit node, fact) pairs, ascending.
    summaries: BTreeMap<(usize, u32), Vec<(usize, u32)>>,
    stats: SolveStats,
    graph: PhantomData<fn() -> S>,
}

impl<S: Supergraph> IfdsResults<S> {
    /// The facts that hold at `node` along some realizable path from a seed,
    /// ascending. The zero fact is among them wherever the zero fact reaches.
    ///
    /// # Panics
    ///
    /// Panics when `node` is not below the supergraph's node bound.
    pub fn facts_at(&self, node: S::Node) -> impl Iterator<Item = u32> + '_ {
        let index = node.index();
        self.facts[self.offsets[index]..self.offsets[index + 1]]
            .iter()
            .copied()
    }

    /// Whether `fact` holds at `node`.
    ///
    /// # Panics
    ///
    /// Panics when `node` is not below the supergraph's node bound.
    #[must_use]
    pub fn holds(&self, node: S::Node, fact: u32) -> bool {
        let index = node.index();
        self.facts[self.offsets[index]..self.offsets[index + 1]]
            .binary_search(&fact)
            .is_ok()
    }

    /// The (exit, fact) pairs that `proc` reaches when it is entered with
    /// `entry_fact`, ascending; nothing when no realizable path entered
    /// `proc` with that fact.
    pub fn summaries(
        &self,
        proc: S::Proc,
        entry_fact: u32,
    ) -> impl Iterator<Item = (S::Node, u32)> + '_ {
        self.summaries
            .get(&(proc.index(), entry_fact))
            .map_or(&[][..], Vec::as_slice)
            .iter()
            .map(|&(exit, fact)| (S::Node::from_index(exit), fact))
    }

    /// What the solve did: its steps are the path edges it processed and
    /// its peak the most path edges it held at once.
    #[must_use]
    pub const fn stats(&self) -> SolveStats {
        self.stats
    }
}

impl<S: Supergraph> fmt::Debug for IfdsResults<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IfdsResults")
            .field("node_bound", &(self.offsets.len() - 1))
            .field("facts", &self.facts.len())
            .field("contexts", &self.summaries.len())
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

/// Solve an IFDS problem by tabulation from `seeds`.
///
/// A seed `(node, fact)` states that `fact` holds at `node` whenever its
/// procedure runs: it is the path edge from the zero fact at the start of
/// the node's procedure to `fact` at `node`, as in the solver of Reps,
/// Horwitz and Sagiv, whose usual seed is the zero fact at the start of
/// `main`. A return from a procedure that no realizable path called is not
/// followed.
///
/// # Errors
///
/// Returns [`SolveError::StepLimitExceeded`] when path edges remain at the
/// configured limit; its `pending_node` is the node of the next path edge.
///
/// # Panics
///
/// Panics when a seed's node or fact, or a fact a flow function pushes, is
/// out of range.
///
/// # Examples
///
/// ```
/// use cfglib::{IfdsConfig, IfdsProblem, Supergraph, solve_ifds};
///
/// // main: 0 = start, 1 = call id(x), 2 = return site (exit).
/// // id:   3 = start (exit).
/// struct Program;
/// impl Supergraph for Program {
///     type Node = usize;
///     type Proc = usize;
///     fn node_bound(&self) -> usize { 4 }
///     fn proc_bound(&self) -> usize { 2 }
///     fn proc_of(&self, node: usize) -> usize { usize::from(node == 3) }
///     fn start(&self, proc: usize) -> usize { [0, 3][proc] }
///     fn is_exit(&self, node: usize) -> bool { node >= 2 }
///     fn successors(&self, node: usize, out: &mut Vec<usize>) {
///         if node < 2 { out.push(node + 1) }
///     }
///     fn callees(&self, node: usize, out: &mut Vec<usize>) {
///         if node == 1 { out.push(1) }
///     }
/// }
///
/// // Taint: fact 1 is "x is tainted"; node 0 taints x.
/// struct Taint;
/// impl IfdsProblem<Program> for Taint {
///     fn fact_bound(&self) -> usize { 2 }
///     fn normal(&self, from: usize, _: usize, fact: u32, out: &mut Vec<u32>) {
///         if from == 0 && fact == 0 { out.push(1) } else if fact != 0 { out.push(fact) }
///     }
///     fn call(&self, _: usize, _: usize, fact: u32, out: &mut Vec<u32>) {
///         if fact != 0 { out.push(fact) }
///     }
///     fn ret(&self, _: usize, _: usize, _: usize, _: usize, fact: u32, out: &mut Vec<u32>) {
///         if fact != 0 { out.push(fact) }
///     }
///     fn call_to_return(&self, _: usize, _: usize, _: u32, _: &mut Vec<u32>) {}
/// }
///
/// let results = solve_ifds(&Program, &Taint, &[(0, 0)], IfdsConfig::default()).unwrap();
/// assert_eq!(results.facts_at(2).collect::<Vec<_>>(), [0, 1]);
/// assert_eq!(results.summaries(1, 1).collect::<Vec<_>>(), [(3, 1)]);
/// ```
pub fn solve_ifds<S: Supergraph, P: IfdsProblem<S>>(
    graph: &S,
    problem: &P,
    seeds: &[(S::Node, u32)],
    config: IfdsConfig,
) -> Result<IfdsResults<S>, SolveError> {
    tabulate::solve(graph, problem, seeds, config)
}

#[cfg(test)]
mod tests;
