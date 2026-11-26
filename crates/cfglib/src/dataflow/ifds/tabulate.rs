//! The tabulation algorithm behind [`solve_ifds`](super::solve_ifds).
//!
//! A *context* is a procedure entered with one fact. A path edge
//! `⟨start, d1⟩ → ⟨n, d2⟩` belongs to the context of `d1` at the start of
//! `n`'s procedure and is stored there as `(n, d2)`. Each context also keeps
//! its end summaries, the `(exit, fact)` pairs it reaches, and its incoming
//! calls, the `(call node, fact, caller context)` triples that entered it.
//! A new end summary returns to every incoming call, and a new incoming call
//! receives every end summary so far, so neither summary edges nor a lookup
//! of the caller's path edges is needed: path edges serve only to process
//! each one once, which is what lets a finished context drop them.

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use super::{IfdsConfig, IfdsProblem, IfdsResults, Supergraph};
use crate::dataflow::solve::{SolveError, SolveStats, count_step};
use crate::graph::view::DenseId;

/// The stored path edges below which no collection pass runs.
const FIRST_PASS: usize = 64;

pub(super) fn solve<S: Supergraph, P: IfdsProblem<S>>(
    graph: &S,
    problem: &P,
    seeds: &[(S::Node, u32)],
    config: IfdsConfig,
) -> Result<IfdsResults<S>, SolveError> {
    let mut tabulation = Tabulation {
        graph,
        problem,
        fact_bound: problem.fact_bound(),
        collect: config.collect_path_edges,
        context_of: BTreeMap::new(),
        contexts: Vec::new(),
        worklist: BTreeSet::new(),
        holds: BTreeSet::new(),
        stored: 0,
        peak: 0,
        idle: false,
        next_pass: FIRST_PASS,
        uncollected: Vec::new(),
        marked: Vec::new(),
        nodes: Vec::new(),
        procs: Vec::new(),
        facts: Vec::new(),
    };
    let node_bound = graph.node_bound();
    for &(node, fact) in seeds {
        assert!(node.index() < node_bound, "seed node is out of range");
        tabulation.check_fact(fact);
        let context = tabulation.context(graph.proc_of(node).index(), 0);
        tabulation.propagate(context, node.index(), fact);
    }

    let mut steps = 0;
    while let Some((node, fact, context)) = tabulation.worklist.pop_first() {
        count_step(config.max_steps, &mut steps, node)?;
        tabulation.contexts[context].pending -= 1;
        tabulation.process(node, fact, context);
        if tabulation.contexts[context].pending == 0 {
            tabulation.idle = true;
        }
        if tabulation.collect && tabulation.idle && tabulation.stored >= tabulation.next_pass {
            tabulation.collect_finished();
        }
    }
    Ok(tabulation.finish(node_bound, steps))
}

/// One procedure entered with one fact.
struct Context {
    proc: usize,
    fact: u32,
    /// Whether the path edge from the entry fact to itself at the start was
    /// propagated, so the procedure is analyzed from its start.
    entered: bool,
    /// Whether the path edges were dropped because nothing can extend them.
    collected: bool,
    /// Path edges of this context on the worklist.
    pending: usize,
    /// The `(node, fact)` targets of the path edges, until collected.
    path_edges: BTreeSet<(usize, u32)>,
    /// The `(exit, fact)` pairs reached.
    end: BTreeSet<(usize, u32)>,
    /// The `(call node, fact at the call, caller context)` triples that
    /// entered this context.
    incoming: BTreeSet<(usize, u32, usize)>,
}

struct Tabulation<'a, S: Supergraph, P> {
    graph: &'a S,
    problem: &'a P,
    fact_bound: usize,
    collect: bool,
    /// Each context by procedure and entry fact.
    context_of: BTreeMap<(usize, u32), usize>,
    contexts: Vec<Context>,
    /// Path edges to process, as `(node, fact, context)`, lowest first.
    worklist: BTreeSet<(usize, u32, usize)>,
    /// Every `(node, fact)` that holds.
    holds: BTreeSet<(usize, u32)>,
    /// Path edges currently stored, and the most stored at once.
    stored: usize,
    peak: usize,
    /// Whether some context ran out of work since the last collection pass.
    idle: bool,
    /// The stored path edges at which the next collection pass may run.
    next_pass: usize,
    /// The contexts whose path edges are kept, in creation order.
    uncollected: Vec<usize>,
    /// Scratch marks of the collection pass, one per context.
    marked: Vec<bool>,
    nodes: Vec<S::Node>,
    procs: Vec<S::Proc>,
    facts: Vec<u32>,
}

impl<S: Supergraph, P: IfdsProblem<S>> Tabulation<'_, S, P> {
    fn check_fact(&self, fact: u32) {
        assert!(
            usize::try_from(fact).is_ok_and(|fact| fact < self.fact_bound),
            "fact is out of range"
        );
    }

    /// The context of `proc` entered with `fact`, made on first use.
    fn context(&mut self, proc: usize, fact: u32) -> usize {
        let next = self.contexts.len();
        let context = *self.context_of.entry((proc, fact)).or_insert(next);
        if context == next {
            self.contexts.push(Context {
                proc,
                fact,
                entered: false,
                collected: false,
                pending: 0,
                path_edges: BTreeSet::new(),
                end: BTreeSet::new(),
                incoming: BTreeSet::new(),
            });
            self.uncollected.push(context);
            self.marked.push(false);
        }
        context
    }

    /// Record the path edge to `fact` at `node` in `context`, queueing it the
    /// first time.
    fn propagate(&mut self, context: usize, node: usize, fact: u32) {
        let entry = &mut self.contexts[context];
        debug_assert!(
            !entry.collected,
            "a collected context has no caller or callee with work left"
        );
        if !entry.path_edges.insert((node, fact)) {
            return;
        }
        if fact == entry.fact && node == self.graph.start(S::Proc::from_index(entry.proc)).index() {
            entry.entered = true;
        }
        entry.pending += 1;
        self.stored += 1;
        self.peak = self.peak.max(self.stored);
        self.holds.insert((node, fact));
        self.worklist.insert((node, fact, context));
    }

    /// Fill `self.facts` with what a flow function pushes for `fact`, plus
    /// the zero fact's identity.
    fn flow(&mut self, fact: u32, function: impl FnOnce(&P, &mut Vec<u32>)) -> Vec<u32> {
        let mut out = core::mem::take(&mut self.facts);
        out.clear();
        function(self.problem, &mut out);
        if fact == 0 {
            out.push(0);
        }
        for &produced in &out {
            self.check_fact(produced);
        }
        out
    }

    /// Hand a flow output buffer back for reuse.
    fn recycle(&mut self, out: Vec<u32>) {
        self.facts = out;
    }

    fn process(&mut self, node: usize, fact: u32, context: usize) {
        let graph = self.graph;
        let here = S::Node::from_index(node);
        if graph.is_exit(here) && self.contexts[context].end.insert((node, fact)) {
            self.return_to_callers(context, node, fact);
        }

        let mut procs = core::mem::take(&mut self.procs);
        procs.clear();
        graph.callees(here, &mut procs);
        let mut successors = core::mem::take(&mut self.nodes);
        successors.clear();
        graph.successors(here, &mut successors);

        if procs.is_empty() {
            for &next in &successors {
                let out = self.flow(fact, |problem, out| problem.normal(here, next, fact, out));
                for &produced in &out {
                    self.propagate(context, next.index(), produced);
                }
                self.recycle(out);
            }
        } else {
            for &callee in &procs {
                self.enter(context, here, callee, fact, &successors);
            }
            for &site in &successors {
                let out = self.flow(fact, |problem, out| {
                    problem.call_to_return(here, site, fact, out);
                });
                for &produced in &out {
                    self.propagate(context, site.index(), produced);
                }
                self.recycle(out);
            }
        }
        self.procs = procs;
        self.nodes = successors;
    }

    /// The call from context `from` at `call` with `fact` into `callee`:
    /// enter the callee's contexts, and return the end summaries they reached
    /// so far.
    fn enter(&mut self, from: usize, call: S::Node, callee: S::Proc, fact: u32, sites: &[S::Node]) {
        let entries = self.flow(fact, |problem, out| problem.call(call, callee, fact, out));
        for &entry in &entries {
            let context = self.context(callee.index(), entry);
            let state = &mut self.contexts[context];
            // A collected context is finished: its end summaries are final, so
            // a new call takes them and need not be remembered.
            if !state.collected && !state.incoming.insert((call.index(), fact, from)) {
                continue;
            }
            if !state.entered {
                let start = self.graph.start(callee).index();
                self.propagate(context, start, entry);
            }
            let ends: Vec<(usize, u32)> = self.contexts[context].end.iter().copied().collect();
            for (exit, exit_fact) in ends {
                let exit = S::Node::from_index(exit);
                for &site in sites {
                    let out = self.flow(exit_fact, |problem, out| {
                        problem.ret(call, callee, exit, site, exit_fact, out);
                    });
                    for &produced in &out {
                        self.propagate(from, site.index(), produced);
                    }
                    self.recycle(out);
                }
            }
        }
        self.recycle(entries);
    }

    /// A new end summary `(exit, fact)` of `context`: return it to every call
    /// that entered the context.
    fn return_to_callers(&mut self, context: usize, exit: usize, fact: u32) {
        let graph = self.graph;
        let callee = S::Proc::from_index(self.contexts[context].proc);
        let exit = S::Node::from_index(exit);
        let incoming: Vec<(usize, u32, usize)> =
            self.contexts[context].incoming.iter().copied().collect();
        let mut sites = Vec::new();
        // The triples are sorted by call node, so the return sites of one
        // call and the facts they receive are computed once for its callers.
        let mut group = 0;
        while group < incoming.len() {
            let call = incoming[group].0;
            let end = group + incoming[group..].partition_point(|entry| entry.0 == call);
            let call = S::Node::from_index(call);
            sites.clear();
            graph.successors(call, &mut sites);
            for &site in &sites {
                let out = self.flow(fact, |problem, out| {
                    problem.ret(call, callee, exit, site, fact, out);
                });
                for &(_, _, caller) in &incoming[group..end] {
                    for &produced in &out {
                        self.propagate(caller, site.index(), produced);
                    }
                }
                self.recycle(out);
            }
            group = end;
        }
    }

    /// Drop the path edges of every entered context from which no context
    /// with work left is reachable through calls: nothing can add to them
    /// again. A context with work, and every caller of one, transitively,
    /// stays. The pass visits only contexts not collected yet, and runs once
    /// the stored path edges have doubled since the last one, so its cost is
    /// amortized over the path edges stored in between.
    fn collect_finished(&mut self) {
        let mut stack: Vec<usize> = self
            .uncollected
            .iter()
            .copied()
            .filter(|&context| self.contexts[context].pending > 0)
            .collect();
        for &context in &stack {
            self.marked[context] = true;
        }
        while let Some(context) = stack.pop() {
            for &(_, _, caller) in &self.contexts[context].incoming {
                if !self.marked[caller] {
                    self.marked[caller] = true;
                    stack.push(caller);
                }
            }
        }
        let contexts = &mut self.contexts;
        let marked = &mut self.marked;
        let stored = &mut self.stored;
        self.uncollected.retain(|&context| {
            let state = &mut contexts[context];
            if core::mem::take(&mut marked[context]) || !state.entered {
                return true;
            }
            *stored -= state.path_edges.len();
            state.path_edges = BTreeSet::new();
            state.incoming = BTreeSet::new();
            state.collected = true;
            false
        });
        self.idle = false;
        self.next_pass = (2 * self.stored).max(FIRST_PASS);
    }

    fn finish(self, node_bound: usize, steps: usize) -> IfdsResults<S> {
        let mut offsets = vec![0; node_bound + 1];
        let mut facts = Vec::with_capacity(self.holds.len());
        for (node, fact) in self.holds {
            offsets[node + 1] += 1;
            facts.push(fact);
        }
        for node in 0..node_bound {
            offsets[node + 1] += offsets[node];
        }
        let summaries = self
            .contexts
            .into_iter()
            .map(|context| {
                (
                    (context.proc, context.fact),
                    context.end.into_iter().collect(),
                )
            })
            .collect();
        IfdsResults {
            offsets,
            facts,
            summaries,
            stats: SolveStats::new(steps, 0, self.peak),
            graph: PhantomData,
        }
    }
}
