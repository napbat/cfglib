//! `solve_ifds` over a synthetic supergraph of 10^4 procedures with 64 facts.
//!
//! Procedure `i` is seven nodes: start, a node that generates fact
//! `1 + i % 63` from the zero fact and kills another, two calls (to
//! procedures `2i + 1` and `2i + 2`, a binary call tree) with their return
//! sites, a branch around the calls, and an exit. A call passes the facts
//! congruent to `i` modulo 4 into the callee and the rest alongside, so
//! facts travel down and back up the tree as a taint analysis's do. The
//! second case solves the same problem collecting finished contexts' path
//! edges and checks that the peak drops and the answer does not change.

use cfglib::{IfdsConfig, IfdsProblem, IfdsResults, Supergraph, solve_ifds};

use super::BenchmarkSuite;
use crate::harness::benchmark_case;

/// Procedures of the supergraph.
const PROCS: usize = 10_000;
/// Nodes of each procedure.
const NODES: usize = 7;
/// Facts, the zero fact included.
const FACTS: usize = 64;

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let seeds = [(0, 0)];
    benchmark_case!(
        suite,
        "api_solve_ifds",
        covers[solve_ifds],
        || solve_ifds(&Tree, &Taint, &seeds, IfdsConfig::default()),
        |results: &Result<IfdsResults<Tree>, _>| check(results.as_ref().expect("unbounded")),
    );
    let collecting = IfdsConfig {
        collect_path_edges: true,
        ..IfdsConfig::default()
    };
    benchmark_case!(
        suite,
        "api_solve_ifds_collecting_path_edges",
        || solve_ifds(&Tree, &Taint, &seeds, collecting),
        |results: &Result<IfdsResults<Tree>, _>| {
            let collected = results.as_ref().expect("unbounded");
            check(collected);
            let kept = solve_ifds(&Tree, &Taint, &seeds, IfdsConfig::default()).expect("unbounded");
            assert_eq!(collected.stats().step_count(), kept.stats().step_count());
            assert!(collected.stats().peak_count() < kept.stats().peak_count());
            for node in 0..PROCS * NODES {
                assert!(collected.facts_at(node).eq(kept.facts_at(node)));
            }
        },
    );
}

/// The zero fact reaches every node, and each procedure's generated fact
/// reaches its first call.
fn check(results: &IfdsResults<Tree>) {
    for proc in 0..PROCS {
        for node in proc * NODES..(proc + 1) * NODES {
            assert!(results.holds(node, 0), "node {node}");
        }
        assert!(results.holds(proc * NODES + 2, generated(proc)));
    }
}

fn generated(proc: usize) -> u32 {
    u32::try_from(1 + proc % (FACTS - 1)).expect("a fact below 64")
}

fn killed(proc: usize) -> u32 {
    u32::try_from(1 + proc * 5 % (FACTS - 1)).expect("a fact below 64")
}

/// The binary call tree of `PROCS` procedures.
struct Tree;

impl Supergraph for Tree {
    type Node = usize;
    type Proc = usize;

    fn node_bound(&self) -> usize {
        PROCS * NODES
    }

    fn proc_bound(&self) -> usize {
        PROCS
    }

    fn proc_of(&self, node: usize) -> usize {
        node / NODES
    }

    fn start(&self, proc: usize) -> usize {
        proc * NODES
    }

    fn is_exit(&self, node: usize) -> bool {
        node % NODES == NODES - 1
    }

    fn successors(&self, node: usize, out: &mut Vec<usize>) {
        // 0 start, 1 generate, 2 call, 3 return site, 4 call, 5 return site,
        // 6 exit; the generating node also branches around both calls.
        let local = node % NODES;
        if local < NODES - 1 {
            out.push(node + 1);
        }
        if local == 1 {
            out.push(node + 4);
        }
    }

    fn callees(&self, node: usize, out: &mut Vec<usize>) {
        let (proc, local) = (node / NODES, node % NODES);
        let callee = match local {
            2 => 2 * proc + 1,
            4 => 2 * proc + 2,
            _ => return,
        };
        if callee < PROCS {
            out.push(callee);
        }
    }
}

/// Sparse taint-like flow functions over `FACTS` facts.
struct Taint;

impl Taint {
    fn passes_in(call: usize, fact: u32) -> bool {
        usize::try_from(fact).expect("a small fact") % 4 == (call / NODES) % 4
    }
}

impl IfdsProblem<Tree> for Taint {
    fn fact_bound(&self) -> usize {
        FACTS
    }

    fn normal(&self, from: usize, _: usize, fact: u32, out: &mut Vec<u32>) {
        let proc = from / NODES;
        if from % NODES == 1 {
            if fact == 0 {
                out.push(generated(proc));
            } else if fact != killed(proc) {
                out.push(fact);
            }
        } else if fact != 0 {
            out.push(fact);
        }
    }

    fn call(&self, call: usize, _: usize, fact: u32, out: &mut Vec<u32>) {
        if fact != 0 && Self::passes_in(call, fact) {
            out.push(fact);
        }
    }

    fn ret(&self, _: usize, _: usize, _: usize, _: usize, fact: u32, out: &mut Vec<u32>) {
        if fact != 0 {
            out.push(fact);
        }
    }

    fn call_to_return(&self, call: usize, _: usize, fact: u32, out: &mut Vec<u32>) {
        if fact != 0 && !Self::passes_in(call, fact) {
            out.push(fact);
        }
    }
}
