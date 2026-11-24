use cfglib::{Graph, NodeId, NonterminationScratch, nontermination_sensitive_controllers};

use super::BenchmarkSuite;
use crate::harness::benchmark_case;

/// Rungs of the ladder of the nontermination-sensitive case.
const LADDER_RUNGS: usize = 1_024;

/// Two rails `a_0 .. a_k` and `b_0 .. b_k`, each a path, with a rung from
/// every `a_i` down to `b_i`; `b_k` is the one exit.
///
/// Every `a_i` decides whether each later block of the `b` rail runs, so the
/// nontermination-sensitive relation holds `k(k + 3) / 2` pairs: the ladder
/// is the documented case where the relation itself is quadratic.
fn ladder(rungs: usize) -> Graph<(), ()> {
    let mut graph = Graph::with_capacity(2 * (rungs + 1), 3 * rungs + 1);
    let left: Vec<NodeId> = (0..=rungs).map(|_| graph.add_node(())).collect();
    let right: Vec<NodeId> = (0..=rungs).map(|_| graph.add_node(())).collect();
    for rung in 0..=rungs {
        if rung < rungs {
            graph.add_edge(left[rung], left[rung + 1], ());
            graph.add_edge(right[rung], right[rung + 1], ());
        }
        graph.add_edge(left[rung], right[rung], ());
    }
    graph
}

fn register_nontermination_sensitive(suite: &mut BenchmarkSuite<'_>) {
    let graph = ladder(LADDER_RUNGS);
    let mut scratch = NonterminationScratch::new();
    let mut controllers = Vec::new();
    benchmark_case!(
        suite,
        "api_nontermination_sensitive_controllers_ladder",
        covers[nontermination_sensitive_controllers],
        || {
            // The whole relation, one dependent at a time.
            graph
                .node_ids()
                .map(|node| {
                    nontermination_sensitive_controllers(
                        &graph,
                        node,
                        &mut scratch,
                        &mut controllers,
                    );
                    controllers.len()
                })
                .sum::<usize>()
        },
        |pairs: &usize| assert_eq!(*pairs, LADDER_RUNGS * (LADDER_RUNGS + 3) / 2)
    );
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    register_nontermination_sensitive(suite);
}
