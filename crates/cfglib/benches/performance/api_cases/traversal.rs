//! Small-run traversal and fan-out shapes.
//!
//! `graph.rs` times the open traversals over one 1,024-node chain. Consumers
//! mostly run them the other way: thousands of walks that each touch a handful
//! of nodes, a fan-out read once per parser step, a chase that ends after two
//! hops. These cases keep that shape, so a per-call fixed cost (an allocation,
//! a copied path) shows up where a long chain hides it.

use std::ops::ControlFlow;

use cfglib::{
    DenseFanout, FoldEnter, MarkScope, OpenFold, OpenFoldConfig, OpenPathsConfig, OpenSearchConfig,
    SearchOrder, StackGraph, StackNodeId, StackReverseIndex, StackSearchConfig, Visit, follow,
    open_breadth_first_paths, open_fold_post_order, open_search,
};

use super::BenchmarkSuite;
use crate::fixtures::fixture_u32;
use crate::harness::benchmark_case;

/// Keys of the dense fan-out tables.
const FANOUT_KEYS: usize = 1_024;
/// Values recorded per key on average.
const FANOUT_FAN: usize = 3;
/// Chases and searches run by one measured operation of the small-walk cases.
const WALKS: usize = 256;
/// Levels of the layered graphs behind the route and fold cases.
const LEVELS: usize = 9;
/// Nodes per level of the layered graphs.
const WIDTH: usize = 2;
/// References, scope hops per reference, and decoy edges of the stack graph.
const STACK_REFERENCES: usize = 256;
const STACK_HOPS: usize = 12;

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    register_fanout(suite);
    register_small_walks(suite);
    register_layered(suite);
    register_stack(suite);
}

/// Pairs scattered over the key space in an order unrelated to the keys, the
/// way a reference index is recorded.
fn fanout_pairs() -> Vec<(usize, u32)> {
    (0..FANOUT_KEYS * FANOUT_FAN)
        .map(|index| (index * 7_919 % FANOUT_KEYS, fixture_u32(index)))
        .collect()
}

fn register_fanout(suite: &mut BenchmarkSuite<'_>) {
    let pairs = fanout_pairs();
    let table = DenseFanout::from_pairs(FANOUT_KEYS, pairs.iter().copied());
    benchmark_case!(
        suite,
        "fanout_dense_get",
        || {
            let mut total = 0_usize;
            for _ in 0..8 {
                for key in 0..FANOUT_KEYS {
                    total += table.get(key).len();
                }
            }
            total
        },
        |total: &usize| assert_eq!(*total, 8 * pairs.len())
    );
    benchmark_case!(
        suite,
        "fanout_dense_from_pairs",
        || DenseFanout::from_pairs(FANOUT_KEYS, pairs.iter().copied()),
        |built: &DenseFanout<u32>| {
            assert_eq!(built.value_count(), pairs.len());
            assert_eq!(built, &table);
        }
    );
    benchmark_case!(
        suite,
        "fanout_dense_from_grouped",
        || {
            DenseFanout::<u32>::from_grouped((0..FANOUT_KEYS).map(|key| {
                (0..key % (2 * FANOUT_FAN)).map(move |slot| u32::try_from(key + slot).unwrap_or(0))
            }))
        },
        |built: &DenseFanout<u32>| {
            assert_eq!(built.bound(), FANOUT_KEYS);
            assert_eq!(built.get(5), [5, 6, 7, 8, 9]);
        }
    );
}

fn register_small_walks(suite: &mut BenchmarkSuite<'_>) {
    // A chain of aliases that ends after four nodes: the length a use's
    // resolved-target chase typically has.
    benchmark_case!(
        suite,
        "follow_short_chains",
        || {
            (0..WALKS)
                .map(|seed| follow(seed * 4, 16, |node| (node % 4 != 3).then(|| node + 1)))
                .sum::<usize>()
        },
        |total: &usize| assert_eq!(*total, (0..WALKS).map(|seed| seed * 4 + 3).sum::<usize>())
    );

    // A barrel re-export walk: each module re-exports the next two, three
    // levels deep, from two seeds.
    let barrel = |node: &(u32, u32), out: &mut Vec<(u32, u32)>| {
        let (module, depth) = *node;
        if depth < 3 {
            out.push((module, depth + 1));
            out.push((module + 1, depth + 1));
        }
    };
    for (name, order) in [
        ("open_search_small_dfs", SearchOrder::DepthFirst),
        ("open_search_small_bfs", SearchOrder::BreadthFirst),
    ] {
        // The case name is a variable here, which `benchmark_case!` takes only
        // as a literal.
        suite.case(
            name,
            &[],
            || {
                let mut visited = 0_usize;
                for walk in 0..WALKS {
                    let base = u32::try_from(walk).unwrap_or(0) * 16;
                    let stopped = open_search(
                        [(base, 0), (base + 8, 0)],
                        OpenSearchConfig::new(order),
                        barrel,
                        |_, _| {
                            visited += 1;
                            ControlFlow::<(), _>::Continue(Visit::Descend)
                        },
                    );
                    assert!(stopped.is_none());
                }
                visited
            },
            |visited: &usize| assert_eq!(*visited, WALKS * 2 * 10),
        );
    }
}

/// The successors of node `node` in the layered graph: every node of the next
/// level, and nothing below the last.
fn layered(node: usize, out: &mut Vec<usize>) {
    let level = node / WIDTH;
    if level + 1 < LEVELS {
        out.extend((0..WIDTH).map(|column| (level + 1) * WIDTH + column));
    }
}

/// Counts the nodes folded below each node; every node marks itself on its own
/// route only, so the layered graph unfolds into a full tree.
struct RouteCount;

impl OpenFold for RouteCount {
    type Node = usize;
    type Mark = usize;
    type Value = usize;
    type Accumulator = usize;

    fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
        layered(*node, out);
    }

    fn mark(&mut self, node: &usize) -> Option<usize> {
        Some(*node)
    }

    fn enter(&mut self, _node: &usize) -> FoldEnter<usize, usize> {
        FoldEnter::Fold {
            accumulator: 1,
            marks: MarkScope::Isolated,
        }
    }

    fn absorb(
        &mut self,
        accumulator: &mut usize,
        _child: &usize,
        value: Option<usize>,
    ) -> ControlFlow<()> {
        *accumulator += value.unwrap_or_default();
        ControlFlow::Continue(())
    }

    fn finish(&mut self, _node: &usize, accumulator: usize) -> Option<usize> {
        Some(accumulator)
    }
}

fn register_layered(suite: &mut BenchmarkSuite<'_>) {
    benchmark_case!(
        suite,
        "open_paths_layered",
        || {
            let mut routes = 0_usize;
            let stopped = open_breadth_first_paths(
                [0_usize],
                OpenPathsConfig::new(),
                |node: &usize, out: &mut Vec<usize>| layered(*node, out),
                |_| {
                    routes += 1;
                    ControlFlow::<(), _>::Continue(Visit::Descend)
                },
            );
            (stopped, routes)
        },
        |(stopped, routes)| {
            assert!(stopped.is_none());
            assert_eq!(*routes, (1 << LEVELS) - 1);
        }
    );
    benchmark_case!(
        suite,
        "open_fold_isolated_layered",
        || open_fold_post_order(&mut RouteCount, 0_usize, OpenFoldConfig::new()),
        |size: &Option<usize>| assert_eq!(*size, Some((1 << LEVELS) - 1))
    );
}

type Stack = StackGraph<(), u32, (), ()>;

/// One file of `STACK_REFERENCES` references. Each pushes its symbol, hops
/// through `STACK_HOPS` scopes, and pops it at its definition; every scope also
/// has an edge to a pop of another symbol, which every path rejects. The decoy
/// is listed first on even hops and last on odd ones.
fn stack_graph() -> (Stack, Vec<(StackNodeId, StackNodeId)>) {
    let mut graph = Stack::new();
    let file = graph.add_file(());
    let mut bindings = Vec::with_capacity(STACK_REFERENCES);
    for index in 0..STACK_REFERENCES {
        let symbol = fixture_u32(index);
        let reference = graph
            .add_push_symbol_node(file, symbol, true, ())
            .expect("known file");
        let definition = graph
            .add_pop_symbol_node(file, symbol, true, ())
            .expect("known file");
        let decoy = graph
            .add_pop_symbol_node(file, u32::MAX, false, ())
            .expect("known file");
        let scopes: Vec<StackNodeId> = (0..STACK_HOPS)
            .map(|_| graph.add_scope_node(file, false, ()).expect("known file"))
            .collect();
        let mut add = |source, target| {
            graph
                .add_edge(source, target, 0, ())
                .expect("file-local edge");
        };
        add(reference, scopes[0]);
        for (hop, &scope) in scopes.iter().enumerate() {
            let onward = scopes.get(hop + 1).copied().unwrap_or(definition);
            if hop % 2 == 0 {
                add(scope, decoy);
                add(scope, onward);
            } else {
                add(scope, onward);
                add(scope, decoy);
            }
        }
        bindings.push((reference, definition));
    }
    (graph, bindings)
}

fn register_stack(suite: &mut BenchmarkSuite<'_>) {
    let (graph, bindings) = stack_graph();
    benchmark_case!(
        suite,
        "stack_reverse_index_linear",
        || StackReverseIndex::try_compute_linear(&graph, StackSearchConfig::new()),
        |index| {
            let index = index.as_ref().expect("every query is linear");
            for &(reference, definition) in &bindings {
                assert_eq!(index.references_to(definition), [reference]);
            }
        }
    );
}
