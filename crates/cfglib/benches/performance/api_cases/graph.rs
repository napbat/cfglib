use std::collections::BTreeSet;
use std::ops::ControlFlow;

use cfglib::{
    CallInfo, Cfg, DominatorTree, EdgeKind, EpochMarks, FoldEnter, Graph, GraphView, MarkScope,
    NodeId, OpenBfsConfig, OpenDfsConfig, OpenFold, OpenFoldConfig, OpenPathsConfig,
    OpenSearchConfig, Rooted, SearchConfig, SearchOrder, SearchScratch, SemanticValidator,
    TraversalDirection, Visit, VisitedPolicy, breadth_first_events, call_graph, canonicalize_loops,
    condensation, condensation_of, depth_first_edges, depth_first_events, depth_first_postorder,
    detect_loops_tagged, find_back_edges_tagged, find_function, follow, follow_path,
    insert_preheader, is_recursive_function, is_reducible, kosaraju_scc, loop_exit_blocks,
    min_label_relaxation, open_breadth_first_events, open_breadth_first_paths,
    open_depth_first_events, open_fold_post_order, open_search, program_dependence_graph,
    propagate_summaries, reachable, reverse_cfg, reverse_postorder, scan_predecessors, search,
    search_with_marks, search_with_scratch, to_view_dot, topological_sort, verify,
    verify_edge_view, verify_view, verify_with, write_view_dot,
};

use super::BenchmarkSuite;
use crate::fixtures::{branchy_cfg, branchy_graph, independent_constants};
use crate::harness::benchmark_case;

const NODE_COUNT: usize = 1_024;

#[derive(Clone, Copy)]
struct CallInst(Option<u32>);

impl CallInfo for CallInst {
    type Callee = u32;

    fn callee(&self) -> Option<Self::Callee> {
        self.0
    }
}

struct NoopValidator;

impl<I, E> SemanticValidator<I, E> for NoopValidator {
    type Error = ();
}

fn chain_graph(node_count: usize) -> Graph<(), ()> {
    let mut graph = Graph::with_capacity(node_count, node_count.saturating_sub(1));
    let nodes: Vec<_> = (0..node_count).map(|_| graph.add_node(())).collect();
    for edge in nodes.windows(2) {
        graph.add_edge(edge[0], edge[1], ());
    }
    graph
}

fn loop_cfg() -> Cfg<u32> {
    let mut cfg = Cfg::new();
    let alternate = cfg.new_block();
    let header = cfg.new_block();
    let body = cfg.new_block();
    let exit = cfg.new_block();
    cfg.add_edge(cfg.entry(), header, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), alternate, EdgeKind::ConditionalFalse);
    cfg.add_edge(alternate, header, EdgeKind::Unconditional);
    cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, header, EdgeKind::Back);
    cfg
}

fn call_cfgs(function_count: usize) -> Vec<Cfg<CallInst>> {
    (0..function_count)
        .map(|index| {
            let mut cfg = Cfg::new();
            let callee = u32::try_from((index + 1) % function_count)
                .expect("benchmark function index must fit in u32");
            cfg.block_mut(cfg.entry()).push(CallInst(Some(callee)));
            cfg
        })
        .collect()
}

fn register_dense_traversals(suite: &mut BenchmarkSuite<'_>) {
    let graph = branchy_graph(NODE_COUNT);
    let root = cfglib::NodeId::from_raw(0);

    benchmark_case!(
        suite,
        "api_depth_first_postorder",
        covers[depth_first_postorder],
        || depth_first_postorder(&graph, root, TraversalDirection::Outgoing),
        |order: &Vec<_>| assert_eq!(order.len(), NODE_COUNT)
    );
    benchmark_case!(
        suite,
        "api_reverse_postorder",
        covers[reverse_postorder],
        || reverse_postorder(&graph, root, TraversalDirection::Outgoing),
        |order: &Vec<_>| assert_eq!(order.len(), NODE_COUNT)
    );
    benchmark_case!(
        suite,
        "api_reachable",
        covers[reachable],
        || reachable(&graph, [root], TraversalDirection::Outgoing),
        |marks: &Vec<bool>| assert!(marks.iter().all(|marked| *marked))
    );
    benchmark_case!(
        suite,
        "api_depth_first_edges",
        covers [depth_first_edges, depth_first_edges_with],
        || depth_first_edges(&graph, root, TraversalDirection::Outgoing),
        |steps: &Vec<_>| assert!(!steps.is_empty())
    );
    benchmark_case!(
        suite,
        "api_breadth_first_events",
        covers[breadth_first_events],
        || {
            let mut events = 0_usize;
            let stopped = breadth_first_events(&graph, root, TraversalDirection::Outgoing, |_| {
                events += 1;
                ControlFlow::<()>::Continue(())
            });
            (stopped, events)
        },
        |(stopped, events)| {
            assert!(stopped.is_none());
            assert!(*events >= NODE_COUNT);
        }
    );
    benchmark_case!(
        suite,
        "api_depth_first_events",
        covers[depth_first_events],
        || {
            let mut events = 0_usize;
            let stopped = depth_first_events(&graph, root, TraversalDirection::Outgoing, |_| {
                events += 1;
                ControlFlow::<()>::Continue(())
            });
            (stopped, events)
        },
        |(stopped, events)| {
            assert!(stopped.is_none());
            assert!(*events >= 2 * NODE_COUNT);
        }
    );
}

fn register_dense_searches(suite: &mut BenchmarkSuite<'_>) {
    let graph = branchy_graph(NODE_COUNT);
    let root = cfglib::NodeId::from_raw(0);

    let config = SearchConfig::new(SearchOrder::BreadthFirst, TraversalDirection::Outgoing);
    benchmark_case!(
        suite,
        "api_search",
        covers[search],
        || {
            let mut visited = 0_usize;
            let stopped = search(&graph, [root], config, |_, _| {
                visited += 1;
                ControlFlow::<(), _>::Continue(Visit::Descend)
            });
            (stopped, visited)
        },
        |(stopped, visited)| {
            assert!(stopped.is_none());
            assert_eq!(*visited, NODE_COUNT);
        }
    );

    let mut marks = EpochMarks::new(NODE_COUNT);
    benchmark_case!(
        suite,
        "api_search_with_marks",
        covers[search_with_marks],
        || {
            let mut visited = 0_usize;
            let stopped = search_with_marks(&graph, [root], config, &mut marks, |_, _| {
                visited += 1;
                ControlFlow::<(), _>::Continue(Visit::Descend)
            });
            (stopped, visited)
        },
        |(stopped, visited)| {
            assert!(stopped.is_none());
            assert_eq!(*visited, NODE_COUNT);
        }
    );

    let mut scratch = SearchScratch::new(NODE_COUNT);
    benchmark_case!(
        suite,
        "api_search_with_scratch",
        covers[search_with_scratch],
        || {
            let mut visited = 0_usize;
            let stopped = search_with_scratch(&graph, [root], config, &mut scratch, |_, _| {
                visited += 1;
                ControlFlow::<(), _>::Continue(Visit::Descend)
            });
            (stopped, visited)
        },
        |(stopped, visited)| {
            assert!(stopped.is_none());
            assert_eq!(*visited, NODE_COUNT);
        }
    );
}

fn register_follow(suite: &mut BenchmarkSuite<'_>) {
    benchmark_case!(
        suite,
        "api_follow",
        covers[follow],
        || follow(0_usize, NODE_COUNT, |node| (*node + 1 < NODE_COUNT)
            .then(|| *node + 1)),
        |last: &usize| assert_eq!(*last, NODE_COUNT - 1)
    );
    benchmark_case!(
        suite,
        "api_follow_path",
        covers[follow_path],
        || follow_path(0_usize, NODE_COUNT, |node| {
            (*node + 1 < NODE_COUNT).then(|| *node + 1)
        }),
        |path: &Vec<usize>| assert_eq!(path.len(), NODE_COUNT)
    );
}

fn register_open_traversals(suite: &mut BenchmarkSuite<'_>) {
    let successors = |node: &usize, out: &mut Vec<usize>| {
        if *node + 1 < NODE_COUNT {
            out.push(*node + 1);
        }
    };

    benchmark_case!(
        suite,
        "api_open_breadth_first_events",
        covers[open_breadth_first_events],
        || {
            let mut visited = 0_usize;
            let stopped =
                open_breadth_first_events([0_usize], OpenBfsConfig::new(), successors, |_| {
                    visited += 1;
                    ControlFlow::<(), _>::Continue(Visit::Descend)
                });
            (stopped, visited)
        },
        |(stopped, visited)| {
            assert!(stopped.is_none());
            assert_eq!(*visited, NODE_COUNT);
        }
    );
    benchmark_case!(
        suite,
        "api_open_depth_first_events",
        covers[open_depth_first_events],
        || {
            let mut events = 0_usize;
            let stopped = open_depth_first_events(
                [0_usize],
                OpenDfsConfig::new(VisitedPolicy::Global),
                successors,
                |_| {
                    events += 1;
                    ControlFlow::<(), _>::Continue(Visit::Descend)
                },
            );
            (stopped, events)
        },
        |(stopped, events)| {
            assert!(stopped.is_none());
            assert_eq!(*events, 2 * NODE_COUNT);
        }
    );
    benchmark_case!(
        suite,
        "api_open_breadth_first_paths",
        covers[open_breadth_first_paths],
        || {
            let mut routes = 0_usize;
            let stopped =
                open_breadth_first_paths([0_usize], OpenPathsConfig::new(), successors, |_| {
                    routes += 1;
                    ControlFlow::<(), _>::Continue(Visit::Descend)
                });
            (stopped, routes)
        },
        |(stopped, routes)| {
            assert!(stopped.is_none());
            assert_eq!(*routes, NODE_COUNT);
        }
    );
    benchmark_case!(
        suite,
        "api_open_fold_post_order",
        covers[open_fold_post_order],
        || open_fold_post_order(&mut SubtreeSize, 0_usize, OpenFoldConfig::new()),
        |size: &Option<usize>| assert_eq!(*size, Some(NODE_COUNT))
    );
    benchmark_case!(
        suite,
        "api_open_search",
        covers[open_search],
        || {
            let mut visited = 0_usize;
            let stopped = open_search(
                [0_usize],
                OpenSearchConfig::new(SearchOrder::DepthFirst),
                successors,
                |_, _| {
                    visited += 1;
                    ControlFlow::<(), _>::Continue(Visit::Descend)
                },
            );
            (stopped, visited)
        },
        |(stopped, visited)| {
            assert!(stopped.is_none());
            assert_eq!(*visited, NODE_COUNT);
        }
    );
}

/// A fold over the same open chain the open-traversal cases walk, answering
/// each subtree's node count.
struct SubtreeSize;

impl OpenFold for SubtreeSize {
    type Node = usize;
    type Mark = usize;
    type Value = usize;
    type Accumulator = usize;

    fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
        if *node + 1 < NODE_COUNT {
            out.push(*node + 1);
        }
    }

    fn mark(&mut self, node: &usize) -> Option<usize> {
        Some(*node)
    }

    fn enter(&mut self, _node: &usize) -> FoldEnter<usize, usize> {
        FoldEnter::Fold {
            accumulator: 1,
            marks: MarkScope::Shared,
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

fn register_graph_algorithms(suite: &mut BenchmarkSuite<'_>) {
    let graph = branchy_graph(NODE_COUNT);
    let root = NodeId::from_raw(0);
    let scanned_target = NodeId::from_index(NODE_COUNT / 2);
    let stored_predecessors: BTreeSet<NodeId> =
        GraphView::predecessors(&graph, scanned_target).collect();

    benchmark_case!(
        suite,
        "api_scan_predecessors",
        covers[scan_predecessors],
        || scan_predecessors(&graph, scanned_target).collect::<BTreeSet<_>>(),
        |scanned: &BTreeSet<NodeId>| assert_eq!(scanned, &stored_predecessors)
    );
    let components = kosaraju_scc(&graph);
    let dag = chain_graph(NODE_COUNT);

    benchmark_case!(
        suite,
        "api_min_label_relaxation",
        covers[min_label_relaxation],
        || min_label_relaxation(
            &graph,
            [(root, 0_usize)],
            TraversalDirection::Outgoing,
            |_, label| Some(*label + 1),
        ),
        |labels: &Vec<Option<usize>>| assert!(labels.iter().all(Option::is_some))
    );
    benchmark_case!(
        suite,
        "api_topological_sort",
        covers[topological_sort],
        || topological_sort(&dag),
        |order: &Option<Vec<_>>| assert_eq!(order.as_ref().map(Vec::len), Some(NODE_COUNT))
    );
    benchmark_case!(
        suite,
        "api_kosaraju_scc",
        covers[kosaraju_scc],
        || kosaraju_scc(&graph),
        |result| assert_eq!(
            result
                .components
                .iter()
                .map(|component| component.nodes.len())
                .sum::<usize>(),
            NODE_COUNT
        )
    );
    benchmark_case!(
        suite,
        "api_condensation",
        covers[condensation],
        || condensation(&graph),
        |result| assert!(topological_sort(result).is_some())
    );
    benchmark_case!(
        suite,
        "api_condensation_of",
        covers[condensation_of],
        || condensation_of(&graph, &components),
        |result| assert_eq!(result.node_count(), components.len())
    );
    benchmark_case!(
        suite,
        "api_propagate_summaries",
        covers[propagate_summaries],
        || propagate_summaries(&graph, &false, |graph, node, summaries| {
            node.index() + 1 == NODE_COUNT
                || graph
                    .successors(node)
                    .any(|successor| summaries[successor.index()])
        }),
        |summaries: &Vec<bool>| assert!(summaries.iter().all(|summary| *summary))
    );
}

fn register_cfg_graphs(suite: &mut BenchmarkSuite<'_>) {
    let cfg = branchy_cfg(NODE_COUNT);
    let graph = branchy_graph(NODE_COUNT);
    let rooted = Rooted::new(&graph, cfglib::NodeId::from_raw(0));
    let dot_graph = chain_graph(128);

    benchmark_case!(
        suite,
        "api_reverse_cfg",
        covers[reverse_cfg],
        || reverse_cfg(&cfg),
        |reversed| assert!(verify(reversed).is_ok())
    );
    benchmark_case!(
        suite,
        "api_program_dependence_graph",
        covers[program_dependence_graph],
        || program_dependence_graph(&independent_constants(NODE_COUNT)),
        |result| assert_eq!(result.node_count(), NODE_COUNT + 1)
    );
    benchmark_case!(
        suite,
        "api_verify",
        covers[verify],
        || verify(&cfg),
        |report| assert!(report.is_ok())
    );
    benchmark_case!(
        suite,
        "api_verify_view",
        covers[verify_view],
        || verify_view(&rooted),
        |report| assert!(report.is_ok())
    );
    benchmark_case!(
        suite,
        "api_verify_edge_view",
        covers[verify_edge_view],
        || verify_edge_view(&rooted),
        |report| assert!(report.is_ok())
    );
    benchmark_case!(
        suite,
        "api_verify_with",
        covers[verify_with],
        || verify_with(&cfg, &NoopValidator),
        |report| assert!(report.is_ok())
    );
    benchmark_case!(
        suite,
        "api_to_view_dot",
        covers[to_view_dot],
        || to_view_dot(&dot_graph, |node| node.index().to_string()),
        |dot: &String| assert!(dot.starts_with("digraph view"))
    );
    benchmark_case!(
        suite,
        "api_write_view_dot",
        covers[write_view_dot],
        || {
            let mut dot = String::new();
            let result = write_view_dot(&dot_graph, &mut dot, |node| node.index().to_string());
            (result, dot)
        },
        |(result, dot)| {
            assert!(result.is_ok());
            assert!(dot.starts_with("digraph view"));
        }
    );
}

fn register_call_graph(suite: &mut BenchmarkSuite<'_>) {
    let cfgs = call_cfgs(256);
    let functions: Vec<_> = cfgs
        .iter()
        .enumerate()
        .map(|(index, cfg)| {
            (
                u32::try_from(index).expect("benchmark function index must fit in u32"),
                cfg,
            )
        })
        .collect();
    let graph = call_graph(&functions);
    let first = cfglib::NodeId::from_raw(0);

    benchmark_case!(
        suite,
        "api_call_graph",
        covers[call_graph],
        || call_graph(&functions),
        |result| {
            assert_eq!(result.node_count(), functions.len());
            assert_eq!(result.edge_count(), functions.len());
        }
    );
    benchmark_case!(
        suite,
        "api_find_function",
        covers[find_function],
        || find_function(&graph, &128_u32),
        |result| assert_eq!(*result, Some(cfglib::NodeId::from_index(128)))
    );
    benchmark_case!(
        suite,
        "api_is_recursive_function",
        covers[is_recursive_function],
        || is_recursive_function(&graph, first),
        |recursive| assert!(*recursive)
    );
}

fn register_loop_structure(suite: &mut BenchmarkSuite<'_>) {
    let cfg = loop_cfg();
    let dominators = DominatorTree::compute(&cfg);
    let loops = detect_loops_tagged(&cfg, &dominators);
    let natural_loop = loops
        .first()
        .expect("benchmark loop fixture must contain a loop")
        .clone();

    benchmark_case!(
        suite,
        "api_find_back_edges_tagged",
        covers[find_back_edges_tagged],
        || find_back_edges_tagged(&cfg, &dominators),
        |edges: &Vec<_>| assert_eq!(edges.len(), 1)
    );
    benchmark_case!(
        suite,
        "api_detect_loops_tagged",
        covers[detect_loops_tagged],
        || detect_loops_tagged(&cfg, &dominators),
        |found: &Vec<_>| assert_eq!(found.len(), 1)
    );
    benchmark_case!(
        suite,
        "api_is_reducible",
        covers[is_reducible],
        || is_reducible(&cfg, &dominators),
        |result| assert!(*result)
    );
    benchmark_case!(
        suite,
        "api_loop_exit_blocks",
        covers[loop_exit_blocks],
        || loop_exit_blocks(&cfg, &natural_loop),
        |exits| assert_eq!(exits.len(), 1)
    );
    benchmark_case!(
        suite,
        "api_insert_preheader",
        covers[insert_preheader],
        || {
            let mut candidate = cfg.clone();
            let preheader = insert_preheader(&mut candidate, &natural_loop);
            (candidate, preheader)
        },
        |(candidate, preheader)| {
            assert!(preheader.is_some());
            assert!(verify(candidate).is_ok());
        }
    );
    benchmark_case!(
        suite,
        "api_canonicalize_loops",
        covers[canonicalize_loops],
        || {
            let mut candidate = cfg.clone();
            let loops = canonicalize_loops(&mut candidate, &dominators);
            (candidate, loops)
        },
        |(candidate, loops)| {
            assert_eq!(loops.len(), 1);
            assert!(verify(candidate).is_ok());
        }
    );
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    register_dense_traversals(suite);
    register_dense_searches(suite);
    register_follow(suite);
    register_open_traversals(suite);
    register_graph_algorithms(suite);
    register_cfg_graphs(suite);
    register_call_graph(suite);
    register_loop_structure(suite);
}
