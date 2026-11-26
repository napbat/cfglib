use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::ops::ControlFlow;

use cfglib::{BlockId, canonical_hash, canonical_order, multiset_fingerprint, wl_colors};
use cfglib::{
    Cfg, Direction, DominatorTree, EdgeKind, Flow, FlowControl, FlowEffect, FoldEnter, Graph,
    GraphView, MarkScope, NodeId, OpenFold, OpenFoldConfig, Rooted, SearchConfig, SearchOrder,
    SearchScratch, SemanticValidator, Visit, call_graph, canonicalize_loops, condensation,
    condensation_of, depth_first_edges, depth_first_events, depth_first_postorder,
    detect_loops_tagged, find_back_edges_tagged, find_function, follow, follow_path,
    insert_preheader, is_recursive_function, is_reducible, kosaraju_scc, loop_exit_blocks,
    min_label_relaxation, open_breadth_first_events, open_breadth_first_paths,
    open_depth_first_events, open_fold_post_order, open_search, propagate_summaries, reachable,
    reverse_postorder, scan_predecessors, search, search_in, to_dot, to_text, topological_sort,
    topological_sort_or_cycle, verify, verify_edge_view, verify_view, verify_with, write_dot,
    write_text,
};
use cfglib::{
    DotEdgeAttributes, DotStyle, EdgeId, FunctionIndex, TextStyle, call_graph_indexed,
    control_flow_edge_attributes, display_label, no_label, parse_cfg_text, parse_text,
    plain_edge_attributes, propagate_summaries_widened, recursive_functions,
};

use super::BenchmarkSuite;
use crate::fixtures::{branchy_cfg, branchy_graph, fixture_u32};
use crate::harness::benchmark_case;

const NODE_COUNT: usize = 1_024;
/// Functions in the mutually recursive ring of the summary case.
const RING_FUNCTIONS: usize = 1_000;
/// Blocks of the CFG the fingerprint cases hash.
const FINGERPRINT_BLOCKS: usize = 10_000;

#[derive(Clone, Copy)]
pub(super) struct CallInst(Option<u32>);

impl FlowControl for CallInst {
    type Target = u32;
    type CaseKey = ();

    fn flow_effect(&self) -> FlowEffect<u32, ()> {
        FlowEffect::Flow(self.0.map_or(Flow::Next, |target| Flow::Call { target }))
    }
}

struct NoopValidator;

impl<I, E> SemanticValidator<I, E> for NoopValidator {
    type Error = ();
}

/// A labeled chain, so the output writers have text to render and the
/// parsers have text to read.
fn chain_graph(node_count: usize) -> Graph<String, String> {
    let mut graph = Graph::with_capacity(node_count, node_count.saturating_sub(1));
    let nodes: Vec<_> = (0..node_count)
        .map(|index| graph.add_node(format!("node {index}")))
        .collect();
    for (index, edge) in nodes.windows(2).enumerate() {
        graph.add_edge(edge[0], edge[1], format!("edge {index}"));
    }
    graph
}

/// One CFG in the line-oriented text form, for the reader to read.
fn chain_cfg_text(block_count: usize) -> String {
    let mut text = String::from("entry bb0\n");
    for index in 0..block_count {
        writeln!(text, "bb{index}:\n    op {index}").unwrap();
    }
    for index in 1..block_count {
        writeln!(text, "bb{} -> bb{index} fallthrough", index - 1).unwrap();
    }
    text
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
            let callee = fixture_u32((index + 1) % function_count);
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
        || depth_first_postorder(&graph, root, Direction::Forward),
        |order: &Vec<_>| assert_eq!(order.len(), NODE_COUNT)
    );
    benchmark_case!(
        suite,
        "api_reverse_postorder",
        covers[reverse_postorder],
        || reverse_postorder(&graph, root, Direction::Forward),
        |order: &Vec<_>| assert_eq!(order.len(), NODE_COUNT)
    );
    benchmark_case!(
        suite,
        "api_reachable",
        covers[reachable],
        || reachable(&graph, [root], Direction::Forward),
        |marks: &Vec<bool>| assert!(marks.iter().all(|marked| *marked))
    );
    benchmark_case!(
        suite,
        "api_depth_first_edges",
        covers[depth_first_edges],
        || depth_first_edges(&graph, root, Direction::Forward),
        |steps: &Vec<_>| assert_ne!(steps.as_slice(), [])
    );
    benchmark_case!(
        suite,
        "api_depth_first_events",
        covers[depth_first_events],
        || {
            let mut events = 0_usize;
            let stopped = depth_first_events(&graph, root, Direction::Forward, |_| {
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

    let config = SearchConfig {
        order: SearchOrder::BreadthFirst,
        ..SearchConfig::default()
    };
    benchmark_case!(
        suite,
        "api_search",
        covers[search],
        || {
            let mut visited = 0_usize;
            let stopped = search(&graph, [root], Direction::Forward, config, |_, _| {
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
        "api_search_depth_first",
        || {
            let mut visited = 0_usize;
            let depth_first = SearchConfig::default();
            let stopped = search(&graph, [root], Direction::Forward, depth_first, |_, _| {
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

    let mut scratch = SearchScratch::default();
    benchmark_case!(
        suite,
        "api_search_in",
        covers[search_in],
        || {
            let mut visited = 0_usize;
            let forward = Direction::Forward;
            let stopped = search_in(&mut scratch, &graph, [root], forward, config, |_, _| {
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
            let stopped = open_breadth_first_events([0_usize], None, successors, |_| {
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
            let stopped =
                open_depth_first_events([0_usize], SearchConfig::default(), successors, |_| {
                    events += 1;
                    ControlFlow::<(), _>::Continue(Visit::Descend)
                });
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
            let stopped = open_breadth_first_paths([0_usize], None, successors, |_| {
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
            let stopped = open_search([0_usize], SearchConfig::default(), successors, |_, _| {
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
        || min_label_relaxation(&graph, [(root, 0_usize)], Direction::Forward, |_, label| {
            Some(*label + 1)
        },),
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
        "api_topological_sort_or_cycle",
        covers[topological_sort_or_cycle],
        || topological_sort_or_cycle(&graph),
        |result: &Result<Vec<NodeId>, Vec<NodeId>>| {
            let cycle = result
                .as_ref()
                .expect_err("the branchy graph's back edges close cycles");
            assert_ne!(*cycle, []);
            for (position, &node) in cycle.iter().enumerate() {
                let next = cycle[(position + 1) % cycle.len()];
                assert!(GraphView::successors(&graph, node).any(|successor| successor == next));
            }
        }
    );
    benchmark_case!(
        suite,
        "api_kosaraju_scc",
        covers[kosaraju_scc],
        || kosaraju_scc(&graph),
        |result| assert_eq!(
            result.components().map(<[_]>::len).sum::<usize>(),
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
}

/// The output writers and the readers that undo them.
fn register_output_formats(suite: &mut BenchmarkSuite<'_>) {
    let cfg = branchy_cfg(NODE_COUNT);
    let output_graph = chain_graph(128);
    let graph_text = output_graph.to_text();
    let cfg_text = chain_cfg_text(128);

    benchmark_case!(
        suite,
        "api_to_dot",
        covers[to_dot, display_label, plain_edge_attributes],
        || to_dot(
            &output_graph,
            &DotStyle::new(display_label, plain_edge_attributes)
        ),
        |dot: &String| assert!(dot.starts_with("digraph view"))
    );
    benchmark_case!(
        suite,
        "api_write_dot",
        covers[write_dot, no_label],
        || {
            let style = DotStyle::new(no_label, |_, label: &String| DotEdgeAttributes {
                label: std::borrow::Cow::Borrowed(label),
                ..DotEdgeAttributes::plain()
            });
            let mut dot = String::new();
            let result = write_dot(&output_graph, &mut dot, &style);
            (result, dot)
        },
        |(result, dot)| {
            assert!(result.is_ok());
            assert!(dot.starts_with("digraph view"));
        }
    );
    benchmark_case!(
        suite,
        "api_control_flow_edge_attributes",
        covers[control_flow_edge_attributes],
        || cfg
            .edge_ids()
            .filter(|&id| control_flow_edge_attributes(id, &cfg[id]).color.is_some())
            .count(),
        |styled: &usize| assert_eq!(*styled, cfg.edge_count())
    );
    benchmark_case!(
        suite,
        "api_to_text",
        covers[to_text],
        || to_text(
            &output_graph,
            &TextStyle::new(
                |_, label: &String| std::borrow::Cow::Borrowed(label),
                no_label
            )
        ),
        |text: &String| assert!(text.starts_with("n0 node 0"))
    );
    benchmark_case!(
        suite,
        "api_write_text",
        covers[write_text],
        || {
            let mut text = String::new();
            let result = write_text(&output_graph, &mut text, &TextStyle::plain());
            (result, text)
        },
        |(result, text)| {
            assert!(result.is_ok());
            assert!(text.starts_with("n0\n"));
        }
    );
    benchmark_case!(
        suite,
        "api_parse_text",
        covers[parse_text],
        || parse_text(&graph_text),
        |parsed: &Result<Graph<String, String>, _>| assert_eq!(
            parsed.as_ref().unwrap().node_count(),
            output_graph.node_count()
        )
    );
    benchmark_case!(
        suite,
        "api_parse_cfg_text",
        covers[parse_cfg_text],
        || parse_cfg_text(&cfg_text, |line| Ok::<_, String>(line.to_owned())),
        |parsed: &Result<Cfg<String>, _>| assert_eq!(parsed.as_ref().unwrap().block_count(), 128)
    );
}

fn register_call_graph(suite: &mut BenchmarkSuite<'_>) {
    let cfgs = call_cfgs(256);
    let functions: Vec<_> = cfgs
        .iter()
        .enumerate()
        .map(|(index, cfg)| (fixture_u32(index), cfg))
        .collect();
    let graph = call_graph(&functions);
    let first = cfglib::NodeId::from_raw(0);
    let index = FunctionIndex::compute(&graph);

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
        "api_call_graph_indexed",
        covers[call_graph_indexed],
        || call_graph_indexed(&functions),
        |(result, index)| {
            assert_eq!(result.node_count(), functions.len());
            assert_eq!(index.get(&128_u32), Some(cfglib::NodeId::from_index(128)));
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
        "api_function_index_get",
        || index.get(&128_u32),
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

fn register_call_tree(suite: &mut BenchmarkSuite<'_>) {
    // A call tree in which every 64th function calls back its caller: the
    // per-function loop a consumer runs to mark the recursive functions.
    let tree_cfgs = call_tree_cfgs(2_048);
    let tree_functions: Vec<_> = tree_cfgs
        .iter()
        .enumerate()
        .map(|(index, cfg)| (fixture_u32(index), cfg))
        .collect();
    let tree = call_graph(&tree_functions);
    let components = kosaraju_scc(&tree);
    let recursive = tree
        .node_ids()
        .filter(|&function| components.component(function).is_cyclic(&tree))
        .count();
    benchmark_case!(
        suite,
        "call_tree_is_recursive_every_function",
        || tree
            .node_ids()
            .filter(|&function| is_recursive_function(&tree, function))
            .count(),
        |count: &usize| assert_eq!(*count, recursive)
    );
    benchmark_case!(
        suite,
        "api_recursive_functions",
        covers[recursive_functions],
        || recursive_functions(&tree),
        |marked: &Vec<bool>| assert_eq!(marked.iter().filter(|&&marked| marked).count(), recursive)
    );

    // Each function's summary is the set of `id % 64` over everything it
    // reaches: a finite-height domain that every recursive cluster of the
    // call tree has to iterate.
    let reach_mask = |graph: &cfglib::Graph<cfglib::FunctionNode<u32>, cfglib::CallMetadata>,
                      node: cfglib::NodeId,
                      summaries: &[u64]| {
        graph
            .successors(node)
            .fold(1_u64 << (node.index() % 64), |mask, callee| {
                mask | summaries[callee.index()]
            })
    };
    let tree_masks = propagate_summaries(&tree, &0, reach_mask);
    benchmark_case!(
        suite,
        "call_tree_propagate_summaries",
        || propagate_summaries(&tree, &0, reach_mask),
        |masks: &Vec<u64>| assert_eq!(*masks, tree_masks)
    );
}

fn register_recursive_ring(suite: &mut BenchmarkSuite<'_>) {
    // One ring of mutually recursive functions whose last member holds the
    // only seed, so the seed has to travel against the id order.
    let mut ring = Graph::<(), ()>::new();
    let members: Vec<_> = (0..RING_FUNCTIONS).map(|_| ring.add_node(())).collect();
    for (index, &caller) in members.iter().enumerate() {
        ring.add_edge(caller, members[(index + 1) % RING_FUNCTIONS], ());
    }
    benchmark_case!(
        suite,
        "ring_propagate_summaries_seed_at_last",
        || propagate_summaries(&ring, &false, |graph, node, summaries| {
            node.index() + 1 == RING_FUNCTIONS
                || graph
                    .successors(node)
                    .any(|callee| summaries[callee.index()])
        }),
        |summaries: &Vec<bool>| assert!(summaries.iter().all(|summary| *summary))
    );
    // The deepest stack over the same ring grows forever. Widening at the
    // member that closes the ring makes every depth unbounded in about two
    // passes; the oracle bounds the `compute` calls the case returns.
    benchmark_case!(
        suite,
        "api_propagate_summaries_widened",
        covers[propagate_summaries_widened],
        || {
            let mut calls = 0_usize;
            let depths = propagate_summaries_widened(
                &ring,
                &None,
                |graph, node, depths: &[Option<u32>]| {
                    calls += 1;
                    let deepest = graph
                        .successors(node)
                        .filter_map(|callee| depths[callee.index()])
                        .max();
                    Some(deepest.unwrap_or(0).saturating_add(1))
                },
                |_, previous, computed| match (*previous, *computed) {
                    (Some(previous), Some(computed)) if computed > previous => Some(u32::MAX),
                    (previous, computed) => previous.max(computed),
                },
            );
            (depths, calls)
        },
        |(depths, calls): &(Vec<Option<u32>>, usize)| {
            assert!(depths.iter().all(|depth| *depth == Some(u32::MAX)));
            assert!(*calls <= 3 * RING_FUNCTIONS, "{calls} compute calls");
        }
    );
}

/// Function `i` calls `2i + 1` and `2i + 2`; every 64th function also calls
/// its caller back.
pub(super) fn call_tree_cfgs(function_count: usize) -> Vec<Cfg<CallInst>> {
    (0..function_count)
        .map(|index| {
            let mut cfg = Cfg::new();
            let entry = cfg.entry();
            for callee in [2 * index + 1, 2 * index + 2] {
                if callee < function_count {
                    cfg.block_mut(entry)
                        .push(CallInst(Some(fixture_u32(callee))));
                }
            }
            if index > 0 && index % 64 == 0 {
                cfg.block_mut(entry)
                    .push(CallInst(Some(fixture_u32((index - 1) / 2))));
            }
            cfg
        })
        .collect()
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

/// `cfg`'s edge kinds as keys.
fn kind_key(cfg: &Cfg<u32>) -> impl Fn(EdgeId) -> u64 + '_ {
    |edge| cfg[edge].kind() as u64
}

/// The wiring of `branchy_cfg(block_count)` with every block but the entry
/// numbered in reverse, and the block that plays the part of each block of
/// the original, by its index.
fn renumbered_branchy_cfg(block_count: usize) -> (Cfg<u32>, Vec<BlockId>) {
    let mut cfg = Cfg::new();
    let mut blocks: Vec<BlockId> = (1..block_count).map(|_| cfg.new_block()).collect();
    blocks.push(cfg.entry());
    blocks.reverse();
    let edges = branchy_cfg(block_count);
    for edge in edges.edge_ids() {
        let edge = edges.edge(edge);
        cfg.add_edge(
            blocks[edge.source().index()],
            blocks[edge.target().index()],
            edge.data().kind(),
        );
    }
    (cfg, blocks)
}

fn register_fingerprints(suite: &mut BenchmarkSuite<'_>) {
    // Each oracle checks the measured call against the same call on a
    // renumbered copy, which the fingerprints must not see.
    let cfg = branchy_cfg(FINGERPRINT_BLOCKS);
    let (copy, part) = renumbered_branchy_cfg(FINGERPRINT_BLOCKS);
    let copied_colors = wl_colors(&copy, 3, |_| 0, kind_key(&copy));
    let copied_fingerprint = multiset_fingerprint(copied_colors.iter().copied());
    let copied_order = canonical_order(&copy, kind_key(&copy));
    let copied_hash = canonical_hash(&copy, |_| 0, kind_key(&copy));
    let colors = wl_colors(&cfg, 3, |_| 0, kind_key(&cfg));

    benchmark_case!(
        suite,
        "api_wl_colors",
        covers[wl_colors],
        || wl_colors(&cfg, 3, |_| 0, kind_key(&cfg)),
        |colors: &Vec<u64>| {
            for (index, color) in colors.iter().enumerate() {
                assert_eq!(*color, copied_colors[part[index].index()]);
            }
        }
    );
    benchmark_case!(
        suite,
        "api_multiset_fingerprint",
        covers[multiset_fingerprint],
        || multiset_fingerprint(colors.iter().copied()),
        |fingerprint: &u64| assert_eq!(*fingerprint, copied_fingerprint)
    );
    benchmark_case!(
        suite,
        "api_canonical_order",
        covers[canonical_order],
        || canonical_order(&cfg, kind_key(&cfg)),
        |order: &Vec<BlockId>| {
            assert_eq!(order.len(), FINGERPRINT_BLOCKS);
            assert!(
                order
                    .iter()
                    .map(|block| part[block.index()])
                    .eq(copied_order.iter().copied())
            );
        }
    );
    benchmark_case!(
        suite,
        "api_canonical_hash",
        covers[canonical_hash],
        || canonical_hash(&cfg, |_| 0, kind_key(&cfg)),
        |hash: &u64| assert_eq!(*hash, copied_hash)
    );
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    register_dense_traversals(suite);
    register_dense_searches(suite);
    register_follow(suite);
    register_open_traversals(suite);
    register_graph_algorithms(suite);
    register_cfg_graphs(suite);
    register_output_formats(suite);
    register_call_graph(suite);
    register_call_tree(suite);
    register_recursive_ring(suite);
    register_loop_structure(suite);
    register_fingerprints(suite);
}
