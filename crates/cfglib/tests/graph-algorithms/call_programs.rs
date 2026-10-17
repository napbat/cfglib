//! `call_graph`, `find_function`, `is_recursive_function` and `propagate_summaries` over a
//! multi-function program.

use std::collections::BTreeSet;

use cfglib::{
    BasicBlock, CallInfo, CallMetadata, Cfg, EdgeKind, FunctionNode, Graph, NodeId, call_graph,
    find_function, is_recursive_function, propagate_summaries,
};

use crate::Lcg;
use crate::digraph::Digraph;

/// A toy instruction set: calls (direct, tail, indirect), observable effects, and plain work.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Op {
    Call(&'static str),
    TailCall(&'static str),
    CallIndirect,
    Effect(&'static str),
    Compute,
}

impl CallInfo for Op {
    type Callee = &'static str;

    fn callee(&self) -> Option<&'static str> {
        match self {
            Self::Call(callee) | Self::TailCall(callee) => Some(callee),
            Self::CallIndirect | Self::Effect(_) | Self::Compute => None,
        }
    }

    fn is_tail_call(&self) -> bool {
        matches!(self, Self::TailCall(_))
    }
}

/// An entry block, a block it branches to, and a block nothing reaches.
fn function(entry: &[Op], branch: &[Op], unreachable: &[Op]) -> Cfg<Op> {
    let mut cfg = Cfg::new();
    let taken = cfg.new_block();
    let orphan = cfg.new_block();
    cfg.block_mut(cfg.entry())
        .instructions_mut()
        .extend_from_slice(entry);
    cfg.block_mut(taken)
        .instructions_mut()
        .extend_from_slice(branch);
    cfg.block_mut(orphan)
        .instructions_mut()
        .extend_from_slice(unreachable);
    cfg.add_edge(cfg.entry(), taken, EdgeKind::ConditionalTrue);
    cfg
}

const NAMES: [&str; 8] = [
    "main", "parse", "lex", "render", "layout", "cleanup", "legacy", "helper",
];

fn program() -> Vec<(&'static str, Cfg<Op>)> {
    use Op::{Call, CallIndirect, Compute, Effect, TailCall};
    vec![
        (
            "main",
            function(
                &[Call("parse"), Call("render"), Call("render")],
                &[TailCall("cleanup")],
                &[Call("legacy")],
            ),
        ),
        (
            "parse",
            function(&[Call("lex"), Effect("alloc")], &[Call("parse")], &[]),
        ),
        ("lex", function(&[Effect("io")], &[], &[])),
        (
            "render",
            function(&[Call("layout"), CallIndirect, Call("missing")], &[], &[]),
        ),
        (
            "layout",
            function(&[Effect("math"), Call("render")], &[], &[]),
        ),
        ("cleanup", function(&[Effect("free")], &[], &[])),
        ("legacy", function(&[Compute], &[], &[])),
        ("helper", function(&[Call("lex")], &[], &[])),
    ]
}

fn build(program: &[(&'static str, Cfg<Op>)]) -> Graph<FunctionNode<&'static str>, CallMetadata> {
    let inputs: Vec<(&'static str, &Cfg<Op>)> =
        program.iter().map(|(name, cfg)| (*name, cfg)).collect();
    call_graph(&inputs)
}

/// The call edges the program text spells out, in the order the instructions appear.
const CALLS: [(&str, &str, bool); 10] = [
    ("main", "parse", false),
    ("main", "render", false),
    ("main", "render", false),
    ("main", "cleanup", true),
    ("main", "legacy", false),
    ("parse", "lex", false),
    ("parse", "parse", false),
    ("render", "layout", false),
    ("layout", "render", false),
    ("helper", "lex", false),
];

fn name(graph: &Graph<FunctionNode<&'static str>, CallMetadata>, node: NodeId) -> &'static str {
    graph.node(node).id
}

#[test]
fn the_call_graph_has_one_node_per_function_in_input_order_and_one_edge_per_resolved_call() {
    let program = program();
    let graph = build(&program);

    let nodes: Vec<&str> = graph.nodes().map(|node| node.id).collect();
    assert_eq!(nodes, NAMES);
    let edges: Vec<(&str, &str, bool)> = graph
        .edges()
        .map(|edge| {
            (
                name(&graph, edge.source()),
                name(&graph, edge.target()),
                edge.payload().is_tail_call,
            )
        })
        .collect();
    assert_eq!(
        edges, CALLS,
        "indirect and unknown calls make no edge; a call in dead code still does"
    );
}

#[test]
fn finding_a_function_returns_its_node_or_nothing() {
    let program = program();
    let graph = build(&program);
    for (index, expected) in NAMES.iter().enumerate() {
        let node = find_function(&graph, expected).expect("every function is in the graph");
        assert_eq!(node.index(), index);
        assert_eq!(graph.node(node).id, *expected);
    }
    assert_eq!(
        find_function(&graph, &"missing"),
        None,
        "a callee outside the program has no node"
    );
}

/// Reachability from a function's callees back to itself, by plain search over the edge list.
fn recursive_by_search(edges: &[(usize, usize)], nodes: usize, function: usize) -> bool {
    let mut seen = vec![false; nodes];
    let mut stack: Vec<usize> = edges
        .iter()
        .filter(|&&(source, _)| source == function)
        .map(|&(_, target)| target)
        .collect();
    while let Some(node) = stack.pop() {
        if std::mem::replace(&mut seen[node], true) {
            continue;
        }
        stack.extend(
            edges
                .iter()
                .filter(|&&(source, _)| source == node)
                .map(|&(_, target)| target),
        );
    }
    seen[function]
}

#[test]
fn direct_and_mutual_recursion_are_recursive_and_everything_else_is_not() {
    let program = program();
    let graph = build(&program);
    let recursive: Vec<&str> = NAMES
        .iter()
        .copied()
        .filter(|candidate| {
            is_recursive_function(&graph, find_function(&graph, candidate).expect("present"))
        })
        .collect();
    assert_eq!(
        recursive,
        ["parse", "render", "layout"],
        "self call, and the render/layout cycle"
    );
}

#[test]
fn recursion_agrees_with_a_search_on_random_call_graphs() {
    for seed in 1..=8 {
        let digraph = Digraph::random(&mut Lcg(seed), 18, 24);
        let graph = random_call_graph(&digraph);
        for function in 0..18 {
            assert_eq!(
                is_recursive_function(&graph, NodeId::from_index(function)),
                recursive_by_search(&digraph.edges, 18, function),
                "seed {seed}, function {function}"
            );
        }
    }
}

fn random_call_graph(digraph: &Digraph) -> Graph<FunctionNode<usize>, CallMetadata> {
    let mut graph = Graph::new();
    let nodes: Vec<NodeId> = (0..digraph.nodes)
        .map(|id| graph.add_node(FunctionNode { id }))
        .collect();
    for &(caller, callee) in &digraph.edges {
        graph.add_edge(
            nodes[caller],
            nodes[callee],
            CallMetadata {
                is_tail_call: false,
            },
        );
    }
    graph
}

/// The effects an instruction list performs.
fn intrinsic_effects(cfg: &Cfg<Op>) -> BTreeSet<&'static str> {
    cfg.blocks()
        .flat_map(BasicBlock::instructions)
        .filter_map(|instruction| match instruction {
            Op::Effect(effect) => Some(*effect),
            _ => None,
        })
        .collect()
}

#[test]
fn summaries_flow_from_callees_to_callers_through_recursion_to_a_fixpoint() {
    let program = program();
    let graph = build(&program);
    let intrinsic: Vec<BTreeSet<&str>> = program
        .iter()
        .map(|(_, cfg)| intrinsic_effects(cfg))
        .collect();

    let summaries = propagate_summaries(&graph, &BTreeSet::new(), |graph, node, summaries| {
        let mut effects = intrinsic[node.index()].clone();
        for callee in graph.successors(node) {
            effects.extend(summaries[callee.index()].iter().copied());
        }
        effects
    });

    let expected: [&[&str]; 8] = [
        &["alloc", "free", "io", "math"],
        &["alloc", "io"],
        &["io"],
        &["math"],
        &["math"],
        &["free"],
        &[],
        &["io"],
    ];
    for (index, effects) in expected.iter().enumerate() {
        let want: BTreeSet<&str> = effects.iter().copied().collect();
        assert_eq!(summaries[index], want, "effects of {}", NAMES[index]);
    }
}

#[test]
fn an_acyclic_program_computes_each_function_once_after_its_callees_are_final() {
    use Op::{Call, Effect};
    let straight_line = [
        (
            "main",
            function(&[Call("parse"), Call("cleanup")], &[], &[]),
        ),
        ("parse", function(&[Call("lex"), Effect("alloc")], &[], &[])),
        ("lex", function(&[Effect("io")], &[], &[])),
        (
            "cleanup",
            function(&[Effect("free"), Call("lex")], &[], &[]),
        ),
    ];
    let graph = build(&straight_line);
    let mut calls = vec![0; 4];
    let mut seen_by_callers: Vec<(usize, usize, BTreeSet<&str>)> = Vec::new();

    let summaries = propagate_summaries(&graph, &BTreeSet::new(), |graph, node, summaries| {
        calls[node.index()] += 1;
        let mut effects = intrinsic_effects(&straight_line[node.index()].1);
        for callee in graph.successors(node) {
            seen_by_callers.push((
                node.index(),
                callee.index(),
                summaries[callee.index()].clone(),
            ));
            effects.extend(summaries[callee.index()].iter().copied());
        }
        effects
    });

    assert_eq!(
        calls,
        [1, 1, 1, 1],
        "no component is cyclic, so nothing is recomputed"
    );
    for (caller, callee, seen) in seen_by_callers {
        assert_eq!(
            seen, summaries[callee],
            "{} read {} before it was final",
            NAMES[caller], NAMES[callee]
        );
    }
    assert_eq!(summaries[0], BTreeSet::from(["alloc", "free", "io"]));
}

#[test]
fn recursive_functions_are_recomputed_until_their_summary_stops_changing() {
    let program = program();
    let graph = build(&program);
    let mut calls = [0; 8];
    let intrinsic: Vec<BTreeSet<&str>> = program
        .iter()
        .map(|(_, cfg)| intrinsic_effects(cfg))
        .collect();
    let summaries = propagate_summaries(&graph, &BTreeSet::new(), |graph, node, summaries| {
        calls[node.index()] += 1;
        let mut effects = intrinsic[node.index()].clone();
        for callee in graph.successors(node) {
            effects.extend(summaries[callee.index()].iter().copied());
        }
        effects
    });
    for recursive in ["parse", "render", "layout"] {
        let node = find_function(&graph, &recursive).expect("present");
        assert!(!summaries[node.index()].is_empty());
        assert!(
            calls[node.index()] >= 2,
            "{recursive}: computed {} time(s)",
            calls[node.index()]
        );
    }
    for plain in ["main", "lex", "cleanup", "legacy", "helper"] {
        let node = find_function(&graph, &plain).expect("present");
        assert_eq!(calls[node.index()], 1, "{plain} is not on a cycle");
    }
}

#[test]
fn summaries_over_random_call_graphs_are_the_reachable_sets_a_search_finds() {
    for seed in 1..=8 {
        let digraph = Digraph::random(&mut Lcg(seed + 20), 20, 30);
        let graph = random_call_graph(&digraph);
        let summaries = propagate_summaries(&graph, &BTreeSet::new(), |graph, node, summaries| {
            let mut reached = BTreeSet::from([node.index()]);
            for callee in graph.successors(node) {
                reached.extend(summaries[callee.index()].iter().copied());
            }
            reached
        });
        for (function, reached) in summaries.iter().enumerate() {
            let expected: BTreeSet<usize> = digraph
                .distances(function, true, None)
                .into_iter()
                .enumerate()
                .filter_map(|(node, distance)| distance.map(|_| node))
                .collect();
            assert_eq!(reached, &expected, "seed {seed}, function {function}");
        }
    }
}

#[test]
fn a_removed_function_keeps_the_bottom_summary_and_recursion_ignores_it() {
    let program = program();
    let mut graph = build(&program);
    let legacy = find_function(&graph, &"legacy").expect("present");
    graph.remove_node(legacy);

    let summaries = propagate_summaries(&graph, &BTreeSet::new(), |graph, node, summaries| {
        let mut reached = BTreeSet::from([name(graph, node)]);
        for callee in graph.successors(node) {
            reached.extend(summaries[callee.index()].iter().copied());
        }
        reached
    });
    assert_eq!(
        summaries.len(),
        graph.node_bound(),
        "sized by the node bound"
    );
    assert!(
        summaries[legacy.index()].is_empty(),
        "the removed slot keeps `bottom`"
    );
    assert!(
        !summaries[0].contains("legacy"),
        "the removed callee leaves its caller's summary"
    );
    assert!(summaries[0].contains("lex"));

    for candidate in NAMES.iter().filter(|&&candidate| candidate != "legacy") {
        let node = find_function(&graph, candidate).expect("present");
        let expected = ["parse", "render", "layout"].contains(candidate);
        assert_eq!(
            is_recursive_function(&graph, node),
            expected,
            "{candidate} after removing legacy"
        );
    }
}
