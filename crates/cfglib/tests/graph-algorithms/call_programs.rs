//! `call_graph`, `call_graph_indexed`, `FunctionIndex`, `find_function`,
//! `is_recursive_function`, `recursive_functions` and `propagate_summaries`, with and without
//! widening, over a multi-function program.

use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;

use cfglib::{
    BasicBlock, CallInfo, CallMetadata, Cfg, EdgeKind, FunctionIndex, FunctionNode, Graph, NodeId,
    SummaryConfig, SummaryProblem, call_graph, call_graph_indexed, find_function,
    is_recursive_function, propagate_summaries, recursive_functions,
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

/// Every key a lookup can meet: each function, a callee outside the program, and a key the
/// input lists twice.
#[test]
fn the_function_index_answers_what_find_function_answers() {
    let program = program();
    let again = function(&[Op::Call("helper")], &[], &[]);
    let mut inputs: Vec<(&'static str, &Cfg<Op>)> =
        program.iter().map(|(name, cfg)| (*name, cfg)).collect();
    inputs.insert(3, ("parse", &again));
    inputs.push(("parse", &again));

    let (graph, index) = call_graph_indexed(&inputs);
    assert_eq!(
        index,
        FunctionIndex::compute(&graph),
        "the index the construction returns is the index of its graph"
    );
    let shape = |graph: &Graph<FunctionNode<&'static str>, CallMetadata>| {
        let nodes: Vec<&str> = graph.nodes().map(|node| node.id).collect();
        let edges: Vec<(NodeId, NodeId, bool)> = graph
            .edges()
            .map(|edge| (edge.source(), edge.target(), edge.payload().is_tail_call))
            .collect();
        (nodes, edges)
    };
    assert_eq!(
        shape(&graph),
        shape(&call_graph(&inputs)),
        "the graph is the one `call_graph` builds"
    );

    for key in NAMES.iter().chain(&["missing"]) {
        assert_eq!(index.get(key), find_function(&graph, key), "{key}");
    }
    assert_eq!(
        index.get(&"parse").map(NodeId::index),
        Some(1),
        "a key listed three times keeps its first position"
    );
    assert_eq!(index.get(&"missing"), None);
}

/// A graph built by hand can repeat an identity and lose nodes: the lowest live node wins, as
/// it does for the scan.
#[test]
fn the_function_index_keeps_the_lowest_live_node_of_a_repeated_identity() {
    let mut graph = Graph::<FunctionNode<usize>, CallMetadata>::new();
    let nodes: Vec<NodeId> = [3, 1, 3, 2, 1, 3]
        .into_iter()
        .map(|id| graph.add_node(FunctionNode { id }))
        .collect();
    graph.remove_node(nodes[0]);
    graph.remove_node(nodes[3]);

    let index = FunctionIndex::compute(&graph);
    for key in 0..=4 {
        assert_eq!(index.get(&key), find_function(&graph, &key), "key {key}");
    }
    assert_eq!(index.get(&3), Some(nodes[2]));
    assert_eq!(index.get(&1), Some(nodes[1]));
    assert_eq!(index.get(&2), None, "the only node of 2 is removed");

    let empty = Graph::<FunctionNode<usize>, CallMetadata>::new();
    assert_eq!(FunctionIndex::compute(&empty).get(&0), None);
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

#[test]
fn the_batch_classification_marks_the_self_call_and_the_cycle() {
    let program = program();
    let graph = build(&program);
    let recursive = recursive_functions(&graph);
    assert_eq!(recursive.len(), graph.node_bound());
    let marked: Vec<&str> = NAMES
        .iter()
        .enumerate()
        .filter_map(|(index, &name)| recursive[index].then_some(name))
        .collect();
    assert_eq!(marked, ["parse", "render", "layout"]);
    assert_eq!(
        recursive_functions(&Graph::<(), ()>::new()),
        Vec::<bool>::new()
    );
}

/// Random call graphs with parallel edges, self-calls, repeated identities and removed slots:
/// the batch forms answer what the per-function queries answer, and the classification agrees
/// with a search over the live edges.
#[test]
fn the_batch_forms_match_the_per_function_queries_on_random_call_graphs() {
    for seed in 1..=60 {
        let mut rng = Lcg(seed);
        let nodes = 1 + rng.below(24);
        let edges = rng.below(2 * nodes + 1);
        let digraph = Digraph::random(&mut rng, nodes, edges);
        let mut graph = Graph::new();
        let ids: Vec<NodeId> = (0..nodes)
            .map(|_| {
                graph.add_node(FunctionNode {
                    id: rng.below(nodes),
                })
            })
            .collect();
        for &(caller, callee) in &digraph.edges {
            graph.add_edge(
                ids[caller],
                ids[callee],
                CallMetadata {
                    is_tail_call: false,
                },
            );
        }
        let removed: Vec<bool> = (0..nodes).map(|_| rng.below(6) == 0).collect();
        for (function, &node) in ids.iter().enumerate() {
            if removed[function] {
                graph.remove_node(node);
            }
        }
        let live_edges: Vec<(usize, usize)> = digraph
            .edges
            .iter()
            .copied()
            .filter(|&(caller, callee)| !removed[caller] && !removed[callee])
            .collect();

        let recursive = recursive_functions(&graph);
        assert_eq!(recursive.len(), graph.node_bound(), "seed {seed}");
        for (function, &node) in ids.iter().enumerate() {
            if removed[function] {
                assert!(!recursive[function], "seed {seed}, removed {function}");
                continue;
            }
            assert_eq!(
                recursive[function],
                is_recursive_function(&graph, node),
                "seed {seed}, function {function}"
            );
            assert_eq!(
                recursive[function],
                recursive_by_search(&live_edges, nodes, function),
                "seed {seed}, function {function}"
            );
        }

        let index = FunctionIndex::compute(&graph);
        for key in 0..=nodes {
            assert_eq!(
                index.get(&key),
                find_function(&graph, &key),
                "seed {seed}, key {key}"
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

/// The effects each function performs itself or through a callee, recording every `compute`
/// call and every callee summary it read.
struct Effects {
    /// The effects of each function's own instructions.
    intrinsic: Vec<BTreeSet<&'static str>>,
    /// How many times `compute` ran for each function.
    calls: RefCell<Vec<usize>>,
    /// Every callee summary a caller read: the caller, the callee and the summary.
    reads: RefCell<Vec<(usize, usize, BTreeSet<&'static str>)>>,
}

impl Effects {
    fn of(program: &[(&'static str, Cfg<Op>)]) -> Self {
        Self {
            intrinsic: program
                .iter()
                .map(|(_, cfg)| intrinsic_effects(cfg))
                .collect(),
            calls: RefCell::new(vec![0; program.len()]),
            reads: RefCell::default(),
        }
    }
}

impl SummaryProblem<Graph<FunctionNode<&'static str>, CallMetadata>> for Effects {
    type Summary = BTreeSet<&'static str>;

    fn bottom(&self, _: &Graph<FunctionNode<&'static str>, CallMetadata>) -> Self::Summary {
        BTreeSet::new()
    }

    fn compute(
        &self,
        graph: &Graph<FunctionNode<&'static str>, CallMetadata>,
        node: NodeId,
        summaries: &[Self::Summary],
    ) -> Self::Summary {
        self.calls.borrow_mut()[node.index()] += 1;
        let mut effects = self.intrinsic[node.index()].clone();
        for callee in graph.successors(node) {
            let summary = &summaries[callee.index()];
            self.reads
                .borrow_mut()
                .push((node.index(), callee.index(), summary.clone()));
            effects.extend(summary.iter().copied());
        }
        effects
    }
}

#[test]
fn summaries_flow_from_callees_to_callers_through_recursion_to_a_fixpoint() {
    let program = program();
    let graph = build(&program);
    let summaries = propagate_summaries(&graph, &Effects::of(&program), SummaryConfig::default());

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
    let effects = Effects::of(&straight_line);
    let summaries = propagate_summaries(&graph, &effects, SummaryConfig::default());

    assert_eq!(
        effects.calls.into_inner(),
        [1, 1, 1, 1],
        "no component is cyclic, so nothing is recomputed"
    );
    for (caller, callee, seen) in effects.reads.into_inner() {
        assert_eq!(
            seen, summaries[callee],
            "{} read {} before it was final",
            NAMES[caller], NAMES[callee]
        );
    }
    assert_eq!(summaries[0], BTreeSet::from(["alloc", "free", "io"]));
}

#[test]
fn recursive_functions_settle_on_a_summary_that_no_longer_changes() {
    let program = program();
    let graph = build(&program);
    let effects = Effects::of(&program);
    let summaries = propagate_summaries(&graph, &effects, SummaryConfig::default());
    let calls = effects.calls.borrow().clone();
    // A recursive cluster is iterated, so one of its members runs again once
    // a callee it read has changed; every member ends on a summary that
    // computing it once more would not change.
    let cluster = ["render", "layout"].map(|name| find_function(&graph, &name).expect("present"));
    assert!(cluster.iter().any(|node| calls[node.index()] >= 2));
    for recursive in ["parse", "render", "layout"] {
        let node = find_function(&graph, &recursive).expect("present");
        assert!(!summaries[node.index()].is_empty());
        assert_eq!(
            effects.compute(&graph, node, &summaries),
            summaries[node.index()],
            "{recursive} is stable"
        );
    }
    for plain in ["main", "lex", "cleanup", "legacy", "helper"] {
        let node = find_function(&graph, &plain).expect("present");
        assert_eq!(calls[node.index()], 1, "{plain} is not on a cycle");
    }
}

/// Frame sizes in bytes, in the order of `NAMES`.
const FRAMES: [u64; 8] = [64, 48, 16, 96, 32, 8, 24, 40];

/// The deepest stack a call reaches: the function's frame plus the deepest stack of its
/// callees, where a callee not computed yet (`None`) adds nothing. `u64::MAX` is unbounded.
fn deepest_stack(
    graph: &Graph<FunctionNode<&'static str>, CallMetadata>,
    node: NodeId,
    depths: &[Option<u64>],
) -> u64 {
    let deepest = graph
        .successors(node)
        .filter_map(|callee| depths[callee.index()])
        .max();
    FRAMES[node.index()].saturating_add(deepest.unwrap_or(0))
}

/// The deepest stack of each function, as `deepest_stack` gives it, counting the `compute`
/// calls. Past `budget` calls every summary freezes, so that a run that would climb forever ends.
struct StackDepths {
    budget: usize,
    calls: Cell<usize>,
}

impl StackDepths {
    fn with_budget(budget: usize) -> Self {
        Self {
            budget,
            calls: Cell::new(0),
        }
    }
}

impl SummaryProblem<Graph<FunctionNode<&'static str>, CallMetadata>> for StackDepths {
    type Summary = Option<u64>;

    fn bottom(&self, _: &Graph<FunctionNode<&'static str>, CallMetadata>) -> Option<u64> {
        None
    }

    fn compute(
        &self,
        graph: &Graph<FunctionNode<&'static str>, CallMetadata>,
        node: NodeId,
        depths: &[Option<u64>],
    ) -> Option<u64> {
        self.calls.set(self.calls.get() + 1);
        if self.calls.get() > self.budget {
            return depths[node.index()];
        }
        Some(deepest_stack(graph, node, depths))
    }

    /// Interval widening on the upper end: a depth that grows again is unbounded.
    fn widen(
        &self,
        _: &Graph<FunctionNode<&'static str>, CallMetadata>,
        _: NodeId,
        previous: &Option<u64>,
        computed: &Option<u64>,
    ) -> Option<u64> {
        match (*previous, *computed) {
            (Some(previous), Some(computed)) if computed > previous => Some(u64::MAX),
            (previous, computed) => previous.max(computed),
        }
    }
}

/// The stack depth of a recursive function grows with every turn of the recursion, so
/// without widening the iteration still climbs after any budget. Widening at the recursion
/// boundaries ends it: the recursive functions and their callers are unbounded, every other
/// function is exact, and no function's depth is below what its callees' depths give.
#[test]
fn stack_depths_climb_forever_over_recursion_and_converge_when_widened() {
    const BUDGET: usize = 10_000;
    let program = program();
    let graph = build(&program);

    let unwidened = StackDepths::with_budget(BUDGET);
    let _ = propagate_summaries(&graph, &unwidened, SummaryConfig::default());
    let calls = unwidened.calls.get();
    assert!(
        calls > BUDGET,
        "unwidened, the depths still change after {BUDGET} computes"
    );

    let widened = StackDepths::with_budget(usize::MAX);
    let widening = SummaryConfig {
        widening: true,
        narrowing: false,
    };
    let depths = propagate_summaries(&graph, &widened, widening);
    let calls = widened.calls.get();
    assert!(calls <= 2 * NAMES.len(), "{calls} computes");
    let unbounded = u64::MAX;
    assert_eq!(
        depths,
        [
            unbounded,
            unbounded,
            16,
            unbounded,
            unbounded,
            8,
            24,
            40 + 16
        ]
        .map(Some),
        "{NAMES:?}"
    );
    for node in graph.node_ids() {
        assert!(
            Some(deepest_stack(&graph, node, &depths)) <= depths[node.index()],
            "{} bounds what its callees give",
            name(&graph, node)
        );
    }
}

/// The functions each function reaches through calls, itself included, by their ids.
struct Reached;

impl<C: Copy + Ord> SummaryProblem<Graph<FunctionNode<C>, CallMetadata>> for Reached {
    type Summary = BTreeSet<C>;

    fn bottom(&self, _: &Graph<FunctionNode<C>, CallMetadata>) -> BTreeSet<C> {
        BTreeSet::new()
    }

    fn compute(
        &self,
        graph: &Graph<FunctionNode<C>, CallMetadata>,
        node: NodeId,
        summaries: &[BTreeSet<C>],
    ) -> BTreeSet<C> {
        let mut reached = BTreeSet::from([graph.node(node).id]);
        for callee in graph.successors(node) {
            reached.extend(summaries[callee.index()].iter().copied());
        }
        reached
    }
}

#[test]
fn summaries_over_random_call_graphs_are_the_reachable_sets_a_search_finds() {
    for seed in 1..=8 {
        let digraph = Digraph::random(&mut Lcg(seed + 20), 20, 30);
        let graph = random_call_graph(&digraph);
        let summaries = propagate_summaries(&graph, &Reached, SummaryConfig::default());
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

    let summaries = propagate_summaries(&graph, &Reached, SummaryConfig::default());
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
