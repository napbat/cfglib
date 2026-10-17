//! The generic graph algorithms over an MLIL view, the way reify runs them.
//!
//! Each answer is compared with a brute-force oracle that reads the successor
//! records of the function, on the fixtures and on random functions with
//! loops, parallel records, unreachable code, and removed slots.

use std::collections::BTreeSet;

use cfglib::{
    DenseId, DominanceFrontiers, DominatorTree, GraphView, PostDominatorTree, PostNode, RootedView,
    TraversalDirection, control_dependence_graph, reverse_postorder, tarjan_scc,
};

use crate::fixtures;
use crate::glint::{Block, Function, Glint};
use crate::oracle::{Cfg, extract};
use crate::random_functions::random_function;

/// The fixtures and four hundred random functions, with names for failures.
fn functions() -> Vec<(String, Function)> {
    let mut all: Vec<(String, Function)> = fixtures::all()
        .into_iter()
        .map(|(name, function)| (name.to_owned(), function))
        .collect();
    all.extend((0..400).map(|seed| (format!("seed {seed}"), random_function(seed))));
    all
}

fn block(index: usize) -> Block {
    Block::from_index(index)
}

#[test]
fn the_dominator_tree_matches_removal_reachability() {
    for (name, function) in functions() {
        let cfg = extract::<Glint, _>(&function);
        let reachable = cfg.reachable();
        let dominators = cfg.dominators();
        let tree = DominatorTree::compute(&function);
        assert_eq!(
            tree.idom(function.entry()),
            None,
            "{name}: the root has no idom"
        );
        for node in cfg.live_nodes() {
            assert_eq!(
                tree.is_reachable(block(node)),
                reachable[node],
                "{name}: {node}"
            );
            if !reachable[node] {
                continue;
            }
            for candidate in cfg.live_nodes() {
                assert_eq!(
                    tree.dominates(block(candidate), block(node)),
                    dominators[node].contains(&candidate),
                    "{name}: does {candidate} dominate {node}"
                );
            }
            assert_eq!(
                tree.idom(block(node)).map(DenseId::index),
                Cfg::immediate_dominator(&dominators, node),
                "{name}: idom of {node}"
            );
        }
    }
}

#[test]
fn dominance_frontiers_match_their_textbook_definition() {
    for (name, function) in functions() {
        let cfg = extract::<Glint, _>(&function);
        let reachable = cfg.reachable();
        let expected = cfg.frontiers();
        let tree = DominatorTree::compute(&function);
        let frontiers = DominanceFrontiers::compute(&function, &tree);
        for node in cfg.live_nodes().filter(|&node| reachable[node]) {
            let actual: BTreeSet<usize> = frontiers
                .frontier(block(node))
                .iter()
                .map(|frontier| frontier.index())
                .collect();
            assert_eq!(actual, expected[node], "{name}: frontier of {node}");
        }
    }
}

#[test]
fn strongly_connected_components_are_the_mutual_reachability_classes_numbered_leaves_first() {
    for (name, function) in functions() {
        let cfg = extract::<Glint, _>(&function);
        let classes = cfg.components();
        let scc = tarjan_scc(&function);
        assert_eq!(scc.len(), classes.len(), "{name}: component count");
        let same_class = |a: usize, b: usize| {
            classes
                .iter()
                .any(|class| class.contains(&a) && class.contains(&b))
        };
        for a in cfg.live_nodes() {
            for b in cfg.live_nodes() {
                assert_eq!(
                    scc.component_index(block(a)) == scc.component_index(block(b)),
                    same_class(a, b),
                    "{name}: {a} and {b}"
                );
            }
            // An edge leaves a component for one that was numbered earlier.
            for &b in &cfg.successors[a] {
                if !same_class(a, b) {
                    assert!(
                        scc.component_index(block(a)) > scc.component_index(block(b)),
                        "{name}: edge {a} -> {b} must go to a lower number"
                    );
                }
            }
        }
        let numbered: usize = scc.components().map(<[Block]>::len).sum();
        assert_eq!(
            numbered,
            cfg.live_nodes().count(),
            "{name}: every live node once"
        );
        let acyclic = classes
            .iter()
            .all(|class| class.len() == 1 && !cfg.successors[class[0]].contains(&class[0]));
        assert_eq!(scc.is_dag(&function), acyclic, "{name}: is_dag");
    }
}

#[test]
fn reverse_postorder_lists_the_reachable_blocks_and_puts_forward_edges_in_order() {
    for (name, function) in functions() {
        let cfg = extract::<Glint, _>(&function);
        let reachable = cfg.reachable();
        let order = reverse_postorder(&function, function.root(), TraversalDirection::Outgoing);
        let listed: BTreeSet<usize> = order.iter().map(|&node| node.index()).collect();
        assert_eq!(order.len(), listed.len(), "{name}: no block twice");
        let expected: BTreeSet<usize> = cfg.live_nodes().filter(|&node| reachable[node]).collect();
        assert_eq!(listed, expected, "{name}: the reachable blocks");
        assert_eq!(order[0], function.root(), "{name}: the root comes first");
        if cfg.is_reducible() {
            // In a reducible graph the retreating edges are the back edges, so
            // every other edge runs forward in the order.
            let position = |node: usize| order.iter().position(|&listed| listed.index() == node);
            let dominators = cfg.dominators();
            for &from in &expected {
                for &to in &cfg.successors[from] {
                    if !dominators[from].contains(&to) {
                        assert!(position(from) < position(to), "{name}: edge {from} -> {to}");
                    }
                }
            }
        }
    }
}

#[test]
fn post_dominators_and_control_dependence_match_their_definitions() {
    for (name, function) in functions() {
        let cfg = extract::<Glint, _>(&function);
        let reachable = cfg.reachable();
        let post = PostDominatorTree::compute(&function);
        let exits: Vec<usize> = post.exits().iter().map(|exit| exit.index()).collect();
        assert_eq!(exits, cfg.exits(), "{name}: exits");
        let Some(pdom) = cfg.post_dominators() else {
            // Some reachable block cannot reach an exit: the tree has a cut.
            assert!(
                !post.cut_roots().is_empty(),
                "{name}: an exit-free region needs a cut"
            );
            for node in cfg.live_nodes().filter(|&node| reachable[node]) {
                assert_eq!(
                    post.reaches_exit(block(node)),
                    cfg.can_reach_exit(node),
                    "{name}: {node}"
                );
            }
            continue;
        };
        assert!(
            post.cut_roots().is_empty(),
            "{name}: no cut when every block reaches an exit"
        );
        for a in cfg.live_nodes().filter(|&node| reachable[node]) {
            assert!(
                post.contains(block(a)) && post.reaches_exit(block(a)),
                "{name}: {a}"
            );
            for b in cfg.live_nodes().filter(|&node| reachable[node]) {
                assert_eq!(
                    post.post_dominates(block(b), block(a)),
                    pdom[a].contains(&b),
                    "{name}: does {b} post-dominate {a}"
                );
            }
            // The immediate post-dominator is the strict one that every other
            // strict post-dominator post-dominates, or the virtual exit.
            let strict: Vec<usize> = pdom[a].iter().copied().filter(|&b| b != a).collect();
            let nearest = strict.iter().copied().find(|&candidate| {
                strict
                    .iter()
                    .all(|&other| other == candidate || pdom[candidate].contains(&other))
            });
            let expected = nearest.map_or(PostNode::Exit, |node| PostNode::Node(block(node)));
            assert_eq!(post.ipdom(block(a)), Some(expected), "{name}: ipdom of {a}");
        }

        let dependences = control_dependence_graph(&function, &post);
        let actual: BTreeSet<(usize, usize)> = dependences
            .edges()
            .map(|edge| {
                (
                    dependences.node(edge.source()).index(),
                    dependences.node(edge.target()).index(),
                )
            })
            .collect();
        assert_eq!(
            actual,
            cfg.control_dependence(&pdom),
            "{name}: control dependence"
        );
    }
}

#[test]
fn a_branch_into_an_endless_loop_gets_a_cut_and_the_loop_depends_on_itself() {
    // Entry branches to `spin`, which jumps to itself, or to `out`, which returns.
    let function = fixtures::with_an_endless_loop();
    let [entry, spin, out] = [0, 1, 2].map(block);
    let post = PostDominatorTree::compute(&function);
    assert_eq!(post.exits(), [out]);
    assert_eq!(post.cut_roots(), [spin]);
    assert!(post.reaches_exit(entry));
    assert!(!post.reaches_exit(spin));
    assert_eq!(post.ipdom(entry), Some(PostNode::Exit));

    let dependences = control_dependence_graph(&function, &post);
    let edges: BTreeSet<(Block, Block)> = dependences
        .edges()
        .map(|edge| {
            (
                *dependences.node(edge.source()),
                *dependences.node(edge.target()),
            )
        })
        .collect();
    // The entry decides whether either arm runs, and the spinning block
    // decides whether it runs again.
    assert_eq!(
        edges,
        BTreeSet::from([(entry, spin), (entry, out), (spin, spin)])
    );
}

#[test]
fn parallel_switch_records_give_one_control_dependence_edge_each() {
    let function = fixtures::switch_with_parallel_records();
    let [entry, shared, other, fallback] = [0, 1, 2, 3].map(block);
    let post = PostDominatorTree::compute(&function);
    let dependences = control_dependence_graph(&function, &post);
    let mut edges: Vec<(Block, Block)> = dependences
        .edges()
        .map(|edge| {
            (
                *dependences.node(edge.source()),
                *dependences.node(edge.target()),
            )
        })
        .collect();
    edges.sort_unstable();
    // Cases 1 and 2 both lead to `shared`, yet the entry controls it once.
    assert_eq!(edges, [(entry, shared), (entry, other), (entry, fallback)]);
}

#[test]
fn a_dag_function_is_acyclic_and_a_loop_is_not() {
    let acyclic = [
        fixtures::absolute_value(),
        fixtures::switch_with_parallel_records(),
        fixtures::single_block(),
        fixtures::with_an_unreachable_block(),
    ];
    for function in &acyclic {
        assert!(tarjan_scc(function).is_dag(function));
    }
    let cyclic = [
        fixtures::counting_sum(),
        fixtures::self_loop(),
        fixtures::with_an_endless_loop(),
        fixtures::irreducible(),
    ];
    for function in &cyclic {
        assert!(!tarjan_scc(function).is_dag(function));
    }
}

#[test]
fn the_graph_view_of_a_function_with_a_removed_block_skips_the_slot() {
    let function = fixtures::with_a_removed_block();
    let order = reverse_postorder(&function, function.root(), TraversalDirection::Outgoing);
    assert_eq!(order, [block(0), block(2)]);
    let scc = tarjan_scc(&function);
    assert_eq!(scc.len(), 2);
    assert_eq!(GraphView::node_bound(&function), 3);
}

#[test]
fn the_random_functions_cover_the_shapes_the_algorithms_must_handle() {
    let mut counts = [0_usize; 6];
    for seed in 0..400 {
        let function = random_function(seed);
        let cfg = extract::<Glint, _>(&function);
        let reachable = cfg.reachable();
        let dominators = cfg.dominators();
        counts[0] += cfg.live.len() - cfg.live_nodes().count();
        counts[1] += cfg.live_nodes().filter(|&node| !reachable[node]).count();
        counts[2] += usize::from(!cfg.is_reducible());
        counts[3] += usize::from(cfg.post_dominators().is_none());
        counts[4] += cfg
            .live_nodes()
            .filter(|&node| {
                let targets: BTreeSet<_> = cfg.successors[node].iter().collect();
                targets.len() < cfg.successors[node].len()
            })
            .count();
        counts[5] += cfg
            .live_nodes()
            .filter(|&node| {
                reachable[node]
                    && cfg.successors[node]
                        .iter()
                        .any(|target| dominators[node].contains(target))
            })
            .count();
    }
    let names = [
        "removed slots",
        "unreachable blocks",
        "irreducible functions",
        "functions with an exit-free region",
        "blocks with parallel records",
        "blocks with a back edge",
    ];
    for (name, count) in names.into_iter().zip(counts) {
        assert!(count >= 20, "only {count} {name} in 400 random functions");
    }
}
