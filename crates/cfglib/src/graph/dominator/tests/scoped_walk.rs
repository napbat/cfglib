//! The scoped preorder walk: its order, its nesting, forests, and depth.

extern crate alloc;
extern crate std;

use alloc::vec::Vec;

use super::forest::two_trees;
use super::support::{Family, Recipe, Rng, random_graph};
use crate::graph::dominator::DominatorTree;
use crate::graph::store::{Graph, NodeId};

/// One step of a walk, as a test records it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    Enter(u32),
    Exit(u32),
}

/// Walk `roots` of `tree`, recording every callback. `enter` gives the scope
/// the depth it was entered at, and `exit` checks it is handed back.
fn record(tree: &DominatorTree<u32>, roots: &[u32]) -> Vec<Step> {
    let mut steps = Vec::new();
    let mut open = Vec::new();
    tree.scoped_walk(
        roots.iter().copied(),
        &mut steps,
        &mut open,
        |steps, node| {
            steps.push(Step::Enter(node));
            node * 10
        },
        |steps, node, scope| {
            assert_eq!(scope, node * 10, "exit gets back what enter returned");
            steps.push(Step::Exit(node));
        },
    );
    assert!(open.is_empty(), "a walk leaves its stack empty");
    steps
}

#[test]
fn a_scope_ends_after_every_node_its_node_dominates() {
    // root -> {first -> grandchild, second}
    let mut graph = Graph::<(), ()>::new();
    let [root, first, second, grandchild, outside] = [(); 5].map(|()| graph.add_node(()));
    graph.add_edge(root, first, ());
    graph.add_edge(root, second, ());
    graph.add_edge(first, grandchild, ());
    let tree = DominatorTree::compute_from(&graph, root);

    let mut log = Vec::new();
    let mut open = Vec::new();
    tree.scoped_walk(
        [root],
        &mut log,
        &mut open,
        |log, node| log.push((true, node)),
        |log, node, ()| log.push((false, node)),
    );

    assert_eq!(
        log,
        [
            (true, root),
            (true, first),
            (true, grandchild),
            (false, grandchild),
            (false, first),
            (true, second),
            (false, second),
            (false, root),
        ]
    );
    let _ = outside;
}

#[test]
fn trees_are_walked_in_the_order_given_and_nodes_in_no_tree_are_skipped() {
    let forest = two_trees();
    assert_eq!(
        record(&forest, &[5, 0, 2, 7]),
        [
            Step::Enter(5),
            Step::Enter(1),
            Step::Exit(1),
            Step::Enter(4),
            Step::Enter(3),
            Step::Exit(3),
            Step::Exit(4),
            Step::Exit(5),
            Step::Enter(2),
            Step::Enter(6),
            Step::Exit(6),
            Step::Exit(2),
        ]
    );
    assert_eq!(record(&forest, &[]), []);
    assert_eq!(record(&forest, &[0, 7]), []);
}

#[test]
fn a_node_is_entered_once_however_often_it_is_named_as_a_walk_ends() {
    // Naming a root twice walks its tree twice, each time as a whole.
    let forest = two_trees();
    let once = record(&forest, &[2]);
    assert_eq!(record(&forest, &[2, 2]), [once.clone(), once].concat());
}

/// Entering every root in ascending order is the preorder, and at every node
/// the open scopes are exactly its dominators.
#[test]
fn the_walk_is_the_preorder_and_nests_by_dominance_on_random_graphs() {
    for seed in 0..60_u64 {
        let mut rng = Rng::new(seed);
        let recipe = Recipe {
            family: Family::ALL[usize::try_from(seed % 3).expect("a residue fits in usize")],
            node_count: [8, 40, 200][usize::try_from(seed % 3).expect("a residue fits in usize")],
            parallel: seed % 2 == 1,
            detached: if seed % 4 >= 2 { 3 } else { 0 },
        };
        let graph = random_graph(&mut rng, recipe).graph();
        let tree = DominatorTree::compute_from(&graph, 0);
        let roots: Vec<u32> = tree
            .preorder()
            .iter()
            .copied()
            .filter(|&node| tree.idom(node).is_none())
            .collect();

        let mut path: Vec<u32> = Vec::new();
        let mut entered = Vec::new();
        let mut open = Vec::new();
        tree.scoped_walk(
            roots.iter().copied(),
            &mut (&mut path, &mut entered),
            &mut open,
            |(path, entered), node| {
                assert_eq!(
                    path.last().copied(),
                    tree.idom(node),
                    "the scope open at {node} is its immediate dominator ({recipe:?} seed {seed})"
                );
                assert_eq!(
                    u32::try_from(path.len()).ok(),
                    tree.level(node),
                    "{recipe:?} seed {seed}"
                );
                path.push(node);
                entered.push(node);
                path.len()
            },
            |(path, _), node, depth| {
                assert_eq!(path.len(), depth, "{recipe:?} seed {seed}");
                assert_eq!(path.pop(), Some(node), "{recipe:?} seed {seed}");
            },
        );
        assert!(path.is_empty(), "{recipe:?} seed {seed}");
        assert_eq!(entered, tree.preorder(), "{recipe:?} seed {seed}");
    }
}

/// A walk does not recurse: a chain of 10⁵ nodes goes through on a thread
/// whose stack could not hold that many frames of anything.
#[test]
fn a_chain_of_a_hundred_thousand_nodes_walks_on_a_small_stack() {
    const LENGTH: usize = 100_000;
    let walked = std::thread::Builder::new()
        .stack_size(64 * 1024)
        .spawn(|| {
            let mut graph = Graph::<(), ()>::new();
            let nodes: Vec<NodeId> = (0..LENGTH).map(|_| graph.add_node(())).collect();
            for pair in nodes.windows(2) {
                graph.add_edge(pair[0], pair[1], ());
            }
            let tree = DominatorTree::compute_from(&graph, nodes[0]);
            let (mut deepest, mut depth, mut exits) = (0_usize, 0_usize, 0_usize);
            let mut open = Vec::new();
            tree.scoped_walk(
                [nodes[0]],
                &mut (&mut deepest, &mut depth, &mut exits),
                &mut open,
                |(deepest, depth, _), _| {
                    **depth += 1;
                    **deepest = (**deepest).max(**depth);
                },
                |(_, depth, exits), _, ()| {
                    **depth -= 1;
                    **exits += 1;
                },
            );
            (deepest, depth, exits)
        })
        .expect("the thread starts")
        .join()
        .expect("the walk does not overflow the stack");
    assert_eq!(walked, (LENGTH, 0, LENGTH));
}
