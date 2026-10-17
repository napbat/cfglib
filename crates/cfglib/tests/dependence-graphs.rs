//! Integration tests: control- and program-dependence graphs over a CFG of a
//! consumer-defined instruction type.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, DependenceKind, DependenceNode, EdgeKind, InstrInfo, NodeId, PostDominatorTree,
    ProgramPoint, TraversalDirection, control_dependence_graph, depth_first_preorder,
    program_dependence_graph,
};

/// A three-address operation over named virtual registers.
#[derive(Debug, Clone)]
struct Op {
    reads: Vec<&'static str>,
    writes: Vec<&'static str>,
}

impl InstrInfo for Op {
    type Variable = &'static str;

    fn uses(&self) -> &[Self::Variable] {
        &self.reads
    }

    fn defs(&self) -> &[Self::Variable] {
        &self.writes
    }
}

fn op(reads: &[&'static str], writes: &[&'static str]) -> Op {
    Op {
        reads: reads.to_vec(),
        writes: writes.to_vec(),
    }
}

/// `entry: a = input; flag = cmp a`, a branch on `flag` to `then_block: b = a + 1`
/// or `else_block: b = a - 1`, and `merge: out = b`.
struct Branch {
    cfg: Cfg<Op>,
    entry: BlockId,
    then_block: BlockId,
    else_block: BlockId,
    merge: BlockId,
}

fn branch() -> Branch {
    let mut cfg = Cfg::<Op>::new();
    let entry = cfg.entry();
    let [then_block, else_block, merge] = [(); 3].map(|()| cfg.new_block());
    cfg.block_mut(entry).push(op(&[], &["a"]));
    cfg.block_mut(entry).push(op(&["a"], &["flag"]));
    cfg.block_mut(then_block).push(op(&["a"], &["b"]));
    cfg.block_mut(else_block).push(op(&["a"], &["b"]));
    cfg.block_mut(merge).push(op(&["b"], &["out"]));
    cfg.add_edge(entry, then_block, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, else_block, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_block, merge, EdgeKind::Fallthrough);
    cfg.add_edge(else_block, merge, EdgeKind::Fallthrough);
    Branch {
        cfg,
        entry,
        then_block,
        else_block,
        merge,
    }
}

fn point(block: BlockId, inst_idx: usize) -> DependenceNode {
    DependenceNode::Instruction(ProgramPoint { block, inst_idx })
}

fn node_of(
    graph: &cfglib::Graph<DependenceNode, DependenceKind>,
    payload: DependenceNode,
) -> NodeId {
    graph
        .node_ids()
        .find(|&node| *graph.node(node) == payload)
        .expect("the dependence node exists")
}

#[test]
fn a_branch_controls_its_arms_and_not_the_merge() {
    let Branch {
        cfg,
        entry,
        then_block,
        else_block,
        merge,
    } = branch();
    let post = PostDominatorTree::compute(&cfg);
    let dependences = control_dependence_graph(&cfg, &post);

    let controlled: BTreeSet<BlockId> = dependences
        .successors(NodeId::from_index(entry.index()))
        .map(|node| dependences[node])
        .collect();
    assert_eq!(controlled, BTreeSet::from([then_block, else_block]));
    assert_eq!(
        dependences
            .predecessors(NodeId::from_index(merge.index()))
            .count(),
        0,
        "the merge runs whichever way the branch goes"
    );
}

#[test]
fn control_edges_start_at_predicates_and_data_edges_feed_them() {
    let Branch {
        cfg,
        entry,
        then_block,
        merge,
        ..
    } = branch();
    let graph = program_dependence_graph(&cfg);

    for edge in graph.edges() {
        if *edge.payload() == DependenceKind::Control {
            assert!(
                matches!(graph.node(edge.source()), DependenceNode::Block(_)),
                "control edge from {:?}",
                graph.node(edge.source())
            );
        }
    }

    let has = |source, target, kind| {
        graph.edges().any(|edge| {
            *graph.node(edge.source()) == source
                && *graph.node(edge.target()) == target
                && *edge.payload() == kind
        })
    };
    assert!(has(
        DependenceNode::Block(entry),
        DependenceNode::Block(then_block),
        DependenceKind::Control
    ));
    assert!(has(
        DependenceNode::Block(entry),
        point(then_block, 0),
        DependenceKind::Control
    ));
    assert!(has(
        point(entry, 1),
        DependenceNode::Block(entry),
        DependenceKind::Data
    ));
    assert!(has(
        point(then_block, 0),
        point(merge, 0),
        DependenceKind::Data
    ));
}

#[test]
fn a_backward_slice_follows_the_branch_back_to_the_values_it_tests() {
    let Branch {
        cfg,
        entry,
        then_block,
        else_block,
        merge,
    } = branch();
    let graph = program_dependence_graph(&cfg);

    let slice_of = |seed: DependenceNode| -> BTreeSet<DependenceNode> {
        depth_first_preorder(&graph, node_of(&graph, seed), TraversalDirection::Incoming)
            .into_iter()
            .map(|node| *graph.node(node))
            .collect()
    };

    // The arm runs because the branch chose it, and the branch tests `flag`,
    // which `cmp` computed from `a`.
    let arm = slice_of(point(then_block, 0));
    for needed in [
        DependenceNode::Block(entry),
        point(entry, 1),
        point(entry, 0),
    ] {
        assert!(
            arm.contains(&needed),
            "{needed:?} is in the slice of the arm"
        );
    }
    assert!(!arm.contains(&point(else_block, 0)));

    // The merge reads `b` from either arm, so both arms' definitions reach it;
    // the merge itself is not control dependent on the branch.
    let merged = slice_of(point(merge, 0));
    assert!(merged.contains(&point(then_block, 0)));
    assert!(merged.contains(&point(else_block, 0)));
    assert!(merged.contains(&point(entry, 0)));
}

#[test]
fn a_removed_block_leaves_no_dependence_nodes() {
    let Branch {
        mut cfg,
        else_block,
        then_block,
        ..
    } = branch();
    let dead = cfg.new_block();
    cfg.block_mut(dead).push(op(&["a"], &["unused"]));
    assert!(cfg.remove_block(dead));

    let graph = program_dependence_graph(&cfg);

    let instruction_count: usize = cfg
        .block_ids()
        .map(|block| cfg.block(block).instructions().len())
        .sum();
    assert_eq!(graph.node_count(), cfg.block_count() + instruction_count);
    assert!(
        graph.node_ids().all(
            |node| !matches!(graph.node(node), DependenceNode::Block(block) if *block == dead)
        )
    );
    assert!(
        graph
            .node_ids()
            .any(|node| *graph.node(node) == point(then_block, 0))
    );
    assert!(
        graph
            .node_ids()
            .any(|node| *graph.node(node) == point(else_block, 0))
    );
}
