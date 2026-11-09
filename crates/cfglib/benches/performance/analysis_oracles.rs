use std::collections::{BTreeSet, VecDeque};

use cfglib::{
    BlockId, Cfg, CommonAncestor, DominanceFrontiers, DominatorTree, EdgeStep, Facts, Graph,
    IntervalAnalysis, NodeId, PostDominatorTree, PostNode, TraversalDirection,
};

use super::fixtures::{WIDE_FACT_WORDS, fixture_u32};
use super::structural_oracles::has_directed_edge;

pub(super) fn directed_distances(
    graph: &Graph<(), ()>,
    start: NodeId,
    direction: TraversalDirection,
) -> (Vec<usize>, Vec<NodeId>) {
    let mut distances = vec![usize::MAX; graph.node_bound()];
    let mut order = Vec::with_capacity(graph.node_bound());
    let mut queue = VecDeque::new();
    distances[start.index()] = 0;
    queue.push_back(start);
    while let Some(node) = queue.pop_front() {
        order.push(node);
        let adjacent: Vec<_> = match direction {
            TraversalDirection::Outgoing => graph.outgoing(node).collect(),
            TraversalDirection::Incoming => graph.incoming(node).collect(),
        };
        for edge_id in adjacent {
            let edge = graph.edge(edge_id);
            let next = match direction {
                TraversalDirection::Outgoing => edge.target(),
                TraversalDirection::Incoming => edge.source(),
            };
            if distances[next.index()] == usize::MAX {
                distances[next.index()] = distances[node.index()] + 1;
                queue.push_back(next);
            }
        }
    }
    (distances, order)
}

pub(super) fn reference_cfg_preorder(cfg: &Cfg<u32>) -> Vec<BlockId> {
    let mut visited = vec![false; cfg.block_bound()];
    let mut order = Vec::with_capacity(cfg.block_count());
    let mut stack = vec![cfg.entry()];
    while let Some(block) = stack.pop() {
        if visited[block.index()] {
            continue;
        }
        visited[block.index()] = true;
        order.push(block);
        let successors: Vec<_> = cfg.successors(block).collect();
        stack.extend(
            successors
                .into_iter()
                .rev()
                .filter(|successor| !visited[successor.index()]),
        );
    }
    order
}

pub(super) fn reference_cfg_breadth_first(cfg: &Cfg<u32>) -> Vec<BlockId> {
    let mut visited = vec![false; cfg.block_bound()];
    let mut order = Vec::with_capacity(cfg.block_count());
    let mut queue = VecDeque::from([cfg.entry()]);
    visited[cfg.entry().index()] = true;
    while let Some(block) = queue.pop_front() {
        order.push(block);
        for successor in cfg.successors(block) {
            if !visited[successor.index()] {
                visited[successor.index()] = true;
                queue.push_back(successor);
            }
        }
    }
    order
}

pub(super) fn assert_edge_traversal(steps: &[EdgeStep], graph: &Graph<(), ()>) {
    assert_eq!(steps.len(), graph.edge_count());
    let mut seen = vec![false; graph.edge_bound()];
    for step in steps {
        assert!(!seen[step.edge.index()], "edge traversal repeated an edge");
        seen[step.edge.index()] = true;
        let edge = graph.edge(step.edge);
        assert_eq!(step.source, edge.source());
        assert_eq!(step.target, edge.target());
    }
    assert!(graph.edge_ids().all(|edge| seen[edge.index()]));

    let mut expected = Vec::with_capacity(graph.edge_count());
    let mut expanded = vec![false; graph.node_bound()];
    let mut queue = VecDeque::from([NodeId::from_raw(0)]);
    expanded[0] = true;
    while let Some(node) = queue.pop_front() {
        for edge_id in graph.outgoing(node) {
            let edge = graph.edge(edge_id);
            expected.push(EdgeStep {
                edge: edge_id,
                source: edge.source(),
                target: edge.target(),
            });
            if !expanded[edge.target().index()] {
                expanded[edge.target().index()] = true;
                queue.push_back(edge.target());
            }
        }
    }
    assert_eq!(steps, expected);
}

pub(super) fn assert_node_path(path: &[NodeId], graph: &Graph<(), ()>, from: NodeId, to: NodeId) {
    assert_eq!(path.first(), Some(&from));
    assert_eq!(path.last(), Some(&to));
    let (distances, _) = directed_distances(graph, from, TraversalDirection::Outgoing);
    assert_eq!(path.len(), distances[to.index()] + 1);
    assert!(
        path.windows(2)
            .all(|pair| has_directed_edge(graph, pair[0], pair[1]))
    );
}

pub(super) fn assert_edge_path(
    path: &[cfglib::EdgeId],
    graph: &Graph<(), ()>,
    from: NodeId,
    to: NodeId,
) {
    let (distances, _) = directed_distances(graph, from, TraversalDirection::Outgoing);
    assert_eq!(path.len(), distances[to.index()]);
    let mut current = from;
    let mut seen = BTreeSet::new();
    for &edge_id in path {
        assert!(seen.insert(edge_id), "shortest path repeated an edge");
        let edge = graph.edge(edge_id);
        assert_eq!(edge.source(), current);
        current = edge.target();
    }
    assert_eq!(current, to);
}

pub(super) fn assert_common_ancestor_results(
    results: &[CommonAncestor<NodeId>],
    graph: &Graph<(), ()>,
    a: NodeId,
    b: NodeId,
) {
    let (from_a, _) = directed_distances(graph, a, TraversalDirection::Incoming);
    let (from_b, b_order) = directed_distances(graph, b, TraversalDirection::Incoming);
    let expected: Vec<_> = b_order
        .into_iter()
        .filter(|node| from_a[node.index()] != usize::MAX)
        .map(|node| CommonAncestor {
            node,
            from_a: from_a[node.index()],
            from_b: from_b[node.index()],
        })
        .collect();
    assert_eq!(results, expected);
}

pub(super) fn assert_branchy_dominators(dominators: &DominatorTree, node_count: usize) {
    for index in 0..node_count {
        let block = BlockId::from_index(index);
        assert!(dominators.is_reachable(block));
        if index == 0 {
            assert_eq!(dominators.idom(block), None);
            assert_eq!(dominators.level(block), Some(0));
        } else {
            let parent = ((index - 1) / 2) * 2;
            assert_eq!(dominators.idom(block), Some(BlockId::from_index(parent)));
            let level = fixture_u32(index.div_ceil(2));
            assert_eq!(dominators.level(block), Some(level));
        }
    }
}

pub(super) fn assert_dominance_frontiers(
    frontiers: &DominanceFrontiers,
    cfg: &Cfg<u32>,
    dominators: &DominatorTree,
) {
    let mut expected = vec![BTreeSet::new(); cfg.block_bound()];
    for block_id in cfg.block_ids() {
        if cfg.incoming(block_id).count() < 2 {
            continue;
        }
        let root = dominators.idom(block_id).unwrap_or(block_id);
        for predecessor in cfg.predecessors(block_id) {
            let mut runner = predecessor;
            while runner != root {
                expected[runner.index()].insert(block_id);
                let Some(parent) = dominators.idom(runner) else {
                    break;
                };
                runner = parent;
            }
        }
    }
    for block_id in cfg.block_ids() {
        let members: Vec<_> = expected[block_id.index()].iter().copied().collect();
        assert_eq!(frontiers.frontier(block_id), members);
    }
}

pub(super) fn assert_branchy_post_dominators(post: &PostDominatorTree, node_count: usize) {
    for index in 0..node_count {
        let block = BlockId::from_index(index);
        assert!(post.contains(block));
        let expected = if index + 1 == node_count {
            PostNode::Exit
        } else if index % 2 == 0 {
            PostNode::Node(BlockId::from_index((index + 2).min(node_count - 1)))
        } else {
            PostNode::Node(BlockId::from_index(index + 1))
        };
        assert_eq!(post.ipdom(block), Some(expected));
    }
}

pub(super) fn assert_control_dependence_graph(
    result: &Graph<BlockId, ()>,
    cfg: &Cfg<u32>,
    post_dominators: &PostDominatorTree,
) {
    assert_eq!(result.node_count(), cfg.block_bound());
    for node in result.node_ids() {
        assert_eq!(*result.node(node), BlockId::from_index(node.index()));
    }

    let mut expected = BTreeSet::new();
    for controller in cfg.block_ids() {
        let immediate = post_dominators.ipdom(controller);
        for target in cfg.successors(controller) {
            let mut dependent = target;
            while Some(PostNode::Node(dependent)) != immediate {
                expected.insert((controller, dependent));
                match post_dominators.ipdom(dependent) {
                    Some(PostNode::Node(parent)) => dependent = parent,
                    _ => break,
                }
            }
        }
    }
    let actual: BTreeSet<_> = result
        .edges()
        .map(|edge| (*result.node(edge.source()), *result.node(edge.target())))
        .collect();
    assert_eq!(actual, expected);
    assert_eq!(result.edge_count(), expected.len());
}

pub(super) fn assert_cfg_intervals(result: &IntervalAnalysis, cfg: &Cfg<u32>) {
    assert_ne!(result.intervals.len(), 0);
    let mut assigned = BTreeSet::new();
    for interval in &result.intervals {
        assert!(interval.blocks.contains(&interval.header));
        for &block in &interval.blocks {
            assert!(assigned.insert(block), "block appeared in two intervals");
            if block != interval.header {
                assert!(
                    cfg.predecessors(block)
                        .all(|predecessor| interval.blocks.contains(&predecessor)),
                    "non-header interval block has an external predecessor"
                );
            }
        }
    }
    assert_eq!(assigned.len(), cfg.block_count());
}

pub(super) fn assert_reverse_chain_intervals(
    result: &IntervalAnalysis<NodeId>,
    node_count: usize,
    root: NodeId,
) {
    assert_eq!(result.intervals.len(), 1);
    let interval = &result.intervals[0];
    assert_eq!(interval.header, root);
    assert_eq!(interval.blocks.len(), node_count);
    assert!((0..node_count).all(|index| interval.blocks.contains(&NodeId::from_index(index))));
}

pub(super) fn assert_bool_node_facts(facts: &Facts<bool>, node_count: usize) {
    for index in 0..node_count {
        let node = NodeId::from_index(index);
        assert!(*facts.fact_in(node));
        assert!(*facts.fact_out(node));
    }
}

pub(super) fn assert_wide_node_facts(facts: &Facts<Vec<u64>>, node_count: usize) {
    for index in 0..node_count {
        let node = NodeId::from_index(index);
        for fact in [facts.fact_in(node), facts.fact_out(node)] {
            assert_eq!(fact.len(), WIDE_FACT_WORDS);
            assert!(fact.iter().all(|&word| word == u64::MAX));
        }
    }
}

pub(super) fn assert_bool_cfg_facts(facts: &Facts<bool>, block_count: usize) {
    for index in 0..block_count {
        let block = BlockId::from_index(index);
        assert!(*facts.fact_in(block));
        assert!(*facts.fact_out(block));
    }
}

pub(super) fn assert_wide_cfg_facts(facts: &Facts<Vec<u64>>, block_count: usize) {
    for index in 0..block_count {
        let block = BlockId::from_index(index);
        for fact in [facts.fact_in(block), facts.fact_out(block)] {
            assert_eq!(fact.len(), WIDE_FACT_WORDS);
            assert!(fact.iter().all(|&word| word == u64::MAX));
        }
    }
}
