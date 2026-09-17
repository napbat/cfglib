use super::fixtures::{WIDE_FACT_WORDS, fixture_u32};
use super::structural_oracles::has_directed_edge;
use super::{
    BTreeSet, BlockId, Cfg, CommonAncestor, ConstantInst, DominanceFrontiers, DominatorTree,
    EdgeStep, Facts, Graph, IntervalAnalysis, NodeFacts, NodeId, PhiPlacements, ProgramPoint,
    SsaForm, SsaValue, TraversalDirection, VecDeque,
};

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
            assert_eq!(dominators.depth(block), Some(0));
        } else {
            let parent = ((index - 1) / 2) * 2;
            assert_eq!(dominators.idom(block), Some(BlockId::from_index(parent)));
            assert_eq!(dominators.depth(block), Some(index.div_ceil(2)));
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
        assert_eq!(frontiers.frontier(block_id), &expected[block_id.index()]);
    }
}

pub(super) fn assert_branchy_post_dominators(dominators: &DominatorTree, node_count: usize) {
    for index in 0..node_count {
        let block = BlockId::from_index(index);
        assert!(dominators.is_reachable(block));
        let expected = if index + 1 == node_count {
            None
        } else if index % 2 == 0 {
            Some(BlockId::from_index((index + 2).min(node_count - 1)))
        } else {
            Some(BlockId::from_index(index + 1))
        };
        assert_eq!(dominators.idom(block), expected);
    }
}

pub(super) fn assert_control_dependence_graph(
    result: &Graph<BlockId, ()>,
    cfg: &Cfg<u32>,
    post_dominators: &DominatorTree,
) {
    assert_eq!(result.node_count(), cfg.block_bound());
    for node in result.node_ids() {
        assert_eq!(*result.node(node), BlockId::from_index(node.index()));
    }

    let mut expected = BTreeSet::new();
    for controller in cfg.block_ids() {
        for target in cfg.successors(controller) {
            if post_dominators.dominates(target, controller) {
                continue;
            }
            let immediate = post_dominators.idom(controller);
            let mut dependent = target;
            loop {
                expected.insert((controller, dependent));
                match post_dominators.idom(dependent) {
                    Some(next) if Some(next) != immediate => dependent = next,
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
    assert_eq!(result.levels.len(), 1);
    assert_ne!(result.levels[0].len(), 0);
    let mut assigned = BTreeSet::new();
    for interval in &result.levels[0] {
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
    assert_eq!(result.is_reducible, result.levels[0].len() <= 1);
}

pub(super) fn assert_reverse_chain_intervals(
    result: &IntervalAnalysis<NodeId>,
    node_count: usize,
    root: NodeId,
) {
    assert_eq!(result.levels.len(), 1);
    assert_eq!(result.levels[0].len(), 1);
    assert!(result.is_reducible);
    let interval = &result.levels[0][0];
    assert_eq!(interval.header, root);
    assert_eq!(interval.blocks.len(), node_count);
    assert!((0..node_count).all(|index| interval.blocks.contains(&NodeId::from_index(index))));
}

pub(super) fn assert_bool_node_facts(facts: &NodeFacts<bool>, node_count: usize) {
    for index in 0..node_count {
        let node = NodeId::from_index(index);
        assert!(*facts.fact_in(node));
        assert!(*facts.fact_out(node));
    }
}

pub(super) fn assert_wide_node_facts(facts: &NodeFacts<Vec<u64>>, node_count: usize) {
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

pub(super) fn assert_linear_ssa(ssa: &SsaForm<u32>, block_count: usize) {
    assert_eq!(ssa.blocks().len(), block_count);
    assert_eq!(ssa.phis().count(), 0);
    for index in 0..block_count {
        let block = BlockId::from_index(index);
        let ssa_block = ssa.block(block);
        assert_eq!(ssa_block.block, block);
        assert_eq!(ssa_block.phis.len(), 0);
        assert_eq!(ssa_block.instructions.len(), 1);
        let instruction = &ssa_block.instructions[0];
        assert_eq!(instruction.point, ProgramPoint { block, inst_idx: 0 });
        assert_eq!(instruction.uses.len(), 0);
        assert_eq!(instruction.defs, [SsaValue::new(0, index + 1)]);
    }
    assert_eq!(ssa.max_version(&0), block_count);
}

pub(super) fn assert_phi_placements(
    placements: &PhiPlacements<u32>,
    layer_count: usize,
    variable_count: usize,
) {
    assert_eq!(placements.len(), layer_count * variable_count);
    for layer in 0..layer_count {
        let left = BlockId::from_index(3 * layer + 1);
        let right = BlockId::from_index(3 * layer + 2);
        let merge = BlockId::from_index(3 * layer + 3);
        let at_merge = placements.at(merge);
        assert_eq!(at_merge.len(), variable_count);
        for (variable, placement) in at_merge.iter().enumerate() {
            assert_eq!(placement.variable, fixture_u32(variable));
            assert_eq!(placement.predecessors, [left, right]);
        }
    }
    for index in 0..=3 * layer_count {
        if index % 3 != 0 || index == 0 {
            assert_eq!(placements.at(BlockId::from_index(index)).len(), 0);
        }
    }
}

pub(super) fn assert_phi_ssa(
    ssa: &SsaForm<u32>,
    source: &Cfg<ConstantInst>,
    layer_count: usize,
    variable_count: usize,
) {
    assert_eq!(ssa.blocks().len(), source.block_bound());
    assert_eq!(ssa.phis().count(), layer_count * variable_count);
    let mut definitions = BTreeSet::new();
    for block_id in source.block_ids() {
        let block = source.block(block_id);
        let ssa_block = ssa.block(block_id);
        assert_eq!(ssa_block.block, block_id);
        assert_eq!(ssa_block.instructions.len(), block.instructions().len());
        for (index, annotation) in ssa_block.instructions.iter().enumerate() {
            assert_eq!(
                annotation.point,
                ProgramPoint {
                    block: block_id,
                    inst_idx: index
                }
            );
            assert_eq!(annotation.uses.len(), 0);
            assert_eq!(annotation.defs.len(), 1);
            assert!(annotation.defs[0].version > 0);
            assert!(definitions.insert(annotation.defs[0].clone()));
        }
        for phi in &ssa_block.phis {
            assert!(phi.result.version > 0);
            assert!(definitions.insert(phi.result.clone()));
            assert_eq!(phi.operands.len(), source.incoming(block_id).count());
            assert_eq!(
                phi.operands
                    .iter()
                    .map(|(block, _)| *block)
                    .collect::<Vec<_>>(),
                source.predecessors(block_id).collect::<Vec<_>>()
            );
            assert!(
                phi.operands
                    .iter()
                    .all(|(_, value)| value.variable == phi.result.variable)
            );
        }
    }
    assert_eq!(definitions.len(), layer_count * variable_count * 3);
    for variable in 0..fixture_u32(variable_count) {
        assert_eq!(ssa.max_version(&variable), layer_count * 3);
    }
}
