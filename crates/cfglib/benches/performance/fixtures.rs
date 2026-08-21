use super::{
    BTreeMap, BlockId, Cfg, CfgBuilder, ConstantFolder, DirectedGraph, Direction, EdgeKind,
    FlowControl, FlowEffect, InstrInfo, NodeId, NodeProblem, Problem, ValueNumberInfo,
};

pub(super) fn branchy_cfg(node_count: usize) -> Cfg<u32> {
    assert!(node_count > 0);
    let mut cfg = Cfg::new();
    let mut nodes = Vec::with_capacity(node_count);
    nodes.push(cfg.entry());
    for _ in 1..node_count {
        nodes.push(cfg.new_block());
    }
    for (index, &node) in nodes.iter().enumerate() {
        cfg.block_mut(node).push(index as u32);
        cfg.block_mut(node).push((index as u32).wrapping_mul(17));
    }
    for index in 0..node_count - 1 {
        cfg.add_edge(nodes[index], nodes[index + 1], EdgeKind::Fallthrough);
    }
    for index in (0..node_count.saturating_sub(2)).step_by(2) {
        cfg.add_edge(nodes[index], nodes[index + 2], EdgeKind::ConditionalTrue);
    }
    for index in (32..node_count).step_by(32) {
        cfg.add_edge(nodes[index], nodes[index - 16], EdgeKind::Back);
    }
    cfg
}

pub(super) fn branchy_graph(node_count: usize) -> DirectedGraph<(), ()> {
    assert!(node_count > 0);
    let edge_capacity = node_count * 2 + node_count / 32;
    let mut graph = DirectedGraph::with_capacity(node_count, edge_capacity);
    let nodes: Vec<_> = (0..node_count).map(|_| graph.add_node(())).collect();
    for index in 0..node_count - 1 {
        graph.add_edge(nodes[index], nodes[index + 1], ());
    }
    for index in (0..node_count.saturating_sub(2)).step_by(2) {
        graph.add_edge(nodes[index], nodes[index + 2], ());
    }
    for index in (32..node_count).step_by(32) {
        graph.add_edge(nodes[index], nodes[index - 16], ());
    }
    graph
}

pub(super) fn reverse_id_chain_graph(node_count: usize) -> (DirectedGraph<(), ()>, NodeId) {
    assert!(node_count > 0);
    let mut graph = DirectedGraph::with_capacity(node_count, node_count - 1);
    let nodes: Vec<_> = (0..node_count).map(|_| graph.add_node(())).collect();
    for index in 1..node_count {
        graph.add_edge(nodes[index], nodes[index - 1], ());
    }
    (graph, nodes[node_count - 1])
}

pub(super) fn linear_cfg(node_count: usize) -> Cfg<u32> {
    assert!(node_count > 0);
    let mut cfg = Cfg::new();
    let mut previous = cfg.entry();
    cfg.block_mut(previous).push(0);
    for index in 1..node_count {
        let next = cfg.new_block();
        cfg.block_mut(next).push(index as u32);
        cfg.add_edge(previous, next, EdgeKind::Fallthrough);
        previous = next;
    }
    cfg
}

pub(super) fn many_exit_cfg(exit_count: usize) -> Cfg<u32> {
    assert!(exit_count > 0);
    let mut cfg = Cfg::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(0);
    for index in 0..exit_count {
        let exit = cfg.new_block();
        cfg.block_mut(exit).push((index + 1) as u32);
        cfg.add_edge(entry, exit, EdgeKind::SwitchCase);
    }
    cfg
}

pub(super) fn empty_chain_cfg(node_count: usize) -> Cfg<u32> {
    let mut cfg = Cfg::new();
    let mut previous = cfg.entry();
    cfg.block_mut(previous).push(0);
    for index in 1..node_count {
        let next = cfg.new_block();
        cfg.add_edge(previous, next, EdgeKind::Fallthrough);
        previous = next;
        if index + 1 == node_count {
            cfg.block_mut(next).push(index as u32);
        }
    }
    cfg
}

pub(super) fn high_fan_in_cfg(predecessor_count: usize) -> (Cfg<u32>, BlockId, BlockId) {
    let mut cfg = Cfg::new();
    let old_target = cfg.new_block();
    let new_target = cfg.new_block();
    for _ in 0..predecessor_count {
        let predecessor = cfg.new_block();
        cfg.add_edge(predecessor, old_target, EdgeKind::Unconditional);
    }
    (cfg, old_target, new_target)
}

pub(super) fn weighted_high_fan_out_cfg(edge_count: usize) -> (Cfg<u32>, BlockId, BlockId) {
    assert!(edge_count >= 2);
    let mut cfg = Cfg::new();
    let source = cfg.entry();
    let target = cfg.new_block();
    let sink = cfg.new_block();
    cfg.block_mut(source).push(0);
    cfg.block_mut(target).push(1);
    cfg.block_mut(sink).push(2);
    cfg.add_edge(source, target, EdgeKind::Fallthrough);
    for index in 0..edge_count - 1 {
        let kind = if index % 2 == 0 {
            EdgeKind::ConditionalTrue
        } else {
            EdgeKind::ConditionalFalse
        };
        cfg.add_weighted_edge(target, sink, kind, index as f64 + 0.25);
    }
    cfg.add_weighted_edge(target, source, EdgeKind::Back, 0.875);
    (cfg, source, target)
}

pub(super) fn irreducible_cfg(cycle_nodes: usize, external_entries: usize) -> Cfg<u32> {
    assert!(cycle_nodes >= 2);
    let mut cfg = Cfg::new();
    let cycle: Vec<_> = (0..cycle_nodes).map(|_| cfg.new_block()).collect();
    cfg.add_edge(cfg.entry(), cycle[0], EdgeKind::ConditionalTrue);
    for edge in cycle.windows(2) {
        cfg.add_edge(edge[0], edge[1], EdgeKind::Fallthrough);
    }
    cfg.add_edge(cycle[cycle_nodes - 1], cycle[0], EdgeKind::Back);

    for _ in 0..external_entries {
        let external = cfg.new_block();
        cfg.add_edge(cfg.entry(), external, EdgeKind::ConditionalFalse);
        cfg.add_edge(external, cycle[1], EdgeKind::Unconditional);
    }
    cfg
}

pub(super) fn weighted_irreducible_cfg() -> Cfg<u32> {
    let mut cfg = Cfg::new();
    let entry = cfg.entry();
    let first = cfg.new_block();
    let second = cfg.new_block();
    let exit = cfg.new_block();
    for (index, block) in [entry, first, second, exit].into_iter().enumerate() {
        cfg.block_mut(block).push(index as u32);
    }

    cfg.add_edge(entry, first, EdgeKind::ConditionalTrue);
    cfg.add_weighted_edge(entry, second, EdgeKind::ConditionalFalse, 0.125);
    cfg.add_edge(first, second, EdgeKind::Fallthrough);
    cfg.add_weighted_edge(second, first, EdgeKind::Back, 0.75);
    cfg.add_weighted_edge(second, exit, EdgeKind::SwitchCase, 0.25);
    cfg
}

pub(super) fn multi_latch_graph(
    chain_nodes: usize,
    latch_count: usize,
) -> (DirectedGraph<(), ()>, NodeId) {
    let mut graph =
        DirectedGraph::with_capacity(chain_nodes + latch_count + 1, chain_nodes + latch_count * 2);
    let header = graph.add_node(());
    let mut tail = header;
    for _ in 0..chain_nodes {
        let next = graph.add_node(());
        graph.add_edge(tail, next, ());
        tail = next;
    }
    for _ in 0..latch_count {
        let latch = graph.add_node(());
        graph.add_edge(tail, latch, ());
        graph.add_edge(latch, header, ());
    }
    (graph, header)
}

pub(super) struct Reachability;

impl NodeProblem<DirectedGraph<(), ()>> for Reachability {
    type Fact = bool;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &DirectedGraph<(), ()>) -> bool {
        false
    }

    fn boundary(&self, _graph: &DirectedGraph<(), ()>) -> bool {
        true
    }

    fn meet(&self, a: &bool, b: &bool) -> bool {
        *a || *b
    }

    fn transfer(&self, _graph: &DirectedGraph<(), ()>, _node: NodeId, input: &bool) -> bool {
        *input
    }
}

pub(super) struct CfgReachability;

impl Problem<u32> for CfgReachability {
    type Fact = bool;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self) -> bool {
        false
    }

    fn entry_fact(&self) -> bool {
        true
    }

    fn meet(&self, a: &bool, b: &bool) -> bool {
        *a || *b
    }

    fn transfer(&self, _cfg: &Cfg<u32>, _block: BlockId, input: &bool) -> bool {
        *input
    }
}

pub(super) const WIDE_FACT_WORDS: usize = 256;

pub(super) struct WideCfgFact;

impl Problem<u32> for WideCfgFact {
    type Fact = Vec<u64>;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self) -> Self::Fact {
        vec![0; WIDE_FACT_WORDS]
    }

    fn entry_fact(&self) -> Self::Fact {
        vec![u64::MAX; WIDE_FACT_WORDS]
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact {
        a.iter().zip(b).map(|(left, right)| left | right).collect()
    }

    fn transfer(&self, _cfg: &Cfg<u32>, _block: BlockId, input: &Self::Fact) -> Self::Fact {
        input.clone()
    }
}

pub(super) struct WideNodeFact;

impl NodeProblem<DirectedGraph<(), ()>> for WideNodeFact {
    type Fact = Vec<u64>;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &DirectedGraph<(), ()>) -> Self::Fact {
        vec![0; WIDE_FACT_WORDS]
    }

    fn boundary(&self, _graph: &DirectedGraph<(), ()>) -> Self::Fact {
        vec![u64::MAX; WIDE_FACT_WORDS]
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact {
        a.iter().zip(b).map(|(left, right)| left | right).collect()
    }

    fn transfer(
        &self,
        _graph: &DirectedGraph<(), ()>,
        _node: NodeId,
        input: &Self::Fact,
    ) -> Self::Fact {
        input.clone()
    }
}

pub(super) struct ConstantInst {
    defs: Vec<u32>,
    uses: Vec<u32>,
    value: u64,
}

impl InstrInfo for ConstantInst {
    type Variable = u32;

    fn uses(&self) -> &[u32] {
        &self.uses
    }

    fn defs(&self) -> &[u32] {
        &self.defs
    }
}

impl ConstantFolder for ConstantInst {
    type Const = u64;

    fn fold_constant(&self, _known: &BTreeMap<u32, u64>) -> Option<(u32, u64)> {
        Some((self.defs[0], self.value))
    }
}

impl ValueNumberInfo for ConstantInst {
    type Operation = u64;

    fn operation(&self) -> u64 {
        self.value
    }

    fn is_pure(&self) -> bool {
        true
    }
}

pub(super) fn independent_constants(instruction_count: usize) -> Cfg<ConstantInst> {
    let mut cfg = Cfg::new();
    for variable in 0..instruction_count as u32 {
        cfg.block_mut(cfg.entry()).push(ConstantInst {
            defs: vec![variable],
            uses: Vec::new(),
            value: u64::from(variable),
        });
    }
    cfg
}

pub(super) fn linear_constants(block_count: usize) -> Cfg<ConstantInst> {
    let mut cfg = Cfg::new();
    let mut block = cfg.entry();
    for index in 0..block_count {
        cfg.block_mut(block).push(ConstantInst {
            defs: vec![0],
            uses: Vec::new(),
            value: index as u64,
        });
        if index + 1 < block_count {
            let next = cfg.new_block();
            cfg.add_edge(block, next, EdgeKind::Fallthrough);
            block = next;
        }
    }
    cfg
}

pub(super) fn phi_storm_cfg(layer_count: usize, variable_count: usize) -> Cfg<ConstantInst> {
    let mut cfg = Cfg::new();
    let mut branch = cfg.entry();
    for layer in 0..layer_count {
        let left = cfg.new_block();
        let right = cfg.new_block();
        let merge = cfg.new_block();
        cfg.add_edge(branch, left, EdgeKind::ConditionalTrue);
        cfg.add_edge(branch, right, EdgeKind::ConditionalFalse);
        cfg.add_edge(left, merge, EdgeKind::Fallthrough);
        cfg.add_edge(right, merge, EdgeKind::Fallthrough);
        for variable in 0..variable_count as u32 {
            for block in [left, right] {
                cfg.block_mut(block).push(ConstantInst {
                    defs: vec![variable],
                    uses: Vec::new(),
                    value: ((layer as u64) << 32) | u64::from(variable),
                });
            }
        }
        branch = merge;
    }
    cfg
}

#[derive(Clone, Copy)]
pub(super) struct BuilderInst(pub(super) FlowEffect);

impl FlowControl for BuilderInst {
    fn flow_effect(&self) -> FlowEffect {
        self.0
    }
}

pub(super) fn build_if_else_chain(region_count: usize) -> Cfg<BuilderInst> {
    CfgBuilder::build((0..region_count).flat_map(|_| {
        [
            BuilderInst(FlowEffect::ConditionalOpen),
            BuilderInst(FlowEffect::Fallthrough),
            BuilderInst(FlowEffect::ConditionalAlternate),
            BuilderInst(FlowEffect::Fallthrough),
            BuilderInst(FlowEffect::ConditionalClose),
        ]
    }))
    .expect("synthetic conditionals are balanced")
}

pub(super) fn build_conditional_break_chain(region_count: usize) -> Cfg<BuilderInst> {
    CfgBuilder::build((0..region_count).flat_map(|_| {
        [
            BuilderInst(FlowEffect::LoopOpen),
            BuilderInst(FlowEffect::ConditionalBreak),
            BuilderInst(FlowEffect::LoopClose),
        ]
    }))
    .expect("synthetic loops are balanced")
}

pub(super) fn build_two_case_switch_chain(region_count: usize) -> Cfg<BuilderInst> {
    CfgBuilder::build((0..region_count).flat_map(|_| {
        [
            BuilderInst(FlowEffect::SwitchOpen),
            BuilderInst(FlowEffect::Fallthrough),
            BuilderInst(FlowEffect::SwitchCase),
            BuilderInst(FlowEffect::Fallthrough),
            BuilderInst(FlowEffect::SwitchClose),
        ]
    }))
    .expect("synthetic switches are balanced")
}

pub(super) fn build_eight_case_switch_chain(region_count: usize) -> Cfg<BuilderInst> {
    CfgBuilder::build((0..region_count).flat_map(|_| {
        (0..17).map(|position| {
            let effect = match position {
                0 => FlowEffect::SwitchOpen,
                16 => FlowEffect::SwitchClose,
                odd if odd % 2 == 1 => FlowEffect::Fallthrough,
                _ => FlowEffect::SwitchCase,
            };
            BuilderInst(effect)
        })
    }))
    .expect("synthetic switches are balanced")
}
