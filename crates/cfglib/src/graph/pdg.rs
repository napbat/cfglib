//! Program-dependence graph construction on [`Graph`].
//!
//! The builder combines control dependences and def-use chains into one graph
//! that can be traversed in either direction for slicing, provenance, clone
//! detection, and deobfuscation. It deliberately returns the common graph
//! storage instead of introducing another graph wrapper.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::def_use::DefUseChains;
use crate::dataflow::{InstrInfo, ProgramPoint};
use crate::graph::cdg::control_dependence_graph;
use crate::graph::dominator::DominatorTree;
use crate::graph::store::{Graph, NodeId};

/// A node in a program-dependence graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DependenceNode {
    /// Synthetic predicate node for a CFG block.
    ///
    /// Empty branch blocks still need a node that can carry control
    /// dependences, so predicates cannot be represented only by instructions.
    Block(BlockId),
    /// A concrete instruction in the source CFG.
    Instruction(ProgramPoint),
}

/// Relation carried by a program-dependence edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DependenceKind {
    /// The source predicate determines whether the target executes.
    Control,
    /// The target reads a value defined by the source instruction.
    Data,
}

/// Build a program-dependence graph for `cfg`.
///
/// Edges point from a cause to its dependent. Reverse traversal therefore
/// computes a backward slice, while forward traversal computes affected code.
/// Every CFG block receives a synthetic predicate node, and every instruction
/// receives an instruction node. For non-empty controlling blocks, the last
/// instruction feeds the synthetic predicate so a backward slice can continue
/// through values used by the branch.
#[must_use]
pub fn program_dependence_graph<I: InstrInfo>(
    cfg: &Cfg<I>,
) -> Graph<DependenceNode, DependenceKind> {
    let instruction_count = cfg
        .blocks()
        .map(|block| block.instructions().len())
        .sum::<usize>();
    let mut graph = Graph::with_capacity(
        cfg.block_count().saturating_add(instruction_count),
        cfg.edge_count().saturating_add(instruction_count),
    );

    // Both tables are addressed by block index, so they span every slot up to
    // the identity bound; a retired slot keeps no predicate and no
    // instructions.
    let mut block_nodes: Vec<Option<NodeId>> = alloc::vec![None; cfg.block_bound()];
    let mut instruction_nodes: Vec<Vec<NodeId>> = alloc::vec![Vec::new(); cfg.block_bound()];
    for block in cfg.block_ids() {
        block_nodes[block.index()] = Some(graph.add_node(DependenceNode::Block(block)));
    }
    for block in cfg.block_ids() {
        instruction_nodes[block.index()] = (0..cfg.block(block).instructions().len())
            .map(|inst_idx| {
                graph.add_node(DependenceNode::Instruction(ProgramPoint {
                    block,
                    inst_idx,
                }))
            })
            .collect();
    }
    let post_dominators = DominatorTree::compute_post(cfg);
    let control = control_dependence_graph(cfg, &post_dominators);
    let mut controllers = BTreeSet::new();
    for edge in control.edges() {
        let controller = *control.node(edge.source());
        let dependent = *control.node(edge.target());
        // The dependence graph spans every block slot, so skip any pair a
        // retired slot contributed: it carries no predicate node.
        let (Some(controller_node), Some(dependent_node)) = (
            block_nodes[controller.index()],
            block_nodes[dependent.index()],
        ) else {
            continue;
        };
        controllers.insert(controller);

        graph.add_edge(controller_node, dependent_node, DependenceKind::Control);
        for &instruction in &instruction_nodes[dependent.index()] {
            graph.add_edge(controller_node, instruction, DependenceKind::Control);
        }
    }

    for controller in controllers {
        let Some(controller_node) = block_nodes[controller.index()] else {
            continue;
        };
        if let Some(&predicate) = instruction_nodes[controller.index()].last() {
            graph.add_edge(predicate, controller_node, DependenceKind::Control);
        }
    }

    let def_use = DefUseChains::compute(cfg);
    for (definition, uses) in &def_use.def_use {
        let source = instruction_nodes[definition.block.index()][definition.inst_idx];
        for use_site in uses {
            let target = instruction_nodes[use_site.block.index()][use_site.inst_idx];
            graph.add_edge(source, target, DependenceKind::Data);
        }
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::EdgeKind;
    use crate::graph::traverse::{TraversalDirection, depth_first_preorder};
    use crate::test_util::{DfInst, df_def, df_use};

    fn find_node(graph: &Graph<DependenceNode, DependenceKind>, payload: DependenceNode) -> NodeId {
        graph
            .node_ids()
            .find(|&node| *graph.node(node) == payload)
            .expect("dependence node must exist")
    }

    #[test]
    fn diamond_contains_control_and_data_edges() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let left = cfg.new_block();
        let right = cfg.new_block();
        let merge = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_def("branch", 0));
        cfg.block_mut(left).push(df_def("def_left", 1));
        cfg.block_mut(right).push(df_def("def_right", 2));
        cfg.block_mut(merge).push(df_use("use_left", 1));
        cfg.add_edge(cfg.entry(), left, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), right, EdgeKind::ConditionalFalse);
        cfg.add_edge(left, merge, EdgeKind::Fallthrough);
        cfg.add_edge(right, merge, EdgeKind::Fallthrough);

        let graph = program_dependence_graph(&cfg);
        let controller = find_node(&graph, DependenceNode::Block(cfg.entry()));
        let left_block = find_node(&graph, DependenceNode::Block(left));
        let left_definition = find_node(
            &graph,
            DependenceNode::Instruction(ProgramPoint {
                block: left,
                inst_idx: 0,
            }),
        );
        let merge_use = find_node(
            &graph,
            DependenceNode::Instruction(ProgramPoint {
                block: merge,
                inst_idx: 0,
            }),
        );

        assert!(graph.successors(controller).any(|node| node == left_block));
        assert!(graph.edges().any(|edge| {
            edge.source() == left_definition
                && edge.target() == merge_use
                && *edge.payload() == DependenceKind::Data
        }));
    }

    #[test]
    fn reverse_traversal_is_a_backward_slice() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let use_block = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_def("def", 0));
        cfg.block_mut(use_block).push(df_use("use", 0));
        cfg.add_edge(cfg.entry(), use_block, EdgeKind::Fallthrough);

        let graph = program_dependence_graph(&cfg);
        let definition = find_node(
            &graph,
            DependenceNode::Instruction(ProgramPoint {
                block: cfg.entry(),
                inst_idx: 0,
            }),
        );
        let seed = find_node(
            &graph,
            DependenceNode::Instruction(ProgramPoint {
                block: use_block,
                inst_idx: 0,
            }),
        );
        let slice = depth_first_preorder(&graph, seed, TraversalDirection::Incoming);

        assert!(slice.contains(&seed));
        assert!(slice.contains(&definition));
    }
}
