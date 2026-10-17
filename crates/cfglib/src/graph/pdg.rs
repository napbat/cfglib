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
use crate::graph::cdg::control_dependence_graph;
use crate::graph::dominator::PostDominatorTree;
use crate::graph::store::{Graph, NodeId};
use crate::{InstrInfo, ProgramPoint};

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
    /// The source predicate determines whether the target executes. Every
    /// control edge starts at a [`DependenceNode::Block`].
    Control,
    /// The target reads a value defined by the source instruction. This
    /// includes the edge from the last instruction of a controlling block to
    /// that block's predicate node, which reads the value the branch tests.
    Data,
}

/// Build a program-dependence graph for `cfg`.
///
/// Edges point from a cause to its dependent. Reverse traversal therefore
/// computes a backward slice, while forward traversal computes affected code.
/// Every CFG block receives a synthetic predicate node, and every instruction
/// receives an instruction node. For non-empty controlling blocks, the last
/// instruction feeds the synthetic predicate through a data edge, so a
/// backward slice can continue through values used by the branch.
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
    let post_dominators = PostDominatorTree::compute(cfg);
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

    // The predicate of a controlling block reads what the block's last
    // instruction produced for its branch, which is a value flowing in, not
    // control flowing out.
    for controller in controllers {
        let Some(controller_node) = block_nodes[controller.index()] else {
            continue;
        };
        if let Some(&predicate) = instruction_nodes[controller.index()].last() {
            graph.add_edge(predicate, controller_node, DependenceKind::Data);
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

    #[test]
    fn a_block_that_loops_back_to_itself_controls_its_own_predicate_and_instructions() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_def("test", 0));
        cfg.add_edge(cfg.entry(), cfg.entry(), EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), exit, EdgeKind::ConditionalFalse);

        let graph = program_dependence_graph(&cfg);
        let predicate = find_node(&graph, DependenceNode::Block(cfg.entry()));
        let instruction = find_node(
            &graph,
            DependenceNode::Instruction(ProgramPoint {
                block: cfg.entry(),
                inst_idx: 0,
            }),
        );
        let has_edge = |source: NodeId, target: NodeId, kind: DependenceKind| {
            graph.edges().any(|edge| {
                edge.source() == source && edge.target() == target && *edge.payload() == kind
            })
        };
        let controls = |source, target| has_edge(source, target, DependenceKind::Control);
        let feeds = |source, target| has_edge(source, target, DependenceKind::Data);

        assert!(controls(predicate, predicate));
        assert!(controls(predicate, instruction));
        assert!(
            !controls(instruction, predicate),
            "an instruction does not control its predicate"
        );
        assert!(feeds(instruction, predicate));
    }

    #[test]
    fn every_control_edge_starts_at_a_predicate() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let [left, right, merge] = [(); 3].map(|()| cfg.new_block());
        cfg.block_mut(cfg.entry()).push(df_def("branch", 0));
        cfg.block_mut(left).push(df_def("left", 1));
        cfg.block_mut(merge).push(df_use("use", 1));
        cfg.add_edge(cfg.entry(), left, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), right, EdgeKind::ConditionalFalse);
        cfg.add_edge(left, merge, EdgeKind::Fallthrough);
        cfg.add_edge(right, merge, EdgeKind::Fallthrough);

        let graph = program_dependence_graph(&cfg);

        let control: Vec<_> = graph
            .edges()
            .filter(|edge| *edge.payload() == DependenceKind::Control)
            .collect();
        assert_ne!(control.len(), 0);
        for edge in control {
            assert!(
                matches!(graph.node(edge.source()), DependenceNode::Block(_)),
                "control edge from {:?}",
                graph.node(edge.source())
            );
        }
    }

    #[test]
    fn a_data_slice_from_a_predicate_reaches_the_values_its_branch_tests() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let [then_block, else_block] = [(); 2].map(|()| cfg.new_block());
        cfg.block_mut(cfg.entry()).push(df_def("operand", 0));
        cfg.block_mut(cfg.entry()).push(df_use("test", 0));
        cfg.add_edge(cfg.entry(), then_block, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), else_block, EdgeKind::ConditionalFalse);

        let graph = program_dependence_graph(&cfg);
        let point = |inst_idx| {
            find_node(
                &graph,
                DependenceNode::Instruction(ProgramPoint {
                    block: cfg.entry(),
                    inst_idx,
                }),
            )
        };
        let mut slice = BTreeSet::from([find_node(&graph, DependenceNode::Block(cfg.entry()))]);
        let mut pending: Vec<_> = slice.iter().copied().collect();
        while let Some(node) = pending.pop() {
            for edge in graph.edges() {
                if edge.target() == node
                    && *edge.payload() == DependenceKind::Data
                    && slice.insert(edge.source())
                {
                    pending.push(edge.source());
                }
            }
        }

        assert!(slice.contains(&point(1)), "the instruction that tests");
        assert!(slice.contains(&point(0)), "the definition it tests");
    }
}
