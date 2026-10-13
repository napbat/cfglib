//! Every instruction-level analysis gives the same answer on a [`CfgView`]
//! that is not a [`Cfg`].
//!
//! The toy view keeps instructions and edges in plain vectors. It shares no
//! code with the store behind [`Cfg`], so a result that matches the one on
//! the [`Cfg`] proves that the analysis reads the graph only through the
//! trait.

extern crate alloc;

use alloc::vec::Vec;

use super::CfgView;
use crate::analysis::dead_code::DeadCode;
use crate::analysis::profile::{CfgProfile, set_uniform_edge_weights};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::constant_propagation::constant_propagation;
use crate::dataflow::def_use::DefUseChains;
use crate::dataflow::fixpoint::solve_problem;
use crate::dataflow::liveness::{Liveness, LivenessProblem};
use crate::dataflow::reaching::ReachingDefs;
use crate::dataflow::ssa::SsaForm;
use crate::edge::{EdgeId, EdgeKind, KindedEdge};
use crate::flow::FlowEffect;
use crate::graph::dominator::DominatorTree;
use crate::graph::edge_view::{EdgeRef, EdgeView};
use crate::graph::view::{GraphView, RootedView};
use crate::test_util::{DfInst, df_const, df_def, df_use, df_with_effect};

/// One edge of the toy storage.
#[derive(Debug)]
struct ToyEdge {
    source: BlockId,
    target: BlockId,
    kind: EdgeKind,
    weight: Option<f64>,
}

impl KindedEdge for ToyEdge {
    fn kind(&self) -> EdgeKind {
        self.kind
    }
}

/// Blocks and edges in plain vectors, with no [`Cfg`] behind them.
struct ToyView {
    entry: BlockId,
    blocks: Vec<Vec<DfInst>>,
    edges: Vec<ToyEdge>,
}

impl ToyView {
    /// Copies the blocks and the edges of `cfg`, keeping their order.
    fn copy_of(cfg: &Cfg<DfInst>) -> Self {
        let mut blocks = alloc::vec![Vec::new(); cfg.block_bound()];
        for block in cfg.block_ids() {
            blocks[block.index()] = cfg.block(block).instructions().to_vec();
        }
        let edges = cfg
            .edge_ids()
            .map(|id| {
                let edge = cfg.edge(id);
                ToyEdge {
                    source: edge.source(),
                    target: edge.target(),
                    kind: edge.kind(),
                    weight: edge.weight(),
                }
            })
            .collect();
        Self {
            entry: cfg.entry(),
            blocks,
            edges,
        }
    }
}

impl GraphView for ToyView {
    type NodeId = BlockId;

    fn node_bound(&self) -> usize {
        self.blocks.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = BlockId> + '_ {
        (0..self.blocks.len()).map(BlockId::from_index)
    }

    fn successors(&self, node: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        self.edges
            .iter()
            .filter(move |edge| edge.source == node)
            .map(|edge| edge.target)
    }

    fn predecessors(&self, node: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        self.edges
            .iter()
            .filter(move |edge| edge.target == node)
            .map(|edge| edge.source)
    }
}

impl EdgeView for ToyView {
    type EdgeId = EdgeId;
    type EdgeData = ToyEdge;

    fn edge_bound(&self) -> usize {
        self.edges.len()
    }

    fn edge_ids(&self) -> impl Iterator<Item = EdgeId> + '_ {
        (0..self.edges.len()).map(EdgeId::from_index)
    }

    fn outgoing(&self, node: BlockId) -> impl Iterator<Item = EdgeId> + '_ {
        self.edge_ids()
            .filter(move |&id| self.edges[id.index()].source == node)
    }

    fn incoming(&self, node: BlockId) -> impl Iterator<Item = EdgeId> + '_ {
        self.edge_ids()
            .filter(move |&id| self.edges[id.index()].target == node)
    }

    fn edge(&self, edge: EdgeId) -> EdgeRef<'_, BlockId, EdgeId, ToyEdge> {
        let data = &self.edges[edge.index()];
        EdgeRef::new(edge, data.source, data.target, data)
    }
}

impl RootedView for ToyView {
    fn root(&self) -> BlockId {
        self.entry
    }
}

impl CfgView for ToyView {
    type Inst = DfInst;

    fn instructions(&self, block: BlockId) -> &[DfInst] {
        &self.blocks[block.index()]
    }

    fn edge_weight(&self, edge: EdgeId) -> Option<f64> {
        self.edges[edge.index()].weight
    }
}

/// A branch, a join, a loop, and an unwind edge to a handler.
///
/// The entry defines `r0` and then `r1` with an instruction that may throw.
/// The handler reads `r1`, so the unwind edge leaves before that definition
/// completes.
fn fixture() -> Cfg<DfInst> {
    let mut cfg: Cfg<DfInst> = Cfg::new();
    let entry = cfg.entry();
    let left = cfg.new_block();
    let right = cfg.new_block();
    let join = cfg.new_block();
    let handler = cfg.new_block();

    cfg.block_mut(entry).push(df_const("r0", 0, 7));
    cfg.block_mut(entry)
        .push(df_with_effect(df_def("r1", 1), FlowEffect::MayThrow));
    cfg.block_mut(left).push(df_use("use r0", 0));
    cfg.block_mut(left).push(df_def("r2 left", 2));
    cfg.block_mut(right).push(df_def("r2 right", 2));
    cfg.block_mut(join).push(df_use("use r2", 2));
    cfg.block_mut(handler).push(df_use("use r1", 1));

    cfg.add_edge(entry, left, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, right, EdgeKind::ConditionalFalse);
    cfg.add_edge(entry, handler, EdgeKind::ExceptionUnwind);
    cfg.add_edge(left, join, EdgeKind::Fallthrough);
    cfg.add_edge(right, join, EdgeKind::Fallthrough);
    cfg.add_edge(join, left, EdgeKind::Back);
    cfg
}

#[test]
fn the_toy_view_reports_the_graph_of_the_cfg() {
    let cfg = fixture();
    let toy = ToyView::copy_of(&cfg);
    assert_eq!(toy.root(), cfg.entry());
    assert_eq!(toy.node_bound(), cfg.block_bound());
    for block in cfg.block_ids() {
        assert_eq!(
            toy.successors(block).collect::<Vec<_>>(),
            cfg.successors(block).collect::<Vec<_>>()
        );
        assert_eq!(
            toy.predecessors(block).collect::<Vec<_>>(),
            cfg.predecessors(block).collect::<Vec<_>>()
        );
        assert_eq!(
            toy.instructions(block).len(),
            CfgView::instructions(&cfg, block).len()
        );
    }
}

#[test]
fn the_block_fixpoint_matches_on_a_view_that_is_not_a_cfg() {
    let cfg = fixture();
    let toy = ToyView::copy_of(&cfg);
    let expected = solve_problem(&cfg, &LivenessProblem).unwrap();
    let actual = solve_problem(&toy, &LivenessProblem).unwrap();
    for block in cfg.block_ids() {
        assert_eq!(actual.fact_in(block), expected.fact_in(block));
        assert_eq!(actual.fact_out(block), expected.fact_out(block));
    }
    assert_eq!(actual.steps(), expected.steps());
}

#[test]
fn liveness_matches_on_a_view_that_is_not_a_cfg() {
    let cfg = fixture();
    let toy = ToyView::copy_of(&cfg);
    let expected = Liveness::compute(&cfg);
    let actual = Liveness::compute(&toy);
    for block in cfg.block_ids() {
        assert_eq!(actual.live_in(block), expected.live_in(block));
        assert_eq!(actual.live_out(block), expected.live_out(block));
        assert_eq!(
            actual.live_on_unwind(&toy, block),
            expected.live_on_unwind(&cfg, block)
        );
        assert_eq!(
            actual.live_before_instructions(&toy, block),
            expected.live_before_instructions(&cfg, block)
        );
    }
    let handler_reads_r1 = actual.live_on_unwind(&toy, toy.root());
    assert!(handler_reads_r1.contains(&1));
}

#[test]
fn reaching_definitions_and_chains_match_on_a_view_that_is_not_a_cfg() {
    let cfg = fixture();
    let toy = ToyView::copy_of(&cfg);
    let expected = ReachingDefs::compute(&cfg);
    let actual = ReachingDefs::compute(&toy);
    for block in cfg.block_ids() {
        assert_eq!(actual.reaching_in(block), expected.reaching_in(block));
        assert_eq!(actual.reaching_out(block), expected.reaching_out(block));
    }
    let expected = DefUseChains::compute(&cfg);
    let actual = DefUseChains::compute(&toy);
    assert_eq!(actual.def_use, expected.def_use);
    assert_eq!(actual.use_def, expected.use_def);
}

#[test]
fn ssa_and_constants_match_on_a_view_that_is_not_a_cfg() {
    let cfg = fixture();
    let toy = ToyView::copy_of(&cfg);
    let expected = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
    let actual = SsaForm::compute(&toy, &DominatorTree::compute(&toy));
    assert_eq!(actual, expected);
    assert!(actual.phis().count() > 0, "the join needs a phi for r2");

    let expected = constant_propagation(&cfg);
    let actual = constant_propagation(&toy);
    for block in cfg.block_ids() {
        assert_eq!(actual.fact_in(block), expected.fact_in(block));
        assert_eq!(actual.fact_out(block), expected.fact_out(block));
    }
}

#[test]
fn dead_code_matches_on_a_view_that_is_not_a_cfg() {
    let cfg = fixture();
    let toy = ToyView::copy_of(&cfg);
    assert_eq!(DeadCode::compute(&toy), DeadCode::compute(&cfg));
}

#[test]
fn profiles_come_from_the_weights_that_the_view_reports() {
    let mut cfg = fixture();
    set_uniform_edge_weights(&mut cfg);
    let toy = ToyView::copy_of(&cfg);
    let expected = CfgProfile::from_edge_weights(&cfg);
    let actual = CfgProfile::from_edge_weights(&toy);
    assert_eq!(actual.edge_weights, expected.edge_weights);
    assert_eq!(actual.block_weights, expected.block_weights);
    assert!(!actual.edge_weights.is_empty());
}
