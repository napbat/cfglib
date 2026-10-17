//! The read-only view of a control-flow graph that the dataflow and analysis
//! algorithms read.
//!
//! [`CfgView`] is the one trait the instruction-level algorithms take. It
//! joins the graph views of the crate with the facts that an
//! instruction-level algorithm needs and a graph view does not give: the
//! ordered instructions of each block, the [`EdgeKind`](crate::EdgeKind) of each edge, and
//! the optional branch weight of each edge.
//! [`Cfg`] implements it, so every caller that holds a [`Cfg`] keeps working.
//! A consumer with its own storage implements it to run the same algorithms
//! without a conversion.

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::{EdgeId, KindedEdge};
use crate::graph::edge_view::EdgeView;
use crate::graph::view::RootedView;

/// A rooted graph of blocks that hold ordered instructions and have typed
/// edges.
///
/// The graph part comes from [`RootedView`] and [`EdgeView`]. The root is the
/// entry block. Blocks are named by [`BlockId`] and edges by [`EdgeId`], so
/// every fact that an analysis reports uses one identity for each. The
/// kind of an edge comes from the [`KindedEdge`] data of [`EdgeView`].
///
/// # Exception regions
///
/// The view has no separate exception model. An analysis finds the
/// exception-region information in two places:
///
/// - An edge of kind [`EdgeKind::ExceptionUnwind`](crate::EdgeKind::ExceptionUnwind)
///   leaves its source block before a throwing instruction. An edge of kind
///   [`EdgeKind::ExceptionHandler`](crate::EdgeKind::ExceptionHandler) enters a handler.
/// - [`InstrInfo::may_unwind`](super::InstrInfo::may_unwind) marks the
///   throwing instructions of a block.
///
/// [`departs_before_throws`](super::departs_before_throws) combines the two.
///
/// # Contract
///
/// - [`instructions`](Self::instructions) accepts every identity that
///   [`node_ids`](crate::GraphView::node_ids) yields. It returns the same
///   slice each time.
/// - The index of an instruction in the slice is the instruction index of
///   the [`ProgramPoint`](super::ProgramPoint) that an analysis reports.
pub trait CfgView:
    RootedView<NodeId = BlockId> + EdgeView<EdgeId = EdgeId, EdgeData: KindedEdge>
{
    /// The instruction type that the blocks hold.
    type Inst;

    /// Returns the instructions of `block` in program order.
    ///
    /// # Panics
    ///
    /// Panics when `block` does not belong to this view.
    fn instructions(&self, block: BlockId) -> &[Self::Inst];

    /// Returns the branch weight of an edge, if the view has one.
    ///
    /// A weight is a branch probability in `0.0..=1.0`: the likelihood that
    /// control takes the edge, relative to the other edges leaving its
    /// source. The default answers `None`, which means that the view carries
    /// no profile.
    ///
    /// # Panics
    ///
    /// Panics when `edge` does not belong to this view.
    fn edge_weight(&self, _edge: EdgeId) -> Option<f64> {
        None
    }
}

impl<I, E> CfgView for Cfg<I, E> {
    type Inst = I;

    fn instructions(&self, block: BlockId) -> &[I] {
        self.block(block).instructions()
    }

    fn edge_weight(&self, edge: EdgeId) -> Option<f64> {
        self.edge(edge).weight()
    }
}

#[cfg(test)]
mod tests;
