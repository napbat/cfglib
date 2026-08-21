//! Generic fixpoint iteration engine for data flow analysis.
//!
//! Supports both **forward** and **backward** analyses via a worklist
//! algorithm that iterates until the solution stabilizes.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;

/// Direction of the data flow analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Forward: information flows from predecessors to successors.
    Forward,
    /// Backward: information flows from successors to predecessors.
    Backward,
}

/// A data flow problem to be solved by the fixpoint engine.
///
/// `F` is the flow fact type (e.g. `BTreeSet<DefSite>` for reaching
/// definitions, or `BTreeSet<I::Variable>` for liveness).
/// Termination requires `meet` and `transfer` to be monotone over a
/// finite-height fact lattice.
pub trait Problem<I> {
    /// The flow fact (lattice element) type.
    type Fact: Clone + PartialEq;

    /// Analysis direction.
    fn direction(&self) -> Direction;

    /// Initial (bottom) value for each block.
    fn bottom(&self) -> Self::Fact;

    /// Initial value for the entry (forward) or each exit (backward) block.
    fn entry_fact(&self) -> Self::Fact;

    /// Meet/join operator: merge information from multiple paths.
    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact;

    /// Transfer function: given the incoming fact for a block, compute
    /// the outgoing fact after the block's instructions are applied.
    fn transfer(&self, cfg: &Cfg<I>, block: BlockId, input: &Self::Fact) -> Self::Fact;
}

/// Result of a fixpoint computation.
#[derive(Debug, Clone)]
pub struct FixpointResult<F> {
    /// The IN fact for each block (indexed by `BlockId::index()`).
    pub block_in: Vec<F>,
    /// The OUT fact for each block (indexed by `BlockId::index()`).
    pub block_out: Vec<F>,
}

impl<F> FixpointResult<F> {
    /// Get the IN fact for a block.
    #[must_use]
    pub fn fact_in(&self, block: BlockId) -> &F {
        &self.block_in[block.index()]
    }

    /// Get the OUT fact for a block.
    #[must_use]
    pub fn fact_out(&self, block: BlockId) -> &F {
        &self.block_out[block.index()]
    }
}

/// Run the fixpoint iteration for the given problem on the CFG.
///
/// # Examples
///
/// See [`Liveness::compute`](crate::dataflow::liveness::Liveness::compute)
/// and [`ReachingDefs::compute`](crate::dataflow::reaching::ReachingDefs::compute)
/// for concrete usage.
///
/// ```
/// # use cfglib::{Cfg, EdgeKind, InstrInfo};
/// # #[derive(Debug, Clone)]
/// # struct Inst { uses: Vec<u16>, defs: Vec<u16> }
/// # impl InstrInfo for Inst {
/// #     type Variable = u16;
/// #     fn uses(&self) -> &[u16] { &self.uses }
/// #     fn defs(&self) -> &[u16] { &self.defs }
/// # }
/// use cfglib::dataflow::liveness::Liveness;
///
/// let mut cfg = Cfg::<Inst>::new();
/// cfg.block_mut(cfg.entry()).push(Inst { uses: vec![], defs: vec![0] });
///
/// let live = Liveness::compute(&cfg);
/// assert!(live.live_in(cfg.entry()).is_empty()); // r0 defined, not used
/// ```
pub fn solve<I, P: Problem<I>>(cfg: &Cfg<I>, problem: &P) -> FixpointResult<P::Fact> {
    let n = cfg.num_blocks();
    let bottom = problem.bottom();

    let mut block_in: Vec<P::Fact> = vec![bottom.clone(); n];
    let mut block_out: Vec<P::Fact> = vec![bottom.clone(); n];

    // Set entry/exit initial fact.
    match problem.direction() {
        Direction::Forward => {
            block_in[cfg.entry().index()] = problem.entry_fact();
            block_out[cfg.entry().index()] =
                problem.transfer(cfg, cfg.entry(), &block_in[cfg.entry().index()]);
        }
        Direction::Backward => {
            // For backward analysis, initialise all exit blocks.
            for b in cfg.blocks() {
                if cfg.successor_edges(b.id()).is_empty() {
                    block_out[b.id().index()] = problem.entry_fact();
                    block_in[b.id().index()] =
                        problem.transfer(cfg, b.id(), &block_out[b.id().index()]);
                }
            }
        }
    }

    // Forward solving ignores blocks unreachable from the entry. Backward
    // solving starts from every allocated block because every exit receives
    // the boundary fact, including exits in disconnected components.
    let mut worklist: BTreeSet<u32> = match problem.direction() {
        Direction::Forward => cfg
            .dfs_preorder()
            .into_iter()
            .map(|block| block.0)
            .collect(),
        Direction::Backward => cfg.blocks().iter().map(|block| block.id().0).collect(),
    };

    while let Some(raw_block) = worklist.pop_first() {
        let block = BlockId(raw_block);
        match problem.direction() {
            Direction::Forward => {
                // IN = meet of all predecessors' OUT.
                let mut preds = cfg.predecessors(block);
                let merged = match preds.next() {
                    None => problem.entry_fact(),
                    Some(first) => {
                        let mut m = block_out[first.index()].clone();
                        for p in preds {
                            m = problem.meet(&m, &block_out[p.index()]);
                        }
                        m
                    }
                };
                block_in[block.index()] = merged;

                let new_out = problem.transfer(cfg, block, &block_in[block.index()]);
                if new_out != block_out[block.index()] {
                    block_out[block.index()] = new_out;
                    for s in cfg.successors(block) {
                        worklist.insert(s.0);
                    }
                }
            }
            Direction::Backward => {
                // OUT = meet of all successors' IN.
                let mut succs = cfg.successors(block);
                let merged = match succs.next() {
                    None => problem.entry_fact(),
                    Some(first) => {
                        let mut m = block_in[first.index()].clone();
                        for s in succs {
                            m = problem.meet(&m, &block_in[s.index()]);
                        }
                        m
                    }
                };
                block_out[block.index()] = merged;

                let new_in = problem.transfer(cfg, block, &block_out[block.index()]);
                if new_in != block_in[block.index()] {
                    block_in[block.index()] = new_in;
                    for p in cfg.predecessors(block) {
                        worklist.insert(p.0);
                    }
                }
            }
        }
    }

    FixpointResult {
        block_in,
        block_out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::EdgeKind;

    #[test]
    fn backward_solver_propagates_through_disconnected_components() {
        struct BackwardReachability;

        impl Problem<()> for BackwardReachability {
            type Fact = bool;

            fn direction(&self) -> Direction {
                Direction::Backward
            }

            fn bottom(&self) -> Self::Fact {
                false
            }

            fn entry_fact(&self) -> Self::Fact {
                true
            }

            fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact {
                *a || *b
            }

            fn transfer(&self, _cfg: &Cfg<()>, _block: BlockId, input: &Self::Fact) -> Self::Fact {
                *input
            }
        }

        let mut cfg = Cfg::new();
        let disconnected_predecessor = cfg.new_block();
        let disconnected_exit = cfg.new_block();
        cfg.add_edge(
            disconnected_predecessor,
            disconnected_exit,
            EdgeKind::Fallthrough,
        );
        let result = solve(&cfg, &BackwardReachability);
        assert!(*result.fact_in(disconnected_predecessor));
        assert!(*result.fact_out(disconnected_predecessor));
        assert!(*result.fact_in(disconnected_exit));
        assert!(*result.fact_out(disconnected_exit));
    }
}
