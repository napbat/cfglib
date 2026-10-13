//! Selection resolution: which candidate an indexed selection picks.
//!
//! An indexed read or write names its candidates and a computed index. The
//! analysis follows the constants of the roots across blocks, evaluates the
//! index of each selection with the [`Fold`] hook of the dialect, and reports
//! the selections whose index is a constant.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::roots::RootTable;
use super::words::normalized;
use crate::graph::traverse::{TraversalDirection, reverse_postorder};
use crate::graph::view::DenseId;
use crate::ir::rtl::{Dialect, Fold, FunctionView};

mod state;
mod transfer;

use state::ConstState;
use transfer::Transfer;

/// A root with a known value on entry to the function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootConstant<'a, D: Dialect> {
    /// The root.
    pub root: D::Root,
    /// The value of the root as little-endian 64-bit words. Bits past the
    /// root width are ignored.
    pub words: &'a [u64],
}

/// One indexed selection whose index is a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResolvedSelection<I> {
    /// The instruction that holds the selection.
    pub inst: I,
    /// The position of the statement among the statements of the record.
    pub statement: usize,
    /// The position of the selected place among the candidates.
    pub candidate: usize,
}

/// The indexed selections of a function whose index is a constant.
///
/// The facts cover the selections of reads and of writes. A selection is
/// absent from the facts when its index varies, when its block is
/// unreachable, or when the constant is not below the number of candidates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionResolution<I> {
    /// The resolved selections, sorted by instruction and statement.
    resolved: Vec<ResolvedSelection<I>>,
}

impl<I: DenseId> SelectionResolution<I> {
    /// Resolves the indexed selections of `function`.
    ///
    /// The analysis starts at the entry block with the roots of `entry` known
    /// and every other root unknown. It tracks the value of a root through
    /// the writes of each block, with these rules:
    ///
    /// - A read of a known root, a constant, and an operator that
    ///   [`Fold::fold`] evaluates over constant operands give constants.
    /// - A write gives its root a constant when every bit it writes is a
    ///   constant. Other writes make their root unknown.
    /// - An indexed write with a constant index changes one candidate. An
    ///   indexed write with another index makes every candidate unknown, and
    ///   a sparse write makes its bank unknown.
    /// - After a call, every root is unknown, because the callee can write
    ///   any root.
    /// - At a join, a root keeps its value when every incoming path agrees.
    ///
    /// A root of `entry` that the function does not name is ignored.
    ///
    /// # Panics
    ///
    /// Panics when `roots` was not collected from `function`.
    #[must_use]
    pub fn compute<D, F>(function: &F, roots: &RootTable<D>, entry: &[RootConstant<'_, D>]) -> Self
    where
        D: Fold,
        F: FunctionView<D, Inst = I>,
    {
        let bound = function.node_bound();
        let order = reverse_postorder(function, function.root(), TraversalDirection::Outgoing);
        let mut incoming: Vec<Option<ConstState>> = vec![None; bound];
        let mut dirty = vec![false; bound];
        let root = function.root().index();
        incoming[root] = Some(entry_state(roots, entry));
        dirty[root] = true;

        let mut transfer = Transfer::new(function, roots);
        let mut changed = true;
        while changed {
            changed = false;
            for &block in &order {
                if !dirty[block.index()] {
                    continue;
                }
                dirty[block.index()] = false;
                let mut state = incoming[block.index()]
                    .clone()
                    .expect("a block with new facts has an entry state");
                transfer.run_block(block, &mut state, None);
                for successor in function.successors(block) {
                    if join(&mut incoming[successor.index()], &state) {
                        dirty[successor.index()] = true;
                        changed = true;
                    }
                }
            }
        }

        let mut resolved = Vec::new();
        for &block in &order {
            if let Some(entry_state) = &incoming[block.index()] {
                let mut state = entry_state.clone();
                transfer.run_block(block, &mut state, Some(&mut resolved));
            }
        }
        resolved.sort_unstable_by_key(|selection: &ResolvedSelection<I>| {
            (selection.inst, selection.statement)
        });
        Self { resolved }
    }

    /// Returns the candidate that the selection at `statement` of `inst`
    /// picks, or `None` when the analysis did not resolve it.
    #[must_use]
    pub fn candidate(&self, inst: I, statement: usize) -> Option<usize> {
        self.resolved
            .binary_search_by_key(&(inst, statement), |selection| {
                (selection.inst, selection.statement)
            })
            .ok()
            .map(|position| self.resolved[position].candidate)
    }

    /// Returns the resolved selections, sorted by instruction and statement.
    #[must_use]
    pub fn resolved(&self) -> &[ResolvedSelection<I>] {
        &self.resolved
    }
}

/// Builds the state on entry to the function.
fn entry_state<D: Dialect>(roots: &RootTable<D>, entry: &[RootConstant<'_, D>]) -> ConstState {
    let mut state = ConstState::default();
    for constant in entry {
        if let Some(index) = roots.index_of(constant.root) {
            state.set(
                index,
                normalized(constant.words, D::root_width(constant.root)),
            );
        }
    }
    state
}

/// Merges the state that leaves a block into the state that enters its
/// successor. Returns whether the successor state changed.
fn join(incoming: &mut Option<ConstState>, leaving: &ConstState) -> bool {
    match incoming {
        None => {
            *incoming = Some(leaving.clone());
            true
        }
        Some(state) => state.intersect(leaving),
    }
}
