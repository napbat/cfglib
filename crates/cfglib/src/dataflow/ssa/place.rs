//! Iterated-dominance-frontier phi placement, and the flat draft the renamer
//! fills in.
//!
//! Placement and renaming are one pass split in two: placement decides which
//! block merges which variable and from which predecessors, and renaming
//! fills every operand it left open. So the placed phis go straight into the
//! storage the renamer will complete — [`PhiDrafts`] — and [`PhiPlacements`],
//! the public answer for a consumer that wants placement alone, is built from
//! that.

extern crate alloc;

use alloc::collections::btree_map::Entry;
use alloc::vec::Vec;
use core::ops::Range;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::{InstrInfo, VariableId};
use crate::graph::dominator::DominatorTree;

use super::SsaValue;
use super::scratch::SsaScratch;

/// A structural phi placement before SSA values are renamed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhiPlacement<V> {
    /// The source-IR variable merged by the phi.
    pub variable: V,
    /// CFG predecessors that contribute operands, in CFG predecessor order.
    pub predecessors: Vec<BlockId>,
}

/// Phi placements indexed by containing block.
#[derive(Debug, Clone)]
pub struct PhiPlacements<V> {
    placements: Vec<Vec<PhiPlacement<V>>>,
}

impl<V> PhiPlacements<V> {
    /// Return phi placements at `block`.
    #[must_use]
    pub fn at(&self, block: BlockId) -> &[PhiPlacement<V>] {
        &self.placements[block.index()]
    }

    /// Return the total number of placed phis.
    #[must_use]
    pub fn len(&self) -> usize {
        self.placements.iter().map(Vec::len).sum()
    }

    /// Return whether no phis were placed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over all `(block, placement)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (BlockId, &PhiPlacement<V>)> {
        self.placements
            .iter()
            .enumerate()
            .flat_map(|(index, phis)| {
                phis.iter()
                    .map(move |phi| (BlockId::from_index(index), phi))
            })
    }
}

impl<V: VariableId> PhiPlacements<V> {
    /// Place phis for every variable defined in the CFG.
    ///
    /// This is the iterated-dominance-frontier phase of SSA construction. Use
    /// [`SsaForm::compute`](super::SsaForm::compute) when renamed definitions
    /// and operands are also required (and see its precondition: the entry
    /// block must not be a branch target).
    #[must_use]
    pub fn compute<I: InstrInfo<Variable = V>, E>(cfg: &Cfg<I, E>, dom: &DominatorTree) -> Self {
        let mut scratch = SsaScratch::new();
        scratch.reset_placement(cfg.block_bound());
        place_phis(&mut scratch, cfg, dom);

        let drafts = &scratch.phis;
        let placements = (0..cfg.block_bound())
            .map(|index| {
                drafts
                    .at(BlockId::from_index(index))
                    .iter()
                    .map(|draft| PhiPlacement {
                        variable: draft.variable.clone(),
                        predecessors: drafts.predecessors(draft).to_vec(),
                    })
                    .collect()
            })
            .collect();
        Self { placements }
    }
}

/// One placed phi, with its operands still open.
///
/// The predecessors and the operand slots live in [`PhiDrafts`]' flat arrays
/// rather than in two vectors of this struct's own: a procedure has as many
/// phis as it has merged variables, and two allocations each is the cost the
/// flat form exists to remove.
#[derive(Debug)]
pub(super) struct PhiDraft<V> {
    /// The source-IR variable this phi merges.
    pub(super) variable: V,
    /// This phi's run in both flat arrays: predecessor `i` of the phi is
    /// `predecessors[operands.start + i]`, and the value arriving along it is
    /// `operands[operands.start + i]`.
    pub(super) operands: Range<usize>,
    /// The SSA value the phi defines, assigned when its block is renamed.
    pub(super) result: Option<SsaValue<V>>,
}

/// Every placed phi of one procedure, in flat storage a scratch reuses.
#[derive(Debug)]
pub(super) struct PhiDrafts<V> {
    /// Per block, its phis in placement order. Never shortened, because a
    /// shorter procedure after a longer one would otherwise release the inner
    /// vectors this exists to keep.
    pub(super) by_block: Vec<Vec<PhiDraft<V>>>,
    /// Every phi's predecessors, end to end.
    pub(super) predecessors: Vec<BlockId>,
    /// One operand slot per entry in [`predecessors`](Self::predecessors),
    /// filled while the predecessor block is renamed.
    pub(super) operands: Vec<Option<SsaValue<V>>>,
}

impl<V> Default for PhiDrafts<V> {
    fn default() -> Self {
        Self {
            by_block: Vec::new(),
            predecessors: Vec::new(),
            operands: Vec::new(),
        }
    }
}

impl<V: VariableId> PhiDrafts<V> {
    /// Empty every run, keeping the storage, and cover `block_bound` blocks.
    pub(super) fn reset(&mut self, block_bound: usize) {
        if self.by_block.len() < block_bound {
            self.by_block.resize_with(block_bound, Vec::new);
        }
        for phis in &mut self.by_block {
            phis.clear();
        }
        self.predecessors.clear();
        self.operands.clear();
    }

    /// Record a phi for `variable` at `block`, merging `predecessors`.
    fn place(&mut self, block: BlockId, variable: V, predecessors: impl Iterator<Item = BlockId>) {
        let start = self.predecessors.len();
        self.predecessors.extend(predecessors);
        let end = self.predecessors.len();
        self.operands.resize(end, None);
        self.by_block[block.index()].push(PhiDraft {
            variable,
            operands: start..end,
            result: None,
        });
    }

    /// The phis placed at `block`, in placement order.
    pub(super) fn at(&self, block: BlockId) -> &[PhiDraft<V>] {
        &self.by_block[block.index()]
    }

    /// One phi's predecessors, in CFG predecessor order.
    pub(super) fn predecessors(&self, draft: &PhiDraft<V>) -> &[BlockId] {
        &self.predecessors[draft.operands.clone()]
    }

    /// Record the value reaching every phi of `block` along the edge from
    /// `predecessor`.
    ///
    /// A parallel edge occupies two operand positions and receives the value
    /// at both, which is what the per-phi map this replaced did by answering
    /// the same value to both of its lookups.
    pub(super) fn set_operands(
        &mut self,
        block: BlockId,
        predecessor: BlockId,
        mut value: impl FnMut(&V) -> SsaValue<V>,
    ) {
        for draft in &self.by_block[block.index()] {
            let mut arriving: Option<SsaValue<V>> = None;
            for index in draft.operands.clone() {
                if self.predecessors[index] != predecessor {
                    continue;
                }
                if arriving.is_none() {
                    arriving = Some(value(&draft.variable));
                }
                self.operands[index].clone_from(&arriving);
            }
        }
    }
}

/// Place a phi for every variable at every iterated dominance frontier of its
/// definitions, leaving the result in `scratch.phis`.
pub(super) fn place_phis<I: InstrInfo, E>(
    scratch: &mut SsaScratch<I::Variable>,
    cfg: &Cfg<I, E>,
    dom: &DominatorTree,
) {
    let SsaScratch {
        frontiers,
        definition_blocks,
        block_pool,
        has_phi,
        placed,
        worklist,
        phis,
        ..
    } = scratch;
    frontiers.rebuild(cfg, dom);

    for block_id in cfg.block_ids() {
        for instruction in cfg.block(block_id).instructions() {
            for variable in instruction.defs() {
                let blocks = match definition_blocks.entry(variable.clone()) {
                    Entry::Vacant(slot) => slot.insert(block_pool.pop().unwrap_or_default()),
                    Entry::Occupied(slot) => slot.into_mut(),
                };
                if blocks.last().copied() != Some(block_id) {
                    blocks.push(block_id);
                }
            }
        }
    }

    for (variable, definitions) in definition_blocks.iter_mut() {
        has_phi.reset();
        placed.reset();
        for &block in definitions.iter() {
            placed.mark(block.index());
        }
        // The definition list is moved rather than copied, so the vector it
        // leaves behind stays in the map and is recycled by the next reset.
        worklist.clear();
        worklist.append(definitions);

        while let Some(block) = worklist.pop() {
            for &frontier_block in frontiers.frontier(block) {
                if has_phi.is_marked(frontier_block.index()) {
                    continue;
                }
                has_phi.mark(frontier_block.index());

                phis.place(
                    frontier_block,
                    variable.clone(),
                    cfg.predecessors(frontier_block),
                );
                if !placed.is_marked(frontier_block.index()) {
                    placed.mark(frontier_block.index());
                    worklist.push(frontier_block);
                }
            }
        }
    }
}
