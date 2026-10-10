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
use crate::dataflow::{
    InstrInfo, ProgramPoint, VariableId, departs_before_throws, first_unwind_point,
};
use crate::edge::EdgeKind;
use crate::graph::dominator::DominatorTree;

use super::SsaValue;
use super::scratch::SsaScratch;

/// A structural phi placement before SSA values are renamed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhiPlacement<V> {
    /// The source-IR variable merged by the phi.
    pub variable: V,
    /// Departure points that contribute operands, in CFG predecessor order.
    pub predecessors: Vec<ProgramPoint>,
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
    pub(super) predecessors: Vec<ProgramPoint>,
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

    /// Record a phi for `variable` at `block`, merging the values that leave
    /// `predecessors`.
    fn place(
        &mut self,
        block: BlockId,
        variable: V,
        predecessors: impl Iterator<Item = ProgramPoint>,
    ) {
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

    /// One phi's departure points, in CFG predecessor order.
    pub(super) fn predecessors(&self, draft: &PhiDraft<V>) -> &[ProgramPoint] {
        &self.predecessors[draft.operands.clone()]
    }

    /// Record the value reaching every phi of `block` from the departure
    /// point `predecessor`.
    ///
    /// Parallel edges occupy several operand positions and receive the value
    /// at each, which is what the per-phi map this replaced did by answering
    /// the same value to every lookup.
    pub(super) fn set_operands(
        &mut self,
        block: BlockId,
        predecessor: ProgramPoint,
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
///
/// A block can dominate its unwind successor while a definition it makes
/// at or after its first throwing instruction does not reach that
/// successor. Such a definition also places a phi at the unwind successor,
/// whose operands name the value before each throwing instruction.
pub(super) fn place_phis<I: InstrInfo, E>(
    scratch: &mut SsaScratch<I::Variable>,
    cfg: &Cfg<I, E>,
    dom: &DominatorTree,
) {
    let SsaScratch {
        frontiers,
        definition_blocks,
        block_pool,
        unwound_definitions,
        has_phi,
        placed,
        worklist,
        phis,
        ..
    } = scratch;
    frontiers.rebuild(cfg, dom);

    for block_id in cfg.block_ids() {
        let first_unwind = first_unwind_point(cfg, block_id);
        for (index, instruction) in cfg.block(block_id).instructions().iter().enumerate() {
            for variable in instruction.defs() {
                let blocks = match definition_blocks.entry(variable.clone()) {
                    Entry::Vacant(slot) => slot.insert(block_pool.pop().unwrap_or_default()),
                    Entry::Occupied(slot) => slot.into_mut(),
                };
                if blocks.last().copied() != Some(block_id) {
                    blocks.push(block_id);
                }
                if first_unwind.is_some_and(|first| index >= first) {
                    unwound_definitions.push((variable.clone(), block_id));
                }
            }
        }
    }
    unwound_definitions.sort_unstable();
    unwound_definitions.dedup();

    let mut unwound = 0;
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

        // Both lists ascend by variable, so one cursor walks the unwound
        // definitions alongside the map.
        while let Some((unwound_variable, source)) = unwound_definitions.get(unwound) {
            if unwound_variable > variable {
                break;
            }
            unwound += 1;
            if unwound_variable < variable {
                continue;
            }
            for edge in cfg.outgoing(*source) {
                let edge = cfg.edge(edge);
                let target = edge.target();
                if edge.kind() != EdgeKind::ExceptionUnwind || has_phi.is_marked(target.index()) {
                    continue;
                }
                has_phi.mark(target.index());
                phis.place(target, variable.clone(), arrivals(cfg, target));
                if !placed.is_marked(target.index()) {
                    placed.mark(target.index());
                    worklist.push(target);
                }
            }
        }

        while let Some(block) = worklist.pop() {
            for &frontier_block in frontiers.frontier(block) {
                if has_phi.is_marked(frontier_block.index()) {
                    continue;
                }
                has_phi.mark(frontier_block.index());

                phis.place(
                    frontier_block,
                    variable.clone(),
                    arrivals(cfg, frontier_block),
                );
                if !placed.is_marked(frontier_block.index()) {
                    placed.mark(frontier_block.index());
                    worklist.push(frontier_block);
                }
            }
        }
    }
}

/// The departure point of every edge into `block`, in CFG predecessor order.
///
/// A normal edge departs at its source's end, named by the source's
/// instruction count. An edge that [departs before
/// throws](departs_before_throws) contributes one departure before each of
/// its source's throwing instructions.
fn arrivals<I: InstrInfo, E>(
    cfg: &Cfg<I, E>,
    block: BlockId,
) -> impl Iterator<Item = ProgramPoint> + '_ {
    cfg.incoming(block).flat_map(move |edge| {
        let edge = cfg.edge(edge);
        let source = edge.source();
        let instructions = cfg.block(source).instructions();
        let before_throws = departs_before_throws(cfg, edge);
        let end = instructions.len();
        (if before_throws { 0 } else { end }..=end)
            .filter(move |&index| {
                !before_throws || instructions.get(index).is_some_and(InstrInfo::may_unwind)
            })
            .map(move |inst_idx| ProgramPoint {
                block: source,
                inst_idx,
            })
    })
}
