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
use smallvec::SmallVec;

use crate::block::BlockId;
use crate::dataflow::cfg_view::{CfgView, OperandView};
use crate::dataflow::unwind::{departs_before_throws_with, first_unwind_point, unwind_edges};
use crate::graph::dominator::DominatorTree;
use crate::graph::view::GraphView;
use crate::{InstrOperands, ProgramPoint, VariableId};

use super::prune::{SsaPruning, mark_live_blocks, run_of};
use super::scratch::SsaScratch;
use super::{INLINE_OPERANDS, SsaValue};

/// A structural phi placement before SSA values are renamed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhiPlacement<V> {
    /// The source-IR variable merged by the phi.
    pub variable: V,
    predecessors: SmallVec<[ProgramPoint; INLINE_OPERANDS]>,
    merges_start: bool,
}

impl<V> PhiPlacement<V> {
    /// Departure points that contribute operands, in CFG predecessor order.
    ///
    /// These are the CFG edges into the block only; see
    /// [`merges_start_value`](Self::merges_start_value) for the input that
    /// has no edge.
    #[must_use]
    pub fn predecessors(&self) -> &[ProgramPoint] {
        &self.predecessors
    }

    /// Whether the phi also merges the value its variable holds when the
    /// function starts, as [`SsaPhi::merges_start_value`](super::SsaPhi::merges_start_value)
    /// reports for the renamed phi.
    ///
    /// That is the case for every phi placed in the CFG entry block, which is
    /// then a branch target. The start input has no departure point, so it is
    /// not among [`predecessors`](Self::predecessors).
    #[must_use]
    pub const fn merges_start_value(&self) -> bool {
        self.merges_start
    }
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
    /// Place phis for every variable defined in the CFG (minimal SSA).
    ///
    /// This is the iterated-dominance-frontier phase of SSA construction. Use
    /// [`SsaForm::compute`](super::SsaForm::compute) when renamed definitions
    /// and operands are also required.
    /// An entry block that is a branch target is placed like any other loop
    /// header, and its phis report
    /// [`merges_start_value`](PhiPlacement::merges_start_value).
    ///
    /// `cfg` is a [`CfgView`] whose instructions implement
    /// [`InstrInfo`](crate::InstrInfo), or a
    /// [`WithOperands`](crate::WithOperands) that reads them through a
    /// context.
    #[must_use]
    pub fn compute<S>(cfg: &S, dom: &DominatorTree) -> Self
    where
        S: OperandView<Variable = V>,
    {
        Self::compute_with_pruning(cfg, dom, SsaPruning::Minimal)
    }

    /// Place the phis that `pruning` keeps.
    ///
    /// This is the placement
    /// [`SsaForm::compute_with_pruning`](super::SsaForm::compute_with_pruning)
    /// renames. [`SsaPruning::Minimal`] gives [`compute`](Self::compute).
    /// `cfg` supplies its operands as it does to [`compute`](Self::compute).
    #[must_use]
    pub fn compute_with_pruning<S>(cfg: &S, dom: &DominatorTree, pruning: SsaPruning) -> Self
    where
        S: OperandView<Variable = V>,
    {
        let (operands, cfg) = (cfg.operands(), cfg.view());
        let mut scratch = SsaScratch::new();
        scratch.reset_placement(cfg.node_bound());
        place_phis(&mut scratch, cfg, operands, dom, pruning);

        let drafts = &scratch.phis;
        let placements = (0..cfg.node_bound())
            .map(|index| {
                drafts
                    .at(BlockId::from_index(index))
                    .iter()
                    .map(|draft| PhiPlacement {
                        variable: draft.variable.clone(),
                        predecessors: drafts.predecessors(draft).iter().copied().collect(),
                        merges_start: draft.merges_start,
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
    /// Whether the phi also merges the value its variable holds when the
    /// function starts: true for every phi in the CFG entry block.
    pub(super) merges_start: bool,
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
    ///
    /// A phi in the `entry` block also merges the value the variable holds
    /// when the function starts. A phi is only placed at a block that some
    /// edge enters, so an entry that gets one is a branch target, and the
    /// function's start is the one way into it that no edge names.
    fn place(
        &mut self,
        entry: BlockId,
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
            merges_start: block == entry,
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
/// definitions that `pruning` keeps, leaving the result in `scratch.phis`.
///
/// A block can dominate its unwind successor while a definition it makes
/// at or after its first throwing instruction does not reach that
/// successor. Such a definition also places a phi at the unwind successor,
/// whose operands name the value before each throwing instruction.
///
/// A pruned form skips a variable no block reads before writing it, and
/// [`SsaPruning::Pruned`] also skips a block the variable is dead on entry
/// to. A skipped block is not a definition of the variable either, so the
/// frontier walk does not continue from it: every path out of it writes the
/// variable before reading it, and each such write places its own phis.
pub(super) fn place_phis<G, O, V>(
    scratch: &mut SsaScratch<V>,
    cfg: &G,
    operands: &O,
    dom: &DominatorTree,
    pruning: SsaPruning,
) where
    G: CfgView,
    O: InstrOperands<G::Inst, Variable = V>,
    V: VariableId,
{
    scan_instructions(scratch, cfg, operands, pruning);
    let SsaScratch {
        frontiers,
        definition_blocks,
        unwound_definitions,
        upward_exposed,
        definitions_before_throw,
        throwing,
        live,
        has_phi,
        placed,
        worklist,
        phis,
        ..
    } = scratch;
    frontiers.rebuild(cfg, dom);
    let root = cfg.root();

    let (mut unwound, mut exposed, mut before_throw) = (0, 0, 0);
    for (variable, definitions) in definition_blocks.iter_mut() {
        let unwound_sources = run_of(unwound_definitions, &mut unwound, variable);
        let reads = run_of(upward_exposed, &mut exposed, variable);
        let writes_before_throw = run_of(definitions_before_throw, &mut before_throw, variable);
        if pruning != SsaPruning::Minimal && reads.is_empty() {
            continue;
        }
        has_phi.reset();
        placed.reset();
        for &block in definitions.iter() {
            placed.mark(block.index());
        }
        if pruning == SsaPruning::Pruned {
            mark_live_blocks(
                cfg,
                reads.iter().map(|&(_, block)| block),
                placed,
                throwing,
                writes_before_throw,
                live,
                worklist,
            );
        }
        let keeps = |block: BlockId| pruning != SsaPruning::Pruned || live.is_marked(block.index());
        // The definition list is moved rather than copied, so the vector it
        // leaves behind stays in the map and is recycled by the next reset.
        worklist.clear();
        worklist.append(definitions);

        for (_, source) in unwound_sources {
            for edge in unwind_edges(cfg, *source) {
                let target = edge.target();
                if has_phi.is_marked(target.index()) || !keeps(target) {
                    continue;
                }
                has_phi.mark(target.index());
                phis.place(
                    root,
                    target,
                    variable.clone(),
                    arrivals(cfg, operands, target),
                );
                if !placed.is_marked(target.index()) {
                    placed.mark(target.index());
                    worklist.push(target);
                }
            }
        }

        while let Some(block) = worklist.pop() {
            for &frontier_block in frontiers.frontier(block) {
                if has_phi.is_marked(frontier_block.index()) || !keeps(frontier_block) {
                    continue;
                }
                has_phi.mark(frontier_block.index());

                phis.place(
                    root,
                    frontier_block,
                    variable.clone(),
                    arrivals(cfg, operands, frontier_block),
                );
                if !placed.is_marked(frontier_block.index()) {
                    placed.mark(frontier_block.index());
                    worklist.push(frontier_block);
                }
            }
        }
    }
}

/// Record, from one pass over the instructions, every list placement reads:
/// each variable's defining blocks, the definitions an unwind can miss, and
/// for a pruned form the reads before any write in their block, the writes
/// before a block's first unwind point and the blocks that can unwind. Every
/// operand is read through `operands`. Every list of pairs is left sorted.
fn scan_instructions<G, O, V>(
    scratch: &mut SsaScratch<V>,
    cfg: &G,
    operands: &O,
    pruning: SsaPruning,
) where
    G: CfgView,
    O: InstrOperands<G::Inst, Variable = V>,
    V: VariableId,
{
    let SsaScratch {
        definition_blocks,
        block_pool,
        unwound_definitions,
        upward_exposed,
        definitions_before_throw,
        throwing,
        ..
    } = scratch;
    for block_id in cfg.node_ids() {
        let first_unwind = first_unwind_point(cfg, operands, block_id);
        if pruning == SsaPruning::Pruned && first_unwind.is_some() {
            throwing.mark(block_id.index());
        }
        for (index, instruction) in cfg.instructions(block_id).iter().enumerate() {
            if pruning != SsaPruning::Minimal {
                // An instruction reads its operands before it writes, so a
                // read is upward-exposed unless an earlier instruction of
                // this block wrote the variable.
                for variable in operands.uses(instruction) {
                    let written_here = definition_blocks
                        .get(variable)
                        .is_some_and(|blocks| blocks.last() == Some(&block_id));
                    if !written_here {
                        upward_exposed.push((variable.clone(), block_id));
                    }
                }
            }
            for variable in operands.defs(instruction) {
                let blocks = match definition_blocks.entry(variable.clone()) {
                    Entry::Vacant(slot) => slot.insert(block_pool.pop().unwrap_or_default()),
                    Entry::Occupied(slot) => slot.into_mut(),
                };
                if blocks.last().copied() != Some(block_id) {
                    blocks.push(block_id);
                }
                match first_unwind {
                    Some(first) if index >= first => {
                        unwound_definitions.push((variable.clone(), block_id));
                    }
                    Some(_) if pruning == SsaPruning::Pruned => {
                        definitions_before_throw.push((variable.clone(), block_id));
                    }
                    _ => {}
                }
            }
        }
    }
    for sorted in [
        unwound_definitions,
        upward_exposed,
        definitions_before_throw,
    ] {
        sorted.sort_unstable();
        sorted.dedup();
    }
}

/// The departure point of every edge into `block`, in CFG predecessor order.
///
/// A normal edge departs at its source's end, named by the source's
/// instruction count. An edge that [departs before
/// throws](crate::departs_before_throws) contributes one departure before
/// each of its source's throwing instructions.
fn arrivals<'a, G, O>(
    cfg: &'a G,
    operands: &'a O,
    block: BlockId,
) -> impl Iterator<Item = ProgramPoint> + 'a
where
    G: CfgView,
    O: InstrOperands<G::Inst>,
{
    cfg.incoming(block).flat_map(move |edge| {
        let edge = cfg.edge(edge);
        let source = edge.source();
        let instructions = cfg.instructions(source);
        let before_throws = departs_before_throws_with(cfg, operands, edge);
        let end = instructions.len();
        (if before_throws { 0 } else { end }..=end)
            .filter(move |&index| {
                !before_throws
                    || instructions
                        .get(index)
                        .is_some_and(|instruction| operands.may_unwind(instruction))
            })
            .map(move |inst_idx| ProgramPoint {
                block: source,
                inst_idx,
            })
    })
}
