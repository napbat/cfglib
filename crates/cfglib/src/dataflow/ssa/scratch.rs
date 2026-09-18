//! Caller-owned working storage for SSA construction.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::graph::dominator::{DominatorChildLinks, DominatorScratch};
use crate::graph::epoch::EpochMarks;

use super::SsaInstruction;
use super::SsaValue;
use super::frontier::FrontierRuns;
use super::place::PhiDrafts;
use super::rename::RenameEvent;

/// Every buffer [`SsaForm::compute_in`](super::SsaForm::compute_in) fills,
/// owned by the caller.
///
/// SSA construction over a four-block procedure costs about sixty
/// allocations, and only a third of them are the form it returns. The rest is
/// the call: a dominance-frontier set per block, a definition list and a
/// renaming stack per variable, a phi list per merge block, a predecessor and
/// an operand vector per phi, a pushed-variable vector per block entered, an
/// operand vector per edge, and half a dozen rows sized by the block bound.
/// A whole-codebase pass pays all of it once per callable, and most callables
/// are that small.
///
/// This moves every one of those out of the call. What remains allocated per
/// call is the form itself — its blocks, its renamed operand vectors, and its
/// version map — plus the tree nodes of the two maps keyed by variable.
///
/// Hand one scratch to every [`compute_in`](super::SsaForm::compute_in) of
/// the pass. The buffers grow to the largest procedure the pass meets and are
/// then reused by every procedure after it.
///
/// # One scratch, many procedures
///
/// The type parameter is the source-IR variable, so one scratch serves every
/// CFG whose instructions name variables the same way, whatever their size
/// and whatever their instruction or edge-payload types. It is [`Send`] when
/// the variable is, so a worker thread owns one for the whole corpus it is
/// handed.
///
/// # Sizing
///
/// Nothing is fixed at construction and nothing is released by a call. Every
/// buffer is cleared and grown on entry, so a scratch used on a large
/// procedure and then on a small one is correct and allocation-free, and the
/// space it holds is the high-water mark of the sequence.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, DominatorTree, SsaForm, SsaScratch};
/// # use cfglib::InstrInfo;
/// # #[derive(Clone)]
/// # struct Inst(Vec<u8>, Vec<u8>);
/// # impl InstrInfo for Inst {
/// #     type Variable = u8;
/// #     fn uses(&self) -> &[u8] { &self.0 }
/// #     fn defs(&self) -> &[u8] { &self.1 }
/// # }
/// let mut cfg = Cfg::<Inst>::new();
/// cfg.block_mut(cfg.entry()).push(Inst(vec![], vec![0]));
/// let dominators = DominatorTree::compute(&cfg);
///
/// let mut scratch = SsaScratch::new();
/// let form = SsaForm::compute_in(&mut scratch, &cfg, &dominators);
/// assert_eq!(form, SsaForm::compute(&cfg, &dominators));
/// ```
#[derive(Debug)]
pub struct SsaScratch<V> {
    /// The dominator buffers of the forest completion, which is the only
    /// dominator tree SSA construction builds for itself.
    pub(super) dominators: DominatorScratch,
    /// The forest's roots: the entry plus one block per source component of
    /// otherwise unreachable code.
    pub(super) forest_roots: Vec<BlockId>,
    /// Every block's dominance frontier, flat.
    pub(super) frontiers: FrontierRuns,
    /// Per variable, the blocks that define it, in ascending block order.
    pub(super) definition_blocks: BTreeMap<V, Vec<BlockId>>,
    /// Definition lists whose variable is gone, kept for the next procedure.
    pub(super) block_pool: Vec<Vec<BlockId>>,
    /// Blocks that already carry a phi for the variable being placed.
    pub(super) has_phi: EpochMarks,
    /// Blocks already on the placement worklist for that variable.
    pub(super) placed: EpochMarks,
    /// The placement worklist.
    pub(super) worklist: Vec<BlockId>,
    /// The placed phis, with room for the operands renaming will fill.
    pub(super) phis: PhiDrafts<V>,
    /// Per block, its renamed instructions. Each vector is moved into the
    /// finished block, so the storage here is the outer row alone.
    pub(super) instructions: Vec<Vec<SsaInstruction<V>>>,
    /// Per variable, the renaming stack of values live at the current point.
    pub(super) stacks: BTreeMap<V, Vec<SsaValue<V>>>,
    /// Renaming stacks whose variable is gone, kept for the next procedure.
    pub(super) value_pool: Vec<Vec<SsaValue<V>>>,
    /// Every variable pushed by a block still on the walk, so unwinding is a
    /// count rather than a vector per block.
    pub(super) pushed: Vec<V>,
    /// Blocks the renaming walk has already entered.
    pub(super) renamed: EpochMarks,
    /// Dominator-tree child adjacency, in descending sibling order.
    pub(super) children: DominatorChildLinks<BlockId>,
    /// The roots the renaming walk starts from.
    pub(super) roots: Vec<BlockId>,
    /// The renaming walk's own frontier.
    pub(super) events: Vec<RenameEvent>,
}

impl<V> Default for SsaScratch<V> {
    fn default() -> Self {
        Self {
            dominators: DominatorScratch::new(),
            forest_roots: Vec::new(),
            frontiers: FrontierRuns::default(),
            definition_blocks: BTreeMap::new(),
            block_pool: Vec::new(),
            has_phi: EpochMarks::default(),
            placed: EpochMarks::default(),
            worklist: Vec::new(),
            phis: PhiDrafts::default(),
            instructions: Vec::new(),
            stacks: BTreeMap::new(),
            value_pool: Vec::new(),
            pushed: Vec::new(),
            renamed: EpochMarks::default(),
            children: DominatorChildLinks::default(),
            roots: Vec::new(),
            events: Vec::new(),
        }
    }
}

impl<V: crate::dataflow::VariableId> SsaScratch<V> {
    /// Scratch holding nothing yet.
    ///
    /// Every buffer is sized by the first procedure it sees, so there is no
    /// block count to state here.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Empty every buffer and grow the block-indexed ones to `block_bound`.
    ///
    /// This is a correctness step, not only hygiene: nothing a previous
    /// procedure left behind may reach this one's answer. The two maps keyed
    /// by variable are emptied into their pools, which is what keeps a
    /// variable's list or stack allocated while its key does not survive.
    pub(super) fn reset(&mut self, block_bound: usize) {
        self.reset_placement(block_bound);
        self.reset_renaming(block_bound);
    }

    /// Ready the buffers phi placement alone uses, which is all
    /// [`PhiPlacements::compute`](super::PhiPlacements::compute) needs.
    pub(super) fn reset_placement(&mut self, block_bound: usize) {
        self.forest_roots.clear();
        for (_, mut blocks) in core::mem::take(&mut self.definition_blocks) {
            blocks.clear();
            self.block_pool.push(blocks);
        }
        for marks in [&mut self.has_phi, &mut self.placed] {
            marks.grow_to(block_bound);
            marks.reset();
        }
        self.worklist.clear();
        self.phis.reset(block_bound);
    }

    /// Ready the buffers the renaming walk adds to those.
    fn reset_renaming(&mut self, block_bound: usize) {
        for (_, mut stack) in core::mem::take(&mut self.stacks) {
            stack.clear();
            self.value_pool.push(stack);
        }
        self.renamed.grow_to(block_bound);
        self.renamed.reset();
        if self.instructions.len() < block_bound {
            self.instructions.resize_with(block_bound, Vec::new);
        }
        for block in &mut self.instructions {
            block.clear();
        }
        self.pushed.clear();
        self.children.reset(block_bound);
        self.roots.clear();
        self.events.clear();
    }
}
