//! Generic Static Single Assignment (SSA) construction.
//!
//! The SSA representation is independent of an instruction set and of the
//! concrete instruction type stored in [`Cfg`]. An [`InstrInfo`] adapter
//! supplies its native variable identity, and [`SsaForm::compute`] produces
//! renamed definitions, uses, and phi operands keyed by [`ProgramPoint`]. The
//! original
//! instructions remain untouched and can be recovered from the source CFG.
//!
//! Construction runs in four phases, one module each: the dominance forest
//! that gives unreachable code its own roots, the dominance frontier table,
//! phi placement into flat drafts, and the dominator-tree renaming walk that
//! fills them. Every buffer any of them needs belongs to an [`SsaScratch`],
//! so a whole-codebase pass builds one form per callable without allocating
//! one set of working storage per callable.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::{InstrInfo, ProgramPoint, VariableId};
use crate::graph::dominator::DominatorTree;

mod forest;
mod frontier;
mod place;
mod rename;
mod scratch;

pub use frontier::DominanceFrontiers;
pub use place::{PhiPlacement, PhiPlacements};
pub use scratch::SsaScratch;

/// A per-variable SSA version number.
pub type SsaVersion = usize;

/// A source-IR variable qualified by an SSA version.
///
/// Version `0` represents the value entering a dominator-tree root before any
/// definition in that root's region. Positive versions are produced by phis
/// and instruction definitions.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SsaValue<V> {
    /// Source-IR variable identity.
    pub variable: V,
    /// SSA version of the variable.
    pub version: SsaVersion,
}

impl<V> SsaValue<V> {
    /// Create an SSA value with an explicit version.
    #[must_use]
    pub const fn new(variable: V, version: SsaVersion) -> Self {
        Self { variable, version }
    }

    /// Create the version-zero live-in value for `variable`.
    #[must_use]
    pub const fn live_in(variable: V) -> Self {
        Self::new(variable, 0)
    }

    /// Return whether this is a version-zero live-in value.
    #[must_use]
    pub const fn is_live_in(&self) -> bool {
        self.version == 0
    }
}

/// A fully renamed SSA phi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsaPhi<V> {
    /// SSA value defined by the phi.
    pub result: SsaValue<V>,
    /// Incoming SSA value for each CFG predecessor.
    pub operands: Vec<(BlockId, SsaValue<V>)>,
}

/// SSA annotations for one source instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsaInstruction<V> {
    /// Position of the original instruction in the source CFG.
    pub point: ProgramPoint,
    /// Renamed operands, preserving the adapter's use order.
    pub uses: Vec<SsaValue<V>>,
    /// Fresh definitions, preserving the adapter's definition order.
    pub defs: Vec<SsaValue<V>>,
}

/// SSA contents associated with one CFG block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsaBlock<V> {
    /// Source CFG block.
    pub block: BlockId,
    /// Renamed phis at the start of the block.
    pub phis: Vec<SsaPhi<V>>,
    /// Renamed instructions in source instruction order.
    pub instructions: Vec<SsaInstruction<V>>,
}

/// An IR-neutral, renamed SSA view of a CFG.
///
/// This type deliberately stores no instruction payload. Each
/// [`SsaInstruction`] carries a [`ProgramPoint`] that maps back to the native
/// instruction in the CFG used to build the form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsaForm<V> {
    blocks: Vec<SsaBlock<V>>,
    max_versions: BTreeMap<V, SsaVersion>,
    /// Immediate dominator of every block, in block-index order, from the
    /// tree the renaming walked. [`SsaForm::value_at`] climbs it.
    idom: Vec<Option<BlockId>>,
}

impl<V: VariableId> SsaForm<V> {
    /// Return all SSA blocks, indexed by source [`BlockId`].
    ///
    /// The slice is sized by the source CFG's block *bound*, so a slot the
    /// source retired holds an empty block rather than shifting its
    /// successors' indices.
    #[must_use]
    pub fn blocks(&self) -> &[SsaBlock<V>] {
        &self.blocks
    }

    /// Return the SSA block corresponding to `block`.
    #[must_use]
    pub fn block(&self, block: BlockId) -> &SsaBlock<V> {
        &self.blocks[block.index()]
    }

    /// Return SSA annotations for a source program point, if it exists.
    #[must_use]
    pub fn instruction(&self, point: ProgramPoint) -> Option<&SsaInstruction<V>> {
        self.blocks
            .get(point.block.index())?
            .instructions
            .get(point.inst_idx)
    }

    /// Iterate over all `(block, phi)` pairs.
    pub fn phis(&self) -> impl Iterator<Item = (BlockId, &SsaPhi<V>)> {
        self.blocks
            .iter()
            .flat_map(|block| block.phis.iter().map(move |phi| (block.block, phi)))
    }

    /// Return the greatest assigned version for `variable`.
    ///
    /// A result of `0` means the variable only occurs as a live-in value or
    /// does not occur in the form.
    #[must_use]
    pub fn max_version(&self, variable: &V) -> SsaVersion {
        self.max_versions.get(variable).copied().unwrap_or(0)
    }

    /// Return the SSA value of `variable` that reaches `point`.
    ///
    /// The answer is the value the instruction at `point` would read, so a
    /// definition of `variable` by that instruction itself is not counted.
    ///
    /// The search follows the order the renaming used:
    ///
    /// 1. the last definition of `variable` earlier in the same block;
    /// 2. the phi of `variable` at the start of that block;
    /// 3. the same two rules in the immediate dominator, and in each of its
    ///    own dominators in turn; and
    /// 4. the version-zero live-in, when no dominator defines `variable`.
    ///
    /// A `point` outside the form answers with the live-in.
    #[must_use]
    pub fn value_at(&self, point: ProgramPoint, variable: &V) -> SsaValue<V> {
        let mut block = Some(point.block);
        let mut limit = point.inst_idx;
        while let Some(current) = block {
            if let Some(found) = self.value_in_block(current, limit, variable) {
                return found;
            }
            block = self.idom.get(current.index()).copied().flatten();
            limit = usize::MAX;
        }
        SsaValue::live_in(variable.clone())
    }

    /// Return the definition of `variable` in `block` before index `limit`.
    fn value_in_block(&self, block: BlockId, limit: usize, variable: &V) -> Option<SsaValue<V>> {
        let contents = self.blocks.get(block.index())?;
        let end = limit.min(contents.instructions.len());
        let definition = contents.instructions[..end]
            .iter()
            .rev()
            .find_map(|instruction| {
                instruction
                    .defs
                    .iter()
                    .rev()
                    .find(|value| value.variable == *variable)
            });
        if let Some(definition) = definition {
            return Some(definition.clone());
        }
        contents
            .phis
            .iter()
            .find(|phi| phi.result.variable == *variable)
            .map(|phi| phi.result.clone())
    }
}

impl<V: VariableId> SsaForm<V> {
    /// Compute a fully renamed SSA view of `cfg`.
    ///
    /// The algorithm performs phi placement followed by classic dominator-tree
    /// renaming. It is iterative rather than recursive, so deeply nested control
    /// flow does not consume the host call stack. Variables read before any
    /// dominating definition receive version `0`. Disconnected source
    /// components become independent roots of an internal dominator forest, so
    /// definitions still flow through unreachable handler/dead-code chains
    /// without inheriting values from the ordinary entry component.
    ///
    /// # Precondition
    ///
    /// The entry block must not be a branch target. Phi operands come from
    /// predecessor edges, so a phi placed AT the entry (entry doubling as a
    /// loop header) has no operand for the version-`0` live-in value and the
    /// value entering the function is dropped from the web. Every builder in
    /// this workspace guarantees the property; direct constructions that
    /// branch to the entry should canonicalize first
    /// ([`insert_preheader`](crate::insert_preheader) /
    /// [`split_block`](crate::Cfg::split_block)).
    #[must_use]
    pub fn compute<I: InstrInfo<Variable = V>, E>(cfg: &Cfg<I, E>, dom: &DominatorTree) -> Self {
        Self::compute_in(&mut SsaScratch::new(), cfg, dom)
    }

    /// Compute a fully renamed SSA view of `cfg` over caller-owned working
    /// storage.
    ///
    /// This is [`compute`](Self::compute) with the call's buffers taken out of
    /// it. The form it returns is the same one, built by the same code: the
    /// allocating entry point is this function over a fresh [`SsaScratch`],
    /// and it keeps the same precondition.
    #[must_use]
    pub fn compute_in<I: InstrInfo<Variable = V>, E>(
        scratch: &mut SsaScratch<V>,
        cfg: &Cfg<I, E>,
        dom: &DominatorTree,
    ) -> Self {
        scratch.reset(cfg.block_bound());
        let complete = forest::complete_dominator_forest(
            &mut scratch.dominators,
            &mut scratch.forest_roots,
            cfg,
            dom,
        );
        let dom = complete.as_ref().unwrap_or(dom);
        place::place_phis(scratch, cfg, dom);
        let mut max_versions = BTreeMap::new();
        rename::rename(scratch, cfg, dom, &mut max_versions);
        let blocks = rename::finish_blocks(scratch, cfg.block_bound(), &mut max_versions);
        let idom = (0..cfg.block_bound())
            .map(|index| dom.idom(BlockId::from_index(index)))
            .collect();
        SsaForm {
            blocks,
            max_versions,
            idom,
        }
    }
}

#[cfg(test)]
mod tests;
