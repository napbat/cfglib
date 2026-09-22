//! Verified generic MLIL function storage and derived analyses.

extern crate alloc;

use alloc::vec::Vec;

use crate::{
    AstNode, BlockExprTrees, Cfg, ConstFact, DefUseChains, DominatorTree, Facts, Liveness,
    ProgramPoint, SccpAnalysis, SsaForm,
};

use super::{
    AnalysisDialect, Dialect, Instruction, InstructionId, ProvenanceMap, Result, Signature,
    Variable, VariableId, VerificationReport, VerifyDialect,
};

/// One semantic function backed by a cfglib control-flow graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function<D: Dialect> {
    pub(super) cfg: Cfg<Instruction<D>, D::Edge>,
    pub(super) variables: Vec<Variable<D>>,
    pub(super) signature: Signature<D>,
    pub(super) provenance: ProvenanceMap<D>,
    /// Where each stable [`InstructionId`] currently sits, or `None` once a
    /// derived graph dropped it. Every door that changes the graph rebuilds
    /// this table, so an identity never resolves to somebody else's
    /// instruction.
    pub(super) instruction_points: Vec<Option<ProgramPoint>>,
}

impl<D: Dialect> Function<D> {
    /// Returns the exact edge-bearing MLIL control-flow graph.
    #[must_use]
    pub const fn cfg(&self) -> &Cfg<Instruction<D>, D::Edge> {
        &self.cfg
    }

    /// Returns the ordered parameter and return signature.
    #[must_use]
    pub const fn signature(&self) -> &Signature<D> {
        &self.signature
    }

    /// Returns the source function identity and coordinate system.
    #[must_use]
    pub fn source(&self) -> &D::Source {
        self.provenance.source()
    }

    /// Returns variables in dense identity order.
    #[must_use]
    pub fn variables(&self) -> &[Variable<D>] {
        &self.variables
    }

    /// Looks up one declared variable.
    #[must_use]
    pub fn variable(&self, id: VariableId) -> Option<&Variable<D>> {
        self.variables.get(id.index())
    }

    /// Returns stable source-to-MLIL provenance.
    #[must_use]
    pub const fn provenance(&self) -> &ProvenanceMap<D> {
        &self.provenance
    }

    /// Returns instructions in stable identity order.
    ///
    /// Identity order is allocation order, which is independent of graph
    /// position; use [`Self::cfg`] for block-structured iteration.
    pub fn instructions(&self) -> impl Iterator<Item = &Instruction<D>> {
        self.instruction_points
            .iter()
            .filter_map(|point| self.at((*point)?))
    }

    /// Looks up one instruction by stable identity.
    ///
    /// `None` once a derived graph has dropped the instruction.
    #[must_use]
    pub fn instruction(&self, id: InstructionId) -> Option<&Instruction<D>> {
        self.at((*self.instruction_points.get(id.index())?)?)
    }

    /// Borrow the instruction stored at one program point.
    fn at(&self, point: ProgramPoint) -> Option<&Instruction<D>> {
        self.cfg
            .contains_block(point.block)
            .then(|| self.cfg.block(point.block))?
            .instructions()
            .get(point.inst_idx)
    }

    /// Rebuild the identity-to-position table from the graph.
    ///
    /// Every door that hands a caller the graph to change works on a clone
    /// and calls this afterwards, which is what keeps
    /// [`instruction`](Self::instruction) honest about a transform that
    /// moved or dropped instructions. A transform that duplicates a block
    /// leaves one identity at several positions; the table then names the
    /// last of them.
    pub(super) fn reindex_instructions(&mut self) {
        self.instruction_points.fill(None);
        for block in self.cfg.block_ids() {
            for (inst_idx, instruction) in self.cfg.block(block).instructions().iter().enumerate() {
                let Some(slot) = self.instruction_points.get_mut(instruction.id().index()) else {
                    continue;
                };
                *slot = Some(ProgramPoint { block, inst_idx });
            }
        }
    }

    /// Returns the number of instructions — the exclusive upper bound of
    /// the function's dense [`InstructionId`] space.
    #[must_use]
    pub fn instruction_count(&self) -> usize {
        self.instruction_points.len()
    }

    /// Returns the current graph location of one stable instruction.
    #[must_use]
    pub fn instruction_point(&self, id: InstructionId) -> Option<ProgramPoint> {
        (*self.instruction_points.get(id.index())?)
            .as_ref()
            .copied()
    }

    /// Computes a dominator tree over ordinary and exceptional control flow.
    #[must_use]
    pub fn dominators(&self) -> DominatorTree {
        DominatorTree::compute(&self.cfg)
    }

    /// Computes definition-to-use and use-to-definition chains.
    #[must_use]
    pub fn def_use(&self) -> DefUseChains {
        DefUseChains::compute(&self.cfg)
    }

    /// Computes block-entry and block-exit liveness.
    #[must_use]
    pub fn liveness(&self) -> Liveness<VariableId> {
        Liveness::compute(&self.cfg)
    }

    /// Recovers structured control flow, retaining explicit labels and gotos
    /// when the graph is irreducible.
    #[must_use]
    pub fn structured_control_flow(&self) -> AstNode<Instruction<D>> {
        crate::lift(&self.cfg)
    }

    /// Recovers structured control flow together with the exact list of
    /// constructs that degraded to unstructured flow.
    #[must_use]
    pub fn structured_control_flow_with_report(
        &self,
    ) -> (AstNode<Instruction<D>>, crate::LiftReport) {
        crate::lift_with_report(&self.cfg)
    }

    /// Returns a derived function whose unknown exception-handler extents
    /// are promoted to their dominated blocks, with the promotion count.
    ///
    /// The canonical function keeps stating the extents are unknown; the
    /// returned view exists so extent-dependent consumers (structured
    /// try/catch recovery, [HLIL lifting](crate::ir::hlil::lift_function))
    /// can run without the frontend inferring extents in canonical
    /// metadata. All identities are unchanged.
    #[must_use]
    pub fn with_promoted_handler_extents(&self) -> (Self, usize) {
        let mut derived = self.clone();
        let promoted = crate::promote_handler_extents(&mut derived.cfg);
        derived.reindex_instructions();
        (derived, promoted)
    }

    /// Returns a derived function whose small shared tails are duplicated
    /// until control flow tree-structures, with the number of blocks
    /// materialized. Short-circuit conditions and shared side-exits stop
    /// producing goto residue, at the price of instruction *values*
    /// appearing at several graph positions — identity-keyed side tables
    /// keep describing the original function, so use this view for
    /// structuring and presentation, never as canonical storage.
    #[must_use]
    pub fn with_duplicated_structuring_tails(&self) -> (Self, usize)
    where
        D::Edge: Clone,
    {
        let mut derived = self.clone();
        let duplicated = crate::duplicate_structuring_tails(&mut derived.cfg);
        derived.reindex_instructions();
        (derived, duplicated)
    }

    /// Returns a derived function whose graph the caller transformed in
    /// place — the generic door for consumer-specific presentation views
    /// (detaching coverage a recovered construct regenerates, specializing
    /// edges). Canonical storage is never mutated, and the derived
    /// function's identity-to-position table is rebuilt from the transformed
    /// graph, so [`instruction`](Self::instruction) stays honest about what
    /// the transform moved or dropped.
    #[must_use]
    pub fn with_derived_cfg<R>(
        &self,
        transform: impl FnOnce(&mut Cfg<Instruction<D>, D::Edge>) -> R,
    ) -> Self {
        let mut derived = self.clone();
        let _ = transform(&mut derived.cfg);
        derived.reindex_instructions();
        derived
    }

    /// Reports effect-aware dead definitions and unreachable blocks.
    #[must_use]
    pub fn dead_code(&self) -> crate::DeadCode {
        crate::DeadCode::compute(&self.cfg)
    }

    /// Returns a graph with effect-free dead definitions removed.
    ///
    /// The canonical function and its stable provenance are not mutated;
    /// instruction positions in the returned view are derived data.
    #[must_use]
    pub fn dead_code_eliminated_cfg(&self) -> (Cfg<Instruction<D>, D::Edge>, usize) {
        let mut cfg = self.cfg.clone();
        let removed = crate::dead_code_elimination(&mut cfg);
        (cfg, removed)
    }
}

impl<D: VerifyDialect> Function<D> {
    /// Verifies structural, control-flow, typing, and dialect invariants.
    #[must_use]
    pub fn verify(&self) -> VerificationReport {
        super::verify::verify_function(self)
    }

    /// Computes a fully renamed SSA view while retaining the source graph.
    ///
    /// # Errors
    ///
    /// Returns a verification report if the stored function is invalid.
    pub fn ssa(&self) -> Result<SsaForm<VariableId>> {
        let report = self.verify();
        if !report.is_ok() {
            return Err(report.into());
        }
        let dominators = self.dominators();
        Ok(SsaForm::compute(&self.cfg, &dominators))
    }

    /// Rebuilds the function with every unaliased memory location promoted
    /// to a mutable variable, per the dialect's
    /// [`PromoteDialect`](super::PromoteDialect) judgment.
    ///
    /// Every non-variable identity is preserved; rewritten accesses become
    /// dialect copies at the same instruction identities. Composes with
    /// [`Self::split_variables`] and
    /// [`lift_function`](crate::ir::hlil::lift_function): promote, split,
    /// then lift for one typed local per promoted lifetime.
    ///
    /// # Errors
    ///
    /// Returns a verification report if the stored function is invalid.
    pub fn promote_memory(&self) -> Result<super::MemoryPromotion<D>>
    where
        D: super::PromoteDialect,
    {
        super::promote::promote_memory(self)
    }

    /// Rebuilds the function with every unaliased memory location promoted
    /// to a mutable variable, per `classify` rather than per the dialect's
    /// [`promotion_access`](super::PromoteDialect::promotion_access).
    ///
    /// It exists because the address judgment is not always a property of
    /// the instruction. A machine frontend cannot tell a frame slot from
    /// any other pointer dereference by looking at `load(add(v9, 0x30))`
    /// alone — that is a slot only once a stack-pointer analysis has
    /// proved what `v9` holds — so the trait method, which sees one
    /// instruction and no analysis context, has nothing to answer with.
    /// The consumer that ran the analysis answers instead, and gets the
    /// same aliasing contract, validation, and identity-preserving rewrite
    /// as [`Self::promote_memory`], which is this door passed the trait
    /// method.
    ///
    /// `classify` is called once per instruction in the collection pass
    /// and once per instruction in the rewrite pass, both in instruction
    /// order, and must answer the same for the same instruction both
    /// times; it is [`FnMut`] so the analysis can be memoized by
    /// [`InstructionId`](super::InstructionId).
    ///
    /// # Errors
    ///
    /// Returns a verification report if the stored function is invalid.
    pub fn promote_memory_with(
        &self,
        classify: impl FnMut(&super::Instruction<D>) -> super::PromotionAccess<D::Location>,
    ) -> Result<super::MemoryPromotion<D>>
    where
        D: super::PromoteDialect,
    {
        super::promote::promote_memory_with(self, classify)
    }

    /// Rebuilds the function with one variable per SSA phi-web, separating
    /// unrelated lifetimes that shared a storage-derived variable.
    ///
    /// Block, edge, instruction, and region identities are preserved; only
    /// variable identities change, and the result maps both directions.
    /// Composes with [`lift_function`](crate::ir::hlil::lift_function) so
    /// structured output gets one local per lifetime.
    ///
    /// # Errors
    ///
    /// Returns a verification report if the stored function is invalid.
    pub fn split_variables(&self) -> Result<super::VariableSplit<D>> {
        super::split::split_variables(self)
    }
}

impl<D: AnalysisDialect> Function<D> {
    /// Computes conservative forward constant facts without changing the function.
    #[must_use]
    pub fn constants(&self) -> Facts<ConstFact<VariableId, D::Constant>> {
        crate::constant_propagation(&self.cfg)
    }

    /// Recovers pure, single-use expression trees independently in each block.
    #[must_use]
    pub fn expressions(
        &self,
    ) -> Vec<BlockExprTrees<VariableId, D::ExpressionOperator, D::Constant>> {
        crate::recover_expressions(&self.cfg)
    }

    /// Returns a copy-propagated graph for presentation and downstream analysis.
    ///
    /// The canonical function and its identity-indexed provenance are not
    /// mutated. Removed copies retain no graph position in the returned view.
    #[must_use]
    pub fn copy_propagated_cfg(
        &self,
    ) -> (Cfg<Instruction<D>, D::Edge>, crate::CopyPropagationStats) {
        let mut cfg = self.cfg.clone();
        let statistics = crate::copy_propagation(&mut cfg);
        (cfg, statistics)
    }
}

impl<D: AnalysisDialect + VerifyDialect> Function<D> {
    /// Computes sparse constants over the derived SSA view.
    ///
    /// # Errors
    ///
    /// Returns a verification report if the stored function is invalid.
    pub fn sparse_constants(&self) -> Result<SccpAnalysis<VariableId, D::Constant>> {
        let ssa = self.ssa()?;
        Ok(SccpAnalysis::compute(&self.cfg, &ssa))
    }
}
