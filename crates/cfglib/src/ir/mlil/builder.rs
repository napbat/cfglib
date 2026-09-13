//! Checked construction of generic MLIL functions.

extern crate alloc;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::region::{Cleanup, Continuation, HandlerKind, HandlerRef, Region, RegionId};
use crate::{BlockId, Cfg, EdgeId, ProgramPoint};

use super::{
    Dialect, EntityId, Error, Function, Instruction, InstructionId, ProvenanceMap, Result,
    Signature, TypedVariable, Variable, VariableId, VerifyDialect,
};

/// Incremental builder that assigns dense stable MLIL identities.
pub struct FunctionBuilder<D: Dialect> {
    cfg: Cfg<Instruction<D>, D::Edge>,
    variables: Vec<Variable<D>>,
    signature: Signature<D>,
    provenance: ProvenanceMap<D>,
    instruction_points: Vec<ProgramPoint>,
}

impl<D: Dialect> FunctionBuilder<D> {
    /// Creates a builder with an empty synthetic root block.
    #[must_use]
    pub fn new(source: D::Source) -> Self {
        let mut cfg = Cfg::with_edge_payload();
        cfg.block_mut(cfg.entry()).set_label("root");
        Self {
            cfg,
            variables: Vec::new(),
            signature: Signature::<D>::default(),
            provenance: ProvenanceMap::new(source),
            instruction_points: Vec::new(),
        }
    }

    /// Declares the ordered parameter and return signature.
    ///
    /// # Errors
    ///
    /// Returns an error when a parameter is undeclared or repeated.
    pub fn set_signature(&mut self, signature: Signature<D>) -> Result<()> {
        let declared = self.variables.len();
        if let Some(issue) = signature
            .parameter_issues(|parameter| parameter.index() < declared)
            .into_iter()
            .next()
        {
            return Err(Error::InvalidConstruction(issue));
        }
        self.signature = signature;
        Ok(())
    }

    /// Attaches one exception region and its handlers.
    ///
    /// The region id is assigned by the function; the value in `region.id` is
    /// ignored. [`HandlerBody::Unknown`](crate::HandlerBody::Unknown) extents
    /// are legal — structuring then leaves the region as ordinary control
    /// flow rather than guessing handler bounds.
    ///
    /// # Errors
    ///
    /// Returns an error when the region protects no blocks, names an invalid
    /// or synthetic-root block, a known handler body omits its own entry, or
    /// the parent is not an already-added region.
    pub fn add_region(&mut self, region: Region) -> Result<RegionId> {
        if region.protected_blocks.is_empty() {
            return Err(Error::InvalidConstruction(
                "region protects no blocks".into(),
            ));
        }
        for &block in &region.protected_blocks {
            self.require_region_block(block, "protected block")?;
        }
        for handler in &region.handlers {
            self.require_region_block(handler.entry, "handler entry")?;
            if let Some(blocks) = handler.body.blocks() {
                for &block in blocks {
                    self.require_region_block(block, "handler body block")?;
                }
                if !blocks.contains(&handler.entry) {
                    return Err(Error::InvalidConstruction(format!(
                        "handler body omits its own entry {}",
                        handler.entry
                    )));
                }
            }
            if let HandlerKind::Filter { filter_block } = handler.kind {
                self.require_region_block(filter_block, "filter block")?;
            }
        }
        if let Some(parent) = region.parent {
            if parent.index() >= self.cfg.regions().len() {
                return Err(Error::InvalidConstruction(format!(
                    "region parent {parent} has not been added"
                )));
            }
        }
        Ok(self.cfg.add_region(region))
    }

    /// Records the block at which one cleanup handler selects a continuation.
    ///
    /// Register the owning region first with [`Self::add_region`]. A cleanup
    /// that never resumes leaves this unset.
    ///
    /// # Errors
    ///
    /// Returns an error when the handler or resume block does not belong to
    /// this function.
    pub fn set_cleanup_resume(&mut self, handler: HandlerRef, resume_from: BlockId) -> Result<()> {
        self.require_handler(handler)?;
        self.require_block(resume_from)?;
        self.cfg.set_cleanup_resume(handler, resume_from);
        Ok(())
    }

    /// Records one reason-tagged route out of a cleanup handler.
    ///
    /// Register the owning region first with [`Self::add_region`]. Repeating
    /// the same route is a no-op.
    ///
    /// # Errors
    ///
    /// Returns an error when the handler or continuation block does not belong
    /// to this function.
    pub fn add_continuation(
        &mut self,
        handler: HandlerRef,
        continuation: Continuation,
    ) -> Result<()> {
        self.require_handler(handler)?;
        self.require_block(continuation.resume)?;
        self.cfg.add_continuation(handler, continuation);
        Ok(())
    }

    fn require_handler(&self, handler: HandlerRef) -> Result<()> {
        if self
            .cfg
            .regions()
            .get(handler.region().index())
            .is_some_and(|region| handler.index() < region.handlers.len())
        {
            Ok(())
        } else {
            Err(Error::InvalidConstruction(format!(
                "cleanup handler {handler} does not exist"
            )))
        }
    }

    fn copy_cleanups(&mut self, cleanups: &[Cleanup]) -> Result<()> {
        for cleanup in cleanups {
            if let Some(resume_from) = cleanup.resume_from {
                self.set_cleanup_resume(cleanup.handler, resume_from)?;
            }
            for &continuation in &cleanup.continuations {
                self.add_continuation(cleanup.handler, continuation)?;
            }
        }
        Ok(())
    }

    /// Mirrors the source block skeleton so rebuilt identities match.
    pub(super) fn copy_blocks(&mut self, source: &Cfg<Instruction<D>, D::Edge>) {
        for block in source.blocks().iter().skip(1) {
            let rebuilt = self.new_block(block.label().unwrap_or(""));
            debug_assert_eq!(rebuilt, block.id());
        }
    }

    /// Copies every edge, exception region, and cleanup route verbatim.
    ///
    /// Blocks must already mirror the source; call after instructions are
    /// appended so edge endpoints and cleanup blocks exist.
    pub(super) fn copy_structure(&mut self, source: &Cfg<Instruction<D>, D::Edge>) -> Result<()> {
        for edge in source.edges() {
            self.add_edge(edge.source(), edge.target(), edge.payload().clone(), None)?;
        }
        for region in source.regions() {
            self.add_region(region.clone())?;
        }
        self.copy_cleanups(source.cleanups())
    }

    /// Copies the signature and every provenance correspondence verbatim.
    pub(super) fn copy_metadata(
        &mut self,
        signature: Signature<D>,
        provenance: &ProvenanceMap<D>,
    ) -> Result<()> {
        self.set_signature(signature)?;
        for entry in provenance.entries() {
            self.map_entity(entry.source.clone(), entry.entity)?;
        }
        Ok(())
    }

    fn require_region_block(&self, block: BlockId, role: &str) -> Result<()> {
        crate::ir::construction::check_semantic_block(&self.cfg, block, role)
            .map_err(Error::InvalidConstruction)
    }

    /// Returns the synthetic root block.
    #[must_use]
    pub fn entry(&self) -> BlockId {
        self.cfg.entry()
    }

    /// Returns the number of blocks, including the synthetic root.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.cfg.block_count()
    }

    /// Allocates a semantic block with a diagnostic label.
    pub fn new_block(&mut self, label: impl Into<String>) -> BlockId {
        let block = self.cfg.new_block();
        self.cfg.block_mut(block).set_label(label);
        block
    }

    /// Declares one mutable pre-SSA variable.
    ///
    /// # Errors
    ///
    /// Returns an error if the function exceeds the compact identity space.
    pub fn declare_variable(
        &mut self,
        role: D::VariableRole,
        native: Option<D::NativeVariable>,
    ) -> Result<VariableId> {
        let raw = u32::try_from(self.variables.len())
            .map_err(|_| Error::InvalidConstruction("variable count exceeds u32::MAX".into()))?;
        let id = VariableId::from_raw(raw);
        self.variables.push(Variable { id, role, native });
        Ok(id)
    }

    /// Appends one typed semantic instruction to a block.
    ///
    /// `may_throw` records possible implicit exceptional transfer independently
    /// of whether the operation explicitly throws.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid block, identity-space exhaustion, or an
    /// invalid source span.
    pub fn append_instruction(
        &mut self,
        block: BlockId,
        operation: D::Operation,
        uses: Vec<TypedVariable<D>>,
        defs: Vec<TypedVariable<D>>,
        may_throw: bool,
        source: Option<D::SourceSpan>,
    ) -> Result<InstructionId> {
        self.require_block(block)?;
        require_source_span::<D>(source.as_ref())?;
        let raw = u32::try_from(self.instruction_points.len())
            .map_err(|_| Error::InvalidConstruction("instruction count exceeds u32::MAX".into()))?;
        let id = InstructionId::from_raw(raw);
        let point = ProgramPoint {
            block,
            inst_idx: self.cfg.block(block).instructions().len(),
        };
        self.cfg
            .block_mut(block)
            .push(Instruction::new(id, operation, uses, defs, may_throw));
        self.instruction_points.push(point);
        if let Some(span) = source {
            self.provenance.insert(span, EntityId::Instruction(id))?;
        }
        Ok(id)
    }

    /// Adds one exact semantic edge.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid endpoint or source span.
    pub fn add_edge(
        &mut self,
        source: BlockId,
        target: BlockId,
        metadata: D::Edge,
        source_span: Option<D::SourceSpan>,
    ) -> Result<EdgeId> {
        self.require_block(source)?;
        self.require_block(target)?;
        require_source_span::<D>(source_span.as_ref())?;
        let kind = D::edge_kind(&metadata);
        let edge = self
            .cfg
            .add_edge_with_payload(source, target, kind, metadata);
        if let Some(span) = source_span {
            self.provenance.insert(span, EntityId::Edge(edge))?;
        }
        Ok(edge)
    }

    /// Records an additional many-to-many source correspondence.
    ///
    /// # Errors
    ///
    /// Returns an error when `source` is empty or reversed.
    pub fn map_entity(&mut self, source: D::SourceSpan, entity: EntityId) -> Result<bool> {
        Ok(self.provenance.insert(source, entity)?)
    }

    fn require_block(&self, block: BlockId) -> Result<()> {
        crate::ir::construction::check_block(&self.cfg, block).map_err(Error::InvalidConstruction)
    }
}

fn require_source_span<D: Dialect>(source: Option<&D::SourceSpan>) -> Result<()> {
    if source.is_some_and(D::span_is_empty) {
        Err(Error::InvalidProvenance(
            "source span is empty or reversed".into(),
        ))
    } else {
        Ok(())
    }
}

impl<D: VerifyDialect> FunctionBuilder<D> {
    /// Completes and strictly verifies the function.
    ///
    /// An empty semantic block is valid when it forwards control unchanged:
    /// it has no outgoing edge and represents an unresolved or opaque exit,
    /// or it has exactly one non-exceptional outgoing edge and is a
    /// trampoline, such as a machine block holding only a jump or a `nop`.
    /// A block that decides between successors must hold the instruction
    /// that decides, and a block with an exceptional edge must hold the
    /// instruction that throws.
    ///
    /// # Errors
    ///
    /// Returns every discovered invariant violation as one report.
    pub fn finish(self) -> Result<Function<D>> {
        let function = Function {
            cfg: self.cfg,
            variables: self.variables,
            signature: self.signature,
            provenance: self.provenance,
            instruction_points: self.instruction_points,
        };
        let report = function.verify();
        if report.is_ok() {
            Ok(function)
        } else {
            Err(report.into())
        }
    }
}
