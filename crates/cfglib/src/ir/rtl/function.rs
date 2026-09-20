//! RTL function storage and checked construction.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::ir::dialect::Vocabulary;
use crate::region::{HandlerKind, Region, RegionId};
use crate::{BlockId, Cfg, EdgeId};

use super::dialect::Dialect;
use super::error::{Error, Result};
use super::expr::{Expr, Place};
use super::statement::{Statement, StatementId, StatementNode};
use super::types::{Constraint, Shape};

/// Deterministic many-to-many source provenance for RTL statements.
pub type ProvenanceMap<D> = crate::ir::provenance::ProvenanceMap<D, StatementId>;

/// Ordered native parameter places and semantic return types of an RTL
/// function.
pub type Signature<D> =
    crate::ir::signature::Signature<Place<D>, <D as crate::ir::dialect::Vocabulary>::ValueType>;

/// One RTL function backed by a cfglib control-flow graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function<D: Dialect> {
    pub(super) cfg: Cfg<StatementNode<D>, D::Edge>,
    pub(super) source: D::Source,
    pub(super) signature: Signature<D>,
    pub(super) provenance: ProvenanceMap<D>,
    pub(super) statements: u32,
}

impl<D: Dialect> Function<D> {
    /// Returns the exact edge-bearing RTL control-flow graph.
    #[must_use]
    pub const fn cfg(&self) -> &Cfg<StatementNode<D>, D::Edge> {
        &self.cfg
    }

    /// Returns the source function identity.
    #[must_use]
    pub const fn source(&self) -> &D::Source {
        &self.source
    }

    /// Returns the native parameter places and semantic return types.
    #[must_use]
    pub const fn signature(&self) -> &Signature<D> {
        &self.signature
    }

    /// Returns every source span mapped to an RTL statement.
    ///
    /// Multiple spans may identify one statement when source operations
    /// fuse, and one span may identify several statements when an
    /// operation expands.
    #[must_use]
    pub const fn provenance(&self) -> &ProvenanceMap<D> {
        &self.provenance
    }

    /// Returns the number of statements — the exclusive upper bound of
    /// the function's dense [`StatementId`] space.
    #[must_use]
    pub const fn statement_count(&self) -> usize {
        self.statements as usize
    }
}

/// Incremental checked builder of one RTL function.
pub struct FunctionBuilder<D: Dialect> {
    cfg: Cfg<StatementNode<D>, D::Edge>,
    source: D::Source,
    signature: Signature<D>,
    provenance: ProvenanceMap<D>,
    statements: u32,
}

impl<D: Dialect> FunctionBuilder<D> {
    /// Creates a builder with an empty synthetic root block.
    #[must_use]
    pub fn new(source: D::Source) -> Self {
        let mut cfg = Cfg::with_edge_payload();
        cfg.block_mut(cfg.entry()).set_label("root");
        Self {
            cfg,
            provenance: ProvenanceMap::new(source.clone()),
            source,
            signature: Signature::<D>::default(),
            statements: 0,
        }
    }

    /// Declares the ordered native parameter places and semantic return
    /// types.
    ///
    /// # Errors
    ///
    /// Returns an error when a parameter place is empty, repeats a lane,
    /// or overlaps another parameter place.
    pub fn set_signature(&mut self, signature: Signature<D>) -> Result<()> {
        let mut occupied = BTreeSet::new();
        for place in &signature.parameters {
            if place.lanes.is_empty() {
                return Err(Error::InvalidConstruction(
                    "signature parameter occupies no lanes".into(),
                ));
            }
            let mut local = BTreeSet::new();
            for &lane in &place.lanes {
                if !local.insert(lane) {
                    return Err(Error::InvalidConstruction(format!(
                        "signature parameter repeats lane {lane} of {:?}",
                        place.storage
                    )));
                }
                if !occupied.insert((place.storage.clone(), lane)) {
                    return Err(Error::InvalidConstruction(format!(
                        "signature parameters overlap lane {lane} of {:?}",
                        place.storage
                    )));
                }
            }
        }
        self.signature = signature;
        Ok(())
    }

    /// Returns the synthetic root block.
    #[must_use]
    pub fn entry(&self) -> BlockId {
        self.cfg.entry()
    }

    /// Allocates a semantic block with a diagnostic label.
    pub fn new_block(&mut self, label: impl Into<String>) -> BlockId {
        let block = self.cfg.new_block();
        self.cfg.block_mut(block).set_label(label);
        block
    }

    /// Adds one exact semantic edge.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid endpoint.
    pub fn add_edge(
        &mut self,
        source: BlockId,
        target: BlockId,
        metadata: D::Edge,
    ) -> Result<EdgeId> {
        self.require_block(source)?;
        self.require_block(target)?;
        let kind = D::edge_kind(&metadata);
        Ok(self
            .cfg
            .add_edge_with_payload(source, target, kind, metadata))
    }

    /// Attaches one exception region and its handlers.
    ///
    /// The region identity is assigned by the function; the value in
    /// `region.id` is ignored.
    ///
    /// # Errors
    ///
    /// Returns an error for empty protection, invalid or root blocks, a
    /// known handler body omitting its entry, or a parent that has not
    /// already been added.
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
        if let Some(parent) = region.parent
            && parent.index() >= self.cfg.regions().len()
        {
            return Err(Error::InvalidConstruction(format!(
                "region parent {parent} has not been added"
            )));
        }
        Ok(self.cfg.add_region(region))
    }

    /// Appends one statement to a block, validating its shapes, and
    /// returns the statement's stable identity.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid block, an assignment-free
    /// transfer, a width mismatch between a place and its value, a
    /// destination lane written twice anywhere in one transfer, a
    /// zero-lane value, a non-scalar branch condition or dispatch
    /// scrutinee, or a total-bit-width-changing reinterpretation.
    pub fn append(
        &mut self,
        block: BlockId,
        statement: Statement<D>,
        span: Option<D::SourceSpan>,
    ) -> Result<StatementId> {
        self.require_block(block)?;
        validate_statement(&statement)?;
        if span.as_ref().is_some_and(D::span_is_empty) {
            return Err(Error::InvalidConstruction(
                "source span is empty or reversed".into(),
            ));
        }
        let id = StatementId::from_raw(self.statements);
        self.statements = self
            .statements
            .checked_add(1)
            .ok_or_else(|| Error::InvalidConstruction("statement count exceeds u32::MAX".into()))?;
        self.cfg
            .block_mut(block)
            .push(StatementNode::new(id, statement, span.clone()));
        if let Some(span) = span {
            self.provenance
                .insert(span, id)
                .map_err(|error| Error::InvalidConstruction(error.to_string()))?;
        }
        Ok(id)
    }

    /// Records an additional source correspondence for one statement.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown statement or an empty/reversed
    /// source span.
    pub fn map_statement(&mut self, source: D::SourceSpan, statement: StatementId) -> Result<bool> {
        if statement.index() >= self.statements as usize {
            return Err(Error::InvalidConstruction(format!(
                "statement {statement:?} is outside a {}-statement function",
                self.statements
            )));
        }
        self.provenance
            .insert(source, statement)
            .map_err(|error| Error::InvalidConstruction(error.to_string()))
    }

    /// Completes the function, validating its control-flow structure.
    ///
    /// Unreachable blocks are legal and preserved — dead code after a
    /// return or throw is source-faithful in managed bytecode, and SSA
    /// annotates such blocks as their own dominator-tree roots.
    ///
    /// # Errors
    ///
    /// Returns an error when a terminating statement sits before the end
    /// of its block, a branch block carries fewer than two outgoing
    /// normal edges, a dispatch block carries none, a return block
    /// carries any edge, a raise block carries a non-exceptional edge,
    /// or a block with exceptional edges lacks exactly one throwing
    /// statement to own them.
    pub fn finish(self) -> Result<Function<D>> {
        for block_id in self.cfg.block_ids() {
            let block = self.cfg.block(block_id);
            let statements = block.instructions();
            for (index, node) in statements.iter().enumerate() {
                if node.statement().is_terminator() && index + 1 != statements.len() {
                    return Err(Error::InvalidConstruction(format!(
                        "control-flow statement is not last in block {block_id}"
                    )));
                }
            }
        }

        let mut normal: Vec<usize> = vec![0; self.cfg.block_bound()];
        let mut exceptional: Vec<usize> = vec![0; self.cfg.block_bound()];
        for edge in self.cfg.edges() {
            let source = edge.source().index();
            if edge.kind().is_exceptional() {
                exceptional[source] += 1;
            } else {
                normal[source] += 1;
            }
        }
        for block_id in self.cfg.block_ids() {
            let block = self.cfg.block(block_id);
            let index = block_id.index();
            match block.instructions().last().map(StatementNode::statement) {
                Some(Statement::Return { .. }) if normal[index] + exceptional[index] != 0 => {
                    return Err(Error::InvalidConstruction(format!(
                        "return block {block_id} has outgoing edges"
                    )));
                }
                Some(Statement::Branch { .. }) if normal[index] < 2 => {
                    return Err(Error::InvalidConstruction(format!(
                        "branch block {} decides between {} outgoing edges",
                        block_id, normal[index]
                    )));
                }
                Some(Statement::Dispatch { .. }) if normal[index] == 0 => {
                    return Err(Error::InvalidConstruction(format!(
                        "dispatch block {block_id} has no outgoing edges"
                    )));
                }
                Some(Statement::Raise { .. }) if normal[index] != 0 => {
                    return Err(Error::InvalidConstruction(format!(
                        "raise block {block_id} has a normal outgoing edge"
                    )));
                }
                _ => {}
            }
            // Exceptional edges need exactly one owning statement, so a
            // handler observes an unambiguous pre-state.
            if exceptional[index] != 0 {
                let throwing = block
                    .instructions()
                    .iter()
                    .filter(|node| node.statement().may_throw())
                    .count();
                if throwing != 1 {
                    return Err(Error::InvalidConstruction(format!(
                        "block {block_id} has exceptional edges but {throwing} throwing \
                         statements"
                    )));
                }
            }
        }

        Ok(Function {
            cfg: self.cfg,
            source: self.source,
            signature: self.signature,
            provenance: self.provenance,
            statements: self.statements,
        })
    }

    fn require_block(&self, block: BlockId) -> Result<()> {
        crate::ir::construction::check_block(&self.cfg, block).map_err(Error::InvalidConstruction)
    }

    fn require_region_block(&self, block: BlockId, role: &str) -> Result<()> {
        crate::ir::construction::check_semantic_block(&self.cfg, block, role)
            .map_err(Error::InvalidConstruction)
    }
}

fn validate_statement<D: Dialect>(statement: &Statement<D>) -> Result<()> {
    match statement {
        Statement::Transfer { assignments, .. } => {
            if assignments.is_empty() {
                return Err(Error::InvalidConstruction(
                    "transfer carries no assignments; an effect-only \
                     instruction is a Statement::Effect"
                        .into(),
                ));
            }
            // A lane written twice anywhere in one parallel transfer has
            // no defined result, whether the writes share a place or not.
            let mut written: BTreeSet<(&<D as Vocabulary>::NativeVariable, u8)> = BTreeSet::new();
            for (place, value) in assignments {
                if place.lanes.is_empty() {
                    return Err(Error::InvalidConstruction(
                        "assignment writes no lanes".into(),
                    ));
                }
                for &lane in &place.lanes {
                    if !written.insert((&place.storage, lane)) {
                        return Err(Error::InvalidConstruction(format!(
                            "transfer writes lane {lane} of {:?} twice",
                            place.storage
                        )));
                    }
                }
                let width = value.shape().lanes;
                if usize::from(width) != place.lanes.len() {
                    return Err(Error::InvalidConstruction(format!(
                        "value width {width} does not match {}-lane destination",
                        place.lanes.len()
                    )));
                }
                validate_expr(value)?;
            }
            Ok(())
        }
        Statement::Effect { operands, .. } | Statement::Raise { operands, .. } => {
            for operand in operands {
                validate_expr(operand)?;
            }
            Ok(())
        }
        Statement::Branch { condition } => {
            if condition.shape().lanes != 1 {
                return Err(Error::InvalidConstruction(
                    "branch condition is not scalar".into(),
                ));
            }
            validate_expr(condition)
        }
        Statement::Dispatch { scrutinee } => {
            if scrutinee.shape().lanes != 1 {
                return Err(Error::InvalidConstruction(
                    "dispatch scrutinee is not scalar".into(),
                ));
            }
            validate_expr(scrutinee)
        }
        Statement::Return { values } => {
            for value in values {
                if value.shape().lanes == 0 {
                    return Err(Error::InvalidConstruction(
                        "return value carries no lanes".into(),
                    ));
                }
                validate_expr(value)?;
            }
            Ok(())
        }
    }
}

fn validate_expr<D: Dialect>(expr: &Expr<D>) -> Result<()> {
    match expr {
        Expr::Read { lanes, .. } => {
            if lanes.is_empty() {
                return Err(Error::InvalidConstruction("read selects no lanes".into()));
            }
            Ok(())
        }
        Expr::Const { bits, shape } => {
            if bits.is_empty() {
                return Err(Error::InvalidConstruction(
                    "constant carries no lanes".into(),
                ));
            }
            let words = shape.scalar.word_count();
            if bits.len() != usize::from(shape.lanes) * words {
                return Err(Error::InvalidConstruction(format!(
                    "constant carries {} words for a {}-lane shape of {words}-word lanes",
                    bits.len(),
                    shape.lanes
                )));
            }
            Ok(())
        }
        Expr::Apply { operands, .. } => {
            for operand in operands {
                validate_expr(operand)?;
            }
            Ok(())
        }
        Expr::Reinterpret { operand, shape } => {
            let source = operand.shape();
            if let (Some(from), Some(to)) = (total_bit_width(&source), total_bit_width(shape))
                && from != to
            {
                return Err(Error::InvalidConstruction(format!(
                    "reinterpretation changes total bit width {from} to {to}"
                )));
            }
            validate_expr(operand)
        }
    }
}

/// Returns the total known bit width of one shape.
fn total_bit_width<C: Constraint>(shape: &Shape<C>) -> Option<u64> {
    shape
        .scalar
        .width()
        .map(|width| u64::from(width) * u64::from(shape.lanes))
}
