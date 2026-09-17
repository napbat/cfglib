//! Lowering of MLIL functions into target RTL.
//!
//! The downward counterpart of [`lift`](super::lift()): the dialect
//! plans target storage for the whole function at once ([`Lower::plan`])
//! and translates each instruction into RTL statements
//! ([`Lower::lower_instruction`]) — legalizing representable semantic
//! operations, staging parallel copies as [`Statement::Transfer`]s, and
//! refusing unsupported semantics with a typed
//! [`Error::Lowering`](super::Error::Lowering) rather than a silent
//! approximation. Edges lower *after* every statement exists, so
//! [`Lower::lower_edge`] can remap instruction identities in edge
//! payloads (an exceptional edge's throw site) onto lowered
//! [`StatementId`]s. Lifetime splitting and coalescing happen at the
//! MLIL level
//! ([`split_variables`](crate::ir::mlil::Function::split_variables),
//! copy propagation) before lowering; instruction selection, layout, and
//! encoding stay frontend-owned below RTL.
//!
//! [`Lowered`] retains the full rewrite map — MLIL block, instruction,
//! and edge to their RTL counterparts — so provenance survives the
//! round trip.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use crate::ir::dialect::Vocabulary;
use crate::ir::mlil::{
    Function as MlilFunction, Instruction as MlilInstruction, InstructionId,
    Variable as MlilVariable, VariableId,
};
use crate::region::{Handler, HandlerBody, HandlerKind, Region};
use crate::{BlockId, EdgeId};

use super::dialect::{Dialect, MlilBridge};
use super::error::{Error, Result};
use super::expr::{Expr, Place};
use super::function::{Function, FunctionBuilder, Signature};
use super::statement::{Statement, StatementId};

/// One whole-function storage assignment: every MLIL variable's target
/// place, planned in one coordinated pass.
#[derive(Debug, Clone, Default)]
pub struct Placement<D: Dialect> {
    places: BTreeMap<u32, Place<D>>,
}

impl<D: Dialect> Placement<D> {
    /// An empty plan.
    #[must_use]
    pub fn new() -> Self {
        Self {
            places: BTreeMap::new(),
        }
    }

    /// Assigns one variable's target place.
    pub fn assign(&mut self, variable: VariableId, place: Place<D>) {
        self.places.insert(variable.raw(), place);
    }

    /// The planned place of one variable.
    #[must_use]
    pub fn place(&self, variable: VariableId) -> Option<&Place<D>> {
        self.places.get(&variable.raw())
    }
}

/// Plans one whole place per MLIL variable, preserving native storage.
///
/// The standard plan of a variable-shaped source: `preserved` answers a
/// variable's retained target storage (a native slot worth keeping through
/// the round trip), every other variable receives `generated(ordinal)` with
/// ordinals dense from zero in declaration order (mint fresh slots by adding
/// a caller-computed base — typically one past the highest preserved slot),
/// and every place spans exactly lane zero.
///
/// # Errors
///
/// Propagates `generated`'s error — typically a target storage space
/// exhausted by the fresh slots.
pub fn plan_whole_places<M: crate::ir::mlil::Dialect, D: Dialect>(
    function: &MlilFunction<M>,
    mut preserved: impl FnMut(&MlilVariable<M>) -> Option<<D as Vocabulary>::NativeVariable>,
    mut generated: impl FnMut(u64) -> Result<<D as Vocabulary>::NativeVariable>,
) -> Result<Placement<D>> {
    let mut placement = Placement::new();
    let mut ordinal = 0_u64;
    for variable in function.variables() {
        let storage = if let Some(storage) = preserved(variable) {
            storage
        } else {
            let storage = generated(ordinal)?;
            ordinal += 1;
            storage
        };
        placement.assign(
            variable.id,
            Place {
                storage,
                lanes: vec![0],
            },
        );
    }
    Ok(placement)
}

/// The lowering contract from a dialect's MLIL onto target RTL.
///
/// Implemented on the target RTL marker, whose [`MlilBridge::Mlil`]
/// selects the semantic source. Both levels share one vocabulary while
/// their edge types stay independent and are translated by
/// [`lower_edge`](Self::lower_edge). Storage-assignment and legalization
/// policy live entirely in the implementation; cfglib supplies the walk,
/// checked construction, and rewrite maps.
pub trait Lower: MlilBridge {
    /// Plans target storage for the whole function at once — coordinated
    /// allocation (JVM local numbering, Dalvik register pressure, wide
    /// pairs) rather than one-variable-at-a-time choices. Every variable
    /// an instruction touches must receive a place; a missing one
    /// surfaces as a typed error when the translation reads it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Lowering`](super::Error::Lowering) when some
    /// variable has no legal storage assignment.
    fn plan(function: &MlilFunction<Self::Mlil>) -> Result<Placement<Self>>;

    /// Translates one MLIL instruction into RTL statements through the
    /// context.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Lowering`](super::Error::Lowering) when the
    /// instruction's semantics have no target representation.
    fn lower_instruction(
        context: &mut LowerContext<'_, Self>,
        instruction: &MlilInstruction<Self::Mlil>,
    ) -> Result<()>;

    /// Translates one MLIL edge into the lowered function's RTL edge
    /// metadata.
    ///
    /// Runs after every statement is emitted, so the context resolves
    /// instructions to lowered [`StatementId`]s — an exceptional edge's
    /// throw site remaps from the MLIL instruction identity into the RTL
    /// statement domain. A dialect sharing one edge type across both
    /// levels clones the metadata.
    ///
    /// # Errors
    ///
    /// Returns an error when edge metadata names an entity that was not
    /// lowered or otherwise has no exact target representation.
    fn lower_edge(
        edge: &<Self::Mlil as crate::ir::mlil::Dialect>::Edge,
        context: &LowerEdgeContext<'_>,
    ) -> Result<<Self as Dialect>::Edge>;
}

/// The context one [`Lower::lower_instruction`] call appends RTL
/// statements through.
pub struct LowerContext<'a, D: Lower> {
    builder: &'a mut FunctionBuilder<D>,
    block: BlockId,
    placement: &'a Placement<D>,
    spans: Vec<<D as crate::ir::dialect::Vocabulary>::SourceSpan>,
    statements: Vec<StatementId>,
}

impl<D: Lower> LowerContext<'_, D> {
    /// The planned target place of one MLIL variable.
    ///
    /// # Errors
    ///
    /// Returns an error for a variable the plan never placed.
    pub fn place(&self, variable: VariableId) -> Result<&Place<D>> {
        self.placement
            .place(variable)
            .ok_or_else(|| Error::Lowering(format!("variable {variable:?} has no target place")))
    }

    /// A whole-place read of one MLIL variable under a constraint.
    ///
    /// # Errors
    ///
    /// Returns an error for a variable without a target place.
    pub fn read(&self, variable: VariableId, scalar: D::Constraint) -> Result<Expr<D>> {
        let place = self.place(variable)?;
        Ok(Expr::Read {
            storage: place.storage.clone(),
            lanes: place.lanes.clone(),
            scalar,
        })
    }

    /// Appends one RTL statement for the instruction, validated by the
    /// checked builder and carrying the instruction's source span.
    ///
    /// # Errors
    ///
    /// Returns an error when the statement fails RTL validation.
    pub fn emit(&mut self, statement: Statement<D>) -> Result<StatementId> {
        let id = self
            .builder
            .append(self.block, statement, self.spans.first().cloned())?;
        for span in self.spans.iter().skip(1) {
            self.builder.map_statement(span.clone(), id)?;
        }
        self.statements.push(id);
        Ok(id)
    }
}

/// The finished mappings available while lowering one edge.
pub struct LowerEdgeContext<'a> {
    blocks: &'a [BlockId],
    statements: &'a [Vec<StatementId>],
    owner: Option<InstructionId>,
}

impl LowerEdgeContext<'_> {
    /// The MLIL instruction owning the edge: the unique throwing
    /// instruction of the source block for an exceptional edge, the
    /// block's final instruction otherwise, `None` for an empty block.
    #[must_use]
    pub const fn owner(&self) -> Option<InstructionId> {
        self.owner
    }

    /// The RTL statements one MLIL instruction became, in order.
    #[must_use]
    pub fn statements(&self, instruction: InstructionId) -> &[StatementId] {
        self.statements
            .get(instruction.index())
            .map_or(&[], Vec::as_slice)
    }

    /// The RTL block one MLIL block became.
    #[must_use]
    pub fn block(&self, block: BlockId) -> Option<BlockId> {
        self.blocks.get(block.index()).copied()
    }
}

/// The result of lowering: a checked RTL function plus the rewrite maps
/// from every MLIL entity to its RTL counterparts.
pub struct Lowered<D: Lower> {
    /// The lowered RTL function.
    pub function: Function<D>,
    /// The storage plan the lowering ran under — including places
    /// reserved for variables no instruction touched, such as unused
    /// parameters holding JVM locals or Dalvik registers.
    pub placement: Placement<D>,
    /// MLIL block index → RTL block.
    blocks: Vec<BlockId>,
    /// MLIL instruction index → RTL statements, in emission order.
    statements: Vec<Vec<StatementId>>,
    /// MLIL edge index → RTL edge.
    edges: Vec<EdgeId>,
}

impl<D: Lower> Lowered<D> {
    /// The RTL block one MLIL block became.
    #[must_use]
    pub fn block(&self, block: BlockId) -> Option<BlockId> {
        self.blocks.get(block.index()).copied()
    }

    /// The RTL statements one MLIL instruction became, in order.
    #[must_use]
    pub fn statements(&self, instruction: InstructionId) -> &[StatementId] {
        self.statements
            .get(instruction.index())
            .map_or(&[], Vec::as_slice)
    }

    /// The RTL edge one MLIL edge became.
    #[must_use]
    pub fn edge(&self, edge: EdgeId) -> Option<EdgeId> {
        self.edges.get(edge.index()).copied()
    }
}

/// Lowers one MLIL function onto target RTL.
///
/// # Errors
///
/// Returns [`Error::Lowering`](super::Error::Lowering) when a variable
/// has no legal storage or an instruction no target representation, and
/// construction errors when the emitted RTL is structurally invalid.
pub fn lower<D: Lower>(function: &MlilFunction<D::Mlil>) -> Result<Lowered<D>> {
    let cfg = function.cfg();
    let placement = D::plan(function)?;

    let mut builder = FunctionBuilder::<D>::new(function.source().clone());
    let parameters = function
        .signature()
        .parameters
        .iter()
        .map(|&variable| {
            placement.place(variable).cloned().ok_or_else(|| {
                Error::Lowering(format!("parameter {variable:?} has no target place"))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    builder.set_signature(Signature::new(
        parameters,
        function.signature().returns.clone(),
    ))?;
    let mut blocks: Vec<BlockId> = Vec::with_capacity(cfg.block_count());
    for block_id in cfg.block_ids() {
        let block = cfg.block(block_id);
        if block_id == cfg.entry() {
            blocks.push(builder.entry());
        } else {
            let label = block.label().unwrap_or("b").to_string();
            blocks.push(builder.new_block(label));
        }
    }

    let mut statements: Vec<Vec<StatementId>> = vec![Vec::new(); function.instruction_count()];
    for block_id in cfg.block_ids() {
        let block = cfg.block(block_id);
        for instruction in block.instructions() {
            let spans = function
                .provenance()
                .mappings_to(crate::ir::mlil::EntityId::Instruction(instruction.id()))
                .map(|entry| entry.source.clone())
                .collect();
            let mut context = LowerContext {
                builder: &mut builder,
                block: blocks[block_id.index()],
                placement: &placement,
                spans,
                statements: Vec::new(),
            };
            D::lower_instruction(&mut context, instruction)?;
            statements[instruction.id().index()] = context.statements;
        }
    }

    let mut edges: Vec<EdgeId> = Vec::new();
    for edge in cfg.edges() {
        let source = cfg.block(edge.source());
        let owner = if edge.kind().is_exceptional() {
            source
                .instructions()
                .iter()
                .find(|instruction| instruction.may_throw())
                .map(MlilInstruction::id)
        } else {
            source.instructions().last().map(MlilInstruction::id)
        };
        let context = LowerEdgeContext {
            blocks: &blocks,
            statements: &statements,
            owner,
        };
        let lowered = builder.add_edge(
            blocks[edge.source().index()],
            blocks[edge.target().index()],
            D::lower_edge(edge.payload(), &context)?,
        )?;
        edges.push(lowered);
    }

    for region in cfg.regions() {
        builder.add_region(map_region(region, &blocks)?)?;
    }

    Ok(Lowered {
        function: builder.finish()?,
        placement,
        blocks,
        statements,
        edges,
    })
}

fn map_region(region: &Region, blocks: &[BlockId]) -> Result<Region> {
    let block = |source: BlockId| {
        blocks
            .get(source.index())
            .copied()
            .ok_or_else(|| Error::Lowering("region block lost during lowering".into()))
    };
    let protected_blocks = region
        .protected_blocks
        .iter()
        .map(|&source| block(source))
        .collect::<Result<_>>()?;
    let handlers = region
        .handlers
        .iter()
        .map(|handler| {
            let body = match &handler.body {
                HandlerBody::Unknown => HandlerBody::Unknown,
                HandlerBody::Known(sources) => HandlerBody::Known(
                    sources
                        .iter()
                        .map(|&source| block(source))
                        .collect::<Result<_>>()?,
                ),
            };
            let kind = match handler.kind {
                HandlerKind::Filter { filter_block } => HandlerKind::Filter {
                    filter_block: block(filter_block)?,
                },
                other => other,
            };
            Ok(Handler {
                entry: block(handler.entry)?,
                body,
                kind,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Region {
        id: region.id,
        protected_blocks,
        handlers,
        parent: region.parent,
    })
}
