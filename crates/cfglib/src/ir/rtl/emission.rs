//! Dialect-driven MLIL emission and the lift's provenance maps.
//!
//! Each serialized statement reaches [`Lift::emit`](super::Lift::emit)
//! through an [`Emission`] context that can append one or more MLIL
//! instructions — the storage-flavored one-operation form is
//! [`Emission::single`], while semantic dialects expand or fuse freely
//! under read-alignment and exceptional-placement validation, splitting
//! normal continuations off with [`Emission::continuation`] so a throw
//! site can stay terminal in its block. [`LiftMaps`] records the block,
//! statement, instruction, and edge correspondences the finished
//! [`Lifting`] hands back for regions, signatures, and diagnostics.

extern crate alloc;

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;
use smallvec::SmallVec;

use crate::ir::dialect::Vocabulary;
use crate::ir::mlil::{
    EntityId, FunctionBuilder as MlilBuilder, InstructionId, TypedVariable, VariableId,
};
use crate::{BlockId, EdgeId, SsaValue};

use super::dialect::{Dialect, Lift, MlilBridge};
use super::error::{Error, Result};
use super::expr::{Expr, Place};
use super::render::Webs;
use super::statement::{Lane, Statement, StatementId};
use super::template::{LiftedStatement, VarExpr, WebInfo};
use super::types::Shape;

type MlilOf<D> = <D as MlilBridge>::Mlil;

/// The stable correspondences one lift established.
///
/// Signatures, exception regions, native provenance, and diagnostics
/// attach through these maps rather than through coincidentally equal
/// dense identities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiftMaps {
    pub(super) blocks: Vec<BlockId>,
    pub(super) block_expansions: Vec<Vec<BlockId>>,
    pub(super) instructions: Vec<Vec<InstructionId>>,
    pub(super) throw_sites: Vec<Option<InstructionId>>,
    pub(super) edges: Vec<EdgeId>,
}

impl LiftMaps {
    /// The MLIL block one RTL block became.
    #[must_use]
    pub fn block(&self, block: BlockId) -> Option<BlockId> {
        self.blocks.get(block.index()).copied()
    }

    /// Every MLIL block one RTL block expanded into, in control-flow
    /// order. The first is the primary block returned by [`Self::block`].
    #[must_use]
    pub fn blocks(&self, block: BlockId) -> &[BlockId] {
        self.block_expansions
            .get(block.index())
            .map_or(&[], Vec::as_slice)
    }

    /// The MLIL instructions one RTL statement emitted, in order.
    #[must_use]
    pub fn instructions(&self, statement: StatementId) -> &[InstructionId] {
        self.instructions
            .get(statement.index())
            .map_or(&[], Vec::as_slice)
    }

    /// The emitted instruction designated as one statement's throw site.
    #[must_use]
    pub fn throw_site(&self, statement: StatementId) -> Option<InstructionId> {
        self.throw_sites.get(statement.index()).copied().flatten()
    }

    /// The MLIL edge one RTL edge became.
    #[must_use]
    pub fn edge(&self, edge: EdgeId) -> Option<EdgeId> {
        self.edges.get(edge.index()).copied()
    }
}

/// The result of lifting: an MLIL builder ready for signature and region
/// assignment and [`finish`](MlilBuilder::finish), the recovered webs,
/// and the full provenance maps.
pub struct Lifting<D: Lift> {
    /// The populated MLIL builder.
    pub builder: MlilBuilder<MlilOf<D>>,
    /// Recovered webs, resolvable by variable identity. Webs flagged
    /// [`live_in`](WebInfo::live_in) are the function's parameters and
    /// implicit input channels, in declaration order.
    pub webs: Webs<D>,
    /// Block, statement, instruction, and edge correspondences.
    pub maps: LiftMaps,
}

/// The finished mappings available while lifting one edge.
pub struct EdgeContext<'a> {
    pub(super) maps: &'a LiftMaps,
    pub(super) owner: Option<StatementId>,
}

impl EdgeContext<'_> {
    /// The RTL statement owning the edge: the unique throwing statement
    /// for an exceptional edge, the block's terminator otherwise, `None`
    /// for a plain fallthrough out of an unterminated block.
    #[must_use]
    pub const fn owner(&self) -> Option<StatementId> {
        self.owner
    }

    /// The MLIL instructions one statement emitted, in order.
    #[must_use]
    pub fn instructions(&self, statement: StatementId) -> &[InstructionId] {
        self.maps.instructions(statement)
    }

    /// The emitted throw site of one statement.
    #[must_use]
    pub fn throw_site(&self, statement: StatementId) -> Option<InstructionId> {
        self.maps.throw_site(statement)
    }

    /// The MLIL block one RTL block became.
    #[must_use]
    pub fn block(&self, block: BlockId) -> Option<BlockId> {
        self.maps.block(block)
    }

    /// Every MLIL block one RTL block expanded into.
    #[must_use]
    pub fn blocks(&self, block: BlockId) -> &[BlockId] {
        self.maps.blocks(block)
    }
}

pub(super) struct Resolver<'a, D: Dialect> {
    pub(super) ids: &'a BTreeMap<(Lane<D>, usize), usize>,
    pub(super) roots: &'a [usize],
    pub(super) web_of_root: &'a [Option<usize>],
}

impl<D: Dialect> Resolver<'_, D> {
    /// Resolves one SSA value to its web index.
    fn web(&self, value: &SsaValue<Lane<D>>) -> Result<usize> {
        let key = (value.variable.clone(), value.version);
        let id = self
            .ids
            .get(&key)
            .ok_or_else(|| Error::Lifting("unregistered SSA value".into()))?;
        self.web_of_root
            .get(self.roots[*id])
            .copied()
            .flatten()
            .ok_or_else(|| Error::Lifting("web lost its variable".into()))
    }
}

pub(super) struct Emitter<'a, D: Lift> {
    pub(super) builder: MlilBuilder<MlilOf<D>>,
    pub(super) webs: Vec<WebInfo<D>>,
    pub(super) web_index: Vec<Option<usize>>,
    pub(super) resolver: Resolver<'a, D>,
    pub(super) maps: LiftMaps,
    /// The MLIL block currently receiving each RTL block's instructions —
    /// its chain entry until [`Emission::continuation`] extends the chain.
    pub(super) tail: Vec<BlockId>,
    /// The MLIL block holding each statement's designated throw site.
    pub(super) throw_blocks: Vec<Option<BlockId>>,
    pub(super) current: Option<StatementId>,
    pub(super) native_defined: bool,
}

/// What one serialized statement defines.
#[derive(Clone, Copy, Debug)]
enum Defined<'a> {
    /// The statement defines nothing.
    Nothing,
    /// One assignment's target web. `merge` marks that unwritten
    /// positions survive, so the target is also the trailing use.
    Target {
        /// The defined web.
        web: usize,
        /// Whether unwritten positions keep their previous value.
        merge: bool,
    },
    /// An effect's written webs, in write order.
    Writes(&'a [usize]),
}

/// One rebuilt assignment awaiting serialization.
struct PendingAssign<D: Dialect> {
    target: usize,
    positions: Vec<u8>,
    value: VarExpr<D>,
    reads: Vec<usize>,
}

/// Exceptional behavior visible while one RTL statement is emitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExceptionalFlow {
    /// The statement cannot throw.
    None,
    /// Failure unwinds out of the function.
    Unwind,
    /// At least one exceptional edge reaches an in-function handler.
    Local,
}

impl ExceptionalFlow {
    fn from_flags(may_throw: bool, has_exceptional_successors: bool) -> Result<Self> {
        match (may_throw, has_exceptional_successors) {
            (false, false) => Ok(Self::None),
            (true, false) => Ok(Self::Unwind),
            (true, true) => Ok(Self::Local),
            (false, true) => Err(Error::Lifting(
                "a non-throwing statement cannot own exceptional successors".into(),
            )),
        }
    }

    const fn may_throw(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// The context one [`Lift::emit`] call appends MLIL instructions
/// through.
///
/// It validates the dialect's expansion: every use must be a variable
/// the statement made available (its reads, its assignment target, its
/// writes, a context temporary, or an earlier definition of the same
/// statement), exactly one appended instruction may carry the
/// statement's exceptional behavior, and no native web may be defined
/// before that throw site — the statement's own writes excepted, because
/// they are defined after its throw point.
pub struct Emission<'a, 'b, D: Lift> {
    emitter: &'a mut Emitter<'b, D>,
    source: usize,
    statement: StatementId,
    reads: Vec<TypedVariable<MlilOf<D>>>,
    target: Option<TypedVariable<MlilOf<D>>>,
    targets: Vec<TypedVariable<MlilOf<D>>>,
    merge: bool,
    exceptional_flow: ExceptionalFlow,
    spans: Vec<<D as Vocabulary>::SourceSpan>,
    allowed: SmallVec<[VariableId; 8]>,
    appended_throwing: bool,
}

impl<D: Lift> Emission<'_, '_, D> {
    /// The RTL statement being emitted.
    #[must_use]
    pub const fn statement(&self) -> StatementId {
        self.statement
    }

    /// The typed web variables the statement's reads consume, in
    /// pre-order — the instruction `uses` of the one-operation form.
    #[must_use]
    pub fn reads(&self) -> &[TypedVariable<MlilOf<D>>] {
        &self.reads
    }

    /// The typed web variable an assignment defines.
    #[must_use]
    pub const fn target(&self) -> Option<&TypedVariable<MlilOf<D>>> {
        self.target.as_ref()
    }

    /// The typed web variables an effect's
    /// [`writes`](super::Statement::Effect::writes) define, in write
    /// order — the instruction `defs` of the one-operation form.
    ///
    /// Empty for every other statement form; an assignment's definition
    /// is [`target`](Self::target).
    #[must_use]
    pub fn targets(&self) -> &[TypedVariable<MlilOf<D>>] {
        &self.targets
    }

    /// Whether the statement can transfer exceptionally — exactly one
    /// appended instruction must then carry `may_throw`.
    #[must_use]
    pub const fn may_throw(&self) -> bool {
        self.exceptional_flow.may_throw()
    }

    /// Whether this statement owns at least one in-function exceptional
    /// successor in the source RTL graph.
    ///
    /// A statement may throw without having such a successor when failure
    /// unwinds out of the function. Dialects can use this distinction to
    /// preserve pre-throw native state only when a local handler can observe
    /// it.
    #[must_use]
    pub const fn has_exceptional_successors(&self) -> bool {
        matches!(self.exceptional_flow, ExceptionalFlow::Local)
    }

    /// Continues emission in a fresh MLIL block, wired from the current
    /// one by an edge carrying `metadata` — the normal-continuation
    /// mechanism for dialects whose throw sites must be terminal in
    /// their block: emit the throwing instruction, split, then commit
    /// native state in the continuation. Every later instruction of the
    /// same RTL block lands in the continuation too, while the block's
    /// exceptional edges keep leaving the throw site's block.
    ///
    /// # Errors
    ///
    /// Returns an error when MLIL construction fails.
    pub fn continuation(
        &mut self,
        metadata: <MlilOf<D> as crate::ir::mlil::Dialect>::Edge,
    ) -> Result<BlockId> {
        let current = self.emitter.tail[self.source];
        let block = self.emitter.builder.new_block("cont");
        self.emitter
            .builder
            .add_edge(current, block, metadata, None)
            .map_err(|error| Error::Lifting(error.to_string()))?;
        self.emitter.tail[self.source] = block;
        self.emitter.maps.block_expansions[self.source].push(block);
        Ok(block)
    }

    /// Declares one dialect temporary for a multi-instruction expansion.
    ///
    /// # Errors
    ///
    /// Returns an error when MLIL declaration fails.
    pub fn temporary(
        &mut self,
        role: <D as Vocabulary>::VariableRole,
        value_type: <D as Vocabulary>::ValueType,
    ) -> Result<TypedVariable<MlilOf<D>>> {
        let variable = self
            .emitter
            .builder
            .declare_variable(role, None)
            .map_err(|error| Error::Lifting(error.to_string()))?;
        if !self.allowed.contains(&variable) {
            self.allowed.push(variable);
        }
        Ok(TypedVariable::new(variable, value_type))
    }

    /// Appends one MLIL instruction for the statement.
    ///
    /// # Errors
    ///
    /// Returns an error when a use references a variable the statement
    /// did not make available, when a second instruction claims the
    /// statement's throw site, or when a native web was already defined
    /// before a throwing append.
    pub fn append(
        &mut self,
        operation: <MlilOf<D> as crate::ir::mlil::Dialect>::Operation,
        uses: Vec<TypedVariable<MlilOf<D>>>,
        defs: Vec<TypedVariable<MlilOf<D>>>,
        may_throw: bool,
    ) -> Result<InstructionId> {
        for used in &uses {
            if !self.allowed.contains(&used.variable) {
                return Err(Error::Lifting(format!(
                    "emission uses a variable outside statement {}'s reads",
                    self.statement.raw()
                )));
            }
        }
        if may_throw {
            if self.emitter.maps.throw_sites[self.statement.index()].is_some() {
                return Err(Error::Lifting(format!(
                    "statement {} designated two throw sites",
                    self.statement.raw()
                )));
            }
            if self.emitter.native_defined {
                return Err(Error::Lifting(format!(
                    "statement {} committed native state before its throw site",
                    self.statement.raw()
                )));
            }
        }
        for defined in &defs {
            if !self.allowed.contains(&defined.variable) {
                self.allowed.push(defined.variable);
            }
            // A statement's own writes are defined *after* its throw
            // point: the operation that throws is the operation that
            // produces them, so committing one is not pre-throw native
            // state. A call that may throw and defines its results is
            // exactly that shape.
            if self
                .targets
                .iter()
                .any(|write| write.variable == defined.variable)
            {
                continue;
            }
            if self
                .emitter
                .web_index
                .get(defined.variable.index())
                .copied()
                .flatten()
                .is_some_and(|web| self.emitter.webs[web].storage.is_some())
            {
                self.emitter.native_defined = true;
            }
        }
        let block = self.emitter.tail[self.source];
        let id = self
            .emitter
            .builder
            .append_instruction(
                block,
                operation,
                uses,
                defs,
                may_throw,
                self.spans.first().cloned(),
            )
            .map_err(|error| Error::Lifting(error.to_string()))?;
        for span in self.spans.iter().skip(1) {
            self.emitter
                .builder
                .map_entity(span.clone(), EntityId::Instruction(id))
                .map_err(|error| Error::Lifting(error.to_string()))?;
        }
        self.emitter.maps.instructions[self.statement.index()].push(id);
        if may_throw {
            self.emitter.maps.throw_sites[self.statement.index()] = Some(id);
            self.emitter.throw_blocks[self.statement.index()] = Some(block);
            self.appended_throwing = true;
        }
        Ok(id)
    }

    /// Appends the one-operation form: uses are the statement's reads
    /// (plus the merging assignment's trailing target), the definitions
    /// are the assignment target or the effect's writes, and the
    /// statement's exceptional behavior lands on this instruction.
    ///
    /// # Errors
    ///
    /// Returns an error when MLIL construction fails.
    pub fn single(
        &mut self,
        operation: <MlilOf<D> as crate::ir::mlil::Dialect>::Operation,
    ) -> Result<InstructionId> {
        let mut uses = self.reads.clone();
        if self.merge
            && let Some(target) = &self.target
        {
            uses.push(target.clone());
        }
        // An assignment target and an effect's writes are mutually
        // exclusive, so the chain is one or the other.
        let defs: Vec<TypedVariable<MlilOf<D>>> =
            self.target.iter().chain(&self.targets).cloned().collect();
        let may_throw = self.may_throw();
        self.append(operation, uses, defs, may_throw)
    }
}

impl<D: Lift> Emitter<'_, D> {
    fn position(&self, web: usize, lane: u8) -> Result<u8> {
        let index = self.webs[web]
            .lanes
            .binary_search(&lane)
            .map_err(|_| Error::Lifting(format!("lane {lane} escaped its web")))?;
        Ok(u8::try_from(index).unwrap_or(u8::MAX))
    }

    fn typed(&self, web: usize) -> TypedVariable<MlilOf<D>> {
        TypedVariable::new(
            self.webs[web].variable,
            D::value_type(self.webs[web].shape.clone()),
        )
    }

    /// Declares one synthetic temporary web of the given shape.
    fn declare_temporary(&mut self, shape: Shape<D::Constraint>) -> Result<usize> {
        let variable = self
            .builder
            .declare_variable(D::web_role(None), None)
            .map_err(|error| Error::Lifting(error.to_string()))?;
        let index = self.webs.len();
        self.webs.push(WebInfo {
            variable,
            storage: None,
            lanes: (0..shape.lanes).collect(),
            shape,
            live_in: false,
        });
        if self.web_index.len() <= variable.index() {
            self.web_index.resize(variable.index() + 1, None);
        }
        self.web_index[variable.index()] = Some(index);
        Ok(index)
    }

    /// Rebuilds one RTL expression over web positions, consuming SSA
    /// uses in the shared deterministic read order and appending each
    /// read's web (in pre-order) to `reads` — the instruction's use
    /// list in the making.
    fn rebuild(
        &self,
        expr: &Expr<D>,
        uses: &[SsaValue<Lane<D>>],
        cursor: &mut usize,
        reads: &mut Vec<usize>,
    ) -> Result<VarExpr<D>> {
        match expr {
            Expr::Read { lanes, scalar, .. } => {
                let mut mapped: Vec<(usize, u8)> = Vec::with_capacity(lanes.len());
                for _ in lanes {
                    let value = uses
                        .get(*cursor)
                        .ok_or_else(|| Error::Lifting("SSA lost a use".into()))?;
                    *cursor += 1;
                    let web = self.resolver.web(value)?;
                    mapped.push((web, self.position(web, value.variable.1)?));
                }
                let mut parts: Vec<VarExpr<D>> = Vec::new();
                let mut part_webs: Vec<usize> = Vec::new();
                for (web, position) in mapped {
                    match (parts.last_mut(), part_webs.last()) {
                        (Some(VarExpr::Read { positions, .. }), Some(&last)) if last == web => {
                            positions.push(position);
                        }
                        _ => {
                            parts.push(VarExpr::Read {
                                positions: alloc::vec![position],
                                scalar: scalar.clone(),
                            });
                            part_webs.push(web);
                        }
                    }
                }
                reads.extend(part_webs);
                if parts.len() == 1 {
                    Ok(parts.remove(0))
                } else {
                    let shape = Shape {
                        scalar: scalar.clone(),
                        lanes: u8::try_from(lanes.len()).unwrap_or(u8::MAX),
                    };
                    Ok(VarExpr::Compose { parts, shape })
                }
            }
            Expr::Const { bits, shape } => Ok(VarExpr::Const {
                bits: bits.clone(),
                shape: shape.clone(),
            }),
            Expr::Apply {
                operator,
                operands,
                shape,
            } => {
                let operands = operands
                    .iter()
                    .map(|operand| self.rebuild(operand, uses, cursor, reads))
                    .collect::<Result<Vec<_>>>()?;
                Ok(VarExpr::Apply {
                    operator: operator.clone(),
                    operands,
                    shape: shape.clone(),
                })
            }
            Expr::Reinterpret { operand, shape } => Ok(VarExpr::Reinterpret {
                operand: Box::new(self.rebuild(operand, uses, cursor, reads)?),
                shape: shape.clone(),
            }),
        }
    }

    /// Hands one serialized statement to the dialect's emission hook.
    #[expect(clippy::too_many_arguments, reason = "one slot per statement facet")]
    fn hand_off(
        &mut self,
        source: usize,
        id: StatementId,
        statement: LiftedStatement<D>,
        reads: &[usize],
        defined: Defined<'_>,
        may_throw: bool,
        has_exceptional_successors: bool,
        spans: Vec<<D as Vocabulary>::SourceSpan>,
    ) -> Result<()> {
        if self.current != Some(id) {
            self.current = Some(id);
            self.native_defined = false;
        }
        let reads: Vec<TypedVariable<MlilOf<D>>> =
            reads.iter().map(|&web| self.typed(web)).collect();
        let (target, targets, merge) = match defined {
            Defined::Nothing => (None, Vec::new(), false),
            Defined::Target { web, merge } => (Some(self.typed(web)), Vec::new(), merge),
            Defined::Writes(webs) => (
                None,
                webs.iter().map(|&web| self.typed(web)).collect(),
                false,
            ),
        };
        let mut allowed = SmallVec::<[VariableId; 8]>::new();
        for variable in reads
            .iter()
            .chain(&target)
            .chain(&targets)
            .map(|typed| typed.variable)
        {
            if !allowed.contains(&variable) {
                allowed.push(variable);
            }
        }
        let exceptional_flow = ExceptionalFlow::from_flags(may_throw, has_exceptional_successors)?;
        let mut emission = Emission {
            emitter: self,
            source,
            statement: id,
            reads,
            target,
            targets,
            merge,
            exceptional_flow,
            spans,
            allowed,
            appended_throwing: false,
        };
        D::emit(&mut emission, statement)?;
        if emission.may_throw() && !emission.appended_throwing {
            return Err(Error::Lifting(format!(
                "the emission dropped statement {}'s exceptional behavior",
                id.raw()
            )));
        }
        Ok(())
    }

    /// Lowers one return value: a whole identity read of a web passes
    /// through, anything else materializes into a temporary so every
    /// returned value is exactly one instruction use.
    fn returnable(
        &mut self,
        source: usize,
        id: StatementId,
        value: VarExpr<D>,
        reads: &[usize],
        spans: &[<D as Vocabulary>::SourceSpan],
    ) -> Result<(VarExpr<D>, usize)> {
        if let VarExpr::Read { positions, scalar } = &value
            && let [web] = reads
        {
            let info = &self.webs[*web];
            let whole = positions.len() == usize::from(info.shape.lanes)
                && positions
                    .iter()
                    .enumerate()
                    .all(|(index, &position)| usize::from(position) == index);
            if whole && scalar == &info.shape.scalar {
                return Ok((value, *web));
            }
        }
        let shape = value.shape();
        let temporary = self.declare_temporary(shape.clone())?;
        let all: Vec<u8> = (0..shape.lanes).collect();
        self.hand_off(
            source,
            id,
            LiftedStatement::Assign {
                positions: all.clone(),
                width: shape.lanes,
                merges: false,
                value,
                effects: Vec::new(),
            },
            reads,
            Defined::Target {
                web: temporary,
                merge: false,
            },
            false,
            false,
            spans.to_vec(),
        )?;
        Ok((
            VarExpr::Read {
                positions: all,
                scalar: shape.scalar,
            },
            temporary,
        ))
    }

    /// Finds pre-state webs whose native lanes a serialized write would clobber.
    fn serialization_hazards(&self, assignments: &[PendingAssign<D>]) -> BTreeSet<usize> {
        let mut hazards = BTreeSet::new();
        for assignment in assignments {
            for &read in &assignment.reads {
                let read_info = &self.webs[read];
                let overlaps_target = assignments.iter().any(|target| {
                    let target_info = &self.webs[target.target];
                    target_info.storage.is_some()
                        && target_info.storage == read_info.storage
                        && target.positions.iter().any(|&position| {
                            target_info
                                .lanes
                                .get(usize::from(position))
                                .is_some_and(|lane| read_info.lanes.contains(lane))
                        })
                });
                if overlaps_target {
                    hazards.insert(read);
                }
            }
        }
        hazards
    }

    /// Serializes one parallel transfer: rebuild every assignment
    /// against pre-statement state, pre-copy hazarded targets, then
    /// emit one MLIL assignment per destination.
    #[expect(clippy::too_many_arguments, reason = "one slot per statement facet")]
    fn transfer(
        &mut self,
        source: usize,
        id: StatementId,
        assignments: &[(Place<D>, Expr<D>)],
        effects: &[<D as Vocabulary>::Effect],
        may_throw: bool,
        has_exceptional_successors: bool,
        spans: &[<D as Vocabulary>::SourceSpan],
        annotation: &crate::SsaInstruction<Lane<D>>,
    ) -> Result<()> {
        let mut use_cursor = 0usize;
        let mut def_cursor = 0usize;
        let mut lifted: Vec<PendingAssign<D>> = Vec::new();
        for (place, value) in assignments {
            let mut reads = Vec::new();
            let value = self.rebuild(value, &annotation.uses, &mut use_cursor, &mut reads)?;
            let target = annotation
                .defs
                .get(def_cursor)
                .ok_or_else(|| Error::Lifting("SSA lost a definition".into()))?;
            let target = self.resolver.web(target)?;
            let mut positions = Vec::with_capacity(place.lanes.len());
            for offset in 0..place.lanes.len() {
                let def = &annotation.defs[def_cursor + offset];
                positions.push(self.position(target, def.variable.1)?);
            }
            def_cursor += place.lanes.len();
            lifted.push(PendingAssign {
                target,
                positions,
                value,
                reads,
            });
        }
        // Serialize: every read whose native lanes overlap a target keeps
        // its pre-state through a synthetic copy. Compare native storage,
        // not web identity: a straight-line definition starts a fresh web,
        // while its sibling still reads the older web in the same location.
        if lifted.len() > 1 {
            for hazard in self.serialization_hazards(&lifted) {
                let shape = self.webs[hazard].shape.clone();
                let temporary = self.declare_temporary(shape.clone())?;
                let all: Vec<u8> = (0..shape.lanes).collect();
                self.hand_off(
                    source,
                    id,
                    LiftedStatement::Assign {
                        positions: all.clone(),
                        width: shape.lanes,
                        merges: false,
                        value: VarExpr::Read {
                            positions: all,
                            scalar: shape.scalar,
                        },
                        effects: Vec::new(),
                    },
                    &[hazard],
                    Defined::Target {
                        web: temporary,
                        merge: false,
                    },
                    false,
                    false,
                    spans.to_vec(),
                )?;
                for pending in &mut lifted {
                    for web in &mut pending.reads {
                        if *web == hazard {
                            *web = temporary;
                        }
                    }
                }
            }
        }
        let mut first = true;
        for pending in lifted {
            let width = self.webs[pending.target].shape.lanes;
            let merges = pending.positions.len() < usize::from(width);
            let statement_effects = if first { effects.to_vec() } else { Vec::new() };
            let throws = first && may_throw;
            let exceptional = first && has_exceptional_successors;
            first = false;
            self.hand_off(
                source,
                id,
                LiftedStatement::Assign {
                    positions: pending.positions,
                    width,
                    merges,
                    value: pending.value,
                    effects: statement_effects,
                },
                &pending.reads,
                Defined::Target {
                    web: pending.target,
                    merge: merges,
                },
                throws,
                exceptional,
                spans.to_vec(),
            )?;
        }
        Ok(())
    }

    #[expect(clippy::too_many_lines, reason = "one arm per statement form")]
    pub(super) fn statement(
        &mut self,
        source: usize,
        id: StatementId,
        statement: &Statement<D>,
        has_exceptional_successors: bool,
        spans: Vec<<D as Vocabulary>::SourceSpan>,
        annotation: &crate::SsaInstruction<Lane<D>>,
    ) -> Result<()> {
        match statement {
            Statement::Transfer {
                assignments,
                effects,
                may_throw,
            } => self.transfer(
                source,
                id,
                assignments,
                effects,
                *may_throw,
                has_exceptional_successors,
                &spans,
                annotation,
            ),
            Statement::Effect {
                operation,
                operands,
                writes,
                effects,
                may_throw,
            } => {
                let mut cursor = 0usize;
                let mut reads = Vec::new();
                let operands = operands
                    .iter()
                    .map(|operand| self.rebuild(operand, &annotation.uses, &mut cursor, &mut reads))
                    .collect::<Result<Vec<_>>>()?;
                // One web per written place: phase 1 united the lanes a
                // place writes together, so its first definition names
                // the whole web.
                let mut written = Vec::with_capacity(writes.len());
                let mut def_cursor = 0usize;
                for place in writes {
                    let value = annotation
                        .defs
                        .get(def_cursor)
                        .ok_or_else(|| Error::Lifting("SSA lost a definition".into()))?;
                    written.push(self.resolver.web(value)?);
                    def_cursor += place.lanes.len();
                }
                self.hand_off(
                    source,
                    id,
                    LiftedStatement::Effect {
                        operation: operation.clone(),
                        operands,
                        effects: effects.clone(),
                    },
                    &reads,
                    Defined::Writes(&written),
                    *may_throw,
                    has_exceptional_successors,
                    spans,
                )
            }
            Statement::Branch { condition } => {
                let mut cursor = 0usize;
                let mut reads = Vec::new();
                let condition =
                    self.rebuild(condition, &annotation.uses, &mut cursor, &mut reads)?;
                self.hand_off(
                    source,
                    id,
                    LiftedStatement::Branch { condition },
                    &reads,
                    Defined::Nothing,
                    false,
                    false,
                    spans,
                )
            }
            Statement::Dispatch { scrutinee } => {
                let mut cursor = 0usize;
                let mut reads = Vec::new();
                let scrutinee =
                    self.rebuild(scrutinee, &annotation.uses, &mut cursor, &mut reads)?;
                self.hand_off(
                    source,
                    id,
                    LiftedStatement::Dispatch { scrutinee },
                    &reads,
                    Defined::Nothing,
                    false,
                    false,
                    spans,
                )
            }
            Statement::Return { values } => {
                let mut cursor = 0usize;
                let mut lowered = Vec::with_capacity(values.len());
                let mut return_reads = Vec::with_capacity(values.len());
                for value in values {
                    let mut reads = Vec::new();
                    let rebuilt = self.rebuild(value, &annotation.uses, &mut cursor, &mut reads)?;
                    let (value, web) = self.returnable(source, id, rebuilt, &reads, &spans)?;
                    lowered.push(value);
                    return_reads.push(web);
                }
                self.hand_off(
                    source,
                    id,
                    LiftedStatement::Return { values: lowered },
                    &return_reads,
                    Defined::Nothing,
                    false,
                    false,
                    spans,
                )
            }
            Statement::Raise {
                operation,
                operands,
                effects,
            } => {
                let mut cursor = 0usize;
                let mut reads = Vec::new();
                let operands = operands
                    .iter()
                    .map(|operand| self.rebuild(operand, &annotation.uses, &mut cursor, &mut reads))
                    .collect::<Result<Vec<_>>>()?;
                self.hand_off(
                    source,
                    id,
                    LiftedStatement::Raise {
                        operation: operation.clone(),
                        operands,
                        effects: effects.clone(),
                    },
                    &reads,
                    Defined::Nothing,
                    true,
                    has_exceptional_successors,
                    spans,
                )
            }
        }
    }
}
