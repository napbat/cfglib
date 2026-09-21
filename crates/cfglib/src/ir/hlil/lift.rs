//! MLIL → HLIL lifting: control-flow structuring plus expression recovery.
//!
//! [`lift_function`] structures an MLIL function's control flow through
//! [`ir::ast`](crate::ir::ast), then translates each flat instruction list
//! into statements and expression trees: single-use pure definitions inline
//! into their consumer, effectful definitions inline only when evaluation
//! order is provably preserved, and everything else materializes as an
//! assignment. The consumer's [`LiftDialect`] supplies the per-operation
//! translation; the library owns ordering, structuring, and provenance.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::borrow::Borrow;

use crate::block::BlockId;
use crate::dataflow::liveness::Liveness;
use crate::ir::ast::{AstNode, LiftReport, LoopKind};
use crate::ir::mlil;
use crate::region;

mod block;

use self::block::{Expect, ListEnd};
use super::{
    Dialect, EntityId, Error, Expression, ExpressionId, ExpressionKind, Function, FunctionBuilder,
    Handler, HandlerKind, RecoverDialect, Result, Signature, StatementId, StatementKind,
    VariableId, VerifyDialect, recover_structure,
};

/// The HLIL translation of one MLIL operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lifted<Operation> {
    /// The instruction applies `operation` to its lifted operands; with one
    /// definition it is a value, with none an effect statement.
    Operation(Operation),
    /// The instruction stores its **last** use into the place formed by
    /// `location` over the preceding uses (`store(addr, v)` becomes
    /// `location(addr) = v`).
    Store {
        /// The lvalue operation forming the written place.
        location: Operation,
    },
    /// A pairwise parallel move: each use's value moves to the definition
    /// at the same position, with every read observed before any write
    /// (stack `dup`/`swap` families, phi-copy commits). Requires
    /// [`LiftDialect::temporary_role`] when the moves overlap.
    ParallelCopy,
    /// The block's conditional branch; the first use is the condition.
    Branch,
    /// The block's conditional branch deciding on `operation` applied to
    /// **all** of its uses — a fused compare-and-branch whose condition
    /// only exists as an expression.
    BranchOperation(Operation),
    /// The block's multi-way dispatch; the first use is the scrutinee.
    Switch,
    /// Function return; the uses are the returned values.
    Return,
    /// Pure control transfer with no HLIL form (jumps, fallthrough
    /// helpers, no-ops).
    ControlFlow,
}

/// Metadata retained while lifting MLIL into HLIL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LiftMetadata {
    /// Preserve MLIL-instruction correspondence and source provenance.
    #[default]
    Preserve,
    /// Omit correspondence and provenance when a consumer needs only HLIL semantics.
    Omit,
}

/// The bridge between one consumer's MLIL and HLIL dialects.
///
/// Implemented on the same type as both level dialects; the shared
/// [`Vocabulary`](crate::ir::dialect::Vocabulary) supertrait already
/// equates value types, effects, variables, and source coordinates, and
/// this trait's supertrait bound equates the constant domains.
pub trait LiftDialect:
    mlil::AnalysisDialect + Dialect<Constant = <Self as mlil::AnalysisDialect>::Constant>
{
    /// Translates one MLIL operation.
    ///
    /// [`AnalysisDialect::constant`](mlil::AnalysisDialect::constant) and
    /// [`AnalysisDialect::is_copy`](mlil::AnalysisDialect::is_copy) are
    /// consulted first, so constants and copies never reach this hook.
    fn lift_operation(
        operation: &<Self as mlil::Dialect>::Operation,
    ) -> Lifted<<Self as Dialect>::Operation>;

    /// The case values carried by one switch-dispatch edge.
    ///
    /// Lifting a switch requires every case edge to yield at least one
    /// value; the explicit default edge is never queried.
    fn case_values(edge: &<Self as mlil::Dialect>::Edge) -> Vec<<Self as Dialect>::Constant>;

    /// The value type of an expression evaluated only for its effects, and
    /// of condition expressions synthesized by
    /// [`Lifted::BranchOperation`].
    fn void_type() -> Self::ValueType;

    /// The role of temporaries staging overlapping
    /// [`Lifted::ParallelCopy`] moves. `None` rejects overlapping moves.
    #[must_use]
    fn temporary_role() -> Option<Self::VariableRole> {
        None
    }

    /// An operation computing the logical negation of its single operand,
    /// used to state exit-on-true loop conditions structurally. `None`
    /// degrades those loops to `loop { …; if (c) break; }`.
    #[must_use]
    fn logical_not() -> Option<<Self as Dialect>::Operation> {
        None
    }

    /// The use position that reads the destination's previous value,
    /// for operations that define only part of their storage — masked
    /// vector lanes, sub-registers — under a read-modify-write
    /// convention. Declaring it makes the merge visible to the lift:
    /// the definition feeding that position never inlines into it (the
    /// operand stays a plain variable reference), and the operation's
    /// own definition never inlines forward (its value is a merge with
    /// prior state, not a pure expression). `None` — the default —
    /// declares a whole-storage definition.
    #[must_use]
    fn previous_value_operand(_operation: &<Self as mlil::Dialect>::Operation) -> Option<usize> {
        None
    }

    /// Tests whether one definition must remain a visible statement.
    ///
    /// Return `true` when moving the value into its consumer would lose a
    /// semantic relation that the expression tree cannot state. The default
    /// permits normal single-use inlining.
    #[must_use]
    fn materialize_value(_operation: &<Self as mlil::Dialect>::Operation) -> bool {
        false
    }

    /// The exact negation of one operation applied to the same operands —
    /// a comparison with its relation inverted. Consulted before
    /// [`logical_not`](Self::logical_not) when a loop condition needs the
    /// opposite polarity, producing `while (a < b)` instead of
    /// `while (!(a >= b))`.
    #[must_use]
    fn negate_operation(
        operation: &<Self as Dialect>::Operation,
    ) -> Option<<Self as Dialect>::Operation> {
        let _ = operation;
        None
    }

    /// Whether moving one computation's evaluation across another preserves
    /// observable behavior.
    ///
    /// Consulted when expression inlining would carry an effectful or
    /// throwing definition past an intervening effectful or throwing
    /// instruction: `moved` describes the definition being evaluated later
    /// than written, `crossed` the instruction it moves across. The default
    /// refuses every pair, keeping all effectful evaluations in program
    /// order; dialects typically allow read-read pairs (two memory loads,
    /// two field reads) so both fold into one expression.
    #[must_use]
    fn evaluation_commutes(
        moved_effects: &[Self::Effect],
        moved_may_throw: bool,
        crossed_effects: &[Self::Effect],
        crossed_may_throw: bool,
    ) -> bool {
        let _ = (
            moved_effects,
            moved_may_throw,
            crossed_effects,
            crossed_may_throw,
        );
        false
    }
}

/// The result of lifting one MLIL function.
#[derive(Debug, Clone)]
pub struct LiftedFunction<D: LiftDialect> {
    /// The lifted function. Variable identities correspond one-to-one by
    /// index with the MLIL variable table.
    pub function: Function<D>,
    /// The control-flow structuring fidelity report.
    pub report: LiftReport,
    /// MLIL instruction → the HLIL entity carrying it: a statement, or an
    /// expression when the definition was inlined. Pure control-transfer
    /// instructions have no entry.
    pub instructions: BTreeMap<mlil::InstructionId, EntityId>,
}

impl<D: LiftDialect + VerifyDialect + RecoverDialect> LiftedFunction<D> {
    /// Rebuilds the lifted function with source-level structure recovered
    /// — counted loops, conditional value selection, paired enter/exit
    /// regions — remapping the instruction table onto the constructs that
    /// carry each instruction now.
    ///
    /// # Errors
    ///
    /// Returns an error when the rebuilt function fails verification.
    pub fn with_recovered_structure(self) -> Result<Self> {
        if !has_recovery_candidates(&self.function) {
            return Ok(self);
        }
        let recovery = recover_structure(&self.function)?;
        let instructions = self
            .instructions
            .into_iter()
            .filter_map(|(instruction, entity)| {
                let entity = match entity {
                    EntityId::Statement(statement) => recovery
                        .statements
                        .get(&statement)
                        .map(|&new| EntityId::Statement(new)),
                    EntityId::Expression(expression) => recovery
                        .expressions
                        .get(&expression)
                        .map(|&new| EntityId::Expression(new)),
                    EntityId::Variable(variable) => Some(EntityId::Variable(variable)),
                }?;
                Some((instruction, entity))
            })
            .collect();
        Ok(Self {
            function: recovery.function,
            report: self.report,
            instructions,
        })
    }
}

/// Lifts one MLIL function into structured, expression-oriented HLIL.
///
/// # Errors
///
/// Returns [`Error::UnsupportedLift`] when the MLIL shape has no HLIL
/// translation (multi-result instructions, conditional blocks whose
/// terminator the dialect did not classify as a branch, switch case edges
/// without values), and a verification error when the assembled function
/// violates an HLIL invariant.
pub fn lift_function<D>(source: &mlil::Function<D>) -> Result<LiftedFunction<D>>
where
    D: LiftDialect + VerifyDialect,
{
    lift_function_with_metadata(source, LiftMetadata::Preserve)
}

/// Lifts one MLIL function with an explicit metadata-retention policy.
///
/// # Errors
///
/// Returns the same structural, translation, and verification failures as
/// [`lift_function`].
pub fn lift_function_with_metadata<D>(
    source: &mlil::Function<D>,
    metadata: LiftMetadata,
) -> Result<LiftedFunction<D>>
where
    D: LiftDialect + VerifyDialect,
{
    if let Some(cleared) = trampolines_cleared(source) {
        let (ast, report) = crate::lift_borrowed_with_report(&cleared);
        return lift_from_structure(source, &ast, report, metadata);
    }
    let (ast, report) = crate::lift_borrowed_with_report(source.cfg());
    lift_from_structure(source, &ast, report, metadata)
}

/// Lifts MLIL while reusing a structured view already computed for the same
/// function. If HLIL must clear jump trampolines, it recomputes only that
/// distinct working view so the normal lifting contract remains unchanged.
/// Instruction payloads may be owned values or borrowed references; this
/// function does not clone either representation.
///
/// # Errors
///
/// Returns the same structural, translation, and verification failures as
/// [`lift_function`].
pub fn lift_function_with_structure<D, P>(
    source: &mlil::Function<D>,
    ast: &AstNode<P>,
    report: &LiftReport,
    metadata: LiftMetadata,
) -> Result<LiftedFunction<D>>
where
    D: LiftDialect + VerifyDialect,
    P: Borrow<mlil::Instruction<D>>,
{
    if let Some(cleared) = trampolines_cleared(source) {
        let (ast, report) = crate::lift_borrowed_with_report(&cleared);
        return lift_from_structure(source, &ast, report, metadata);
    }
    lift_from_structure(source, ast, report.clone(), metadata)
}

fn lift_from_structure<D, P>(
    source: &mlil::Function<D>,
    ast: &AstNode<P>,
    report: LiftReport,
    metadata: LiftMetadata,
) -> Result<LiftedFunction<D>>
where
    D: LiftDialect + VerifyDialect,
    P: Borrow<mlil::Instruction<D>>,
{
    let mut lifter = Lifter::new(source, metadata)?;
    let body = lifter.translate_node(ast)?;
    lifter.builder.set_body(body)?;
    let function = lifter.builder.finish()?;
    Ok(LiftedFunction {
        function,
        report,
        instructions: lifter.instructions,
    })
}

fn has_recovery_candidates<D: Dialect>(function: &Function<D>) -> bool {
    function.statements().iter().any(|statement| {
        matches!(
            statement.kind(),
            StatementKind::If { .. } | StatementKind::While { .. } | StatementKind::Try { .. }
        )
    })
}

/// The lift's working view of the source CFG, with jump trampolines
/// emptied — or `None` when no block qualifies, avoiding the clone.
///
/// A block whose every instruction lifts to [`Lifted::ControlFlow`],
/// defines nothing, and cannot throw carries no semantics beyond its
/// outgoing edges: emission drops such instructions without consuming
/// their uses, so clearing them changes no lifted output. It does change
/// structuring — the [`ir::ast`](crate::ir::ast) walk forwards `break`/
/// `continue` resolutions through instruction-free single-jump blocks
/// (javac routes `break` through such trampolines), which would
/// otherwise degrade to gotos. The definition and throw guards make the
/// direct [`LiftDialect::lift_operation`] query safe here: constants and
/// copies — normally intercepted before that hook — always define.
fn trampolines_cleared<D: LiftDialect>(
    source: &mlil::Function<D>,
) -> Option<crate::Cfg<mlil::Instruction<D>, <D as mlil::Dialect>::Edge>> {
    let cfg = source.cfg();
    let doomed: Vec<BlockId> = cfg
        .block_ids()
        .filter(|&block| {
            let block = cfg.block(block);
            !block.instructions().is_empty()
                && block.instructions().iter().all(|instruction| {
                    instruction.defs().is_empty()
                        && !instruction.may_throw()
                        && matches!(
                            D::lift_operation(instruction.operation()),
                            Lifted::ControlFlow
                        )
                })
        })
        .collect();
    if doomed.is_empty() {
        return None;
    }
    let mut cleared = cfg.clone();
    for block in doomed {
        cleared.block_mut(block).instructions_mut().clear();
    }
    Some(cleared)
}

pub(super) struct Lifter<'a, D: LiftDialect> {
    pub(super) source: &'a mlil::Function<D>,
    pub(super) builder: FunctionBuilder<D>,
    pub(super) liveness: Liveness<mlil::VariableId>,
    pub(super) spans: BTreeMap<mlil::InstructionId, Vec<D::SourceSpan>>,
    pub(super) instructions: BTreeMap<mlil::InstructionId, EntityId>,
    metadata: LiftMetadata,
    contexts: Vec<Context>,
}

/// One enclosing breakable construct during statement translation.
struct Context {
    /// The loop's label when the construct is a loop; `None` for a switch.
    loop_label: Option<String>,
    /// Set when an unlabeled AST break had to name this loop explicitly.
    needs_label: bool,
}

impl<'a, D: LiftDialect + VerifyDialect> Lifter<'a, D> {
    fn new(source: &'a mlil::Function<D>, metadata: LiftMetadata) -> Result<Self> {
        let mut builder = FunctionBuilder::new(source.source().clone());
        for variable in source.variables() {
            builder.declare_variable(variable.role.clone(), variable.native.clone(), None)?;
        }
        let signature = source.signature();
        builder.set_signature(Signature::<D>::new(
            signature
                .parameters
                .iter()
                .map(|parameter| VariableId::from_raw(parameter.raw()))
                .collect(),
            signature.returns.clone(),
        ))?;
        let mut spans: BTreeMap<mlil::InstructionId, Vec<D::SourceSpan>> = BTreeMap::new();
        if metadata == LiftMetadata::Preserve {
            for entry in source.provenance().entries() {
                match entry.entity {
                    mlil::EntityId::Instruction(instruction) => {
                        spans
                            .entry(instruction)
                            .or_default()
                            .push(entry.source.clone());
                    }
                    mlil::EntityId::Variable(variable) => {
                        builder.map_entity(
                            entry.source.clone(),
                            EntityId::Variable(VariableId::from_raw(variable.raw())),
                        )?;
                    }
                    mlil::EntityId::Block(_) | mlil::EntityId::Edge(_) => {}
                }
            }
        }
        Ok(Self {
            source,
            builder,
            liveness: source.liveness(),
            spans,
            instructions: BTreeMap::new(),
            metadata,
            contexts: Vec::new(),
        })
    }

    /// Records the HLIL entity carrying one MLIL instruction, composing its
    /// source spans onto the new entity.
    pub(super) fn map_instruction(
        &mut self,
        instruction: mlil::InstructionId,
        entity: EntityId,
    ) -> Result<()> {
        if self.metadata == LiftMetadata::Omit {
            return Ok(());
        }
        self.instructions.insert(instruction, entity);
        if let Some(spans) = self.spans.get(&instruction) {
            for span in spans {
                self.builder.map_entity(span.clone(), entity)?;
            }
        }
        Ok(())
    }

    fn block_label(&self, block: BlockId) -> String {
        self.source
            .cfg()
            .block(block)
            .label()
            .map_or_else(|| format!(".bb{}", block.index()), String::from)
    }

    fn translate_nodes<P>(&mut self, nodes: &[AstNode<P>]) -> Result<Vec<StatementId>>
    where
        P: Borrow<mlil::Instruction<D>>,
    {
        let mut statements = Vec::new();
        let mut index = 0;
        while index < nodes.len() {
            let mut end = index;
            while matches!(nodes.get(end), Some(AstNode::Block { .. })) {
                end += 1;
            }
            if end == index {
                statements.extend(self.translate_node(&nodes[index])?);
                index += 1;
                continue;
            }
            // A run of straight-line blocks translates as one list — and
            // absorbs a directly following return or decision list — so
            // expression inlining sees the whole linear region instead of
            // one fragment per frontend block.
            let mut list: Vec<&mlil::Instruction<D>> = Vec::new();
            let mut final_block = None;
            for node in &nodes[index..end] {
                if let AstNode::Block { id, instructions } = node {
                    list.extend(instructions.iter().map(Borrow::borrow));
                    final_block = Some(*id);
                }
            }
            let final_block = final_block.expect("the run contains at least one block node");
            match nodes.get(end) {
                Some(AstNode::Return { id, instructions }) => {
                    list.extend(instructions.iter().map(Borrow::borrow));
                    let (result, _) = self.translate_list(*id, &list, Expect::Statements)?;
                    statements.extend(result);
                    index = end + 1;
                }
                Some(AstNode::IfThenElse {
                    condition,
                    condition_instructions,
                    then_body,
                    else_body,
                }) => {
                    list.extend(condition_instructions.iter().map(Borrow::borrow));
                    statements.extend(self.translate_if(*condition, &list, then_body, else_body)?);
                    index = end + 1;
                }
                Some(AstNode::Switch {
                    condition,
                    condition_instructions,
                    cases,
                    default_body,
                    ..
                }) => {
                    list.extend(condition_instructions.iter().map(Borrow::borrow));
                    statements.extend(self.translate_switch(
                        *condition,
                        &list,
                        cases,
                        default_body,
                    )?);
                    index = end + 1;
                }
                _ => {
                    // Liveness beyond the merged list is the last block's
                    // live-out: defs feeding a later block of the same run
                    // are still single-use inside the list.
                    let (result, _) =
                        self.translate_list(final_block, &list, Expect::Statements)?;
                    statements.extend(result);
                    index = end;
                }
            }
        }
        Ok(statements)
    }

    fn translate_node<P>(&mut self, node: &AstNode<P>) -> Result<Vec<StatementId>>
    where
        P: Borrow<mlil::Instruction<D>>,
    {
        match node {
            AstNode::Sequence { body } => self.translate_nodes(body),
            AstNode::Block { id, instructions } | AstNode::Return { id, instructions } => {
                let (statements, _) = self.translate_list(*id, instructions, Expect::Statements)?;
                Ok(statements)
            }
            AstNode::IfThenElse {
                condition,
                condition_instructions,
                then_body,
                else_body,
            } => self.translate_if(*condition, condition_instructions, then_body, else_body),
            AstNode::Loop { header, kind, body } => self.translate_loop(*header, kind, body, None),
            AstNode::Switch {
                condition,
                condition_instructions,
                cases,
                default_body,
                ..
            } => self.translate_switch(*condition, condition_instructions, cases, default_body),
            AstNode::Break { label } => {
                let label = match label {
                    Some(label) => Some(label.clone()),
                    None => self.break_label(),
                };
                Ok(vec![
                    self.builder
                        .add_statement(StatementKind::Break { label }, None)?,
                ])
            }
            AstNode::Continue { label } => Ok(vec![self.builder.add_statement(
                StatementKind::Continue {
                    label: label.clone(),
                },
                None,
            )?]),
            AstNode::Label { name, body } => {
                // A label wrapping exactly one loop names that loop, so
                // labeled breaks resolve without a second wrapper.
                if let [AstNode::Loop { header, kind, body }] = body.as_slice() {
                    return self.translate_loop(*header, kind, body, Some(name.clone()));
                }
                let statements = self.translate_nodes(body)?;
                Ok(vec![self.builder.add_statement(
                    StatementKind::Labeled {
                        label: name.clone(),
                        body: statements,
                    },
                    None,
                )?])
            }
            AstNode::Goto { target } => Ok(vec![self.builder.add_statement(
                StatementKind::Goto {
                    label: target.clone(),
                },
                None,
            )?]),
            AstNode::TryCatch {
                try_body,
                handlers,
                finally_body,
            } => self.translate_try(try_body, handlers, finally_body),
            AstNode::Guarded { .. } => Err(Error::UnsupportedLift(
                "predicated regions have no HLIL translation".into(),
            )),
        }
    }

    fn translate_if<C, P>(
        &mut self,
        condition: BlockId,
        condition_instructions: &[C],
        then_body: &[AstNode<P>],
        else_body: &[AstNode<P>],
    ) -> Result<Vec<StatementId>>
    where
        C: Borrow<mlil::Instruction<D>>,
        P: Borrow<mlil::Instruction<D>>,
    {
        let (mut statements, end) =
            self.translate_list(condition, condition_instructions, Expect::Branch)?;
        let ListEnd {
            value: condition_expression,
            instruction,
        } = end.expect("Expect::Branch guarantees a branch end");
        let then_statements = self.translate_nodes(then_body)?;
        let else_statements = self.translate_nodes(else_body)?;
        let statement = self.builder.add_statement(
            StatementKind::If {
                condition: condition_expression,
                then_body: then_statements,
                else_body: else_statements,
            },
            None,
        )?;
        self.map_instruction(instruction, EntityId::Statement(statement))?;
        statements.push(statement);
        Ok(statements)
    }

    fn translate_try<P>(
        &mut self,
        try_body: &[AstNode<P>],
        handlers: &[crate::ir::ast::CatchHandler<P>],
        finally_body: &[AstNode<P>],
    ) -> Result<Vec<StatementId>>
    where
        P: Borrow<mlil::Instruction<D>>,
    {
        let body = self.translate_nodes(try_body)?;
        let mut lifted_handlers = Vec::new();
        for handler in handlers {
            let kind = match handler.kind {
                region::HandlerKind::Catch => HandlerKind::Catch,
                region::HandlerKind::CatchAll => HandlerKind::CatchAll,
                region::HandlerKind::Fault => HandlerKind::Fault,
                // The funclet is emitted separately by the sweep; its
                // predicate has no structured body here.
                region::HandlerKind::Filter { .. } => HandlerKind::Filter {
                    filter_body: Vec::new(),
                },
                region::HandlerKind::Finally => {
                    return Err(Error::UnsupportedLift(
                        "a finally handler arm outside the finally body".into(),
                    ));
                }
            };
            lifted_handlers.push(Handler {
                kind,
                binding: None,
                caught_types: Vec::new(),
                body: self.translate_nodes(&handler.body)?,
            });
        }
        let finally_statements = self.translate_nodes(finally_body)?;
        Ok(vec![self.builder.add_statement(
            StatementKind::Try {
                body,
                handlers: lifted_handlers,
                finally_body: finally_statements,
            },
            None,
        )?])
    }

    /// The explicit label an unlabeled AST break needs: at the AST level a
    /// plain break always exits the innermost **loop**, while an unlabeled
    /// HLIL break exits the innermost loop *or switch* — so a break inside
    /// an intervening switch names its loop.
    fn break_label(&mut self) -> Option<String> {
        let mut crossed_switch = false;
        for context in self.contexts.iter_mut().rev() {
            match &context.loop_label {
                None => crossed_switch = true,
                Some(label) => {
                    if crossed_switch {
                        context.needs_label = true;
                        return Some(label.clone());
                    }
                    return None;
                }
            }
        }
        None
    }

    fn translate_switch<C, P>(
        &mut self,
        condition: BlockId,
        condition_instructions: &[C],
        cases: &[crate::ir::ast::SwitchCase<P>],
        default_body: &[AstNode<P>],
    ) -> Result<Vec<StatementId>>
    where
        C: Borrow<mlil::Instruction<D>>,
        P: Borrow<mlil::Instruction<D>>,
    {
        let (mut statements, end) =
            self.translate_list(condition, condition_instructions, Expect::Switch)?;
        let ListEnd {
            value: scrutinee,
            instruction,
        } = end.expect("Expect::Switch guarantees a dispatch end");
        self.contexts.push(Context {
            loop_label: None,
            needs_label: false,
        });
        let mut arms = Vec::new();
        for case in cases {
            let mut values = Vec::new();
            for &edge in &case.edges {
                values.extend(D::case_values(self.source.cfg().edge(edge).payload()));
            }
            if values.is_empty() {
                self.contexts.pop();
                return Err(Error::UnsupportedLift(format!(
                    "switch case at block {} has no case values",
                    case.id
                )));
            }
            let body = self.translate_nodes(&case.body)?;
            arms.push(super::SwitchArm { values, body });
        }
        let default_statements = self.translate_nodes(default_body)?;
        self.contexts.pop();
        let statement = self.builder.add_statement(
            StatementKind::Switch {
                scrutinee,
                cases: arms,
                default_body: default_statements,
            },
            None,
        )?;
        self.map_instruction(instruction, EntityId::Statement(statement))?;
        statements.push(statement);
        Ok(statements)
    }

    fn translate_loop<P>(
        &mut self,
        header: BlockId,
        kind: &LoopKind<P>,
        body: &[AstNode<P>],
        outer_label: Option<String>,
    ) -> Result<Vec<StatementId>>
    where
        P: Borrow<mlil::Instruction<D>>,
    {
        let label = outer_label
            .clone()
            .unwrap_or_else(|| self.block_label(header));
        self.contexts.push(Context {
            loop_label: Some(label.clone()),
            needs_label: false,
        });
        let lifted = self.translate_loop_kind(kind, body);
        let context = self
            .contexts
            .pop()
            .expect("the loop context pushed above is still on the stack");
        let statement = lifted?;
        if context.needs_label && outer_label.is_none() {
            return Ok(vec![self.builder.add_statement(
                StatementKind::Labeled {
                    label,
                    body: vec![statement],
                },
                None,
            )?]);
        }
        if let Some(label) = outer_label {
            return Ok(vec![self.builder.add_statement(
                StatementKind::Labeled {
                    label,
                    body: vec![statement],
                },
                None,
            )?]);
        }
        Ok(vec![statement])
    }

    fn translate_loop_kind<P>(
        &mut self,
        kind: &LoopKind<P>,
        body: &[AstNode<P>],
    ) -> Result<StatementId>
    where
        P: Borrow<mlil::Instruction<D>>,
    {
        match kind {
            LoopKind::Endless => {
                let statements = self.translate_nodes(body)?;
                Ok(self
                    .builder
                    .add_statement(StatementKind::Loop { body: statements }, None)?)
            }
            LoopKind::While {
                condition_block,
                condition,
                exit_on_true,
            } => {
                let (condition_statements, end) =
                    self.translate_list(*condition_block, condition, Expect::Branch)?;
                let ListEnd { value, instruction } =
                    end.expect("Expect::Branch guarantees a branch end");
                let body_statements = self.translate_nodes(body)?;
                if condition_statements.is_empty() {
                    if let Some(value) = self.loop_condition(value, *exit_on_true)? {
                        let statement = self.builder.add_statement(
                            StatementKind::While {
                                condition: value,
                                body: body_statements,
                            },
                            None,
                        )?;
                        self.map_instruction(instruction, EntityId::Statement(statement))?;
                        return Ok(statement);
                    }
                }
                // The condition needs statements (or an unavailable
                // negation): state the test explicitly at the loop top.
                let test = self.exit_test(value, *exit_on_true, instruction)?;
                let mut loop_body = condition_statements;
                loop_body.push(test);
                loop_body.extend(body_statements);
                Ok(self
                    .builder
                    .add_statement(StatementKind::Loop { body: loop_body }, None)?)
            }
            LoopKind::DoWhile {
                latch,
                condition,
                continue_on_true,
            } => {
                let body_statements = self.translate_nodes(body)?;
                let (condition_statements, end) =
                    self.translate_list(*latch, condition, Expect::Branch)?;
                let ListEnd { value, instruction } =
                    end.expect("Expect::Branch guarantees a branch end");
                if condition_statements.is_empty() {
                    if let Some(value) = self.loop_condition(value, !*continue_on_true)? {
                        let statement = self.builder.add_statement(
                            StatementKind::DoWhile {
                                body: body_statements,
                                condition: value,
                            },
                            None,
                        )?;
                        self.map_instruction(instruction, EntityId::Statement(statement))?;
                        return Ok(statement);
                    }
                }
                // State the post-test explicitly at the loop bottom.
                let test = self.exit_test(value, !*continue_on_true, instruction)?;
                let mut loop_body = body_statements;
                loop_body.extend(condition_statements);
                loop_body.push(test);
                Ok(self
                    .builder
                    .add_statement(StatementKind::Loop { body: loop_body }, None)?)
            }
        }
    }

    /// The loop-continuation condition, negating when the raw condition
    /// exits on true; `None` when negation is needed but unavailable.
    fn loop_condition(
        &mut self,
        condition: ExpressionId,
        exit_on_true: bool,
    ) -> Result<Option<ExpressionId>> {
        if !exit_on_true {
            return Ok(Some(condition));
        }
        // Exact inversion first: the same operands under the negated
        // operation, with no wrapper node.
        if let Some(ExpressionKind::Operation { operation, .. }) =
            self.builder.expression(condition).map(Expression::kind)
            && let Some(negated) = D::negate_operation(operation)
        {
            self.builder.replace_operation(condition, negated)?;
            return Ok(Some(condition));
        }
        let Some(negation) = D::logical_not() else {
            return Ok(None);
        };
        let value_type = self
            .builder
            .expression(condition)
            .map_or_else(D::void_type, |expression| expression.value_type().clone());
        Ok(Some(self.builder.add_expression(
            ExpressionKind::Operation {
                operation: negation,
                operands: vec![condition],
            },
            value_type,
        )?))
    }

    /// An explicit `if (…) break;` exit test with the given polarity.
    fn exit_test(
        &mut self,
        condition: ExpressionId,
        exit_on_true: bool,
        instruction: mlil::InstructionId,
    ) -> Result<StatementId> {
        let break_statement = self
            .builder
            .add_statement(StatementKind::Break { label: None }, None)?;
        let (then_body, else_body) = if exit_on_true {
            (vec![break_statement], Vec::new())
        } else {
            (Vec::new(), vec![break_statement])
        };
        let statement = self.builder.add_statement(
            StatementKind::If {
                condition,
                then_body,
                else_body,
            },
            None,
        )?;
        self.map_instruction(instruction, EntityId::Statement(statement))?;
        Ok(statement)
    }
}
