//! The one algorithm that turns the statements of an instruction into its
//! effect.

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::graph::view::DenseId;
use crate::ir::rtl::{
    Action, BitSlice, Bits, Control, Dialect, RecordView, Selection, Statement, WriteTarget,
};

use super::defs::{Def, Defs};
use super::dialect::DialectEffects;
use super::exprs::{ExprId, Exprs, Opaque};
use super::vocabulary::{
    Access, Address, Effect, Exit, RaiseOrigin, Raised, StateTouch, Touch, Write, WriteKind,
};

/// The effect that the [`Summarizer`] gives for dialect `D` with the effect
/// knowledge `C`: places are roots, locations are addresses in the address
/// spaces of `C`, and outcomes are the outcomes of `D`.
pub type RtlEffect<D, C> =
    Effect<<D as Dialect>::Root, Address<<C as DialectEffects<D>>::Space>, <D as Dialect>::Outcome>;

/// Turns the statements of one instruction into its [`Effect`].
///
/// One set of rules applies to every source of statements:
///
/// - **Reads.** Every read observes the state before the instruction, so
///   every read is upward-exposed. A direct read touches its place. An
///   indexed read touches every candidate, and a sparse read touches its bank.
///   The effect lists each root once, with the union of its read bits.
/// - **Writes.** One span of a [`WriteTarget::Root`] write is
///   [`WriteKind::Always`] when it takes an input value, and
///   [`WriteKind::Conditional`] when that value is a select (see
///   [`DialectEffects::select`]) whose `otherwise` operand reads the same
///   bits before the instruction. A span that takes a nondeterministic choice,
///   either through [`Action::Nondeterministic`] or as the value of a `nondet`
///   statement, is [`WriteKind::Undefined`]. An [`Action::PreservePreState`]
///   span is no write. Neighbor spans of one write with the same kind form
///   one [`Write`]. Every candidate of an indexed write, and the bank of a
///   sparse write, is [`WriteKind::Indexed`].
/// - **Memory.** [`DialectEffects::access`] classifies each effect statement.
///   An effect statement that is not an access is an intrinsic, and
///   [`DialectEffects::intrinsic_state`] gives the state that it can change.
/// - **Raises.** Each raise statement is a [`RaiseOrigin::Raise`] with its
///   guard. Each outcome of an effect statement is a
///   [`RaiseOrigin::Environment`], with the guard of a conditional access.
///   An outcome belongs to the instruction check when the check effect
///   declares it, or when its guard depends on a result of the check effect.
/// - **Exit.** The kind of the control boundary.
///
/// The expressions of the guards and the addresses are added to
/// [`Effect::exprs`] in the order that the rules need them, and a value that
/// two rules need has one expression. So two sources that give the same
/// statements give equal effects, whatever their value handles are.
#[derive(Debug)]
pub struct Summarizer<'c, D, C> {
    dialect: &'c C,
    marker: PhantomData<fn(D)>,
}

impl<D, C> Clone for Summarizer<'_, D, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D, C> Copy for Summarizer<'_, D, C> {}

/// The data of the summarizer for one value: the memoized expression and
/// whether the value depends on the instruction check.
#[derive(Debug, Clone, Copy)]
struct Extra {
    expr: Option<ExprId>,
    check: bool,
}

impl<'c, D: Dialect, C: DialectEffects<D>> Summarizer<'c, D, C> {
    /// Creates a summarizer that classifies effects through `dialect`.
    #[must_use]
    pub const fn new(dialect: &'c C) -> Self {
        Self {
            dialect,
            marker: PhantomData,
        }
    }

    /// Summarizes one sealed record.
    ///
    /// This is [`Self::summarize`] over the statements, the control, and the
    /// widths of `record`.
    ///
    /// # Panics
    ///
    /// Panics as [`Self::summarize`] does.
    #[must_use]
    pub fn summarize_record<R: RecordView<D>>(&self, record: &R) -> RtlEffect<D, C> {
        self.summarize(record.statements(), record.control(), |value| {
            record.width(value)
        })
    }

    /// Summarizes the statements of one instruction and its control boundary.
    ///
    /// `statements` yields the statements in the order that they were built.
    /// `width` gives the type of each value that the statements define.
    ///
    /// # Panics
    ///
    /// Panics when an operand is not a value that an earlier statement
    /// defines, or when [`DialectEffects::access`] or
    /// [`DialectEffects::select`] names an operand position past the
    /// operands. A sealed record meets these conditions.
    #[must_use]
    pub fn summarize<'a, V, I>(
        &self,
        statements: I,
        control: Control<V>,
        width: impl Fn(V) -> Bits,
    ) -> RtlEffect<D, C>
    where
        D: 'a,
        V: DenseId + 'a,
        I: IntoIterator<Item = Statement<'a, D, V>>,
    {
        let mut run = Run {
            dialect: self.dialect,
            defs: Defs::new(),
            effect: Effect {
                reads: Vec::new(),
                writes: Vec::new(),
                memory: Vec::new(),
                raises: Vec::new(),
                exit: Exit::from(control),
                nondeterministic: false,
                intrinsic_state: StateTouch::Preserved,
                instruction_check: false,
                exprs: Exprs::new(),
            },
            widths: Vec::new(),
            stack: Vec::new(),
            operands: Vec::new(),
        };
        for statement in statements {
            run.statement(&statement, &width);
        }
        run.finish()
    }
}

/// The state of one summary.
struct Run<'a, 'c, D: Dialect, C: DialectEffects<D>, V> {
    dialect: &'c C,
    defs: Defs<'a, D, V, Extra>,
    effect: RtlEffect<D, C>,
    /// Scratch for the operand and result widths of one effect.
    widths: Vec<Bits>,
    /// Scratch for the walk that builds one expression.
    stack: Vec<(V, bool)>,
    /// Scratch for the operands of one operator expression.
    operands: Vec<ExprId>,
}

impl<'a, D: Dialect, C: DialectEffects<D>, V: DenseId> Run<'a, '_, D, C, V> {
    fn statement(&mut self, statement: &Statement<'a, D, V>, width: &impl Fn(V) -> Bits) {
        let check = match *statement {
            Statement::Effect { operation, .. } => self.dialect.is_instruction_check(operation),
            _ => false,
        };
        let tainted = check || self.depends_on_check(statement);
        self.defs.define(statement, width, |_, _| Extra {
            expr: None,
            check: tainted,
        });
        match *statement {
            Statement::Read { source, .. } => self.read(source),
            Statement::Nondet { .. } => self.effect.nondeterministic = true,
            Statement::Effect {
                results,
                operation,
                args,
                outcomes,
            } => {
                self.widths.clear();
                self.widths
                    .extend(args.iter().map(|&arg| self.defs.slot(arg).width));
                let operands = self.widths.len();
                self.widths.extend(results.iter().map(width));
                let (arg_widths, result_widths) = self.widths.split_at(operands);
                let shape = self.dialect.access(operation, arg_widths, result_widths);
                let mut guard = None;
                if let Some(shape) = shape {
                    let address = self.expr(args[shape.address]);
                    guard = shape.guard.map(|position| self.expr(args[position]));
                    self.effect.memory.push(Access {
                        kind: shape.kind,
                        atomicity: shape.atomicity,
                        location: Address {
                            space: shape.space,
                            address,
                        },
                        bytes: shape.bytes,
                        guard,
                    });
                } else {
                    let touch = self.dialect.intrinsic_state(operation);
                    self.effect.intrinsic_state = self.effect.intrinsic_state.join(touch);
                }
                self.effect.instruction_check |= check;
                for &outcome in outcomes {
                    self.effect.raises.push(Raised {
                        outcome,
                        guard,
                        origin: RaiseOrigin::Environment,
                        check,
                    });
                }
            }
            Statement::Raise(raise) => {
                let check = raise
                    .guard
                    .is_some_and(|guard| self.defs.slot(guard).extra.check);
                let guard = raise.guard.map(|guard| self.expr(guard));
                self.effect.raises.push(Raised {
                    outcome: raise.outcome,
                    guard,
                    origin: RaiseOrigin::Raise,
                    check,
                });
            }
            Statement::Write { target, .. } => self.write(target),
            Statement::Constant { .. } | Statement::Apply { .. } => {}
        }
    }

    /// Returns whether the results of the operator `statement` depend on a
    /// result of the instruction check.
    fn depends_on_check(&self, statement: &Statement<'a, D, V>) -> bool {
        match *statement {
            Statement::Apply { args, .. } => {
                args.iter().any(|&arg| self.defs.slot(arg).extra.check)
            }
            _ => false,
        }
    }

    fn read(&mut self, source: Selection<'a, D, V>) {
        match source {
            Selection::Direct(place) => self.effect.reads.push(Touch {
                place: place.root,
                bits: place.slice,
            }),
            Selection::FiniteIndexed { candidates, .. } => {
                self.effect
                    .reads
                    .extend(candidates.iter().map(|place| Touch {
                        place: place.root,
                        bits: place.slice,
                    }));
            }
            Selection::SparseIndexed { bank, slice, .. } => self.effect.reads.push(Touch {
                place: bank,
                bits: slice,
            }),
        }
    }

    fn write(&mut self, target: WriteTarget<'a, D, V>) {
        match target {
            WriteTarget::Root { root, plan } => {
                let first = self.effect.writes.len();
                for span in plan {
                    let kind = match span.action {
                        Action::PreservePreState => continue,
                        Action::Nondeterministic(_) => {
                            self.effect.nondeterministic = true;
                            WriteKind::Undefined
                        }
                        Action::TakeInput { value, offset } => {
                            self.input_kind(root, span.span, value, offset)
                        }
                    };
                    self.push_write(first, root, span.span, kind);
                }
            }
            WriteTarget::FiniteIndexed { candidates, .. } => {
                self.effect
                    .writes
                    .extend(candidates.iter().map(|place| Write {
                        place: place.root,
                        bits: place.slice,
                        kind: WriteKind::Indexed,
                    }));
            }
            WriteTarget::SparseIndexed { bank, slice, .. } => self.effect.writes.push(Write {
                place: bank,
                bits: slice,
                kind: WriteKind::Indexed,
            }),
        }
    }

    /// Returns the kind of a span that takes bits of `value` from `offset`.
    fn input_kind(&mut self, root: D::Root, span: BitSlice, value: V, offset: u16) -> WriteKind {
        if matches!(self.defs.slot(value).def, Def::Nondet) {
            return WriteKind::Undefined;
        }
        match self
            .defs
            .conditional(self.dialect, root, span.offset, value, offset)
        {
            Some((guard, _)) => WriteKind::Conditional(self.expr(guard)),
            None => WriteKind::Always,
        }
    }

    /// Appends a write, or widens the last write of the same target when it
    /// ends where `span` starts and has the same kind.
    fn push_write(&mut self, first: usize, root: D::Root, span: BitSlice, kind: WriteKind) {
        if let Some(last) = self.effect.writes[first..].last_mut()
            && last.kind == kind
            && last.bits.end() == u32::from(span.offset)
        {
            last.bits.width = add_widths(last.bits.width, span.width);
            return;
        }
        self.effect.writes.push(Write {
            place: root,
            bits: span,
            kind,
        });
    }

    /// Returns the expression of `value`, and adds it and the expressions of
    /// its operands to the arena when they are not there yet.
    fn expr(&mut self, value: V) -> ExprId {
        if let Some(expr) = self.defs.slot(value).extra.expr {
            return expr;
        }
        let mut stack = core::mem::take(&mut self.stack);
        stack.push((value, false));
        while let Some((value, expanded)) = stack.pop() {
            let slot = *self.defs.slot(value);
            if slot.extra.expr.is_some() {
                continue;
            }
            let expr = match slot.def {
                Def::Apply { args, .. } if !expanded => {
                    stack.push((value, true));
                    stack.extend(
                        args.iter()
                            .rev()
                            .filter(|&&arg| self.defs.slot(arg).extra.expr.is_none())
                            .map(|&arg| (arg, false)),
                    );
                    continue;
                }
                Def::Apply {
                    operator,
                    args,
                    result,
                } => {
                    let name = self.dialect.operator_name(operator);
                    self.operands.clear();
                    self.operands.extend(args.iter().map(|&arg| {
                        self.defs
                            .slot(arg)
                            .extra
                            .expr
                            .expect("the walk adds every operand before its user")
                    }));
                    self.effect
                        .exprs
                        .op(name, &self.operands, result, slot.width)
                }
                Def::Read(Selection::Direct(place)) => self.effect.exprs.read(Touch {
                    place: place.root,
                    bits: place.slice,
                }),
                Def::Read(_) => self.effect.exprs.opaque(Opaque::IndexedRead, slot.width),
                Def::Const { words } => self.effect.exprs.constant(words, slot.width),
                Def::Nondet => self.effect.exprs.opaque(Opaque::Nondet, slot.width),
                Def::Effect { ordinal, result } => self
                    .effect
                    .exprs
                    .opaque(Opaque::Effect { ordinal, result }, slot.width),
            };
            self.defs.slot_mut(value).extra.expr = Some(expr);
        }
        self.stack = stack;
        self.defs
            .slot(value)
            .extra
            .expr
            .expect("the walk adds the requested value")
    }

    fn finish(mut self) -> RtlEffect<D, C> {
        self.effect.reads = coalesce(self.effect.reads);
        self.effect
    }
}

/// Returns the sum of two widths whose bits lie in one root.
fn add_widths(left: Bits, right: Bits) -> Bits {
    Bits::new(left.width() + right.width()).expect("two neighbor spans of one root fit in the root")
}

/// Sorts the touches by place and offset, and merges the touches of one
/// place that overlap or meet.
fn coalesce<S: Copy + Ord>(mut reads: Vec<Touch<S>>) -> Vec<Touch<S>> {
    reads.sort_unstable();
    let mut merged: Vec<Touch<S>> = Vec::with_capacity(reads.len());
    for touch in reads {
        if let Some(last) = merged.last_mut()
            && last.place == touch.place
            && u32::from(touch.bits.offset) <= last.bits.end()
        {
            let end = last.bits.end().max(touch.bits.end());
            let width = u16::try_from(end - u32::from(last.bits.offset))
                .ok()
                .and_then(Bits::new)
                .expect("the merged bits lie in one root");
            last.bits.width = width;
            continue;
        }
        merged.push(touch);
    }
    merged
}
