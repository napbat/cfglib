//! The one algorithm that turns the statements of an instruction into its
//! effect.

extern crate alloc;

use alloc::vec::Vec;

use crate::graph::view::DenseId;
use crate::ir::rtl::{
    Action, BitSlice, Bits, Dialect, Place, RecordView, Selection, Statement, WriteTarget,
};

use super::defs::{Def, Defs};
use super::dialect::DialectEffects;
use super::exprs::{ExprId, Exprs, Opaque};
use super::vocabulary::{
    Access, Effect, MemoryAddress, RaiseOrigin, Raised, StateTouch, Write, WriteKind,
};

/// The effect that [`Effect::compute`] gives for dialect `D` with the effect
/// knowledge `C`: places are bits of roots, locations are addresses in the
/// address spaces of `C`, and outcomes are the outcomes of `D`.
pub type RtlEffect<D, C> = Effect<
    <D as Dialect>::Root,
    MemoryAddress<<C as DialectEffects<D>>::Space>,
    <D as Dialect>::Outcome,
>;

impl<S, P, O> Effect<S, MemoryAddress<P>, O> {
    /// Computes the effect of one sealed record.
    ///
    /// One set of rules applies to every source of statements:
    ///
    /// - **Reads.** Every read observes the state before the instruction, so
    ///   every read is upward-exposed. A direct read touches its place. An
    ///   indexed read touches every candidate, and a sparse read touches its
    ///   bank. The effect lists each root once, with the union of its read
    ///   bits.
    /// - **Writes.** One span of a [`WriteTarget::Root`] write is
    ///   [`WriteKind::Always`] when it takes an input value, and
    ///   [`WriteKind::Conditional`] when that value is a select (see
    ///   [`DialectEffects::select`]) whose `otherwise` operand reads the same
    ///   bits before the instruction. A span that takes a nondeterministic
    ///   choice, either through [`Action::Nondeterministic`] or as the value
    ///   of a `nondet` statement, is [`WriteKind::Undefined`]. An
    ///   [`Action::PreservePreState`] span is no write. Neighbor spans of one
    ///   write with the same kind form one [`Write`]. Every candidate of an
    ///   indexed write, and the bank of a sparse write, is
    ///   [`WriteKind::Indexed`].
    /// - **Memory.** [`DialectEffects::access`] classifies each effect
    ///   statement. An effect statement that is not an access is an
    ///   intrinsic, and [`DialectEffects::intrinsic_state`] gives the state
    ///   that it can change.
    /// - **Raises.** Each raise statement is a [`RaiseOrigin::Raise`] with
    ///   its guard. Each outcome of an effect statement is a
    ///   [`RaiseOrigin::Environment`] that names the effect, with the guard
    ///   of a conditional access.
    /// - **Exit.** The control boundary of the record without its values.
    ///
    /// The expressions of the guards and the addresses are added to
    /// [`Effect::exprs`] in the order that the rules need them, and a value
    /// that two rules need has one expression. So two sources that give the
    /// same statements give equal effects, whatever their value handles are.
    ///
    /// # Panics
    ///
    /// Panics when an operand is not a value that an earlier statement
    /// defines, or when [`DialectEffects::access`] or
    /// [`DialectEffects::select`] names an operand position past the
    /// operands. A sealed record meets these conditions.
    #[must_use]
    pub fn compute<D, C, R>(dialect: &C, record: &R) -> Self
    where
        D: Dialect<Root = S, Outcome = O>,
        C: DialectEffects<D, Space = P>,
        R: RecordView<D>,
    {
        let mut run = Run {
            dialect,
            defs: Defs::new(),
            effect: Effect {
                reads: Vec::new(),
                writes: Vec::new(),
                memory: Vec::new(),
                raises: Vec::new(),
                exit: record.control().map(|_| ()),
                nondeterministic: false,
                intrinsic_state: StateTouch::Preserved,
                exprs: Exprs::new(),
            },
            widths: Vec::new(),
            stack: Vec::new(),
            operands: Vec::new(),
        };
        let width = |value| record.width(value);
        for statement in record.statements() {
            run.statement(&statement, &width);
        }
        run.finish()
    }
}

/// The state of one summary. The data of a value is its memoized
/// expression.
struct Run<'a, 'c, D: Dialect, C: DialectEffects<D>, V> {
    dialect: &'c C,
    defs: Defs<'a, D, V, Option<ExprId>>,
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
        self.defs.define(statement, width, None);
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
                        mode: shape.mode,
                        location: MemoryAddress {
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
                let effect = self.defs.last_effect();
                for &outcome in outcomes {
                    self.effect.raises.push(Raised {
                        outcome,
                        guard,
                        origin: RaiseOrigin::Environment { effect },
                    });
                }
            }
            Statement::Raise(raise) => {
                let guard = raise.guard.map(|guard| self.expr(guard));
                self.effect.raises.push(Raised {
                    outcome: raise.outcome,
                    guard,
                    origin: RaiseOrigin::Raise,
                });
            }
            Statement::Write { target, .. } => self.write(target),
            Statement::Constant { .. } | Statement::Apply { .. } => {}
        }
    }

    fn read(&mut self, source: Selection<'a, D, V>) {
        match source {
            Selection::Direct(place) => self.effect.reads.push(place),
            Selection::FiniteIndexed { candidates, .. } => {
                self.effect.reads.extend_from_slice(candidates);
            }
            Selection::SparseIndexed { bank, slice, .. } => {
                self.effect.reads.push(Place { root: bank, slice });
            }
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
                    .extend(candidates.iter().map(|&place| Write {
                        place,
                        kind: WriteKind::Indexed,
                    }));
            }
            WriteTarget::SparseIndexed { bank, slice, .. } => self.effect.writes.push(Write {
                place: Place { root: bank, slice },
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
            && last.place.slice.end() == u32::from(span.offset)
        {
            last.place.slice.width = add_widths(last.place.slice.width, span.width);
            return;
        }
        self.effect.writes.push(Write {
            place: Place { root, slice: span },
            kind,
        });
    }

    /// Returns the expression of `value`, and adds it and the expressions of
    /// its operands to the arena when they are not there yet.
    fn expr(&mut self, value: V) -> ExprId {
        if let Some(expr) = self.defs.slot(value).extra {
            return expr;
        }
        let mut stack = core::mem::take(&mut self.stack);
        stack.push((value, false));
        while let Some((value, expanded)) = stack.pop() {
            let slot = *self.defs.slot(value);
            if slot.extra.is_some() {
                continue;
            }
            let expr = match slot.def {
                Def::Apply { args, .. } if !expanded => {
                    stack.push((value, true));
                    stack.extend(
                        args.iter()
                            .rev()
                            .filter(|&&arg| self.defs.slot(arg).extra.is_none())
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
                            .expect("the walk adds every operand before its user")
                    }));
                    self.effect
                        .exprs
                        .op(name, &self.operands, result, slot.width)
                }
                Def::Read(Selection::Direct(place)) => self.effect.exprs.read(place),
                Def::Read(_) => self.effect.exprs.opaque(Opaque::IndexedRead, slot.width),
                Def::Const { words } => self.effect.exprs.constant(words, slot.width),
                Def::Nondet => self.effect.exprs.opaque(Opaque::Nondet, slot.width),
                Def::Effect { ordinal, result } => self
                    .effect
                    .exprs
                    .opaque(Opaque::Effect { ordinal, result }, slot.width),
            };
            self.defs.slot_mut(value).extra = Some(expr);
        }
        self.stack = stack;
        self.defs
            .slot(value)
            .extra
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

/// Sorts the places by root and offset, and merges the places of one root
/// that overlap or meet.
fn coalesce<S: Copy + Ord>(mut reads: Vec<Place<S>>) -> Vec<Place<S>> {
    reads.sort_unstable();
    let mut merged: Vec<Place<S>> = Vec::with_capacity(reads.len());
    for place in reads {
        if let Some(last) = merged.last_mut()
            && last.root == place.root
            && u32::from(place.slice.offset) <= last.slice.end()
        {
            let end = last.slice.end().max(place.slice.end());
            let width = u16::try_from(end - u32::from(last.slice.offset))
                .ok()
                .and_then(Bits::new)
                .expect("the merged bits lie in one root");
            last.slice.width = width;
            continue;
        }
        merged.push(place);
    }
    merged
}
