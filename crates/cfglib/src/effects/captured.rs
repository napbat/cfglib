//! The owned statements of one instruction.

extern crate alloc;

use alloc::vec::Vec;
use core::convert::Infallible;

use crate::graph::view::DenseId;
use crate::ir::rtl::{
    BitSlice, Bits, Control, Dialect, InstructionBuilder, Place, Raise, RecordView, Selection,
    SpanAction, Statement, ValueRange, WriteTarget,
};

/// A value of a [`Captured`] instruction. The values of one capture are dense
/// from zero, in the order that the statements define them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CapturedValue(u32);

impl DenseId for CapturedValue {
    fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("a capture holds at most u32::MAX values"))
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// A run of one pool of a [`Captured`] instruction.
#[derive(Debug, Clone, Copy)]
struct Run {
    start: usize,
    end: usize,
}

impl Run {
    fn push<T: Clone>(pool: &mut Vec<T>, items: &[T]) -> Self {
        let start = pool.len();
        pool.extend_from_slice(items);
        Self {
            start,
            end: pool.len(),
        }
    }

    fn get<T>(self, pool: &[T]) -> &[T] {
        &pool[self.start..self.end]
    }
}

#[derive(Debug, Clone, Copy)]
enum StoredSource<D: Dialect> {
    Direct(Place<D>),
    Finite {
        index: CapturedValue,
        candidates: Run,
    },
    Sparse {
        bank: D::Root,
        key: CapturedValue,
        slice: BitSlice,
    },
}

#[derive(Debug, Clone, Copy)]
enum StoredTarget<D: Dialect> {
    Root {
        root: D::Root,
        plan: Run,
    },
    Finite {
        index: CapturedValue,
        candidates: Run,
        value: CapturedValue,
    },
    Sparse {
        bank: D::Root,
        key: CapturedValue,
        slice: BitSlice,
        value: CapturedValue,
    },
}

/// One stored statement. Its slices are runs of the pools of the capture.
#[derive(Debug, Clone, Copy)]
enum Stored<D: Dialect> {
    Read {
        result: CapturedValue,
        source: StoredSource<D>,
    },
    Constant {
        result: CapturedValue,
        words: Run,
        width: Bits,
    },
    Apply {
        results: ValueRange<CapturedValue>,
        operator: usize,
        args: Run,
    },
    Nondet {
        result: CapturedValue,
        width: Bits,
        domain: D::NondetDomain,
    },
    Effect {
        results: ValueRange<CapturedValue>,
        operation: usize,
        args: Run,
        outcomes: Run,
    },
    Raise {
        outcome: D::Outcome,
        guard: Option<CapturedValue>,
        payload: Run,
        binds: Run,
    },
    Write {
        target: StoredTarget<D>,
        policy: D::WritePolicy,
    },
}

/// The statements of one instruction as owned data.
///
/// `&mut Captured` implements [`InstructionBuilder`], so instruction
/// semantics can emit into it as into any storage. The capture keeps the
/// values with their widths, the statements in the order of emission, and the
/// control boundary. It checks no rule of the builder contract, and it holds
/// one instruction only. An indexed read without a candidate has no width, so
/// the capture panics at such a read. [`Captured::clear`] empties it for the next
/// instruction and keeps its buffers.
///
/// The capture implements [`RecordView`], so the
/// [`Summarizer`](super::Summarizer) and the [`StatementTree`](super::StatementTree) read it as
/// they read a sealed record.
#[derive(Debug, Clone)]
pub struct Captured<D: Dialect> {
    widths: Vec<Bits>,
    statements: Vec<Stored<D>>,
    args: Vec<CapturedValue>,
    places: Vec<Place<D>>,
    plans: Vec<SpanAction<D, CapturedValue>>,
    words: Vec<u64>,
    operators: Vec<D::Operator>,
    operations: Vec<D::EffectOp>,
    outcomes: Vec<D::Outcome>,
    binds: Vec<(D::Root, CapturedValue)>,
    control: Option<Control<CapturedValue>>,
}

impl<D: Dialect> Default for Captured<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Dialect> Captured<D> {
    /// Creates an empty capture.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            widths: Vec::new(),
            statements: Vec::new(),
            args: Vec::new(),
            places: Vec::new(),
            plans: Vec::new(),
            words: Vec::new(),
            operators: Vec::new(),
            operations: Vec::new(),
            outcomes: Vec::new(),
            binds: Vec::new(),
            control: None,
        }
    }

    /// Removes every statement, value, and the control boundary. The buffers
    /// keep their capacity.
    pub fn clear(&mut self) {
        self.widths.clear();
        self.statements.clear();
        self.args.clear();
        self.places.clear();
        self.plans.clear();
        self.words.clear();
        self.operators.clear();
        self.operations.clear();
        self.outcomes.clear();
        self.binds.clear();
        self.control = None;
    }

    /// Returns how many values the statements define.
    #[must_use]
    pub fn value_count(&self) -> usize {
        self.widths.len()
    }

    /// Returns how many statements the capture holds.
    #[must_use]
    pub fn statement_count(&self) -> usize {
        self.statements.len()
    }

    /// Returns the control boundary, or `None` before
    /// [`finish`](InstructionBuilder::finish).
    #[must_use]
    pub fn finished_control(&self) -> Option<Control<CapturedValue>> {
        self.control
    }

    /// Iterates over the statements in the order of emission.
    pub fn statements(
        &self,
    ) -> impl ExactSizeIterator<Item = Statement<'_, D, CapturedValue>> + '_ {
        self.statements.iter().map(|stored| self.decode(stored))
    }

    /// Returns the type of `value`.
    ///
    /// # Panics
    ///
    /// Panics when `value` is not a value of this capture.
    #[must_use]
    pub fn width(&self, value: CapturedValue) -> Bits {
        self.widths[value.index()]
    }

    fn fresh(&mut self, widths: &[Bits]) -> ValueRange<CapturedValue> {
        let first = CapturedValue::from_index(self.widths.len());
        self.widths.extend_from_slice(widths);
        let len =
            u32::try_from(widths.len()).expect("one statement defines at most u32::MAX values");
        ValueRange::new(first, len)
    }

    fn decode(&self, stored: &Stored<D>) -> Statement<'_, D, CapturedValue> {
        match *stored {
            Stored::Read { result, source } => Statement::Read {
                result,
                source: match source {
                    StoredSource::Direct(place) => Selection::Direct(place),
                    StoredSource::Finite { index, candidates } => Selection::FiniteIndexed {
                        index,
                        candidates: candidates.get(&self.places),
                    },
                    StoredSource::Sparse { bank, key, slice } => {
                        Selection::SparseIndexed { bank, key, slice }
                    }
                },
            },
            Stored::Constant {
                result,
                words,
                width,
            } => Statement::Constant {
                result,
                words: words.get(&self.words),
                width,
            },
            Stored::Apply {
                results,
                operator,
                args,
            } => Statement::Apply {
                results,
                operator: &self.operators[operator],
                args: args.get(&self.args),
            },
            Stored::Nondet {
                result,
                width,
                domain,
            } => Statement::Nondet {
                result,
                width,
                domain,
            },
            Stored::Effect {
                results,
                operation,
                args,
                outcomes,
            } => Statement::Effect {
                results,
                operation: &self.operations[operation],
                args: args.get(&self.args),
                outcomes: outcomes.get(&self.outcomes),
            },
            Stored::Raise {
                outcome,
                guard,
                payload,
                binds,
            } => Statement::Raise(Raise {
                outcome,
                guard,
                payload: payload.get(&self.args),
                binds: binds.get(&self.binds),
            }),
            Stored::Write { target, policy } => Statement::Write {
                target: match target {
                    StoredTarget::Root { root, plan } => WriteTarget::Root {
                        root,
                        plan: plan.get(&self.plans),
                    },
                    StoredTarget::Finite {
                        index,
                        candidates,
                        value,
                    } => WriteTarget::FiniteIndexed {
                        index,
                        candidates: candidates.get(&self.places),
                        value,
                    },
                    StoredTarget::Sparse {
                        bank,
                        key,
                        slice,
                        value,
                    } => WriteTarget::SparseIndexed {
                        bank,
                        key,
                        slice,
                        value,
                    },
                },
                policy,
            },
        }
    }
}

impl<D: Dialect> InstructionBuilder<D> for &mut Captured<D> {
    type Value = CapturedValue;
    type Error = Infallible;

    fn read(&mut self, source: Selection<'_, D, CapturedValue>) -> CapturedValue {
        let (width, source) = match source {
            Selection::Direct(place) => (place.slice.width, StoredSource::Direct(place)),
            Selection::FiniteIndexed { index, candidates } => (
                candidates
                    .first()
                    .expect("an indexed read has a candidate, which gives its width")
                    .slice
                    .width,
                StoredSource::Finite {
                    index,
                    candidates: Run::push(&mut self.places, candidates),
                },
            ),
            Selection::SparseIndexed { bank, key, slice } => {
                (slice.width, StoredSource::Sparse { bank, key, slice })
            }
        };
        let result = self.fresh(&[width]).first;
        self.statements.push(Stored::Read { result, source });
        result
    }

    fn constant(&mut self, words: &[u64], width: Bits) -> CapturedValue {
        let result = self.fresh(&[width]).first;
        let words = Run::push(&mut self.words, words);
        self.statements.push(Stored::Constant {
            result,
            words,
            width,
        });
        result
    }

    fn apply(
        &mut self,
        operator: &D::Operator,
        args: &[CapturedValue],
        results: &[Bits],
    ) -> ValueRange<CapturedValue> {
        let range = self.fresh(results);
        let args = Run::push(&mut self.args, args);
        self.operators.push(operator.clone());
        self.statements.push(Stored::Apply {
            results: range,
            operator: self.operators.len() - 1,
            args,
        });
        range
    }

    fn nondet(&mut self, width: Bits, domain: D::NondetDomain) -> CapturedValue {
        let result = self.fresh(&[width]).first;
        self.statements.push(Stored::Nondet {
            result,
            width,
            domain,
        });
        result
    }

    fn effect(
        &mut self,
        operation: &D::EffectOp,
        args: &[CapturedValue],
        results: &[Bits],
        outcomes: &[D::Outcome],
    ) -> ValueRange<CapturedValue> {
        let range = self.fresh(results);
        let args = Run::push(&mut self.args, args);
        let outcomes = Run::push(&mut self.outcomes, outcomes);
        self.operations.push(operation.clone());
        self.statements.push(Stored::Effect {
            results: range,
            operation: self.operations.len() - 1,
            args,
            outcomes,
        });
        range
    }

    fn raise(&mut self, raise: Raise<'_, D, CapturedValue>) {
        let payload = Run::push(&mut self.args, raise.payload);
        let binds = Run::push(&mut self.binds, raise.binds);
        self.statements.push(Stored::Raise {
            outcome: raise.outcome,
            guard: raise.guard,
            payload,
            binds,
        });
    }

    fn write(&mut self, target: WriteTarget<'_, D, CapturedValue>, policy: D::WritePolicy) {
        let target = match target {
            WriteTarget::Root { root, plan } => StoredTarget::Root {
                root,
                plan: Run::push(&mut self.plans, plan),
            },
            WriteTarget::FiniteIndexed {
                index,
                candidates,
                value,
            } => StoredTarget::Finite {
                index,
                candidates: Run::push(&mut self.places, candidates),
                value,
            },
            WriteTarget::SparseIndexed {
                bank,
                key,
                slice,
                value,
            } => StoredTarget::Sparse {
                bank,
                key,
                slice,
                value,
            },
        };
        self.statements.push(Stored::Write { target, policy });
    }

    /// Stores the control boundary. The capture is then finished.
    fn finish(self, control: Control<CapturedValue>) -> Result<(), Infallible> {
        self.control = Some(control);
        Ok(())
    }
}

impl<D: Dialect> RecordView<D> for Captured<D> {
    type Value = CapturedValue;

    fn statements(&self) -> impl ExactSizeIterator<Item = Statement<'_, D, CapturedValue>> + '_ {
        Captured::statements(self)
    }

    /// Returns the control boundary.
    ///
    /// # Panics
    ///
    /// Panics when the capture is not finished: the producer did not call
    /// [`finish`](InstructionBuilder::finish).
    fn control(&self) -> Control<CapturedValue> {
        self.control
            .expect("a capture is read as a record only after the producer finished it")
    }

    fn width(&self, value: CapturedValue) -> Bits {
        Captured::width(self, value)
    }
}
