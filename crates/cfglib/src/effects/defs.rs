//! The definitions of the values of one instruction, which the summarizer and
//! the tree builder share.

extern crate alloc;

use alloc::vec::Vec;

use crate::graph::view::DenseId;
use crate::ir::rtl::{Bits, Dialect, Selection, Statement};

use super::dialect::DialectEffects;

/// What defines one value.
#[derive(Debug)]
pub(super) enum Def<'a, D: Dialect, V> {
    /// A read of the state before the instruction.
    Read(Selection<'a, D, V>),
    /// A constant.
    Const { words: &'a [u64] },
    /// One result of a pure operator.
    Apply {
        operator: &'a D::Operator,
        args: &'a [V],
        result: u32,
    },
    /// A nondeterministic choice.
    Nondet,
    /// One result of an ordered effect.
    Effect { ordinal: u32, result: u32 },
}

#[expect(
    clippy::expl_impl_clone_on_copy,
    reason = "a derive bounds the dialect types that the fields only borrow"
)]
impl<D: Dialect, V: Copy> Clone for Def<'_, D, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: Dialect, V: Copy> Copy for Def<'_, D, V> {}

/// One defined value: its definition, its type, and the data of the reader.
#[derive(Debug)]
pub(super) struct Slot<'a, D: Dialect, V, X> {
    pub(super) def: Def<'a, D, V>,
    pub(super) width: Bits,
    pub(super) extra: X,
}

#[expect(
    clippy::expl_impl_clone_on_copy,
    reason = "a derive bounds the dialect types that the fields only borrow"
)]
impl<D: Dialect, V: Copy, X: Copy> Clone for Slot<'_, D, V, X> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: Dialect, V: Copy, X: Copy> Copy for Slot<'_, D, V, X> {}

/// The defined values of one instruction, keyed by the dense index of the
/// value.
///
/// The table stores one slot for each index between the lowest and the
/// highest defined index. The values of one record are consecutive in every
/// known source, so the table holds no gap there. A source with sparse
/// values costs memory in proportion to the spread of its indices.
pub(super) struct Defs<'a, D: Dialect, V, X> {
    base: usize,
    slots: Vec<Option<Slot<'a, D, V, X>>>,
    effects: u32,
}

impl<'a, D: Dialect, V: DenseId, X: Copy> Defs<'a, D, V, X> {
    pub(super) const fn new() -> Self {
        Self {
            base: 0,
            slots: Vec::new(),
            effects: 0,
        }
    }

    /// Returns the slot of `value`.
    ///
    /// # Panics
    ///
    /// Panics when no earlier statement defines `value`.
    pub(super) fn slot(&self, value: V) -> &Slot<'a, D, V, X> {
        self.try_slot(value)
            .expect("an operand is a value that an earlier statement of the instruction defines")
    }

    fn try_slot(&self, value: V) -> Option<&Slot<'a, D, V, X>> {
        let index = value.index().checked_sub(self.base)?;
        self.slots.get(index)?.as_ref()
    }

    /// Returns the slot of `value` for a change of its extra data.
    pub(super) fn slot_mut(&mut self, value: V) -> &mut Slot<'a, D, V, X> {
        let index = value
            .index()
            .checked_sub(self.base)
            .expect("an operand is a value that an earlier statement of the instruction defines");
        self.slots[index]
            .as_mut()
            .expect("an operand is a value that an earlier statement of the instruction defines")
    }

    fn insert(&mut self, value: V, slot: Slot<'a, D, V, X>) {
        let index = value.index();
        if self.slots.is_empty() {
            self.base = index;
        } else if index < self.base {
            let gap = self.base - index;
            self.slots.splice(0..0, core::iter::repeat_n(None, gap));
            self.base = index;
        }
        let local = index - self.base;
        if local >= self.slots.len() {
            self.slots.resize(local + 1, None);
        }
        self.slots[local] = Some(slot);
    }

    /// Records the values that `statement` defines.
    ///
    /// `width` gives the type of an operator or effect result, and every new
    /// value starts with the data `extra` of the reader.
    pub(super) fn define(
        &mut self,
        statement: &Statement<'a, D, V>,
        width: impl Fn(V) -> Bits,
        extra: X,
    ) {
        match *statement {
            Statement::Read { result, source } => {
                let slot = Slot {
                    def: Def::Read(source),
                    width: width(result),
                    extra,
                };
                self.insert(result, slot);
            }
            Statement::Constant {
                result,
                words,
                width,
            } => {
                let slot = Slot {
                    def: Def::Const { words },
                    width,
                    extra,
                };
                self.insert(result, slot);
            }
            Statement::Apply {
                results,
                operator,
                args,
            } => {
                for (result, value) in (0..).zip(results.iter()) {
                    let slot = Slot {
                        def: Def::Apply {
                            operator,
                            args,
                            result,
                        },
                        width: width(value),
                        extra,
                    };
                    self.insert(value, slot);
                }
            }
            Statement::Nondet { result, width, .. } => {
                let slot = Slot {
                    def: Def::Nondet,
                    width,
                    extra,
                };
                self.insert(result, slot);
            }
            Statement::Effect { results, .. } => {
                let ordinal = self.effects;
                self.effects += 1;
                for (result, value) in (0..).zip(results.iter()) {
                    let slot = Slot {
                        def: Def::Effect { ordinal, result },
                        width: width(value),
                        extra,
                    };
                    self.insert(value, slot);
                }
            }
            Statement::Raise(_) | Statement::Write { .. } => {}
        }
    }

    /// Returns the position of the last effect that [`Self::define`] saw.
    pub(super) fn last_effect(&self) -> u32 {
        self.effects - 1
    }

    /// Recognizes a conditional write of one span of `root`.
    ///
    /// The span starts at bit `span_offset` of the root and receives the bits
    /// of `value` from bit `offset`. The write is conditional when `value` is
    /// a select whose `otherwise` operand reads the same bits of `root`
    /// before the instruction. Returns the guard and the `taken` operand.
    pub(super) fn conditional<C: DialectEffects<D>>(
        &self,
        dialect: &C,
        root: D::Root,
        span_offset: u16,
        value: V,
        offset: u16,
    ) -> Option<(V, V)> {
        let Def::Apply {
            operator,
            args,
            result: 0,
        } = self.slot(value).def
        else {
            return None;
        };
        let shape = dialect.select(operator)?;
        let Def::Read(Selection::Direct(old)) = self.slot(args[shape.otherwise]).def else {
            return None;
        };
        let same = old.root == root
            && u32::from(old.slice.offset) + u32::from(offset) == u32::from(span_offset);
        same.then(|| (args[shape.guard], args[shape.taken]))
    }
}
