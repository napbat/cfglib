//! What an execution observes, and the semantics of the dialect's
//! operations.
//!
//! The two interpreters (structured and flat) differ in how they move
//! control and share what an operation means, so a difference between their
//! outcomes can only come from the control flow that the library built or
//! rebuilt.

use std::collections::BTreeMap;

use crate::dialect::{Lit, Op};

/// A runtime value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Value {
    Int(i64),
    Bool(bool),
}

impl Value {
    pub(crate) fn of(literal: Lit) -> Self {
        match literal {
            Lit::Int(value) => Self::Int(value),
            Lit::Bool(value) => Self::Bool(value),
        }
    }

    pub(crate) fn int(self) -> i64 {
        match self {
            Self::Int(value) => value,
            Self::Bool(_) => panic!("an integer was expected"),
        }
    }

    pub(crate) fn truth(self) -> bool {
        match self {
            Self::Bool(value) => value,
            Self::Int(_) => panic!("a boolean was expected"),
        }
    }
}

/// One observable step of an execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Event {
    Emitted(Value),
    Acquired(i64),
    Released(i64),
}

/// How an execution ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum End {
    Returned(Vec<Value>),
    /// An exception that no handler caught.
    Uncaught(i64),
    /// Control ran past the last statement or block.
    FellOff,
    /// The step budget ran out.
    Diverged,
}

/// Everything an execution observes: the output trace, the way it ended,
/// and the final memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Outcome {
    pub(crate) events: Vec<Event>,
    pub(crate) end: End,
    pub(crate) memory: BTreeMap<i64, i64>,
}

/// An exception in flight, with its code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Thrown(pub(crate) i64);

/// The mutable state that operations act on.
pub(crate) struct Machine {
    events: Vec<Event>,
    memory: BTreeMap<i64, i64>,
    /// The code of the exception being handled.
    pub(crate) exception: i64,
    fuel: u32,
}

/// Steps an execution may take before it counts as diverged.
const FUEL: u32 = 200_000;

impl Machine {
    pub(crate) fn new() -> Self {
        Self {
            events: Vec::new(),
            memory: BTreeMap::new(),
            exception: 0,
            fuel: FUEL,
        }
    }

    /// Spends one step. Returns `false` when the budget is gone.
    pub(crate) fn tick(&mut self) -> bool {
        self.fuel = self.fuel.saturating_sub(1);
        self.fuel > 0
    }

    pub(crate) fn finish(self, end: End) -> Outcome {
        Outcome {
            events: self.events,
            end,
            memory: self.memory,
        }
    }

    pub(crate) fn load(&self, cell: i64) -> i64 {
        self.memory.get(&cell).copied().unwrap_or(0)
    }

    pub(crate) fn store(&mut self, cell: i64, value: i64) {
        self.memory.insert(cell, value);
    }

    pub(crate) fn acquire(&mut self, subject: i64) {
        self.events.push(Event::Acquired(subject));
    }

    pub(crate) fn release(&mut self, subject: i64) {
        self.events.push(Event::Released(subject));
    }

    /// Applies `op` to operands that are all evaluated. The structured
    /// interpreter evaluates only one arm of a `Pick`, so it never reaches
    /// that case here, but flattened code has already computed every arm.
    pub(crate) fn apply(&mut self, op: Op, args: &[Value]) -> Result<Value, Thrown> {
        let int = |position: usize| args[position].int();
        Ok(match op {
            Op::Add => Value::Int(int(0).wrapping_add(int(1))),
            Op::Sub => Value::Int(int(0).wrapping_sub(int(1))),
            Op::Mul => Value::Int(int(0).wrapping_mul(int(1))),
            Op::Div if int(1) == 0 => return Err(Thrown(-1)),
            Op::Div => Value::Int(int(0).wrapping_div(int(1))),
            Op::Rem if int(1) == 0 => return Err(Thrown(-2)),
            Op::Rem => Value::Int(int(0).wrapping_rem(int(1))),
            Op::Lt => Value::Bool(int(0) < int(1)),
            Op::Ge => Value::Bool(int(0) >= int(1)),
            Op::Eq => Value::Bool(args[0] == args[1]),
            Op::Ne => Value::Bool(args[0] != args[1]),
            Op::Not => Value::Bool(!args[0].truth()),
            Op::Cell => Value::Int(self.load(int(0))),
            Op::Emit => {
                self.events.push(Event::Emitted(args[0]));
                Value::Int(0)
            }
            Op::Raise => return Err(Thrown(int(0))),
            Op::Caught => Value::Int(self.exception),
            Op::Acquire => {
                self.acquire(int(0));
                Value::Int(0)
            }
            Op::Release => {
                self.release(int(0));
                Value::Int(0)
            }
            Op::Pick => {
                if args[0].truth() {
                    args[1]
                } else {
                    args[2]
                }
            }
            Op::Guard => panic!("a region operation is not applied to values"),
        })
    }
}
