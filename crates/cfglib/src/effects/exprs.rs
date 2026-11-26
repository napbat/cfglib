//! The expression arena of one effect.

extern crate alloc;

use alloc::vec::Vec;

use crate::identity::define_dense_id;
use crate::ir::rtl::Bits;

use super::vocabulary::Touch;

define_dense_id! {
    /// The identity of one expression of an [`Exprs`] arena.
    pub struct ExprId(u32);
}

/// A value that an expression cannot state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Opaque {
    /// A result of an ordered effect, such as a loaded value.
    Effect {
        /// The position of the effect among the effects of the code, from
        /// zero.
        ordinal: u32,
        /// The position of the value among the results of the effect.
        result: u32,
    },
    /// A nondeterministic choice.
    Nondet,
    /// A read whose place an index value selects.
    IndexedRead,
}

/// One expression, as [`Exprs::get`] lends it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Expr<'e, S> {
    /// A read of the value from before the code.
    Read(Touch<S>),
    /// A constant.
    Const {
        /// The little-endian 64-bit words of the value. Bits past the width
        /// are zero.
        words: &'e [u64],
        /// The type of the value.
        width: Bits,
    },
    /// One result of an operator.
    Op {
        /// The name of the operator, which the dialect gives.
        name: &'static str,
        /// The operands, in operator order.
        args: &'e [ExprId],
        /// The position of this value among the results of the operator.
        result: u32,
        /// The type of the value.
        width: Bits,
    },
    /// A value that the expression cannot state.
    Opaque {
        /// Where the value comes from.
        origin: Opaque,
        /// The type of the value.
        width: Bits,
    },
}

/// A run of one pool of an [`Exprs`] arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Run {
    start: u32,
    len: u32,
}

impl Run {
    fn push<T: Copy>(pool: &mut Vec<T>, items: &[T]) -> Self {
        let start =
            u32::try_from(pool.len()).expect("an expression pool holds at most u32::MAX items");
        pool.extend_from_slice(items);
        let len =
            u32::try_from(items.len()).expect("an expression pool holds at most u32::MAX items");
        Self { start, len }
    }

    fn get<T>(self, pool: &[T]) -> &[T] {
        let start = self.start as usize;
        &pool[start..start + self.len as usize]
    }
}

/// One stored expression. Its slices are runs of the pools of the arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Node<S> {
    Read(Touch<S>),
    Const {
        words: Run,
        width: Bits,
    },
    Op {
        name: &'static str,
        args: Run,
        result: u32,
        width: Bits,
    },
    Opaque {
        origin: Opaque,
        width: Bits,
    },
}

/// A small arena of expressions over places of type `S`.
///
/// The guards and the addresses of an [`Effect`](super::Effect) point into
/// it. An expression refers only to expressions that were added before it,
/// so the arena is a directed acyclic graph. Two arenas are equal when they
/// hold the same expressions in the same order.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Exprs<S> {
    nodes: Vec<Node<S>>,
    words: Vec<u64>,
    args: Vec<ExprId>,
}

impl<S> Default for Exprs<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> Exprs<S> {
    /// Creates an empty arena.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            words: Vec::new(),
            args: Vec::new(),
        }
    }

    /// Returns how many expressions the arena holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns whether the arena holds no expression.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn push(&mut self, node: Node<S>) -> ExprId {
        let id =
            u32::try_from(self.nodes.len()).expect("an arena holds at most u32::MAX expressions");
        self.nodes.push(node);
        ExprId(id)
    }

    /// Adds a read of `touch`.
    pub fn read(&mut self, touch: Touch<S>) -> ExprId {
        self.push(Node::Read(touch))
    }

    /// Adds a constant of type `width` from little-endian 64-bit words.
    pub fn constant(&mut self, words: &[u64], width: Bits) -> ExprId {
        let words = Run::push(&mut self.words, words);
        self.push(Node::Const { words, width })
    }

    /// Adds result `result` of the operator `name` over `args`.
    ///
    /// # Panics
    ///
    /// Panics when an operand is not an expression of this arena.
    pub fn op(&mut self, name: &'static str, args: &[ExprId], result: u32, width: Bits) -> ExprId {
        assert!(
            args.iter().all(|arg| arg.index() < self.nodes.len()),
            "an operand is an earlier expression of the arena"
        );
        let args = Run::push(&mut self.args, args);
        self.push(Node::Op {
            name,
            args,
            result,
            width,
        })
    }

    /// Adds a value that an expression cannot state.
    pub fn opaque(&mut self, origin: Opaque, width: Bits) -> ExprId {
        self.push(Node::Opaque { origin, width })
    }
}

impl<S: Copy> Exprs<S> {
    /// Returns the expression `id`.
    ///
    /// # Panics
    ///
    /// Panics when `id` is not an expression of this arena.
    #[must_use]
    pub fn get(&self, id: ExprId) -> Expr<'_, S> {
        match self.nodes[id.index()] {
            Node::Read(touch) => Expr::Read(touch),
            Node::Const { words, width } => Expr::Const {
                words: words.get(&self.words),
                width,
            },
            Node::Op {
                name,
                args,
                result,
                width,
            } => Expr::Op {
                name,
                args: args.get(&self.args),
                result,
                width,
            },
            Node::Opaque { origin, width } => Expr::Opaque { origin, width },
        }
    }

    /// Iterates over the identities of every expression in the order they were
    /// added.
    pub fn ids(&self) -> impl Iterator<Item = ExprId> + '_ {
        (0..).zip(&self.nodes).map(|(index, _)| ExprId(index))
    }
}
