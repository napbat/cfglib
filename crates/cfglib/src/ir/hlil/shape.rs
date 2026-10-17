//! The borrowed shapes of structured statements and expressions.
//!
//! A [`FunctionView`](super::FunctionView) lends these types out of the
//! consumer's storage, and the [`FunctionBuilder`](super::FunctionBuilder)
//! takes the same parts as input. Every slice inside them is flat, so a
//! consumer keeps its operands and bodies in dense pools and lends them back
//! without rebuilding them.

use super::dialect::Dialect;

/// The shape of one expression node.
///
/// The library owns only the three universal shapes. Everything else (calls,
/// memory access, arithmetic, conversions, allocation) is a dialect operation
/// over operand expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExprShape<'a, D: Dialect, E, V> {
    /// One read occurrence of a variable.
    Variable(V),
    /// One dialect constant literal.
    Constant(&'a D::Constant),
    /// One dialect operation over ordered operand expressions.
    Operation {
        /// The consumer-defined operation.
        op: &'a D::Op,
        /// The operands in operation order.
        operands: &'a [E],
    },
}

/// One arm of a switch.
///
/// An arm does not fall through: a body that ends transfers to the switch
/// continuation. Source fallthrough lowers to explicit gotos.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwitchArm<'a, D: Dialect, S> {
    /// The constant values that select this arm. Never empty: the default arm
    /// is [`StmtShape::Switch::default_body`](StmtShape::Switch).
    pub values: &'a [D::Constant],
    /// The statements of the arm.
    pub body: &'a [S],
}

/// One handler of a [`StmtShape::Try`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Handler<'a, D: Dialect, S, V> {
    /// The classification of the handler.
    pub kind: HandlerKind<'a, S>,
    /// The variable bound to the delivered exception at handler entry.
    pub binding: Option<V>,
    /// The exception types that the handler accepts, such as a multi-catch.
    /// Empty for a catch-all, a fault, and a filter handler.
    pub caught_types: &'a [D::Type],
    /// The statements of the handler.
    pub body: &'a [S],
}

/// The classification of one [`Handler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandlerKind<'a, S> {
    /// Catches the types that [`Handler::caught_types`] names.
    Catch,
    /// Catches any exception.
    CatchAll,
    /// Runs on exceptional entry only.
    Fault,
    /// A predicate decides whether this handler accepts the exception.
    ///
    /// [`lift`](fn@super::lift) does not structure the predicate funclet: a
    /// lifted filter handler has an empty `filter_body`, and the funclet's
    /// blocks follow the function as a labeled group that nothing targets.
    /// [`lower`](fn@super::lower) needs the body, so it rejects an empty one;
    /// a consumer that lowers a lifted function restores the predicate first.
    Filter {
        /// The statements that evaluate the predicate.
        filter_body: &'a [S],
    },
}

/// The shape of one statement.
///
/// The control shapes are universal and owned by the library; the open
/// semantic vocabulary lives in expressions, and [`Region`](Self::Region) is
/// the escape hatch for dialect statements that wrap a body (`synchronized`,
/// `lock`, `using`, `with`, `unsafe`).
///
/// The arms of a switch and the handlers of a try are structured sequences, so
/// the shape does not hold them. A view lends them through
/// [`FunctionView::switch_arms`](super::FunctionView::switch_arms) and
/// [`FunctionView::handlers`](super::FunctionView::handlers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StmtShape<'a, D: Dialect, S, E> {
    /// Evaluates an expression for its effects: calls, stores modeled as
    /// operations, throws.
    Expression(E),

    /// Stores `value` into the place that `target` names: a variable
    /// occurrence or a dialect place operation (dereference, field, element),
    /// never a constant.
    Assign {
        /// The place written.
        target: E,
        /// The value stored.
        value: E,
    },

    /// A conditional branch.
    If {
        /// The decided condition.
        condition: E,
        /// The statements executed when the condition is true.
        then_body: &'a [S],
        /// The statements executed when the condition is false. Empty for no
        /// `else`.
        else_body: &'a [S],
    },

    /// A pre-tested loop.
    While {
        /// The condition evaluated before every iteration.
        condition: E,
        /// The loop body.
        body: &'a [S],
    },

    /// A post-tested loop.
    DoWhile {
        /// The loop body.
        body: &'a [S],
        /// The condition evaluated after every iteration.
        condition: E,
    },

    /// An endless loop. Iteration ends only through an inner transfer.
    Loop {
        /// The loop body.
        body: &'a [S],
    },

    /// A counted-style loop with explicit initializer and update sections.
    For {
        /// The statements executed once before the first test.
        initializer: &'a [S],
        /// The condition evaluated before every iteration. `None` iterates
        /// unconditionally.
        condition: Option<E>,
        /// The statements executed after every iteration, before the next
        /// test.
        update: &'a [S],
        /// The loop body.
        body: &'a [S],
    },

    /// A multi-way branch over constant case values. The arms are
    /// [`FunctionView::switch_arms`](super::FunctionView::switch_arms).
    Switch {
        /// The dispatched value.
        scrutinee: E,
        /// The statements executed when no arm matches. Empty when the
        /// default transfers straight to the continuation.
        default_body: &'a [S],
    },

    /// Breaks out of the innermost enclosing loop or switch, or out of the
    /// enclosing [`Labeled`](Self::Labeled) statement that `label` names.
    Break {
        /// The named enclosing statement. `None` breaks the innermost loop or
        /// switch.
        label: Option<&'a str>,
    },

    /// Continues the innermost enclosing loop, or the enclosing labeled loop
    /// that `label` names. A label names a loop only when it is the label of a
    /// [`Labeled`](Self::Labeled) statement whose body is exactly that loop.
    Continue {
        /// The named enclosing loop. `None` continues the innermost loop.
        label: Option<&'a str>,
    },

    /// Returns from the function with zero or more values.
    Return {
        /// The returned values in result order.
        values: &'a [E],
    },

    /// A labeled statement group: the target of [`Goto`](Self::Goto) and of a
    /// labeled [`Break`](Self::Break) or [`Continue`](Self::Continue).
    ///
    /// A group whose body is exactly one loop statement labels that loop.
    /// Every other group labels a block, which a `break` or `goto` can
    /// target but a `continue` cannot.
    Labeled {
        /// The unique label name.
        label: &'a str,
        /// The labeled statements.
        body: &'a [S],
    },

    /// An unconditional transfer to the [`Labeled`](Self::Labeled) statement
    /// with the matching name: the structured residue of irreducible control
    /// flow.
    Goto {
        /// The target label name.
        label: &'a str,
    },

    /// A protected region. The handlers are
    /// [`FunctionView::handlers`](super::FunctionView::handlers).
    Try {
        /// The protected statements.
        body: &'a [S],
        /// The statements executed on every completion path. Empty for no
        /// `finally`.
        finally_body: &'a [S],
    },

    /// A dialect statement that wraps a body, such as `synchronized`, `lock`,
    /// `using`, or `with`.
    Region {
        /// The consumer-defined region operation.
        op: &'a D::Op,
        /// The operand expressions, such as the monitor or the resource. They
        /// are evaluated once, on entry, and the end of the region releases
        /// the values they had then, even when the body reassigns a variable
        /// that an operand reads.
        operands: &'a [E],
        /// The wrapped statements.
        body: &'a [S],
    },
}
