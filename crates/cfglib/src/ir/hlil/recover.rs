//! Structural recovery over a structured function.
//!
//! Binary and bytecode frontends encode source constructs as flatter shapes:
//! a counted loop arrives as `init; while (c) { ...; update }`, a conditional
//! expression as an assigning diamond, and a language region protocol
//! (`synchronized`, `lock`) as paired enter and exit operations with an
//! exceptional cleanup handler. [`recover_structure`] rebuilds a function with
//! those source-level shapes restored: [`For`] statements, value selection
//! through the dialect's [`select`](RecoverDialect::select) operation, and
//! [`Region`] statements. It leaves everything that the dialect does not
//! claim untouched.
//!
//! Every transformation is a pure re-expression. A recovered `for` runs the
//! same statements in the same order, a selection still evaluates its
//! condition once and exactly one arm, and a recovered region re-owns the
//! release-and-rethrow cleanup that its construct regenerates.
//!
//! The rebuild reads the source only through [`FunctionView`] and writes only
//! through [`FunctionBuilder`], so it never edits the function it reads.
//!
//! [`For`]: StmtShape::For
//! [`Region`]: StmtShape::Region

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::graph::view::DenseId;

use super::builder::FunctionBuilder;
use super::dialect::{Dialect, RecoverDialect};
use super::error::RecoverError;
use super::shape::{ExprShape, StmtShape};
use super::view::{FunctionView, for_each_child_body};

mod counted;
mod region;
mod select;
mod walk;

/// The result of one structure recovery.
#[derive(Debug, Clone)]
pub struct Recovery<F, S, E, V> {
    /// The rebuilt function.
    pub function: F,
    /// The statement that carries each source statement now, indexed by the
    /// source statement's [`DenseId::index`]: its copy, or the recovered
    /// construct that consumed it. `None` for a statement the rebuild did not
    /// reach.
    pub statements: Vec<Option<S>>,
    /// The copy of each source expression, indexed by the source expression's
    /// index. The expressions of a cleanup that a recovered construct
    /// regenerates have no copy; their statements map to the construct.
    pub expressions: Vec<Option<E>>,
    /// The copy of each source variable, indexed by the source variable's
    /// index. Every variable has one.
    pub variables: Vec<V>,
    /// The number of counted loops recovered.
    pub for_loops: usize,
    /// The number of conditional value selections recovered.
    pub selects: usize,
    /// The number of paired enter and exit regions recovered.
    pub regions: usize,
}

impl<F, S: Copy, E: Copy, V: Copy> Recovery<F, S, E, V> {
    /// Returns the statement that carries `source` now, if the rebuild
    /// reached it.
    #[must_use]
    pub fn statement(&self, source: impl DenseId) -> Option<S> {
        self.statements.get(source.index()).copied().flatten()
    }

    /// Returns the copy of the source expression `source`, if it has one.
    #[must_use]
    pub fn expression(&self, source: impl DenseId) -> Option<E> {
        self.expressions.get(source.index()).copied().flatten()
    }

    /// Returns the copy of the source variable `source`.
    ///
    /// # Panics
    ///
    /// Panics when `source` is not a variable of the source function.
    #[must_use]
    pub fn variable(&self, source: impl DenseId) -> V {
        self.variables[source.index()]
    }
}

/// Returns whether [`recover_structure`] could change `function`: whether it
/// has an `if`, a `while`, or a `try`, the shapes that the recoveries start
/// from. A caller that finds none keeps the function it has instead of
/// rebuilding an identical copy.
#[must_use]
pub fn has_recovery_candidates<H, F>(function: &F) -> bool
where
    H: Dialect,
    F: FunctionView<H>,
{
    (0..function.stmt_count()).any(|index| {
        matches!(
            function.stmt(F::Stmt::from_index(index)),
            StmtShape::If { .. } | StmtShape::While { .. } | StmtShape::Try { .. }
        )
    })
}

/// Rebuilds `source` through `builder` with source-level structure recovered:
/// counted loops, conditional value selection, and paired enter and exit
/// regions. The source is never changed.
///
/// # Errors
///
/// Returns [`RecoverError::Malformed`] when `source` names a statement,
/// expression, or variable that it does not have, and
/// [`RecoverError::Builder`] when the builder refuses to seal the function.
pub fn recover_structure<H, S, B>(source: &S, mut builder: B) -> Recovered<H, B>
where
    H: RecoverDialect,
    S: FunctionView<H>,
    B: FunctionBuilder<H>,
{
    let variables: Vec<B::Var> = (0..source.variable_count())
        .map(|index| {
            let variable = S::Var::from_index(index);
            builder.declare_variable(source.variable(variable), source.declared_type(variable))
        })
        .collect();
    let parameters = source
        .parameters()
        .iter()
        .map(|parameter| {
            variables
                .get(parameter.index())
                .copied()
                .ok_or_else(|| malformed("variable"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    builder.set_signature(&parameters, source.returns());

    let mut rebuilder = Rebuilder {
        source,
        builder,
        variables,
        statements: vec![None; source.stmt_count()],
        expressions: vec![None; source.expr_count()],
        active_regions: Vec::new(),
        statement_stack: Vec::new(),
        expression_stack: Vec::new(),
        frames: Vec::new(),
        ends: Vec::new(),
        selections: Vec::new(),
        for_loops: 0,
        selects: 0,
        regions: 0,
        dialect: PhantomData,
    };
    rebuilder.rebuild_root(source.body())?;
    let function = rebuilder
        .builder
        .finish(&rebuilder.statement_stack)
        .map_err(RecoverError::Builder)?;
    Ok(Recovery {
        function,
        statements: rebuilder.statements,
        expressions: rebuilder.expressions,
        variables: rebuilder.variables,
        for_loops: rebuilder.for_loops,
        selects: rebuilder.selects,
        regions: rebuilder.regions,
    })
}

fn malformed<E>(what: &str) -> RecoverError<E> {
    RecoverError::Malformed(alloc::format!("unresolvable {what} identity"))
}

type Fallible<T, E> = Result<T, RecoverError<E>>;

/// What [`recover_structure`] returns for the builder `B`.
type Recovered<H, B> = Fallible<
    Recovery<
        <B as FunctionBuilder<H>>::Function,
        <B as FunctionBuilder<H>>::Stmt,
        <B as FunctionBuilder<H>>::Expr,
        <B as FunctionBuilder<H>>::Var,
    >,
    <B as FunctionBuilder<H>>::Error,
>;

/// One region under construction: its enter operation, the variables that
/// currently hold the entered object (grown by copies and shrunk by
/// reassignment, in program order), and the exit statements that its
/// construct re-owns.
struct ActiveRegion<Op, V, S> {
    enter: Op,
    aliases: BTreeSet<V>,
    exits: Vec<S>,
}

/// The state of one rebuild.
///
/// The rebuilt handles of operands and child statements wait on two stacks
/// until their parent is built. A parent hands its builder the slice of the
/// stack that holds its children and then truncates the stack, so a rebuild
/// allocates for nothing but the source-sized tables and the switches and
/// tries that it meets. The nesting of statements waits on a third stack of
/// frames, so the depth of a function costs no call frames either.
struct Rebuilder<'a, H, S, B>
where
    H: RecoverDialect,
    S: FunctionView<H>,
    B: FunctionBuilder<H>,
{
    source: &'a S,
    builder: B,
    variables: Vec<B::Var>,
    statements: Vec<Option<B::Stmt>>,
    expressions: Vec<Option<B::Expr>>,
    active_regions: Vec<ActiveRegion<H::Op, S::Var, S::Stmt>>,
    statement_stack: Vec<B::Stmt>,
    expression_stack: Vec<B::Expr>,
    /// The bodies and constructs of the statement walk that are still open.
    frames: Vec<walk::Frame<'a, H, S, B>>,
    /// Where on `statement_stack` each finished child body of the open
    /// constructs ends, in the order that the bodies finished.
    ends: Vec<usize>,
    /// The selection diamonds that are being built.
    selections: Vec<select::OpenSelection<'a, H, S>>,
    for_loops: usize,
    selects: usize,
    regions: usize,
    dialect: PhantomData<fn() -> H>,
}

/// An operation expression whose operands are being rebuilt.
struct OpenOperation<'a, H: RecoverDialect, E> {
    expr: E,
    op: &'a H::Op,
    operands: &'a [E],
    next: usize,
    mark: usize,
}

impl<'a, H, S, B> Rebuilder<'a, H, S, B>
where
    H: RecoverDialect,
    S: FunctionView<H>,
    B: FunctionBuilder<H>,
{
    fn stmt(&self, id: S::Stmt) -> Fallible<StmtShape<'a, H, S::Stmt, S::Expr>, B::Error> {
        let source: &'a S = self.source;
        if id.index() < source.stmt_count() {
            Ok(source.stmt(id))
        } else {
            Err(malformed("statement"))
        }
    }

    fn expr(&self, id: S::Expr) -> Fallible<ExprShape<'a, H, S::Expr, S::Var>, B::Error> {
        let source: &'a S = self.source;
        if id.index() < source.expr_count() {
            Ok(source.expr(id))
        } else {
            Err(malformed("expression"))
        }
    }

    fn variable(&self, id: S::Var) -> Fallible<B::Var, B::Error> {
        self.variables
            .get(id.index())
            .copied()
            .ok_or_else(|| malformed("variable"))
    }

    /// Whether any node of the source expression tree satisfies `predicate`.
    fn any_expression_node(
        &self,
        root: S::Expr,
        mut predicate: impl FnMut(&ExprShape<'a, H, S::Expr, S::Var>) -> bool,
    ) -> Fallible<bool, B::Error> {
        let mut pending = vec![root];
        while let Some(expr) = pending.pop() {
            let shape = self.expr(expr)?;
            if predicate(&shape) {
                return Ok(true);
            }
            if let ExprShape::Operation { operands, .. } = shape {
                pending.extend(operands.iter().copied());
            }
        }
        Ok(false)
    }

    /// Whether any statement of the source subtrees under `roots` satisfies
    /// `predicate`. Nested statements are included, which keeps the checks
    /// that use this conservative.
    fn any_statement(
        &self,
        roots: &[S::Stmt],
        mut predicate: impl FnMut(&StmtShape<'a, H, S::Stmt, S::Expr>) -> Fallible<bool, B::Error>,
    ) -> Fallible<bool, B::Error> {
        let source: &'a S = self.source;
        let mut pending = roots.to_vec();
        while let Some(stmt) = pending.pop() {
            if predicate(&self.stmt(stmt)?)? {
                return Ok(true);
            }
            for_each_child_body(source, stmt, |body, _| pending.extend(body.iter().copied()));
        }
        Ok(false)
    }

    /// Whether one assignment can appear in a source-language expression
    /// clause rather than expanding into a statement sequence.
    fn assignment_is_single_expression(
        &self,
        target: S::Expr,
        value: S::Expr,
    ) -> Fallible<bool, B::Error> {
        self.expr(target)?;
        if !H::single_expression_assignment(self.source.ty(target)) {
            return Ok(false);
        }
        let expands = self.any_expression_node(value, |shape| {
            matches!(shape, ExprShape::Operation { op, .. } if !H::single_expression_operation(op))
        })?;
        Ok(!expands)
    }

    /// Whether the statement is an exit of a region under construction. The
    /// recovered construct re-owns it.
    fn suppress_region_exit(&mut self, id: S::Stmt) -> Fallible<bool, B::Error> {
        let StmtShape::Expression(expression) = self.stmt(id)? else {
            return Ok(false);
        };
        let ExprShape::Operation { op, operands } = self.expr(expression)? else {
            return Ok(false);
        };
        let &[released] = operands else {
            return Ok(false);
        };
        let ExprShape::Variable(released) = self.expr(released)? else {
            return Ok(false);
        };
        let position = self.active_regions.iter().rposition(|region| {
            region.aliases.contains(&released) && H::releases(&region.enter, op)
        });
        if let Some(position) = position {
            self.active_regions[position].exits.push(id);
            return Ok(true);
        }
        Ok(false)
    }

    /// Follows the entered object through copies: an assignment from an
    /// alias adds its target to the region's alias set, and any other
    /// assignment of a tracked variable removes it.
    fn track_region_aliases(&mut self, id: S::Stmt) -> Fallible<(), B::Error> {
        if self.active_regions.is_empty() {
            return Ok(());
        }
        let StmtShape::Assign { target, value } = self.stmt(id)? else {
            return Ok(());
        };
        let ExprShape::Variable(target) = self.expr(target)? else {
            return Ok(());
        };
        let source = match self.expr(value)? {
            ExprShape::Variable(source) => Some(source),
            _ => None,
        };
        for region in &mut self.active_regions {
            match source {
                Some(source) if region.aliases.contains(&source) => {
                    region.aliases.insert(target);
                }
                _ => {
                    region.aliases.remove(&target);
                }
            }
        }
        Ok(())
    }

    /// Rebuilds each source expression onto the expression stack and returns
    /// where they start.
    fn rebuild_operands(&mut self, operands: &[S::Expr]) -> Fallible<usize, B::Error> {
        let start = self.expression_stack.len();
        for &operand in operands {
            let rebuilt = self.rebuild_expression(operand)?;
            self.expression_stack.push(rebuilt);
        }
        Ok(start)
    }

    /// Rebuilds one expression tree, operands first, without recursing: the
    /// open operations wait on an explicit stack.
    fn rebuild_expression(&mut self, root: S::Expr) -> Fallible<B::Expr, B::Error> {
        let source: &'a S = self.source;
        let mut open: Vec<OpenOperation<'a, H, S::Expr>> = Vec::new();
        let mut current = root;
        loop {
            let mut finished = match self.expr(current)? {
                ExprShape::Variable(variable) => {
                    let variable = self.variable(variable)?;
                    Some((
                        current,
                        self.builder.read_variable(variable, source.ty(current)),
                    ))
                }
                ExprShape::Constant(constant) => {
                    Some((current, self.builder.constant(constant, source.ty(current))))
                }
                ExprShape::Operation { op, operands } => {
                    open.push(OpenOperation {
                        expr: current,
                        op,
                        operands,
                        next: 0,
                        mark: self.expression_stack.len(),
                    });
                    None
                }
            };
            loop {
                if let Some((expr, rebuilt)) = finished.take() {
                    self.expressions[expr.index()] = Some(rebuilt);
                    if open.is_empty() {
                        return Ok(rebuilt);
                    }
                    self.expression_stack.push(rebuilt);
                }
                let operation = open
                    .last_mut()
                    .expect("an operation stays open until the root is rebuilt");
                if let Some(&operand) = operation.operands.get(operation.next) {
                    operation.next += 1;
                    current = operand;
                    break;
                }
                let operation = open
                    .pop()
                    .expect("the operation that was just inspected is still open");
                let rebuilt = self.builder.apply(
                    operation.op,
                    &self.expression_stack[operation.mark..],
                    source.ty(operation.expr),
                );
                self.expression_stack.truncate(operation.mark);
                finished = Some((operation.expr, rebuilt));
            }
        }
    }
}
