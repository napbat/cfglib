//! Conditional value selection: an assigning or returning diamond becomes
//! one dialect `select` expression that evaluates its condition once and
//! exactly one arm, the source shape of `?:` and if-expressions.
//!
//! Every walk over the nested diamonds keeps its own stack, so a chain of
//! nested diamonds as deep as generated code can produce costs no call frames.

extern crate alloc;

use alloc::vec::Vec;

use crate::graph::view::DenseId;

use super::{
    ExprShape, Fallible, FunctionBuilder, FunctionView, Rebuilder, RecoverDialect, RecoverError,
    StmtShape,
};

/// The shape of the statement of a selection arm.
type ArmShape<'a, H, S> =
    StmtShape<'a, H, <S as FunctionView<H>>::Stmt, <S as FunctionView<H>>::Expr>;

/// What the leaves of a selection do with their value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Leaves {
    /// Each assigns it to the variable that the selection stores.
    Assign,
    /// Each returns it.
    Return,
}

/// A diamond whose arms are being built. Its condition waits on the
/// expression stack from `mark`, and so does its then-value once that is built.
pub(super) struct OpenSelection<'a, H, S>
where
    H: RecoverDialect,
    S: FunctionView<H>,
{
    mark: usize,
    else_body: &'a [S::Stmt],
    /// The type of the then-arm, which is the type of the selection, once the
    /// then-arm is built.
    then_type: Option<&'a H::Type>,
}

/// The next step of building a selection.
enum Step<'a, H, S, B>
where
    H: RecoverDialect,
    S: FunctionView<H>,
    B: FunctionBuilder<H>,
{
    /// Builds a diamond: its condition, then its arms.
    Diamond {
        condition: S::Expr,
        then_body: &'a [S::Stmt],
        else_body: &'a [S::Stmt],
    },
    /// Builds one arm.
    Arm(&'a [S::Stmt]),
    /// Hands a built value and its type to the diamond that waits for it.
    Value(B::Expr, &'a H::Type),
}

impl<'a, H, S, B> Rebuilder<'a, H, S, B>
where
    H: RecoverDialect,
    S: FunctionView<H>,
    B: FunctionBuilder<H>,
{
    /// Recovers `if (c) x = a; else x = b;` as `x = select(c, a, b);` and the
    /// all-arms-return diamond as `return select(c, a, b);`, through nested
    /// one-statement diamonds.
    pub(super) fn try_select(
        &mut self,
        if_id: S::Stmt,
        condition: S::Expr,
        then_body: &'a [S::Stmt],
        else_body: &'a [S::Stmt],
    ) -> Fallible<Option<B::Stmt>, B::Error> {
        if H::select().is_none() {
            return Ok(None);
        }
        if let Some(target) = self.assigned_target(then_body)?
            && self.assigns_shape(then_body, target)?
            && self.assigns_shape(else_body, target)?
        {
            // Validated: build the selection tree and the single assign.
            let (value, value_type) =
                self.build_selection(Leaves::Assign, condition, then_body, else_body)?;
            let variable = self.variable(target)?;
            let new_target = self.builder.read_variable(variable, value_type);
            let assign = self.builder.assign(new_target, value);
            self.map_selection(if_id, then_body, else_body, assign)?;
            self.selects += 1;
            return Ok(Some(assign));
        }
        if self.returns_shape(then_body)? && self.returns_shape(else_body)? {
            let (value, _) =
                self.build_selection(Leaves::Return, condition, then_body, else_body)?;
            let statement = self.builder.return_(&[value]);
            self.map_selection(if_id, then_body, else_body, statement)?;
            self.selects += 1;
            return Ok(Some(statement));
        }
        Ok(None)
    }

    /// The shape of the only statement of the arm `ids`, or `None` when the
    /// arm holds anything else.
    fn sole_statement(&self, ids: &[S::Stmt]) -> Fallible<Option<ArmShape<'a, H, S>>, B::Error> {
        match *ids {
            [only] => self.stmt(only).map(Some),
            _ => Ok(None),
        }
    }

    /// The same for an arm that the shape checks have validated, which holds
    /// exactly one statement.
    fn validated_arm(&self, ids: &[S::Stmt]) -> Fallible<ArmShape<'a, H, S>, B::Error> {
        self.sole_statement(ids)?.ok_or_else(|| {
            RecoverError::Malformed("a validated selection arm is a single statement".into())
        })
    }

    /// The variable the then-arm chain assigns, when there is one.
    fn assigned_target(&self, ids: &[S::Stmt]) -> Fallible<Option<S::Var>, B::Error> {
        let mut arm = ids;
        loop {
            match self.sole_statement(arm)? {
                Some(StmtShape::Assign { target, .. }) => {
                    return Ok(match self.expr(target)? {
                        ExprShape::Variable(variable) => Some(variable),
                        _ => None,
                    });
                }
                Some(StmtShape::If { then_body, .. }) => arm = then_body,
                _ => return Ok(None),
            }
        }
    }

    /// Whether every arm of the diamond chain `ids` is a single-expression
    /// assignment of `target`.
    fn assigns_shape(&self, ids: &[S::Stmt], target: S::Var) -> Fallible<bool, B::Error> {
        let mut else_arms: Vec<&'a [S::Stmt]> = Vec::new();
        let mut arm = ids;
        loop {
            match self.sole_statement(arm)? {
                Some(StmtShape::Assign {
                    target: assigned,
                    value,
                }) => {
                    let ExprShape::Variable(variable) = self.expr(assigned)? else {
                        return Ok(false);
                    };
                    if variable != target
                        || !self.assignment_is_single_expression(assigned, value)?
                    {
                        return Ok(false);
                    }
                }
                Some(StmtShape::If {
                    then_body,
                    else_body,
                    ..
                }) => {
                    else_arms.push(else_body);
                    arm = then_body;
                    continue;
                }
                _ => return Ok(false),
            }
            let Some(next) = else_arms.pop() else {
                return Ok(true);
            };
            arm = next;
        }
    }

    /// Whether every arm of the diamond chain `ids` returns a single value.
    fn returns_shape(&self, ids: &[S::Stmt]) -> Fallible<bool, B::Error> {
        let mut else_arms: Vec<&'a [S::Stmt]> = Vec::new();
        let mut arm = ids;
        loop {
            match self.sole_statement(arm)? {
                Some(StmtShape::Return { values }) => {
                    if values.len() != 1 {
                        return Ok(false);
                    }
                }
                Some(StmtShape::If {
                    then_body,
                    else_body,
                    ..
                }) => {
                    else_arms.push(else_body);
                    arm = then_body;
                    continue;
                }
                _ => return Ok(false),
            }
            let Some(next) = else_arms.pop() else {
                return Ok(true);
            };
            arm = next;
        }
    }

    /// Builds the selection expression of a validated diamond and returns it
    /// with the type of its value, the type of its leftmost leaf. Each diamond
    /// builds its condition, its then-value, its else-value, and then itself.
    fn build_selection(
        &mut self,
        leaves: Leaves,
        condition: S::Expr,
        then_body: &'a [S::Stmt],
        else_body: &'a [S::Stmt],
    ) -> Fallible<(B::Expr, &'a H::Type), B::Error> {
        let op = H::select().expect("selection recovery requires the select operation");
        let base = self.selections.len();
        let mut step = Step::<H, S, B>::Diamond {
            condition,
            then_body,
            else_body,
        };
        loop {
            step = match step {
                Step::Diamond {
                    condition,
                    then_body,
                    else_body,
                } => {
                    let mark = self.expression_stack.len();
                    let condition = self.rebuild_expression(condition)?;
                    self.expression_stack.push(condition);
                    self.selections.push(OpenSelection {
                        mark,
                        else_body,
                        then_type: None,
                    });
                    Step::Arm(then_body)
                }
                Step::Arm(ids) => match self.validated_arm(ids)? {
                    StmtShape::If {
                        condition,
                        then_body,
                        else_body,
                    } => Step::Diamond {
                        condition,
                        then_body,
                        else_body,
                    },
                    shape => {
                        let (value, value_type) = self.leaf_value(leaves, shape)?;
                        Step::Value(value, value_type)
                    }
                },
                Step::Value(value, value_type) => {
                    if self.selections.len() == base {
                        return Ok((value, value_type));
                    }
                    self.expression_stack.push(value);
                    let open = self
                        .selections
                        .last_mut()
                        .expect("a diamond waits for the value that was just built");
                    match open.then_type {
                        None => {
                            open.then_type = Some(value_type);
                            Step::Arm(open.else_body)
                        }
                        Some(then_type) => {
                            let mark = open.mark;
                            self.selections.pop();
                            let selected =
                                self.builder
                                    .apply(&op, &self.expression_stack[mark..], then_type);
                            self.expression_stack.truncate(mark);
                            Step::Value(selected, then_type)
                        }
                    }
                }
            };
        }
    }

    /// Builds the value of one validated leaf arm and returns it with its
    /// type: the assigned value under the type of its target, or the
    /// returned value.
    fn leaf_value(
        &mut self,
        leaves: Leaves,
        shape: ArmShape<'a, H, S>,
    ) -> Fallible<(B::Expr, &'a H::Type), B::Error> {
        let source: &'a S = self.source;
        match (leaves, shape) {
            (Leaves::Assign, StmtShape::Assign { target, value }) => {
                Ok((self.rebuild_expression(value)?, source.ty(target)))
            }
            (Leaves::Return, StmtShape::Return { values }) => {
                let &[value] = values else {
                    return Err(RecoverError::Malformed(
                        "a validated selection arm returns one value".into(),
                    ));
                };
                Ok((self.rebuild_expression(value)?, source.ty(value)))
            }
            (Leaves::Assign, _) => Err(RecoverError::Malformed(
                "a validated selection arm assigns or nests".into(),
            )),
            (Leaves::Return, _) => Err(RecoverError::Malformed(
                "a validated selection arm returns or nests".into(),
            )),
        }
    }

    /// Maps every statement that the selection consumed onto the new
    /// statement.
    fn map_selection(
        &mut self,
        if_id: S::Stmt,
        then_body: &[S::Stmt],
        else_body: &[S::Stmt],
        statement: B::Stmt,
    ) -> Fallible<(), B::Error> {
        let mut pending: Vec<S::Stmt> = Vec::from([if_id]);
        pending.extend(then_body.iter().copied());
        pending.extend(else_body.iter().copied());
        while let Some(id) = pending.pop() {
            let shape = self.stmt(id)?;
            self.statements[id.index()] = Some(statement);
            if let StmtShape::If {
                then_body,
                else_body,
                ..
            } = shape
            {
                pending.extend(then_body.iter().copied());
                pending.extend(else_body.iter().copied());
            }
        }
        Ok(())
    }
}
