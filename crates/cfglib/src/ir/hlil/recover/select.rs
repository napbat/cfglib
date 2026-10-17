//! Conditional value selection: an assigning or returning diamond becomes
//! one dialect `select` expression that evaluates its condition once and
//! exactly one arm, the source shape of `?:` and if-expressions.

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

impl<'a, H, S, B> Rebuilder<'a, H, S, B>
where
    H: RecoverDialect,
    S: FunctionView<H>,
    B: FunctionBuilder<H>,
{
    /// Recovers `if (c) x = a; else x = b;` as `x = select(c, a, b);` and the
    /// all-arms-return diamond as `return select(c, a, b);`, recursing
    /// through nested one-statement diamonds.
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
            let (variable, value_type) = self.selection_target_node(then_body)?;
            let value = self.build_select(condition, then_body, else_body, target)?;
            let variable = self.variable(variable)?;
            let new_target = self.builder.read_variable(variable, value_type);
            let assign = self.builder.assign(new_target, value);
            self.map_selection(if_id, then_body, else_body, assign)?;
            self.selects += 1;
            return Ok(Some(assign));
        }
        if self.returns_shape(then_body)? && self.returns_shape(else_body)? {
            let (value, _) = self.build_return_select(condition, then_body, else_body)?;
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
        let Some(shape) = self.sole_statement(ids)? else {
            return Ok(None);
        };
        match shape {
            StmtShape::Assign { target, .. } => Ok(match self.expr(target)? {
                ExprShape::Variable(variable) => Some(variable),
                _ => None,
            }),
            StmtShape::If { then_body, .. } => self.assigned_target(then_body),
            _ => Ok(None),
        }
    }

    /// The variable and occurrence type of the leftmost assignment.
    fn selection_target_node(&self, ids: &[S::Stmt]) -> Fallible<(S::Var, &'a H::Type), B::Error> {
        match self.validated_arm(ids)? {
            StmtShape::Assign { target, .. } => {
                let ExprShape::Variable(variable) = self.expr(target)? else {
                    return Err(RecoverError::Malformed(
                        "a validated selection arm assigns a variable".into(),
                    ));
                };
                let source: &'a S = self.source;
                Ok((variable, source.ty(target)))
            }
            StmtShape::If { then_body, .. } => self.selection_target_node(then_body),
            _ => Err(RecoverError::Malformed(
                "a validated selection arm assigns or nests".into(),
            )),
        }
    }

    /// Whether one arm is a single-expression assignment of `target`, or a
    /// nested diamond of such arms.
    fn assigns_shape(&self, ids: &[S::Stmt], target: S::Var) -> Fallible<bool, B::Error> {
        let Some(shape) = self.sole_statement(ids)? else {
            return Ok(false);
        };
        match shape {
            StmtShape::Assign {
                target: assigned,
                value,
            } => {
                let ExprShape::Variable(variable) = self.expr(assigned)? else {
                    return Ok(false);
                };
                Ok(variable == target && self.assignment_is_single_expression(assigned, value)?)
            }
            StmtShape::If {
                then_body,
                else_body,
                ..
            } => {
                Ok(self.assigns_shape(then_body, target)?
                    && self.assigns_shape(else_body, target)?)
            }
            _ => Ok(false),
        }
    }

    /// Whether one arm returns a single value, or nests such diamonds.
    fn returns_shape(&self, ids: &[S::Stmt]) -> Fallible<bool, B::Error> {
        let Some(shape) = self.sole_statement(ids)? else {
            return Ok(false);
        };
        match shape {
            StmtShape::Return { values } => Ok(values.len() == 1),
            StmtShape::If {
                then_body,
                else_body,
                ..
            } => Ok(self.returns_shape(then_body)? && self.returns_shape(else_body)?),
            _ => Ok(false),
        }
    }

    /// Builds the selection expression for one validated assigning arm.
    fn arm_value(&mut self, ids: &[S::Stmt], target: S::Var) -> Fallible<B::Expr, B::Error> {
        match self.validated_arm(ids)? {
            StmtShape::Assign { value, .. } => self.rebuild_expression(value),
            StmtShape::If {
                condition,
                then_body,
                else_body,
            } => self.build_select(condition, then_body, else_body, target),
            _ => Err(RecoverError::Malformed(
                "a validated selection arm assigns or nests".into(),
            )),
        }
    }

    fn build_select(
        &mut self,
        condition: S::Expr,
        then_body: &[S::Stmt],
        else_body: &[S::Stmt],
        target: S::Var,
    ) -> Fallible<B::Expr, B::Error> {
        let op = H::select().expect("selection recovery requires the select operation");
        let (_, value_type) = self.selection_target_node(then_body)?;
        let condition = self.rebuild_expression(condition)?;
        let when_true = self.arm_value(then_body, target)?;
        let when_false = self.arm_value(else_body, target)?;
        Ok(self
            .builder
            .apply(&op, &[condition, when_true, when_false], value_type))
    }

    /// Builds the selection expression for one validated returning arm and
    /// returns it with the type of its value.
    fn return_value(&mut self, ids: &[S::Stmt]) -> Fallible<(B::Expr, &'a H::Type), B::Error> {
        match self.validated_arm(ids)? {
            StmtShape::Return { values } => {
                let &[value] = values else {
                    return Err(RecoverError::Malformed(
                        "a validated selection arm returns one value".into(),
                    ));
                };
                let source: &'a S = self.source;
                Ok((self.rebuild_expression(value)?, source.ty(value)))
            }
            StmtShape::If {
                condition,
                then_body,
                else_body,
            } => self.build_return_select(condition, then_body, else_body),
            _ => Err(RecoverError::Malformed(
                "a validated selection arm returns or nests".into(),
            )),
        }
    }

    fn build_return_select(
        &mut self,
        condition: S::Expr,
        then_body: &[S::Stmt],
        else_body: &[S::Stmt],
    ) -> Fallible<(B::Expr, &'a H::Type), B::Error> {
        let op = H::select().expect("selection recovery requires the select operation");
        let condition = self.rebuild_expression(condition)?;
        let (when_true, value_type) = self.return_value(then_body)?;
        let (when_false, _) = self.return_value(else_body)?;
        let selected = self
            .builder
            .apply(&op, &[condition, when_true, when_false], value_type);
        Ok((selected, value_type))
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
