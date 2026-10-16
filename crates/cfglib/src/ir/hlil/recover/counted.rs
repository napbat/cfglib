//! Counted-loop recovery: `init; while (c) { ...; update }` becomes a
//! [`For`](StmtShape::For) statement when the initializer writes the
//! variable that the condition reads and the body's trailing statement
//! updates it.

use crate::graph::view::DenseId;

use super::{
    ExprShape, Fallible, FunctionBuilder, FunctionView, Rebuilder, RecoverDialect, StmtShape,
};

impl<H, S, B> Rebuilder<'_, H, S, B>
where
    H: RecoverDialect,
    S: FunctionView<H>,
    B: FunctionBuilder<H>,
{
    /// Recovers `ids[index]; ids[index + 1]` as one `for` statement.
    ///
    /// The transformation is exact only when no `continue` reaches the loop:
    /// [`For`](StmtShape::For) runs its update on `continue`, while the
    /// `while` form skips the trailing statement.
    pub(super) fn try_for(
        &mut self,
        ids: &[S::Stmt],
        index: usize,
    ) -> Fallible<Option<B::Stmt>, B::Error> {
        let init = ids[index];
        let loop_id = ids[index + 1];
        let StmtShape::Assign { target, value } = self.stmt(init)? else {
            return Ok(None);
        };
        let ExprShape::Variable(inductive) = self.expr(target)? else {
            return Ok(None);
        };
        if !self.assignment_is_single_expression(target, value)? {
            return Ok(None);
        }
        let StmtShape::While { condition, body } = self.stmt(loop_id)? else {
            return Ok(None);
        };
        if !self.expression_reads(condition, inductive)? {
            return Ok(None);
        }
        let Some((&update, rest)) = body.split_last() else {
            return Ok(None);
        };
        let StmtShape::Assign {
            target: update_target,
            value: update_value,
        } = self.stmt(update)?
        else {
            return Ok(None);
        };
        let ExprShape::Variable(updated) = self.expr(update_target)? else {
            return Ok(None);
        };
        if updated != inductive
            || !self.assignment_is_single_expression(update_target, update_value)?
        {
            return Ok(None);
        }
        if self.contains_continue(rest)? {
            return Ok(None);
        }

        let first = self.statement_stack.len();
        let initializer = self.rebuild_statement(init)?;
        let condition = self.rebuild_expression(condition)?;
        let body = self.rebuild_body(rest)?;
        let update = self.rebuild_statement(update)?;
        let counted = self.builder.for_(
            &[initializer],
            Some(condition),
            &[update],
            &self.statement_stack[body],
        );
        self.statement_stack.truncate(first);
        self.statements[loop_id.index()] = Some(counted);
        self.for_loops += 1;
        Ok(Some(counted))
    }

    /// Whether any node of the source expression tree reads the variable.
    fn expression_reads(&self, expression: S::Expr, variable: S::Var) -> Fallible<bool, B::Error> {
        self.any_expression_node(
            expression,
            |shape| matches!(shape, ExprShape::Variable(read) if *read == variable),
        )
    }

    /// Whether any statement of the source subtree is a `continue`. Nested
    /// loops are included, which keeps the check conservative.
    fn contains_continue(&self, ids: &[S::Stmt]) -> Fallible<bool, B::Error> {
        self.any_statement(ids, |shape| Ok(matches!(shape, StmtShape::Continue { .. })))
    }
}
