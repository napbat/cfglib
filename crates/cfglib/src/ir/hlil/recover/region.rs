//! Paired enter and exit region recovery: `enter(x); try { ...; exit(x) }
//! catch-any { exit(x); rethrow }` becomes one
//! [`Region`](StmtShape::Region) statement, the lowered shape of
//! `synchronized`, `lock`, and similar language constructs, whose source form
//! regenerates the release-and-rethrow cleanup exactly.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::walk::{Kind, Open};
use super::{
    ActiveRegion, ExprShape, Fallible, FunctionBuilder, FunctionView, Rebuilder, RecoverDialect,
    StmtShape,
};
use crate::ir::hlil::HandlerKind;

impl<'a, H, S, B> Rebuilder<'a, H, S, B>
where
    H: RecoverDialect,
    S: FunctionView<H>,
    B: FunctionBuilder<H>,
{
    /// Opens `ids[index]; ids[index + 1]` as one region statement, or
    /// returns `None` when they are not a region. The region is a statement
    /// once its body has been rebuilt.
    pub(super) fn try_region(
        &mut self,
        ids: &[S::Stmt],
        index: usize,
    ) -> Fallible<Option<Open<'a, H, S, B>>, B::Error> {
        let enter_id = ids[index];
        let guarded_id = ids[index + 1];
        let source: &'a S = self.source;
        let StmtShape::Expression(enter_expression) = self.stmt(enter_id)? else {
            return Ok(None);
        };
        let ExprShape::Operation {
            op: enter,
            operands,
        } = self.expr(enter_expression)?
        else {
            return Ok(None);
        };
        let Some(region_op) = H::region_enter(enter) else {
            return Ok(None);
        };
        let &[subject] = operands else {
            return Ok(None);
        };
        let ExprShape::Variable(entered) = self.expr(subject)? else {
            return Ok(None);
        };
        let StmtShape::Try { body, finally_body } = self.stmt(guarded_id)? else {
            return Ok(None);
        };
        let mut handlers = source.handlers(guarded_id);
        let (Some(handler), None) = (handlers.next(), handlers.next()) else {
            return Ok(None);
        };
        if !finally_body.is_empty() || !matches!(handler.kind, HandlerKind::CatchAll) {
            return Ok(None);
        }
        let Some(released) = self.cleanup_release(enter, handler.body)? else {
            return Ok(None);
        };
        // The released view must provably hold the entered object: the same
        // variable, or one copied from the other beforehand.
        if released != entered && !self.copies_link(&ids[..index], entered, released)? {
            return Ok(None);
        }
        // Neither view may be reassigned inside the guarded body.
        if self.assigns_variable(body, entered)?
            || (released != entered && self.assigns_variable(body, released)?)
        {
            return Ok(None);
        }

        let subject = self.rebuild_expression(subject)?;
        self.active_regions.push(ActiveRegion {
            enter: enter.clone(),
            aliases: BTreeSet::from([entered, released]),
            exits: Vec::new(),
        });
        Ok(Some(self.open(Kind::Guarded {
            enter_id,
            guarded_id,
            cleanup: handler.body,
            subject,
            region_op,
            body,
        })))
    }

    /// The variable that the cleanup handler releases before it rethrows the
    /// delivered exception, when the handler is exactly that cleanup.
    fn cleanup_release(
        &self,
        enter: &H::Op,
        body: &[S::Stmt],
    ) -> Fallible<Option<S::Var>, B::Error> {
        let (exit, throw, bound) = match *body {
            [exit, throw] => (exit, throw, None),
            [assign, exit, throw] => {
                let StmtShape::Assign { target, value } = self.stmt(assign)? else {
                    return Ok(None);
                };
                let ExprShape::Variable(bound) = self.expr(target)? else {
                    return Ok(None);
                };
                if !self.contains_exception_materialization(value)? {
                    return Ok(None);
                }
                (exit, throw, Some(bound))
            }
            _ => return Ok(None),
        };
        let StmtShape::Expression(exit) = self.stmt(exit)? else {
            return Ok(None);
        };
        let ExprShape::Operation { op, operands } = self.expr(exit)? else {
            return Ok(None);
        };
        if !H::releases(enter, op) {
            return Ok(None);
        }
        let &[released] = operands else {
            return Ok(None);
        };
        let ExprShape::Variable(released) = self.expr(released)? else {
            return Ok(None);
        };
        let StmtShape::Expression(throw) = self.stmt(throw)? else {
            return Ok(None);
        };
        let ExprShape::Operation {
            op: thrown_op,
            operands: thrown,
        } = self.expr(throw)?
        else {
            return Ok(None);
        };
        if !H::is_throw(thrown_op) {
            return Ok(None);
        }
        let &[thrown] = thrown else {
            return Ok(None);
        };
        let rethrows = match self.expr(thrown)? {
            ExprShape::Variable(variable) => Some(variable) == bound,
            _ => self.contains_exception_materialization(thrown)?,
        };
        Ok(rethrows.then_some(released))
    }

    /// Whether one expression tree materializes the delivered exception.
    fn contains_exception_materialization(&self, expression: S::Expr) -> Fallible<bool, B::Error> {
        self.any_expression_node(expression, |shape| {
            matches!(shape, ExprShape::Operation { op, .. } if H::is_exception_materialization(op))
        })
    }

    /// Whether the preceding statements copy one variable into the other.
    fn copies_link(
        &self,
        preceding: &[S::Stmt],
        first: S::Var,
        second: S::Var,
    ) -> Fallible<bool, B::Error> {
        for &id in preceding {
            let StmtShape::Assign { target, value } = self.stmt(id)? else {
                continue;
            };
            let ExprShape::Variable(target) = self.expr(target)? else {
                continue;
            };
            let ExprShape::Variable(value) = self.expr(value)? else {
                continue;
            };
            let pair = (target, value);
            if pair == (first, second) || pair == (second, first) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Whether any statement of the source subtree assigns the variable.
    fn assigns_variable(&self, ids: &[S::Stmt], variable: S::Var) -> Fallible<bool, B::Error> {
        self.any_statement(ids, |shape| {
            let StmtShape::Assign { target, .. } = shape else {
                return Ok(false);
            };
            Ok(matches!(
                self.expr(*target)?,
                ExprShape::Variable(assigned) if assigned == variable
            ))
        })
    }
}
