//! Builds a [`Sketch`] into a [`ToyHlil`] of either toy dialect.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::ir::hlil::{FunctionBuilder, Handler, HandlerKind, SwitchArm};

use super::sketch::{Class, Ex, Sketch, Sn};
use super::{ExprId, Kit, StmtId, ToyHlil, ToyHlilBuilder, Ty, VarId};

/// Builds statements and expressions bottom-up, as the builder contract asks.
struct Cx<D: Kit> {
    builder: ToyHlilBuilder<D>,
    variables: Vec<VarId>,
}

impl<D: Kit> Cx<D> {
    fn variable(&self, number: u8) -> VarId {
        self.variables[usize::from(number)]
    }

    fn expr(&mut self, expr: &Ex) -> ExprId {
        match expr {
            Ex::Var(number) => {
                let variable = self.variable(*number);
                self.builder.read(variable)
            }
            Ex::Num(value) => self.builder.number(*value),
            Ex::Apply(prim, ty, operands) => {
                let operands = self.exprs(operands);
                self.builder.call(*prim, &operands, *ty)
            }
        }
    }

    fn exprs(&mut self, exprs: &[Ex]) -> Vec<ExprId> {
        exprs.iter().map(|expr| self.expr(expr)).collect()
    }

    fn body(&mut self, nodes: &[Sn]) -> Vec<StmtId> {
        nodes.iter().map(|node| self.stmt(node)).collect()
    }

    fn stmt(&mut self, node: &Sn) -> StmtId {
        match node {
            Sn::Eval(expr) => {
                let expr = self.expr(expr);
                self.builder.expression_statement(expr)
            }
            Sn::Assign(number, value) => {
                let value = self.expr(value);
                self.builder.store(self.variable(*number), value)
            }
            Sn::Put(prim, operands, value) => {
                let operands = self.exprs(operands);
                let place = self.builder.call(*prim, &operands, Ty::Integer);
                let value = self.expr(value);
                self.builder.assign(place, value)
            }
            Sn::If(condition, then_body, else_body) => {
                let condition = self.expr(condition);
                let then_body = self.body(then_body);
                let else_body = self.body(else_body);
                self.builder.if_(condition, &then_body, &else_body)
            }
            Sn::While(condition, body) => {
                let condition = self.expr(condition);
                let body = self.body(body);
                self.builder.while_(condition, &body)
            }
            Sn::DoWhile(body, condition) => {
                let body = self.body(body);
                let condition = self.expr(condition);
                self.builder.do_while(&body, condition)
            }
            Sn::Loop(body) => {
                let body = self.body(body);
                self.builder.loop_(&body)
            }
            Sn::For(initializer, condition, update, body) => {
                let initializer = self.body(initializer);
                let condition = condition.as_ref().map(|condition| self.expr(condition));
                let update = self.body(update);
                let body = self.body(body);
                self.builder.for_(&initializer, condition, &update, &body)
            }
            Sn::Switch(scrutinee, arms, default_body) => self.switch(scrutinee, arms, default_body),
            Sn::Break(label) => self.builder.break_(*label),
            Sn::Continue(label) => self.builder.continue_(*label),
            Sn::Return(values) => {
                let values = self.exprs(values);
                self.builder.return_(&values)
            }
            Sn::Labeled(label, body) => {
                let body = self.body(body);
                self.builder.labeled(label, &body)
            }
            Sn::Goto(label) => self.builder.goto(label),
            Sn::Try(body, handlers, finally_body) => self.try_(body, handlers, finally_body),
            Sn::Region(prim, operands, body) => {
                let operands = self.exprs(operands);
                let body = self.body(body);
                self.builder.region(&D::op(*prim), &operands, &body)
            }
        }
    }

    fn switch(
        &mut self,
        scrutinee: &Ex,
        arms: &[(Vec<i64>, Vec<Sn>)],
        default_body: &[Sn],
    ) -> StmtId {
        let scrutinee = self.expr(scrutinee);
        let arms: Vec<_> = arms
            .iter()
            .map(|(values, body)| {
                let values: Vec<_> = values.iter().map(|&value| D::constant(value)).collect();
                (values, self.body(body))
            })
            .collect();
        let views: Vec<_> = arms
            .iter()
            .map(|(values, body)| SwitchArm { values, body })
            .collect();
        let default_body = self.body(default_body);
        self.builder.switch(scrutinee, &views, &default_body)
    }

    fn try_(&mut self, body: &[Sn], handlers: &[super::sketch::Hn], finally_body: &[Sn]) -> StmtId {
        let body = self.body(body);
        let built: Vec<_> = handlers
            .iter()
            .map(|handler| {
                let filter = match &handler.class {
                    Class::Filter(filter) => self.body(filter),
                    _ => Vec::new(),
                };
                (filter, self.body(&handler.body))
            })
            .collect();
        let views: Vec<_> = handlers
            .iter()
            .zip(&built)
            .map(|(handler, (filter, body))| Handler {
                kind: match handler.class {
                    Class::Catch => HandlerKind::Catch,
                    Class::CatchAll => HandlerKind::CatchAll,
                    Class::Fault => HandlerKind::Fault,
                    Class::Filter(_) => HandlerKind::Filter {
                        filter_body: filter,
                    },
                },
                binding: handler.binding.map(|number| self.variable(number)),
                caught_types: &[],
                body,
            })
            .collect();
        let finally_body = self.body(finally_body);
        self.builder.try_(&body, &views, &finally_body)
    }
}

impl Sketch {
    /// The sketch as a function that went through the checked builder.
    pub(crate) fn build<D: Kit>(&self) -> ToyHlil<D> {
        let (cx, body) = self.assemble::<D>();
        cx.builder
            .finish(&body)
            .expect("a sketch is a valid function")
    }

    /// The sketch sealed without verification, so that a test can hand
    /// lowering a function that the checked builder would refuse.
    pub(crate) fn build_unchecked<D: Kit>(&self) -> ToyHlil<D> {
        let (cx, body) = self.assemble::<D>();
        cx.builder.finish_unchecked(&body)
    }

    fn assemble<D: Kit>(&self) -> (Cx<D>, Vec<StmtId>) {
        let mut builder = ToyHlilBuilder::<D>::new();
        let variables: Vec<VarId> = (0..self.variables)
            .map(|number| {
                if self.untyped.contains(&number) {
                    builder.declare_variable(&D::variable(number), None)
                } else {
                    builder.int_variable(number)
                }
            })
            .collect();
        let parameters: Vec<VarId> = self
            .parameters
            .iter()
            .map(|&number| variables[usize::from(number)])
            .collect();
        builder.set_signature(&parameters, &vec![D::ty(Ty::Integer); self.returns]);
        let mut cx = Cx { builder, variables };
        let body = cx.body(&self.body);
        (cx, body)
    }
}
