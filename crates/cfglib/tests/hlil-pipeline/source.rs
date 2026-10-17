//! A small source tree for authoring programs, and its compilation into any
//! `FunctionBuilder`.
//!
//! The tree names variables by string, so a program reads like source. It
//! compiles through the same builder protocol that the library drives, which
//! also makes every program a use of `FunctionBuilder`.

use cfglib::ir::hlil::{FunctionBuilder, Handler, HandlerKind, SwitchArm};

use crate::dialect::{Calc, Lit, Local, Op, Ty};

/// An expression of the source tree.
#[derive(Debug, Clone)]
pub(crate) enum E {
    Var(&'static str),
    Lit(Lit),
    Apply(Op, Vec<E>),
}

pub(crate) fn var(name: &'static str) -> E {
    E::Var(name)
}

pub(crate) fn int(value: i64) -> E {
    E::Lit(Lit::Int(value))
}

pub(crate) fn apply(op: Op, operands: Vec<E>) -> E {
    E::Apply(op, operands)
}

pub(crate) fn add(left: E, right: E) -> E {
    apply(Op::Add, vec![left, right])
}

pub(crate) fn sub(left: E, right: E) -> E {
    apply(Op::Sub, vec![left, right])
}

pub(crate) fn mul(left: E, right: E) -> E {
    apply(Op::Mul, vec![left, right])
}

pub(crate) fn div(left: E, right: E) -> E {
    apply(Op::Div, vec![left, right])
}

pub(crate) fn rem(left: E, right: E) -> E {
    apply(Op::Rem, vec![left, right])
}

pub(crate) fn lt(left: E, right: E) -> E {
    apply(Op::Lt, vec![left, right])
}

pub(crate) fn ge(left: E, right: E) -> E {
    apply(Op::Ge, vec![left, right])
}

pub(crate) fn eq(left: E, right: E) -> E {
    apply(Op::Eq, vec![left, right])
}

pub(crate) fn ne(left: E, right: E) -> E {
    apply(Op::Ne, vec![left, right])
}

pub(crate) fn pick(condition: E, when_true: E, when_false: E) -> E {
    apply(Op::Pick, vec![condition, when_true, when_false])
}

pub(crate) fn cell(index: E) -> E {
    apply(Op::Cell, vec![index])
}

pub(crate) fn caught() -> E {
    apply(Op::Caught, vec![])
}

/// A handler of the source tree.
#[derive(Debug, Clone)]
pub(crate) struct Catch {
    pub(crate) all: bool,
    pub(crate) binding: Option<&'static str>,
    pub(crate) body: Vec<S>,
}

/// A statement of the source tree.
#[derive(Debug, Clone)]
pub(crate) enum S {
    Set(&'static str, E),
    /// Stores the second expression into the memory cell the first names.
    Store(E, E),
    Do(E),
    If(E, Vec<S>, Vec<S>),
    While(E, Vec<S>),
    DoWhile(Vec<S>, E),
    Loop(Vec<S>),
    For(Vec<S>, Option<E>, Vec<S>, Vec<S>),
    Switch(E, Vec<(Vec<i64>, Vec<S>)>, Vec<S>),
    Break(Option<&'static str>),
    Continue(Option<&'static str>),
    Return(Vec<E>),
    Label(&'static str, Vec<S>),
    Goto(&'static str),
    Try(Vec<S>, Vec<Catch>, Vec<S>),
    Region(Op, Vec<E>, Vec<S>),
}

pub(crate) fn set(name: &'static str, value: E) -> S {
    S::Set(name, value)
}

pub(crate) fn emit(value: E) -> S {
    S::Do(apply(Op::Emit, vec![value]))
}

pub(crate) fn raise(code: E) -> S {
    S::Do(apply(Op::Raise, vec![code]))
}

pub(crate) fn ret(value: E) -> S {
    S::Return(vec![value])
}

pub(crate) fn if_then(condition: E, then_body: Vec<S>) -> S {
    S::If(condition, then_body, Vec::new())
}

pub(crate) fn catch_all(binding: Option<&'static str>, body: Vec<S>) -> Catch {
    Catch {
        all: true,
        binding,
        body,
    }
}

pub(crate) fn catch(binding: Option<&'static str>, body: Vec<S>) -> Catch {
    Catch {
        all: false,
        binding,
        body,
    }
}

/// One function of the source tree: integer parameters, typed locals, and a
/// body.
#[derive(Debug, Clone)]
pub(crate) struct Program {
    pub(crate) name: &'static str,
    pub(crate) parameters: Vec<&'static str>,
    pub(crate) locals: Vec<(&'static str, Ty)>,
    pub(crate) returns: Vec<Ty>,
    pub(crate) body: Vec<S>,
}

impl Program {
    /// A program that takes integers and returns one integer.
    pub(crate) fn new(
        name: &'static str,
        parameters: &[&'static str],
        locals: &[&'static str],
        body: Vec<S>,
    ) -> Self {
        Self {
            name,
            parameters: parameters.to_vec(),
            locals: locals.iter().map(|&name| (name, Ty::Int)).collect(),
            returns: vec![Ty::Int],
            body,
        }
    }

    /// Compiles the program through `builder`.
    ///
    /// # Errors
    ///
    /// Returns the builder's refusal to seal the function.
    pub(crate) fn build<B>(&self, builder: B) -> Result<B::Function, B::Error>
    where
        B: FunctionBuilder<Calc>,
    {
        let mut compiler = Compiler {
            program: self,
            builder,
            variables: Vec::new(),
        };
        for &name in &self.parameters {
            compiler.declare(name, Ty::Int);
        }
        for &(name, ty) in &self.locals {
            compiler.declare(name, ty);
        }
        let parameters: Vec<B::Var> = compiler
            .variables
            .iter()
            .take(self.parameters.len())
            .map(|&(_, variable, _)| variable)
            .collect();
        compiler.builder.set_signature(&parameters, &self.returns);
        let body = compiler.statements(&self.body);
        compiler.builder.finish(&body)
    }
}

struct Compiler<'p, B: FunctionBuilder<Calc>> {
    program: &'p Program,
    builder: B,
    variables: Vec<(&'static str, B::Var, Ty)>,
}

impl<B: FunctionBuilder<Calc>> Compiler<'_, B> {
    fn declare(&mut self, name: &'static str, ty: Ty) {
        let variable = self
            .builder
            .declare_variable(&Local::named(name), Some(&ty));
        self.variables.push((name, variable, ty));
    }

    fn lookup(&self, name: &str) -> (B::Var, Ty) {
        let Some(&(_, variable, ty)) = self
            .variables
            .iter()
            .find(|(declared, ..)| *declared == name)
        else {
            panic!(
                "program {} uses the undeclared variable {name}",
                self.program.name
            );
        };
        (variable, ty)
    }

    /// Builds the tree under `expr` operands first and returns its handle and type.
    fn expr(&mut self, expr: &E) -> (B::Expr, Ty) {
        match expr {
            E::Var(name) => {
                let (variable, ty) = self.lookup(name);
                (self.builder.read_variable(variable, &ty), ty)
            }
            E::Lit(literal) => {
                let ty = match literal {
                    Lit::Int(_) => Ty::Int,
                    Lit::Bool(_) => Ty::Bool,
                };
                (self.builder.constant(literal, &ty), ty)
            }
            E::Apply(op, operands) => {
                let built: Vec<(B::Expr, Ty)> =
                    operands.iter().map(|operand| self.expr(operand)).collect();
                let types: Vec<Ty> = built.iter().map(|&(_, ty)| ty).collect();
                let (_, result) = op.signature(&types);
                let handles: Vec<B::Expr> = built.into_iter().map(|(handle, _)| handle).collect();
                (self.builder.apply(op, &handles, &result), result)
            }
        }
    }

    fn handle(&mut self, expr: &E) -> B::Expr {
        self.expr(expr).0
    }

    fn statements(&mut self, statements: &[S]) -> Vec<B::Stmt> {
        statements
            .iter()
            .map(|statement| self.statement(statement))
            .collect()
    }

    fn statement(&mut self, statement: &S) -> B::Stmt {
        match statement {
            S::Set(name, value) => {
                let (variable, ty) = self.lookup(name);
                let target = self.builder.read_variable(variable, &ty);
                let value = self.handle(value);
                self.builder.assign(target, value)
            }
            S::Store(index, value) => {
                let (index, _) = self.expr(index);
                let target = self.builder.apply(&Op::Cell, &[index], &Ty::Int);
                let value = self.handle(value);
                self.builder.assign(target, value)
            }
            S::Do(expr) => {
                let expression = self.handle(expr);
                self.builder.expression_statement(expression)
            }
            S::If(condition, then_body, else_body) => {
                let condition = self.handle(condition);
                let then_body = self.statements(then_body);
                let else_body = self.statements(else_body);
                self.builder.if_(condition, &then_body, &else_body)
            }
            S::While(condition, body) => {
                let condition = self.handle(condition);
                let body = self.statements(body);
                self.builder.while_(condition, &body)
            }
            S::DoWhile(body, condition) => {
                let body = self.statements(body);
                let condition = self.handle(condition);
                self.builder.do_while(&body, condition)
            }
            S::Loop(body) => {
                let body = self.statements(body);
                self.builder.loop_(&body)
            }
            S::For(initializer, condition, update, body) => {
                let initializer = self.statements(initializer);
                let condition = condition.as_ref().map(|condition| self.handle(condition));
                let update = self.statements(update);
                let body = self.statements(body);
                self.builder.for_(&initializer, condition, &update, &body)
            }
            S::Switch(scrutinee, arms, default_body) => self.switch(scrutinee, arms, default_body),
            S::Break(label) => self.builder.break_(*label),
            S::Continue(label) => self.builder.continue_(*label),
            S::Return(values) => {
                let values: Vec<B::Expr> = values.iter().map(|value| self.handle(value)).collect();
                self.builder.return_(&values)
            }
            S::Label(label, body) => {
                let body = self.statements(body);
                self.builder.labeled(label, &body)
            }
            S::Goto(label) => self.builder.goto(label),
            S::Try(body, catches, finally_body) => self.try_(body, catches, finally_body),
            S::Region(op, operands, body) => {
                let operands: Vec<B::Expr> = operands
                    .iter()
                    .map(|operand| self.handle(operand))
                    .collect();
                let body = self.statements(body);
                self.builder.region(op, &operands, &body)
            }
        }
    }

    fn switch(
        &mut self,
        scrutinee: &E,
        arms: &[(Vec<i64>, Vec<S>)],
        default_body: &[S],
    ) -> B::Stmt {
        let scrutinee = self.handle(scrutinee);
        let built: Vec<(Vec<Lit>, Vec<B::Stmt>)> = arms
            .iter()
            .map(|(values, body)| {
                let values = values.iter().map(|&value| Lit::Int(value)).collect();
                (values, self.statements(body))
            })
            .collect();
        let default_body = self.statements(default_body);
        let arms: Vec<SwitchArm<'_, Calc, B::Stmt>> = built
            .iter()
            .map(|(values, body)| SwitchArm { values, body })
            .collect();
        self.builder.switch(scrutinee, &arms, &default_body)
    }

    fn try_(&mut self, body: &[S], catches: &[Catch], finally_body: &[S]) -> B::Stmt {
        let body = self.statements(body);
        let built: Vec<BuiltCatch<B>> = catches
            .iter()
            .map(|handler| BuiltCatch {
                all: handler.all,
                binding: handler.binding.map(|name| self.lookup(name).0),
                body: self.statements(&handler.body),
            })
            .collect();
        let finally_body = self.statements(finally_body);
        let handlers: Vec<Handler<'_, Calc, B::Stmt, B::Var>> = built
            .iter()
            .map(|handler| Handler {
                kind: if handler.all {
                    HandlerKind::CatchAll
                } else {
                    HandlerKind::Catch
                },
                binding: handler.binding,
                caught_types: &[],
                body: &handler.body,
            })
            .collect();
        self.builder.try_(&body, &handlers, &finally_body)
    }
}

/// A handler whose bodies are already built.
struct BuiltCatch<B: FunctionBuilder<Calc>> {
    all: bool,
    binding: Option<B::Var>,
    body: Vec<B::Stmt>,
}
