//! The reference interpreter of a structured function, written against
//! `FunctionView` alone.
//!
//! It is the oracle of the structured side. It never calls a library walk:
//! it moves control by the documented meaning of each shape, so a function
//! that `lift` or `recover_structure` produced is judged by what the source
//! language says, not by what the library believes.
//!
//! A `goto` finds its label in any enclosing statement list, and the walk
//! then resumes inside the labeled statement, entering the statements around
//! it without running their conditions. That is the C-style meaning of a
//! jump into a nested block, which the structured residue of irreducible
//! control flow needs.

use cfglib::DenseId;
use cfglib::ir::hlil::{ExprShape, FunctionView, HandlerKind, StmtShape};

use crate::dialect::{Calc, Lit, Op, Ty};
use crate::machine::{End, Machine, Outcome, Thrown, Value};

/// How control leaves a statement other than by completing it.
enum Flow<'f> {
    Break(Option<&'f str>),
    Continue(Option<&'f str>),
    Goto(&'f str),
    Return(Vec<Value>),
    Throw(i64),
    /// The step budget ran out.
    Halt,
}

type Step<'f> = Result<(), Flow<'f>>;

/// The parts of a loop statement that the loop driver needs.
struct LoopShape<'f, F: FunctionView<Calc>> {
    initializer: &'f [F::Stmt],
    condition: Option<F::Expr>,
    update: &'f [F::Stmt],
    body: &'f [F::Stmt],
    post_tested: bool,
}

/// Runs `function` on integer `inputs` and reports what it observes.
pub(crate) fn run_structured<F: FunctionView<Calc>>(function: &F, inputs: &[i64]) -> Outcome {
    let mut variables: Vec<Value> = (0..function.variable_count())
        .map(
            |index| match function.declared_type(F::Var::from_index(index)) {
                Some(Ty::Bool) => Value::Bool(false),
                _ => Value::Int(0),
            },
        )
        .collect();
    for (&parameter, &input) in function.parameters().iter().zip(inputs) {
        variables[parameter.index()] = Value::Int(input);
    }
    let mut interpreter = Interpreter {
        function,
        variables,
        machine: Machine::new(),
    };
    let end = match interpreter.exec_list(function.body(), &mut None) {
        Ok(()) => End::FellOff,
        Err(Flow::Return(values)) => End::Returned(values),
        Err(Flow::Throw(code)) => End::Uncaught(code),
        Err(Flow::Halt) => End::Diverged,
        Err(Flow::Break(_) | Flow::Continue(_) | Flow::Goto(_)) => {
            panic!("a transfer escaped the function: verify should have caught it")
        }
    };
    interpreter.machine.finish(end)
}

/// The child statement lists of `stmt`, in the order the shapes document.
pub(crate) fn bodies<F: FunctionView<Calc>>(function: &F, stmt: F::Stmt) -> Vec<&[F::Stmt]> {
    match function.stmt(stmt) {
        StmtShape::If {
            then_body,
            else_body,
            ..
        } => vec![then_body, else_body],
        StmtShape::While { body, .. }
        | StmtShape::DoWhile { body, .. }
        | StmtShape::Loop { body }
        | StmtShape::Labeled { body, .. }
        | StmtShape::Region { body, .. } => vec![body],
        StmtShape::For {
            initializer,
            update,
            body,
            ..
        } => vec![initializer, body, update],
        StmtShape::Switch { default_body, .. } => {
            let mut lists: Vec<&[F::Stmt]> =
                function.switch_arms(stmt).map(|arm| arm.body).collect();
            lists.push(default_body);
            lists
        }
        StmtShape::Try { body, finally_body } => {
            let mut lists = vec![body];
            for handler in function.handlers(stmt) {
                if let HandlerKind::Filter { filter_body } = handler.kind {
                    lists.push(filter_body);
                }
                lists.push(handler.body);
            }
            lists.push(finally_body);
            lists
        }
        _ => Vec::new(),
    }
}

/// Whether `stmt` is, or contains, the statement labeled `label`.
fn defines<F: FunctionView<Calc>>(function: &F, stmt: F::Stmt, label: &str) -> bool {
    if let StmtShape::Labeled { label: own, .. } = function.stmt(stmt)
        && own == label
    {
        return true;
    }
    bodies(function, stmt)
        .into_iter()
        .any(|list| defines_in(function, list, label))
}

fn defines_in<F: FunctionView<Calc>>(function: &F, list: &[F::Stmt], label: &str) -> bool {
    list.iter().any(|&stmt| defines(function, stmt, label))
}

struct Interpreter<'f, F: FunctionView<Calc>> {
    function: &'f F,
    variables: Vec<Value>,
    machine: Machine,
}

impl<'f, F: FunctionView<Calc>> Interpreter<'f, F> {
    fn eval(&mut self, expr: F::Expr) -> Result<Value, Flow<'f>> {
        if !self.machine.tick() {
            return Err(Flow::Halt);
        }
        let function = self.function;
        match function.expr(expr) {
            ExprShape::Variable(variable) => Ok(self.variables[variable.index()]),
            ExprShape::Constant(literal) => Ok(Value::of(*literal)),
            ExprShape::Operation {
                op: Op::Pick,
                operands,
            } => {
                let chosen = if self.eval(operands[0])?.truth() {
                    1
                } else {
                    2
                };
                self.eval(operands[chosen])
            }
            ExprShape::Operation { op, operands } => {
                let mut args = Vec::with_capacity(operands.len());
                for &operand in operands {
                    args.push(self.eval(operand)?);
                }
                self.machine
                    .apply(*op, &args)
                    .map_err(|Thrown(code)| Flow::Throw(code))
            }
        }
    }

    /// Runs the statements of one list. While `seek` names a label, the list
    /// skips every statement that does not contain it.
    fn exec_list(&mut self, list: &'f [F::Stmt], seek: &mut Option<&'f str>) -> Step<'f> {
        let function = self.function;
        let mut index = 0;
        while index < list.len() {
            let stmt = list[index];
            if let Some(label) = *seek
                && !defines(function, stmt, label)
            {
                index += 1;
                continue;
            }
            match self.exec_stmt(stmt, seek) {
                Ok(()) => index += 1,
                Err(Flow::Goto(label)) if defines_in(function, list, label) => {
                    *seek = Some(label);
                    index = 0;
                }
                Err(flow) => return Err(flow),
            }
        }
        Ok(())
    }

    fn exec_stmt(&mut self, stmt: F::Stmt, seek: &mut Option<&'f str>) -> Step<'f> {
        if !self.machine.tick() {
            return Err(Flow::Halt);
        }
        let function = self.function;
        match function.stmt(stmt) {
            StmtShape::Expression(expr) => self.eval(expr).map(drop),
            StmtShape::Assign { target, value } => self.assign(target, value),
            StmtShape::If {
                condition,
                then_body,
                else_body,
            } => {
                let take_then = match *seek {
                    Some(label) => defines_in(function, then_body, label),
                    None => self.eval(condition)?.truth(),
                };
                self.exec_list(if take_then { then_body } else { else_body }, seek)
            }
            StmtShape::While { .. }
            | StmtShape::DoWhile { .. }
            | StmtShape::Loop { .. }
            | StmtShape::For { .. } => self.exec_loop(stmt, None, seek),
            StmtShape::Switch {
                scrutinee,
                default_body,
            } => {
                let mut lists: Vec<(&[Lit], &'f [F::Stmt])> = function
                    .switch_arms(stmt)
                    .map(|arm| (arm.values, arm.body))
                    .collect();
                lists.push((&[], default_body));
                let chosen = if let Some(label) = *seek {
                    lists
                        .iter()
                        .find(|(_, body)| defines_in(function, body, label))
                        .map(|&(_, body)| body)
                        .expect("a sought label is inside one arm")
                } else {
                    let value = Lit::Int(self.eval(scrutinee)?.int());
                    lists
                        .iter()
                        .find(|(values, _)| values.contains(&value))
                        .map_or(default_body, |&(_, body)| body)
                };
                match self.exec_list(chosen, seek) {
                    Err(Flow::Break(None)) => Ok(()),
                    other => other,
                }
            }
            StmtShape::Break { label } => Err(Flow::Break(label)),
            StmtShape::Continue { label } => Err(Flow::Continue(label)),
            StmtShape::Return { values } => {
                let mut returned = Vec::with_capacity(values.len());
                for &value in values {
                    returned.push(self.eval(value)?);
                }
                Err(Flow::Return(returned))
            }
            StmtShape::Labeled { label, body } => {
                if *seek == Some(label) {
                    *seek = None;
                }
                let result = match body {
                    &[only] if is_loop(function, only) => self.exec_loop(only, Some(label), seek),
                    _ => self.exec_list(body, seek),
                };
                match result {
                    Err(Flow::Break(Some(target))) if target == label => Ok(()),
                    other => other,
                }
            }
            StmtShape::Goto { label } => Err(Flow::Goto(label)),
            StmtShape::Try { body, finally_body } => self.exec_try(stmt, body, finally_body, seek),
            StmtShape::Region { operands, body, .. } => {
                assert!(
                    seek.is_none(),
                    "this interpreter does not enter a region by goto"
                );
                let subject = self.eval(operands[0])?.int();
                self.machine.acquire(subject);
                let result = self.exec_list(body, &mut None);
                self.machine.release(subject);
                result
            }
        }
    }

    fn assign(&mut self, target: F::Expr, value: F::Expr) -> Step<'f> {
        let function = self.function;
        match function.expr(target) {
            ExprShape::Variable(variable) => {
                let value = self.eval(value)?;
                self.variables[variable.index()] = value;
            }
            ExprShape::Operation {
                op: Op::Cell,
                operands,
            } => {
                let cell = self.eval(operands[0])?.int();
                let value = self.eval(value)?.int();
                self.machine.store(cell, value);
            }
            _ => panic!("an assignment targets a variable or a memory cell"),
        }
        Ok(())
    }

    fn exec_try(
        &mut self,
        stmt: F::Stmt,
        body: &'f [F::Stmt],
        finally_body: &'f [F::Stmt],
        seek: &mut Option<&'f str>,
    ) -> Step<'f> {
        let function = self.function;
        if let Some(label) = *seek {
            assert!(
                defines_in(function, body, label),
                "this interpreter enters a try only through its body"
            );
        }
        let mut result = self.exec_list(body, seek);
        if let Err(Flow::Throw(code)) = result {
            let handler = function
                .handlers(stmt)
                .find(|handler| matches!(handler.kind, HandlerKind::Catch | HandlerKind::CatchAll));
            if let Some(handler) = handler {
                let outer = self.machine.exception;
                self.machine.exception = code;
                if let Some(binding) = handler.binding {
                    self.variables[binding.index()] = Value::Int(code);
                }
                result = self.exec_list(handler.body, &mut None);
                self.machine.exception = outer;
            }
        }
        if let Err(Flow::Goto(label)) = result
            && defines_in(function, finally_body, label)
        {
            // A jump from the body or a handler to a label inside the
            // finally body resumes there.
            self.exec_list(finally_body, &mut Some(label))?;
            return Ok(());
        }
        if !finally_body.is_empty() {
            self.exec_list(finally_body, &mut None)?;
        }
        result
    }

    /// Runs the loop statement `stmt`, which a `Labeled` statement may name.
    fn exec_loop(
        &mut self,
        stmt: F::Stmt,
        label: Option<&'f str>,
        seek: &mut Option<&'f str>,
    ) -> Step<'f> {
        let shape = match self.function.stmt(stmt) {
            StmtShape::While { condition, body } => LoopShape::<F> {
                initializer: &[],
                condition: Some(condition),
                update: &[],
                body,
                post_tested: false,
            },
            StmtShape::DoWhile { body, condition } => LoopShape::<F> {
                initializer: &[],
                condition: Some(condition),
                update: &[],
                body,
                post_tested: true,
            },
            StmtShape::Loop { body } => LoopShape::<F> {
                initializer: &[],
                condition: None,
                update: &[],
                body,
                post_tested: false,
            },
            StmtShape::For {
                initializer,
                condition,
                update,
                body,
            } => LoopShape::<F> {
                initializer,
                condition,
                update,
                body,
                post_tested: false,
            },
            _ => unreachable!("only a loop statement reaches the loop driver"),
        };
        if seek.is_none() {
            self.exec_list(shape.initializer, &mut None)?;
        }
        // A jump into the body resumes mid-iteration: no test first.
        let mut resuming = seek.is_some();
        loop {
            // An empty body costs a step too, so `loop {}` runs out of fuel.
            if !self.machine.tick() {
                return Err(Flow::Halt);
            }
            if !shape.post_tested
                && !resuming
                && let Some(condition) = shape.condition
                && !self.eval(condition)?.truth()
            {
                return Ok(());
            }
            resuming = false;
            match self.exec_list(shape.body, seek) {
                Ok(()) | Err(Flow::Continue(None)) => {}
                Err(Flow::Continue(Some(target))) if Some(target) == label => {}
                Err(Flow::Break(None)) => return Ok(()),
                Err(other) => return Err(other),
            }
            self.exec_list(shape.update, &mut None)?;
            if shape.post_tested
                && let Some(condition) = shape.condition
                && !self.eval(condition)?.truth()
            {
                return Ok(());
            }
        }
    }
}

fn is_loop<F: FunctionView<Calc>>(function: &F, stmt: F::Stmt) -> bool {
    matches!(
        function.stmt(stmt),
        StmtShape::While { .. }
            | StmtShape::DoWhile { .. }
            | StmtShape::Loop { .. }
            | StmtShape::For { .. }
    )
}

/// Every statement of `function` in source order: a statement, then the
/// statements of its child lists.
pub(crate) fn preorder<F: FunctionView<Calc>>(function: &F) -> Vec<F::Stmt> {
    let mut order = Vec::new();
    let mut pending: Vec<F::Stmt> = function.body().iter().rev().copied().collect();
    while let Some(stmt) = pending.pop() {
        order.push(stmt);
        for list in bodies(function, stmt).into_iter().rev() {
            pending.extend(list.iter().rev().copied());
        }
    }
    order
}
