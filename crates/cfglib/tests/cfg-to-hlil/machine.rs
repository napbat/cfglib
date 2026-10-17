//! Reference interpreters for the source CFG and for the structured
//! function, so that a lift is judged by what the two programs observe.
//!
//! The structured one follows the documented meaning of each shape. A `goto`
//! finds its label in any enclosing statement list and resumes inside the
//! labeled statement, entering the statements around it without running
//! their conditions: the C meaning of a jump into a nested block, which the
//! residue of an irreducible graph needs.

use cfglib::ir::hlil::{ExprShape, FunctionView, HandlerKind as ShapeHandler, StmtShape};
use cfglib::{BlockId, Cfg, HandlerKind};

use crate::dialect::{Form, Link, Prim, Stmt};
use crate::storage::Hlil;

/// What an execution observes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Run {
    pub(crate) output: Vec<i64>,
    pub(crate) end: End,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum End {
    Returned(i64),
    Uncaught(i64),
    FellOff,
    Diverged,
}

const FUEL: u32 = 100_000;

/// The semantics of one primitive over evaluated operands.
fn apply(prim: Prim, args: &[i64], output: &mut Vec<i64>) -> Result<i64, i64> {
    Ok(match prim {
        Prim::Plus => args[0].wrapping_add(args[1]),
        Prim::Less => i64::from(args[0] < args[1]),
        Prim::Show => {
            output.push(args[0]);
            0
        }
        Prim::Quot if args[1] == 0 => return Err(-1),
        Prim::Quot => args[0].wrapping_div(args[1]),
    })
}

/// Runs the source CFG from its entry block.
pub(crate) fn run_cfg(
    cfg: &Cfg<Stmt, Link>,
    variable_count: usize,
    parameters: &[usize],
    inputs: &[i64],
) -> Run {
    let mut variables = vec![0_i64; variable_count];
    for (&parameter, &input) in parameters.iter().zip(inputs) {
        variables[parameter] = input;
    }
    let mut output = Vec::new();
    let mut block = cfg.entry();
    let mut fuel = FUEL;
    let end = 'run: loop {
        let mut next = None;
        for stmt in cfg.block(block).instructions() {
            fuel -= 1;
            if fuel == 0 {
                break 'run End::Diverged;
            }
            let args: Vec<i64> = stmt.uses.iter().map(|&used| variables[used]).collect();
            match stmt.form {
                Form::Const(value) => variables[stmt.defs[0]] = value,
                Form::Apply(prim) => match apply(prim, &args, &mut output) {
                    Ok(value) => {
                        if let Some(&definition) = stmt.defs.first() {
                            variables[definition] = value;
                        }
                    }
                    Err(code) => match handler_of(cfg, block) {
                        Some(entry) => {
                            next = Some(entry);
                            break;
                        }
                        None => break 'run End::Uncaught(code),
                    },
                },
                Form::Test => {
                    let wanted = if args[0] != 0 {
                        Link::True
                    } else {
                        Link::False
                    };
                    next = target(cfg, block, |link| *link == wanted);
                }
                Form::Dispatch => {
                    next = target(
                        cfg,
                        block,
                        |link| matches!(link, Link::Case(values) if values.contains(&args[0])),
                    )
                    .or_else(|| target(cfg, block, |link| *link == Link::Default));
                }
                Form::Ret => break 'run End::Returned(args[0]),
            }
        }
        match next.or_else(|| target(cfg, block, |link| !matches!(link, Link::Unwind))) {
            Some(following) => block = following,
            None => break End::FellOff,
        }
    };
    Run { output, end }
}

fn target(
    cfg: &Cfg<Stmt, Link>,
    block: BlockId,
    wanted: impl Fn(&Link) -> bool,
) -> Option<BlockId> {
    cfg.outgoing(block)
        .map(|id| cfg.edge(id))
        .find(|edge| wanted(edge.payload()))
        .map(|edge| edge.target())
}

/// The handler that catches an exception thrown in `block`: the first one
/// that is not a `finally` of the innermost region that protects it.
fn handler_of(cfg: &Cfg<Stmt, Link>, block: BlockId) -> Option<BlockId> {
    let region = cfg
        .regions()
        .iter()
        .filter(|region| region.protected_blocks.contains(&block))
        .min_by_key(|region| region.protected_blocks.len())?;
    let handler = region
        .handlers
        .iter()
        .find(|handler| !matches!(handler.kind, HandlerKind::Finally))
        .or_else(|| region.handlers.first())?;
    Some(handler.entry)
}

/// How control leaves a statement other than by completing it.
enum Flow<'f> {
    Break(Option<&'f str>),
    Continue(Option<&'f str>),
    Goto(&'f str),
    Return(i64),
    Throw(i64),
    Halt,
}

type Step<'f> = Result<(), Flow<'f>>;

struct Machine<'f> {
    function: &'f Hlil,
    variables: Vec<i64>,
    output: Vec<i64>,
    fuel: u32,
}

/// Runs the structured function on integer `inputs`.
pub(crate) fn run_hlil(function: &Hlil, inputs: &[i64]) -> Run {
    let mut variables = vec![0_i64; function.names.len()];
    for (&parameter, &input) in function.parameters.iter().zip(inputs) {
        variables[parameter] = input;
    }
    let mut machine = Machine {
        function,
        variables,
        output: Vec::new(),
        fuel: FUEL,
    };
    let end = match machine.list(function.body(), &mut None) {
        Ok(()) => End::FellOff,
        Err(Flow::Return(value)) => End::Returned(value),
        Err(Flow::Throw(code)) => End::Uncaught(code),
        Err(Flow::Halt) => End::Diverged,
        Err(Flow::Break(_) | Flow::Continue(_) | Flow::Goto(_)) => panic!("a transfer escaped"),
    };
    Run {
        output: machine.output,
        end,
    }
}

/// Whether the statement is, or contains, the statement labeled `label`.
fn defines(function: &Hlil, stmt: usize, label: &str) -> bool {
    match function.stmt(stmt) {
        StmtShape::Labeled { label: own, body } => {
            own == label || defines_in(function, body, label)
        }
        StmtShape::If {
            then_body,
            else_body,
            ..
        } => defines_in(function, then_body, label) || defines_in(function, else_body, label),
        StmtShape::While { body, .. }
        | StmtShape::DoWhile { body, .. }
        | StmtShape::Loop { body }
        | StmtShape::Region { body, .. }
        | StmtShape::For { body, .. } => defines_in(function, body, label),
        StmtShape::Switch { default_body, .. } => {
            defines_in(function, default_body, label)
                || function
                    .switch_arms(stmt)
                    .any(|arm| defines_in(function, arm.body, label))
        }
        StmtShape::Try { body, finally_body } => {
            defines_in(function, body, label)
                || defines_in(function, finally_body, label)
                || function
                    .handlers(stmt)
                    .any(|handler| defines_in(function, handler.body, label))
        }
        _ => false,
    }
}

fn defines_in(function: &Hlil, list: &[usize], label: &str) -> bool {
    list.iter().any(|&stmt| defines(function, stmt, label))
}

impl<'f> Machine<'f> {
    fn spend(&mut self) -> Step<'f> {
        self.fuel -= 1;
        if self.fuel == 0 {
            Err(Flow::Halt)
        } else {
            Ok(())
        }
    }

    fn eval(&mut self, expr: usize) -> Result<i64, Flow<'f>> {
        self.spend()?;
        let function = self.function;
        match function.expr(expr) {
            ExprShape::Variable(variable) => Ok(self.variables[variable]),
            ExprShape::Constant(value) => Ok(*value),
            ExprShape::Operation { op, operands } => {
                let mut args = Vec::new();
                for &operand in operands {
                    args.push(self.eval(operand)?);
                }
                apply(*op, &args, &mut self.output).map_err(Flow::Throw)
            }
        }
    }

    fn list(&mut self, list: &'f [usize], seek: &mut Option<&'f str>) -> Step<'f> {
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
            match self.stmt(stmt, seek) {
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

    fn stmt(&mut self, stmt: usize, seek: &mut Option<&'f str>) -> Step<'f> {
        self.spend()?;
        let function = self.function;
        match function.stmt(stmt) {
            StmtShape::Expression(expr) => self.eval(expr).map(drop),
            StmtShape::Assign { target, value } => {
                let ExprShape::Variable(variable) = function.expr(target) else {
                    panic!("the toy assigns variables only");
                };
                self.variables[variable] = self.eval(value)?;
                Ok(())
            }
            StmtShape::If {
                condition,
                then_body,
                else_body,
            } => {
                let take_then = match *seek {
                    Some(label) => defines_in(function, then_body, label),
                    None => self.eval(condition)? != 0,
                };
                self.list(if take_then { then_body } else { else_body }, seek)
            }
            StmtShape::While { .. }
            | StmtShape::DoWhile { .. }
            | StmtShape::Loop { .. }
            | StmtShape::For { .. } => self.repeat(stmt, None, seek),
            StmtShape::Switch {
                scrutinee,
                default_body,
            } => {
                let mut arms: Vec<(&[i64], &'f [usize])> = function
                    .switch_arms(stmt)
                    .map(|arm| (arm.values, arm.body))
                    .collect();
                arms.push((&[], default_body));
                let chosen = if let Some(label) = *seek {
                    arms.iter()
                        .find(|(_, body)| defines_in(function, body, label))
                        .map(|&(_, body)| body)
                        .expect("a sought label is inside one arm")
                } else {
                    let value = self.eval(scrutinee)?;
                    arms.iter()
                        .find(|(values, _)| values.contains(&value))
                        .map_or(default_body, |&(_, body)| body)
                };
                match self.list(chosen, seek) {
                    Err(Flow::Break(None)) => Ok(()),
                    other => other,
                }
            }
            StmtShape::Break { label } => Err(Flow::Break(label)),
            StmtShape::Continue { label } => Err(Flow::Continue(label)),
            StmtShape::Return { values } => Err(Flow::Return(self.eval(values[0])?)),
            StmtShape::Labeled { label, body } => {
                if *seek == Some(label) {
                    *seek = None;
                }
                let result = match body {
                    &[only]
                        if matches!(
                            function.stmt(only),
                            StmtShape::While { .. }
                                | StmtShape::DoWhile { .. }
                                | StmtShape::Loop { .. }
                                | StmtShape::For { .. }
                        ) =>
                    {
                        self.repeat(only, Some(label), seek)
                    }
                    _ => self.list(body, seek),
                };
                match result {
                    Err(Flow::Break(Some(target))) if target == label => Ok(()),
                    other => other,
                }
            }
            StmtShape::Goto { label } => Err(Flow::Goto(label)),
            StmtShape::Try { body, finally_body } => self.guarded(stmt, body, finally_body, seek),
            StmtShape::Region { .. } => panic!("the toy has no regions"),
        }
    }

    fn guarded(
        &mut self,
        stmt: usize,
        body: &'f [usize],
        finally_body: &'f [usize],
        seek: &mut Option<&'f str>,
    ) -> Step<'f> {
        let function = self.function;
        let mut result = self.list(body, seek);
        if let Err(Flow::Throw(_)) = result
            && let Some(handler) = function.handlers(stmt).find(|handler| {
                matches!(handler.kind, ShapeHandler::Catch | ShapeHandler::CatchAll)
            })
        {
            result = self.list(handler.body, &mut None);
        }
        if let Err(Flow::Goto(label)) = result
            && defines_in(function, finally_body, label)
        {
            self.list(finally_body, &mut Some(label))?;
            return Ok(());
        }
        self.list(finally_body, &mut None)?;
        result
    }

    /// Runs the loop statement `stmt`, which a labeled statement may name.
    fn repeat(
        &mut self,
        stmt: usize,
        label: Option<&'f str>,
        seek: &mut Option<&'f str>,
    ) -> Step<'f> {
        let function = self.function;
        let (initializer, condition, update, body, post_tested): (
            &[usize],
            Option<usize>,
            &[usize],
            &[usize],
            bool,
        ) = match function.stmt(stmt) {
            StmtShape::While { condition, body } => (&[], Some(condition), &[], body, false),
            StmtShape::DoWhile { body, condition } => (&[], Some(condition), &[], body, true),
            StmtShape::Loop { body } => (&[], None, &[], body, false),
            StmtShape::For {
                initializer,
                condition,
                update,
                body,
            } => (initializer, condition, update, body, false),
            _ => unreachable!("only a loop reaches the loop driver"),
        };
        if seek.is_none() {
            self.list(initializer, &mut None)?;
        }
        let mut resuming = seek.is_some();
        loop {
            if !post_tested
                && !resuming
                && let Some(condition) = condition
                && self.eval(condition)? == 0
            {
                return Ok(());
            }
            resuming = false;
            match self.list(body, seek) {
                Ok(()) | Err(Flow::Continue(None)) => {}
                Err(Flow::Continue(Some(target))) if Some(target) == label => {}
                Err(Flow::Break(None)) => return Ok(()),
                Err(other) => return Err(other),
            }
            self.list(update, &mut None)?;
            if post_tested
                && let Some(condition) = condition
                && self.eval(condition)? == 0
            {
                return Ok(());
            }
        }
    }
}
