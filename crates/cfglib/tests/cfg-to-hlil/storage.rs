//! The structured side of the toy: plain `usize` handles into `Vec` arenas,
//! read through `FunctionView` and written through `FunctionBuilder`.

use cfglib::ir::hlil::{
    ExprShape, FunctionBuilder, FunctionView, Handler, HandlerKind, StmtShape, SwitchArm,
    VerificationReport, verify,
};

use crate::dialect::{Prim, Tiny};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExprNode {
    Var(usize),
    Lit(i64),
    Apply(Prim, Vec<usize>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Catcher {
    pub(crate) all: bool,
    pub(crate) filter: Vec<usize>,
    pub(crate) binding: Option<usize>,
    pub(crate) body: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StmtNode {
    Expression(usize),
    Assign(usize, usize),
    If(usize, Vec<usize>, Vec<usize>),
    While(usize, Vec<usize>),
    DoWhile(Vec<usize>, usize),
    Loop(Vec<usize>),
    For(Vec<usize>, Option<usize>, Vec<usize>, Vec<usize>),
    Switch(usize, Vec<(Vec<i64>, Vec<usize>)>, Vec<usize>),
    Break(Option<String>),
    Continue(Option<String>),
    Return(Vec<usize>),
    Labeled(String, Vec<usize>),
    Goto(String),
    Try(Vec<usize>, Vec<Catcher>, Vec<usize>),
    Region(Prim, Vec<usize>, Vec<usize>),
}

/// A sealed structured function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Hlil {
    pub(crate) names: Vec<&'static str>,
    pub(crate) exprs: Vec<ExprNode>,
    pub(crate) stmts: Vec<StmtNode>,
    pub(crate) body: Vec<usize>,
    pub(crate) parameters: Vec<usize>,
}

impl FunctionView<Tiny> for Hlil {
    type Stmt = usize;
    type Expr = usize;
    type Var = usize;

    fn stmt_count(&self) -> usize {
        self.stmts.len()
    }

    fn expr_count(&self) -> usize {
        self.exprs.len()
    }

    fn variable_count(&self) -> usize {
        self.names.len()
    }

    fn body(&self) -> &[usize] {
        &self.body
    }

    fn stmt(&self, stmt: usize) -> StmtShape<'_, Tiny, usize, usize> {
        match &self.stmts[stmt] {
            StmtNode::Expression(expr) => StmtShape::Expression(*expr),
            StmtNode::Assign(target, value) => StmtShape::Assign {
                target: *target,
                value: *value,
            },
            StmtNode::If(condition, then_body, else_body) => StmtShape::If {
                condition: *condition,
                then_body,
                else_body,
            },
            StmtNode::While(condition, body) => StmtShape::While {
                condition: *condition,
                body,
            },
            StmtNode::DoWhile(body, condition) => StmtShape::DoWhile {
                body,
                condition: *condition,
            },
            StmtNode::Loop(body) => StmtShape::Loop { body },
            StmtNode::For(initializer, condition, update, body) => StmtShape::For {
                initializer,
                condition: *condition,
                update,
                body,
            },
            StmtNode::Switch(scrutinee, _, default_body) => StmtShape::Switch {
                scrutinee: *scrutinee,
                default_body,
            },
            StmtNode::Break(label) => StmtShape::Break {
                label: label.as_deref(),
            },
            StmtNode::Continue(label) => StmtShape::Continue {
                label: label.as_deref(),
            },
            StmtNode::Return(values) => StmtShape::Return { values },
            StmtNode::Labeled(label, body) => StmtShape::Labeled { label, body },
            StmtNode::Goto(label) => StmtShape::Goto { label },
            StmtNode::Try(body, _, finally_body) => StmtShape::Try { body, finally_body },
            StmtNode::Region(op, operands, body) => StmtShape::Region { op, operands, body },
        }
    }

    fn switch_arms(
        &self,
        stmt: usize,
    ) -> impl ExactSizeIterator<Item = SwitchArm<'_, Tiny, usize>> + '_ {
        let arms: &[(Vec<i64>, Vec<usize>)] = match &self.stmts[stmt] {
            StmtNode::Switch(_, arms, _) => arms,
            _ => &[],
        };
        arms.iter().map(|(values, body)| SwitchArm { values, body })
    }

    fn handlers(
        &self,
        stmt: usize,
    ) -> impl ExactSizeIterator<Item = Handler<'_, Tiny, usize, usize>> + '_ {
        let handlers: &[Catcher] = match &self.stmts[stmt] {
            StmtNode::Try(_, handlers, _) => handlers,
            _ => &[],
        };
        handlers.iter().map(|handler| Handler {
            kind: if handler.all {
                HandlerKind::CatchAll
            } else {
                HandlerKind::Catch
            },
            binding: handler.binding,
            caught_types: &[],
            body: &handler.body,
        })
    }

    fn expr(&self, expr: usize) -> ExprShape<'_, Tiny, usize, usize> {
        match &self.exprs[expr] {
            ExprNode::Var(variable) => ExprShape::Variable(*variable),
            ExprNode::Lit(value) => ExprShape::Constant(value),
            ExprNode::Apply(op, operands) => ExprShape::Operation { op, operands },
        }
    }

    fn ty(&self, _: usize) -> &() {
        &()
    }

    fn variable(&self, variable: usize) -> &&'static str {
        &self.names[variable]
    }

    fn declared_type(&self, _: usize) -> Option<&()> {
        None
    }

    fn parameters(&self) -> &[usize] {
        &self.parameters
    }

    fn returns(&self) -> &[()] {
        &[()]
    }
}

/// Builds an [`Hlil`], refusing a function that does not verify.
#[derive(Debug, Default)]
pub(crate) struct Builder {
    function: Option<Hlil>,
}

impl Builder {
    pub(crate) fn new() -> Self {
        Self {
            function: Some(Hlil {
                names: Vec::new(),
                exprs: Vec::new(),
                stmts: Vec::new(),
                body: Vec::new(),
                parameters: Vec::new(),
            }),
        }
    }

    fn function(&mut self) -> &mut Hlil {
        self.function
            .as_mut()
            .expect("the function is present until it is sealed")
    }

    fn expr(&mut self, node: ExprNode) -> usize {
        let function = self.function();
        function.exprs.push(node);
        function.exprs.len() - 1
    }

    fn stmt(&mut self, node: StmtNode) -> usize {
        let function = self.function();
        function.stmts.push(node);
        function.stmts.len() - 1
    }
}

impl FunctionBuilder<Tiny> for Builder {
    type Stmt = usize;
    type Expr = usize;
    type Var = usize;
    type Function = Hlil;
    type Error = VerificationReport;

    fn declare_variable(&mut self, variable: &&'static str, _: Option<&()>) -> usize {
        let function = self.function();
        function.names.push(variable);
        function.names.len() - 1
    }

    fn set_signature(&mut self, parameters: &[usize], _: &[()]) {
        self.function().parameters = parameters.to_vec();
    }

    fn read_variable(&mut self, variable: usize, (): &()) -> usize {
        self.expr(ExprNode::Var(variable))
    }

    fn constant(&mut self, constant: &i64, (): &()) -> usize {
        self.expr(ExprNode::Lit(*constant))
    }

    fn apply(&mut self, op: &Prim, operands: &[usize], (): &()) -> usize {
        self.expr(ExprNode::Apply(*op, operands.to_vec()))
    }

    fn expression_statement(&mut self, expression: usize) -> usize {
        self.stmt(StmtNode::Expression(expression))
    }

    fn assign(&mut self, target: usize, value: usize) -> usize {
        self.stmt(StmtNode::Assign(target, value))
    }

    fn if_(&mut self, condition: usize, then_body: &[usize], else_body: &[usize]) -> usize {
        self.stmt(StmtNode::If(
            condition,
            then_body.to_vec(),
            else_body.to_vec(),
        ))
    }

    fn while_(&mut self, condition: usize, body: &[usize]) -> usize {
        self.stmt(StmtNode::While(condition, body.to_vec()))
    }

    fn do_while(&mut self, body: &[usize], condition: usize) -> usize {
        self.stmt(StmtNode::DoWhile(body.to_vec(), condition))
    }

    fn loop_(&mut self, body: &[usize]) -> usize {
        self.stmt(StmtNode::Loop(body.to_vec()))
    }

    fn for_(
        &mut self,
        initializer: &[usize],
        condition: Option<usize>,
        update: &[usize],
        body: &[usize],
    ) -> usize {
        self.stmt(StmtNode::For(
            initializer.to_vec(),
            condition,
            update.to_vec(),
            body.to_vec(),
        ))
    }

    fn switch(
        &mut self,
        scrutinee: usize,
        arms: &[SwitchArm<'_, Tiny, usize>],
        default_body: &[usize],
    ) -> usize {
        let arms = arms
            .iter()
            .map(|arm| (arm.values.to_vec(), arm.body.to_vec()))
            .collect();
        self.stmt(StmtNode::Switch(scrutinee, arms, default_body.to_vec()))
    }

    fn break_(&mut self, label: Option<&str>) -> usize {
        self.stmt(StmtNode::Break(label.map(str::to_owned)))
    }

    fn continue_(&mut self, label: Option<&str>) -> usize {
        self.stmt(StmtNode::Continue(label.map(str::to_owned)))
    }

    fn return_(&mut self, values: &[usize]) -> usize {
        self.stmt(StmtNode::Return(values.to_vec()))
    }

    fn labeled(&mut self, label: &str, body: &[usize]) -> usize {
        self.stmt(StmtNode::Labeled(label.to_owned(), body.to_vec()))
    }

    fn goto(&mut self, label: &str) -> usize {
        self.stmt(StmtNode::Goto(label.to_owned()))
    }

    fn try_(
        &mut self,
        body: &[usize],
        handlers: &[Handler<'_, Tiny, usize, usize>],
        finally_body: &[usize],
    ) -> usize {
        let handlers = handlers
            .iter()
            .map(|handler| Catcher {
                all: matches!(handler.kind, HandlerKind::CatchAll),
                filter: match handler.kind {
                    HandlerKind::Filter { filter_body } => filter_body.to_vec(),
                    _ => Vec::new(),
                },
                binding: handler.binding,
                body: handler.body.to_vec(),
            })
            .collect();
        self.stmt(StmtNode::Try(
            body.to_vec(),
            handlers,
            finally_body.to_vec(),
        ))
    }

    fn region(&mut self, op: &Prim, operands: &[usize], body: &[usize]) -> usize {
        self.stmt(StmtNode::Region(*op, operands.to_vec(), body.to_vec()))
    }

    fn finish(mut self, body: &[usize]) -> Result<Hlil, VerificationReport> {
        let mut function = self
            .function
            .take()
            .expect("the function is present until it is sealed");
        function.body = body.to_vec();
        let report = verify::<Tiny, _>(&function);
        if report.is_ok() {
            Ok(function)
        } else {
            Err(report)
        }
    }
}
