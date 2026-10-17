//! The consumer's structured storage: dense arenas of expressions and
//! statements, read through `FunctionView` and written through
//! `FunctionBuilder`.
//!
//! The fields are visible to the tests on purpose: a test corrupts a sealed
//! function by editing its arenas, which is how a buggy frontend would break
//! the contract that `verify` checks.

use cfglib::ir::hlil::{
    ExprShape, FunctionBuilder, FunctionView, Handler, HandlerKind, StmtShape, SwitchArm,
    VerificationReport, verify,
};
use cfglib::{Id, IdTag};

use crate::dialect::{Calc, Lit, Local, Op, Ty};

macro_rules! handle {
    ($(#[$doc:meta])* $tag:ident, $alias:ident, $prefix:literal) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub(crate) struct $tag;

        impl IdTag for $tag {
            const PREFIX: &'static str = $prefix;
        }

        pub(crate) type $alias = Id<$tag>;
    };
}

handle!(
    /// Marks a statement handle.
    StmtTag, StmtId, "s"
);
handle!(
    /// Marks an expression handle.
    ExprTag, ExprId, "e"
);
handle!(
    /// Marks a variable handle.
    VarTag, VarId, "v"
);

/// One expression node and the type of its value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Expr {
    pub(crate) node: ExprNode,
    pub(crate) ty: Ty,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExprNode {
    Var(VarId),
    Const(Lit),
    Op { op: Op, operands: Vec<ExprId> },
}

/// One switch arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Arm {
    pub(crate) values: Vec<Lit>,
    pub(crate) body: Vec<StmtId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CatchKind {
    Catch,
    CatchAll,
    Fault,
    Filter(Vec<StmtId>),
}

/// One handler of a `Try`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Catch {
    pub(crate) kind: CatchKind,
    pub(crate) binding: Option<VarId>,
    pub(crate) body: Vec<StmtId>,
}

/// One statement node. Bodies list their statements by handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StmtNode {
    Expression(ExprId),
    Assign {
        target: ExprId,
        value: ExprId,
    },
    If {
        condition: ExprId,
        then_body: Vec<StmtId>,
        else_body: Vec<StmtId>,
    },
    While {
        condition: ExprId,
        body: Vec<StmtId>,
    },
    DoWhile {
        body: Vec<StmtId>,
        condition: ExprId,
    },
    Loop {
        body: Vec<StmtId>,
    },
    For {
        initializer: Vec<StmtId>,
        condition: Option<ExprId>,
        update: Vec<StmtId>,
        body: Vec<StmtId>,
    },
    Switch {
        scrutinee: ExprId,
        arms: Vec<Arm>,
        default_body: Vec<StmtId>,
    },
    Break(Option<String>),
    Continue(Option<String>),
    Return(Vec<ExprId>),
    Labeled {
        label: String,
        body: Vec<StmtId>,
    },
    Goto(String),
    Try {
        body: Vec<StmtId>,
        handlers: Vec<Catch>,
        finally_body: Vec<StmtId>,
    },
    Region {
        op: Op,
        operands: Vec<ExprId>,
        body: Vec<StmtId>,
    },
}

/// One declared variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Var {
    pub(crate) local: Local,
    pub(crate) declared: Option<Ty>,
}

/// A sealed structured function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Function {
    pub(crate) vars: Vec<Var>,
    pub(crate) exprs: Vec<Expr>,
    pub(crate) stmts: Vec<StmtNode>,
    pub(crate) body: Vec<StmtId>,
    pub(crate) parameters: Vec<VarId>,
    pub(crate) returns: Vec<Ty>,
}

impl FunctionView<Calc> for Function {
    type Stmt = StmtId;
    type Expr = ExprId;
    type Var = VarId;

    fn stmt_count(&self) -> usize {
        self.stmts.len()
    }

    fn expr_count(&self) -> usize {
        self.exprs.len()
    }

    fn variable_count(&self) -> usize {
        self.vars.len()
    }

    fn body(&self) -> &[StmtId] {
        &self.body
    }

    fn stmt(&self, stmt: StmtId) -> StmtShape<'_, Calc, StmtId, ExprId> {
        match &self.stmts[stmt.index()] {
            StmtNode::Expression(expr) => StmtShape::Expression(*expr),
            StmtNode::Assign { target, value } => StmtShape::Assign {
                target: *target,
                value: *value,
            },
            StmtNode::If {
                condition,
                then_body,
                else_body,
            } => StmtShape::If {
                condition: *condition,
                then_body,
                else_body,
            },
            StmtNode::While { condition, body } => StmtShape::While {
                condition: *condition,
                body,
            },
            StmtNode::DoWhile { body, condition } => StmtShape::DoWhile {
                body,
                condition: *condition,
            },
            StmtNode::Loop { body } => StmtShape::Loop { body },
            StmtNode::For {
                initializer,
                condition,
                update,
                body,
            } => StmtShape::For {
                initializer,
                condition: *condition,
                update,
                body,
            },
            StmtNode::Switch {
                scrutinee,
                default_body,
                ..
            } => StmtShape::Switch {
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
            StmtNode::Labeled { label, body } => StmtShape::Labeled { label, body },
            StmtNode::Goto(label) => StmtShape::Goto { label },
            StmtNode::Try {
                body, finally_body, ..
            } => StmtShape::Try { body, finally_body },
            StmtNode::Region { op, operands, body } => StmtShape::Region { op, operands, body },
        }
    }

    fn switch_arms(
        &self,
        stmt: StmtId,
    ) -> impl ExactSizeIterator<Item = SwitchArm<'_, Calc, StmtId>> + '_ {
        let arms: &[Arm] = match &self.stmts[stmt.index()] {
            StmtNode::Switch { arms, .. } => arms,
            _ => &[],
        };
        arms.iter().map(|arm| SwitchArm {
            values: &arm.values,
            body: &arm.body,
        })
    }

    fn handlers(
        &self,
        stmt: StmtId,
    ) -> impl ExactSizeIterator<Item = Handler<'_, Calc, StmtId, VarId>> + '_ {
        let handlers: &[Catch] = match &self.stmts[stmt.index()] {
            StmtNode::Try { handlers, .. } => handlers,
            _ => &[],
        };
        handlers.iter().map(|handler| Handler {
            kind: match &handler.kind {
                CatchKind::Catch => HandlerKind::Catch,
                CatchKind::CatchAll => HandlerKind::CatchAll,
                CatchKind::Fault => HandlerKind::Fault,
                CatchKind::Filter(filter_body) => HandlerKind::Filter { filter_body },
            },
            binding: handler.binding,
            caught_types: &[],
            body: &handler.body,
        })
    }

    fn expr(&self, expr: ExprId) -> ExprShape<'_, Calc, ExprId, VarId> {
        match &self.exprs[expr.index()].node {
            ExprNode::Var(variable) => ExprShape::Variable(*variable),
            ExprNode::Const(literal) => ExprShape::Constant(literal),
            ExprNode::Op { op, operands } => ExprShape::Operation { op, operands },
        }
    }

    fn ty(&self, expr: ExprId) -> &Ty {
        &self.exprs[expr.index()].ty
    }

    fn variable(&self, variable: VarId) -> &Local {
        &self.vars[variable.index()].local
    }

    fn declared_type(&self, variable: VarId) -> Option<&Ty> {
        self.vars[variable.index()].declared.as_ref()
    }

    fn parameters(&self) -> &[VarId] {
        &self.parameters
    }

    fn returns(&self) -> &[Ty] {
        &self.returns
    }
}

impl Function {
    /// The variable that the source named `name`.
    pub(crate) fn variable_named(&self, name: &str) -> VarId {
        let index = self
            .vars
            .iter()
            .position(|variable| variable.local.name == name)
            .unwrap_or_else(|| panic!("no variable is named {name}"));
        VarId::from_index(index)
    }
}

/// Builds a [`Function`]. `finish` seals it and runs `verify`, so a builder
/// that is handed a function that breaks the contract refuses it. An
/// unchecked builder seals whatever it was given, which is how a test makes
/// the function of a sloppy frontend.
#[derive(Debug)]
pub(crate) struct Builder {
    function: Function,
    checked: bool,
}

impl Builder {
    pub(crate) fn new() -> Self {
        Self {
            function: Function {
                vars: Vec::new(),
                exprs: Vec::new(),
                stmts: Vec::new(),
                body: Vec::new(),
                parameters: Vec::new(),
                returns: Vec::new(),
            },
            checked: true,
        }
    }

    pub(crate) fn unchecked() -> Self {
        Self {
            checked: false,
            ..Self::new()
        }
    }

    fn function(&mut self) -> &mut Function {
        &mut self.function
    }

    /// Seals the function without checking it.
    pub(crate) fn seal(mut self, body: &[StmtId]) -> Function {
        self.function.body = body.to_vec();
        self.function
    }

    fn push_expr(&mut self, node: ExprNode, ty: Ty) -> ExprId {
        let function = self.function();
        function.exprs.push(Expr { node, ty });
        ExprId::from_index(function.exprs.len() - 1)
    }

    fn push_stmt(&mut self, node: StmtNode) -> StmtId {
        let function = self.function();
        function.stmts.push(node);
        StmtId::from_index(function.stmts.len() - 1)
    }
}

impl FunctionBuilder<Calc> for Builder {
    type Stmt = StmtId;
    type Expr = ExprId;
    type Var = VarId;
    type Function = Function;
    type Error = VerificationReport;

    fn declare_variable(&mut self, variable: &Local, declared_type: Option<&Ty>) -> VarId {
        let function = self.function();
        function.vars.push(Var {
            local: variable.clone(),
            declared: declared_type.copied(),
        });
        VarId::from_index(function.vars.len() - 1)
    }

    fn set_signature(&mut self, parameters: &[VarId], returns: &[Ty]) {
        let function = self.function();
        function.parameters = parameters.to_vec();
        function.returns = returns.to_vec();
    }

    fn read_variable(&mut self, variable: VarId, ty: &Ty) -> ExprId {
        self.push_expr(ExprNode::Var(variable), *ty)
    }

    fn constant(&mut self, constant: &Lit, ty: &Ty) -> ExprId {
        self.push_expr(ExprNode::Const(*constant), *ty)
    }

    fn apply(&mut self, op: &Op, operands: &[ExprId], ty: &Ty) -> ExprId {
        let node = ExprNode::Op {
            op: *op,
            operands: operands.to_vec(),
        };
        self.push_expr(node, *ty)
    }

    fn expression_statement(&mut self, expression: ExprId) -> StmtId {
        self.push_stmt(StmtNode::Expression(expression))
    }

    fn assign(&mut self, target: ExprId, value: ExprId) -> StmtId {
        self.push_stmt(StmtNode::Assign { target, value })
    }

    fn if_(&mut self, condition: ExprId, then_body: &[StmtId], else_body: &[StmtId]) -> StmtId {
        self.push_stmt(StmtNode::If {
            condition,
            then_body: then_body.to_vec(),
            else_body: else_body.to_vec(),
        })
    }

    fn while_(&mut self, condition: ExprId, body: &[StmtId]) -> StmtId {
        self.push_stmt(StmtNode::While {
            condition,
            body: body.to_vec(),
        })
    }

    fn do_while(&mut self, body: &[StmtId], condition: ExprId) -> StmtId {
        self.push_stmt(StmtNode::DoWhile {
            body: body.to_vec(),
            condition,
        })
    }

    fn loop_(&mut self, body: &[StmtId]) -> StmtId {
        self.push_stmt(StmtNode::Loop {
            body: body.to_vec(),
        })
    }

    fn for_(
        &mut self,
        initializer: &[StmtId],
        condition: Option<ExprId>,
        update: &[StmtId],
        body: &[StmtId],
    ) -> StmtId {
        self.push_stmt(StmtNode::For {
            initializer: initializer.to_vec(),
            condition,
            update: update.to_vec(),
            body: body.to_vec(),
        })
    }

    fn switch(
        &mut self,
        scrutinee: ExprId,
        arms: &[SwitchArm<'_, Calc, StmtId>],
        default_body: &[StmtId],
    ) -> StmtId {
        self.push_stmt(StmtNode::Switch {
            scrutinee,
            arms: arms
                .iter()
                .map(|arm| Arm {
                    values: arm.values.to_vec(),
                    body: arm.body.to_vec(),
                })
                .collect(),
            default_body: default_body.to_vec(),
        })
    }

    fn break_(&mut self, label: Option<&str>) -> StmtId {
        self.push_stmt(StmtNode::Break(label.map(str::to_owned)))
    }

    fn continue_(&mut self, label: Option<&str>) -> StmtId {
        self.push_stmt(StmtNode::Continue(label.map(str::to_owned)))
    }

    fn return_(&mut self, values: &[ExprId]) -> StmtId {
        self.push_stmt(StmtNode::Return(values.to_vec()))
    }

    fn labeled(&mut self, label: &str, body: &[StmtId]) -> StmtId {
        self.push_stmt(StmtNode::Labeled {
            label: label.to_owned(),
            body: body.to_vec(),
        })
    }

    fn goto(&mut self, label: &str) -> StmtId {
        self.push_stmt(StmtNode::Goto(label.to_owned()))
    }

    fn try_(
        &mut self,
        body: &[StmtId],
        handlers: &[Handler<'_, Calc, StmtId, VarId>],
        finally_body: &[StmtId],
    ) -> StmtId {
        let handlers = handlers
            .iter()
            .map(|handler| Catch {
                kind: match handler.kind {
                    HandlerKind::Catch => CatchKind::Catch,
                    HandlerKind::CatchAll => CatchKind::CatchAll,
                    HandlerKind::Fault => CatchKind::Fault,
                    HandlerKind::Filter { filter_body } => CatchKind::Filter(filter_body.to_vec()),
                },
                binding: handler.binding,
                body: handler.body.to_vec(),
            })
            .collect();
        self.push_stmt(StmtNode::Try {
            body: body.to_vec(),
            handlers,
            finally_body: finally_body.to_vec(),
        })
    }

    fn region(&mut self, op: &Op, operands: &[ExprId], body: &[StmtId]) -> StmtId {
        self.push_stmt(StmtNode::Region {
            op: *op,
            operands: operands.to_vec(),
            body: body.to_vec(),
        })
    }

    fn finish(self, body: &[StmtId]) -> Result<Function, VerificationReport> {
        let checked = self.checked;
        let function = self.seal(body);
        let report = verify::<Calc, _>(&function);
        if report.is_ok() || !checked {
            Ok(function)
        } else {
            Err(report)
        }
    }
}
