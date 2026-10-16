//! `Vec`-backed structured storage for the toy dialects.
//!
//! The storage keeps statements, expressions, and variables in owned vectors
//! and lends them back through [`FunctionView`]. It shares no code with any
//! production consumer, so a test that passes over it cannot depend on one
//! consumer's layout.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use crate::graph::view::DenseId;
use crate::ir::hlil::{
    Dialect, ExprShape, FunctionBuilder, FunctionView, Handler, HandlerKind, StmtShape, SwitchArm,
    VerificationReport, VerifyDialect, verify,
};

macro_rules! dense_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub(crate) struct $name(pub(crate) u32);

        impl DenseId for $name {
            fn from_index(index: usize) -> Self {
                Self(u32::try_from(index).unwrap())
            }

            fn index(self) -> usize {
                usize::try_from(self.0).unwrap()
            }
        }
    };
}

dense_id!(
    /// A statement of a toy function.
    StmtId
);
dense_id!(
    /// An expression of a toy function.
    ExprId
);
dense_id!(
    /// A variable of a toy function.
    VarId
);

/// One expression node and the type of its occurrence.
#[derive(Debug)]
struct ExprNode<D: Dialect> {
    kind: ExprKind<D>,
    ty: D::Type,
}

#[derive(Debug)]
enum ExprKind<D: Dialect> {
    Variable(VarId),
    Constant(D::Constant),
    Operation { op: D::Op, operands: Vec<ExprId> },
}

#[derive(Debug)]
struct ArmNode<D: Dialect> {
    values: Vec<D::Constant>,
    body: Vec<StmtId>,
}

#[derive(Debug)]
enum HandlerClass {
    Catch,
    CatchAll,
    Fault,
    Filter(Vec<StmtId>),
}

#[derive(Debug)]
struct HandlerNode<D: Dialect> {
    class: HandlerClass,
    binding: Option<VarId>,
    caught_types: Vec<D::Type>,
    body: Vec<StmtId>,
}

#[derive(Debug)]
enum StmtNode<D: Dialect> {
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
        arms: Vec<ArmNode<D>>,
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
        handlers: Vec<HandlerNode<D>>,
        finally_body: Vec<StmtId>,
    },
    Region {
        op: D::Op,
        operands: Vec<ExprId>,
        body: Vec<StmtId>,
    },
}

/// A structured function in `Vec`-backed storage.
#[derive(Debug)]
pub(crate) struct ToyHlil<D: Dialect> {
    variables: Vec<(D::Variable, Option<D::Type>)>,
    parameters: Vec<VarId>,
    returns: Vec<D::Type>,
    exprs: Vec<ExprNode<D>>,
    stmts: Vec<StmtNode<D>>,
    body: Vec<StmtId>,
}

impl<D: Dialect> FunctionView<D> for ToyHlil<D> {
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
        self.variables.len()
    }

    fn body(&self) -> &[StmtId] {
        &self.body
    }

    fn stmt(&self, stmt: StmtId) -> StmtShape<'_, D, StmtId, ExprId> {
        match &self.stmts[stmt.index()] {
            &StmtNode::Expression(expression) => StmtShape::Expression(expression),
            &StmtNode::Assign { target, value } => StmtShape::Assign { target, value },
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
    ) -> impl ExactSizeIterator<Item = SwitchArm<'_, D, StmtId>> + '_ {
        let arms: &[ArmNode<D>] = match &self.stmts[stmt.index()] {
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
    ) -> impl ExactSizeIterator<Item = Handler<'_, D, StmtId, VarId>> + '_ {
        let handlers: &[HandlerNode<D>] = match &self.stmts[stmt.index()] {
            StmtNode::Try { handlers, .. } => handlers,
            _ => &[],
        };
        handlers.iter().map(|handler| Handler {
            kind: match &handler.class {
                HandlerClass::Catch => HandlerKind::Catch,
                HandlerClass::CatchAll => HandlerKind::CatchAll,
                HandlerClass::Fault => HandlerKind::Fault,
                HandlerClass::Filter(filter_body) => HandlerKind::Filter { filter_body },
            },
            binding: handler.binding,
            caught_types: &handler.caught_types,
            body: &handler.body,
        })
    }

    fn expr(&self, expr: ExprId) -> ExprShape<'_, D, ExprId, VarId> {
        match &self.exprs[expr.index()].kind {
            &ExprKind::Variable(variable) => ExprShape::Variable(variable),
            ExprKind::Constant(constant) => ExprShape::Constant(constant),
            ExprKind::Operation { op, operands } => ExprShape::Operation { op, operands },
        }
    }

    fn ty(&self, expr: ExprId) -> &D::Type {
        &self.exprs[expr.index()].ty
    }

    fn variable(&self, variable: VarId) -> &D::Variable {
        &self.variables[variable.index()].0
    }

    fn declared_type(&self, variable: VarId) -> Option<&D::Type> {
        self.variables[variable.index()].1.as_ref()
    }

    fn parameters(&self) -> &[VarId] {
        &self.parameters
    }

    fn returns(&self) -> &[D::Type] {
        &self.returns
    }
}

/// Builds a [`ToyHlil`] bottom-up and verifies it when it is sealed.
///
/// The builder is write-only. A handle is a plain index, so nothing stops a
/// test from using one twice or from inventing one; [`finish`] reports such a
/// function through [`verify`].
///
/// [`finish`]: FunctionBuilder::finish
pub(crate) struct ToyHlilBuilder<D: Dialect> {
    function: ToyHlil<D>,
}

impl<D: VerifyDialect> ToyHlilBuilder<D> {
    /// Creates an empty builder.
    pub(crate) const fn new() -> Self {
        Self {
            function: ToyHlil {
                variables: Vec::new(),
                parameters: Vec::new(),
                returns: Vec::new(),
                exprs: Vec::new(),
                stmts: Vec::new(),
                body: Vec::new(),
            },
        }
    }

    fn push_expr(&mut self, kind: ExprKind<D>, ty: &D::Type) -> ExprId {
        let id = ExprId::from_index(self.function.exprs.len());
        self.function.exprs.push(ExprNode {
            kind,
            ty: ty.clone(),
        });
        id
    }

    fn push_stmt(&mut self, node: StmtNode<D>) -> StmtId {
        let id = StmtId::from_index(self.function.stmts.len());
        self.function.stmts.push(node);
        id
    }

    /// Seals the function without verifying it, so a test can hand a
    /// malformed function to an algorithm.
    pub(crate) fn finish_unchecked(mut self, body: &[StmtId]) -> ToyHlil<D> {
        self.function.body = body.to_vec();
        self.function
    }
}

impl<D: VerifyDialect> FunctionBuilder<D> for ToyHlilBuilder<D> {
    type Stmt = StmtId;
    type Expr = ExprId;
    type Var = VarId;
    type Function = ToyHlil<D>;
    type Error = VerificationReport;

    fn declare_variable(
        &mut self,
        variable: &D::Variable,
        declared_type: Option<&D::Type>,
    ) -> VarId {
        let id = VarId::from_index(self.function.variables.len());
        self.function
            .variables
            .push((variable.clone(), declared_type.cloned()));
        id
    }

    fn set_signature(&mut self, parameters: &[VarId], returns: &[D::Type]) {
        self.function.parameters = parameters.to_vec();
        self.function.returns = returns.to_vec();
    }

    fn read_variable(&mut self, variable: VarId, ty: &D::Type) -> ExprId {
        self.push_expr(ExprKind::Variable(variable), ty)
    }

    fn constant(&mut self, constant: &D::Constant, ty: &D::Type) -> ExprId {
        self.push_expr(ExprKind::Constant(constant.clone()), ty)
    }

    fn apply(&mut self, op: &D::Op, operands: &[ExprId], ty: &D::Type) -> ExprId {
        let kind = ExprKind::Operation {
            op: op.clone(),
            operands: operands.to_vec(),
        };
        self.push_expr(kind, ty)
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
        arms: &[SwitchArm<'_, D, StmtId>],
        default_body: &[StmtId],
    ) -> StmtId {
        let arms = arms
            .iter()
            .map(|arm| ArmNode {
                values: arm.values.to_vec(),
                body: arm.body.to_vec(),
            })
            .collect();
        self.push_stmt(StmtNode::Switch {
            scrutinee,
            arms,
            default_body: default_body.to_vec(),
        })
    }

    fn break_(&mut self, label: Option<&str>) -> StmtId {
        self.push_stmt(StmtNode::Break(label.map(String::from)))
    }

    fn continue_(&mut self, label: Option<&str>) -> StmtId {
        self.push_stmt(StmtNode::Continue(label.map(String::from)))
    }

    fn return_(&mut self, values: &[ExprId]) -> StmtId {
        self.push_stmt(StmtNode::Return(values.to_vec()))
    }

    fn labeled(&mut self, label: &str, body: &[StmtId]) -> StmtId {
        self.push_stmt(StmtNode::Labeled {
            label: label.into(),
            body: body.to_vec(),
        })
    }

    fn goto(&mut self, label: &str) -> StmtId {
        self.push_stmt(StmtNode::Goto(label.into()))
    }

    fn try_(
        &mut self,
        body: &[StmtId],
        handlers: &[Handler<'_, D, StmtId, VarId>],
        finally_body: &[StmtId],
    ) -> StmtId {
        let handlers = handlers
            .iter()
            .map(|handler| HandlerNode {
                class: match handler.kind {
                    HandlerKind::Catch => HandlerClass::Catch,
                    HandlerKind::CatchAll => HandlerClass::CatchAll,
                    HandlerKind::Fault => HandlerClass::Fault,
                    HandlerKind::Filter { filter_body } => {
                        HandlerClass::Filter(filter_body.to_vec())
                    }
                },
                binding: handler.binding,
                caught_types: handler.caught_types.to_vec(),
                body: handler.body.to_vec(),
            })
            .collect();
        self.push_stmt(StmtNode::Try {
            body: body.to_vec(),
            handlers,
            finally_body: finally_body.to_vec(),
        })
    }

    fn region(&mut self, op: &D::Op, operands: &[ExprId], body: &[StmtId]) -> StmtId {
        self.push_stmt(StmtNode::Region {
            op: op.clone(),
            operands: operands.to_vec(),
            body: body.to_vec(),
        })
    }

    fn finish(self, body: &[StmtId]) -> Result<ToyHlil<D>, VerificationReport> {
        let function = self.finish_unchecked(body);
        let report = verify::<D, _>(&function);
        if report.is_ok() {
            Ok(function)
        } else {
            Err(report)
        }
    }
}
