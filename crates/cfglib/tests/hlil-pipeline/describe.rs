//! One-line descriptions of expressions and statements, for tests that ask
//! whether two entities of different functions read the same.

use crate::dialect::Lit;
use crate::structured::{ExprId, ExprNode, Function, StmtId, StmtNode};

/// The expression as `op(operand, ...)`, `v3`, or a literal.
pub(crate) fn expression(function: &Function, expr: ExprId) -> String {
    match &function.exprs[expr.index()].node {
        ExprNode::Var(variable) => format!("v{}", variable.index()),
        ExprNode::Const(Lit::Int(value)) => value.to_string(),
        ExprNode::Const(Lit::Bool(value)) => value.to_string(),
        ExprNode::Op { op, operands } => {
            let operands: Vec<String> = operands
                .iter()
                .map(|&operand| expression(function, operand))
                .collect();
            format!("{}({})", op.mnemonic(), operands.join(", "))
        }
    }
}

/// The statement without its child bodies.
pub(crate) fn head(function: &Function, stmt: StmtId) -> String {
    let show = |expr: ExprId| expression(function, expr);
    match &function.stmts[stmt.index()] {
        StmtNode::Expression(expr) => show(*expr),
        StmtNode::Assign { target, value } => format!("{} = {}", show(*target), show(*value)),
        StmtNode::If { condition, .. } => format!("if ({})", show(*condition)),
        StmtNode::While { condition, .. } => format!("while ({})", show(*condition)),
        StmtNode::DoWhile { condition, .. } => format!("do while ({})", show(*condition)),
        StmtNode::Loop { .. } => "loop".to_owned(),
        StmtNode::For { condition, .. } => {
            format!("for ({})", condition.map_or(String::new(), show))
        }
        StmtNode::Switch { scrutinee, .. } => format!("switch ({})", show(*scrutinee)),
        StmtNode::Break(label) => format!("break {}", label.as_deref().unwrap_or("")),
        StmtNode::Continue(label) => format!("continue {}", label.as_deref().unwrap_or("")),
        StmtNode::Return(values) => {
            let values: Vec<String> = values.iter().map(|&value| show(value)).collect();
            format!("return {}", values.join(", "))
        }
        StmtNode::Labeled { label, .. } => format!("{label}:"),
        StmtNode::Goto(label) => format!("goto {label}"),
        StmtNode::Try { .. } => "try".to_owned(),
        StmtNode::Region { op, operands, .. } => {
            let operands: Vec<String> = operands.iter().map(|&operand| show(operand)).collect();
            format!("@{}({})", op.mnemonic(), operands.join(", "))
        }
    }
}
