//! A minimal instruction language: `dest = operator(operands)`.

use cfglib::{ExprInstr, ExprNode, InstrInfo};

pub type Tree = ExprNode<u32, Operator, i64>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    Negate,
}

pub struct Inst {
    operand: [u32; 1],
    defs: [u32; 1],
}

impl Inst {
    /// `dest = negate(operand)`.
    pub fn negate(dest: u32, operand: u32) -> Self {
        Self {
            operand: [operand],
            defs: [dest],
        }
    }
}

impl InstrInfo for Inst {
    type Variable = u32;

    fn uses(&self) -> &[u32] {
        &self.operand
    }

    fn defs(&self) -> &[u32] {
        &self.defs
    }
}

impl ExprInstr for Inst {
    type Operator = Operator;
    type Const = i64;

    fn as_expr(&self) -> Option<(Operator, &[u32])> {
        Some((Operator::Negate, &self.operand))
    }
}
