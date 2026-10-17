//! The pipeline the way a consumer drives it: one call per library
//! algorithm, each unwrapped with a message that names the program.

use cfglib::ir::hlil::{
    LiftMetadata, LiftedFunction, LoweredFunction, Recovery, lift, lower, pseudocode,
    recover_structure,
};

use crate::dialect::Calc;
use crate::flat::{Code, CodeBuilder, Reg};
use crate::source::Program;
use crate::structured::{Builder, ExprId, Function, StmtId, VarId};

pub(crate) type Lowered = LoweredFunction<Code, StmtId, ExprId, VarId, Reg>;
pub(crate) type Lifted = LiftedFunction<Function, StmtId, ExprId, VarId>;
pub(crate) type Recovered = Recovery<Function, StmtId, ExprId, VarId>;

/// The structured function that `program` compiles to.
pub(crate) fn structured(program: &Program) -> Function {
    program
        .build(Builder::new())
        .unwrap_or_else(|report| panic!("{} does not verify: {report}", program.name))
}

pub(crate) fn lowered(function: &Function) -> Lowered {
    lower::<Calc, _, _>(function, CodeBuilder::new())
        .unwrap_or_else(|error| panic!("lowering failed: {error}"))
}

pub(crate) fn lifted(code: &Code) -> Lifted {
    lift::<Calc, _, _>(code, Builder::new(), LiftMetadata::Preserve)
        .unwrap_or_else(|error| panic!("lifting failed: {error}"))
}

pub(crate) fn recovered(function: &Function) -> Recovered {
    recover_structure::<Calc, _, _>(function, Builder::new())
        .unwrap_or_else(|error| panic!("recovery failed: {error}"))
}

pub(crate) fn text(function: &Function) -> String {
    pseudocode::<Calc, _>(function)
}
