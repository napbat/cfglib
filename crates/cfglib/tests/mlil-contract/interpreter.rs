//! An interpreter that runs a Glint function through `ir::mlil::FunctionView`
//! alone.
//!
//! Running a function is an independent check of what the view promises: the
//! successor arguments land in the parameters of the target in order, all at
//! once, and the operands and results name the values the instructions read
//! and define.

use cfglib::DenseId;
use cfglib::ir::mlil::FunctionView;

use crate::glint::{Arm, Glint, Op};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Returned(i32),
    Discarded,
    /// The function ran for more blocks than the limit.
    OutOfSteps,
}

/// Runs `function` on `arguments`, which become the parameters of the entry
/// block. Booleans are stored as `0` and `1`.
pub fn run<F: FunctionView<Glint>>(function: &F, arguments: &[i32]) -> Outcome {
    let mut values: Vec<Option<i32>> = vec![None; function.value_bound()];
    let mut block = cfglib::RootedView::root(function);
    for (&param, &argument) in function.params(block).iter().zip(arguments) {
        values[param.index()] = Some(argument);
    }
    let read = |values: &[Option<i32>], value: F::Value| {
        values[value.index()].expect("a value is defined before it is read")
    };
    for _ in 0..10_000 {
        let mut next = None;
        for &inst in function.insts(block) {
            let operands: Vec<i32> = function
                .operands(inst)
                .iter()
                .map(|&operand| read(&values, operand))
                .collect();
            let results = function.results(inst);
            let records = FunctionView::successors(function, block);
            match function.op(inst) {
                Op::Const(value) => values[results[0].index()] = Some(*value),
                Op::Add => values[results[0].index()] = Some(operands[0].wrapping_add(operands[1])),
                Op::Sub => values[results[0].index()] = Some(operands[0].wrapping_sub(operands[1])),
                Op::Mul => values[results[0].index()] = Some(operands[0].wrapping_mul(operands[1])),
                Op::Lt => values[results[0].index()] = Some(i32::from(operands[0] < operands[1])),
                Op::Select => {
                    let chosen = if operands[0] == 1 {
                        operands[1]
                    } else {
                        operands[2]
                    };
                    values[results[0].index()] = Some(chosen);
                }
                Op::Return => return Outcome::Returned(operands[0]),
                Op::Discard => return Outcome::Discarded,
                Op::Jump | Op::Branch | Op::Switch => {
                    let taken = records
                        .into_iter()
                        .find(|record| match record.edge {
                            Arm::Always | Arm::Default => true,
                            Arm::Taken => operands[0] == 1,
                            Arm::NotTaken => operands[0] != 1,
                            Arm::Case(case) => *case == operands[0],
                        })
                        .expect("a terminator takes one of its records");
                    // The arguments are read before any parameter changes.
                    let passed: Vec<i32> =
                        taken.args.iter().map(|&arg| read(&values, arg)).collect();
                    for (&param, value) in function.params(taken.target).iter().zip(passed) {
                        values[param.index()] = Some(value);
                    }
                    next = Some(taken.target);
                }
            }
        }
        block = next.expect("a block ends in a terminator that continues or stops");
    }
    Outcome::OutOfSteps
}
