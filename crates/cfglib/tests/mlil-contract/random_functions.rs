//! Random valid Glint functions, for sweeping the algorithms.
//!
//! The generator is a fixed xorshift sequence, so a failure names the seed and
//! reproduces. Blocks branch anywhere (loops, self loops, parallel switch
//! records, unreachable code), some slots are removed, and every record passes
//! one argument per target parameter, chosen among the values that are
//! defined at the end of its block: the entry values, the parameters of the
//! block, and its own results. That keeps every function in SSA form
//! whatever its shape.

use crate::glint::{Arm, Block, Function, Op, Ty, Value};
use crate::rng::Rng;

/// One of `values`, drawn uniformly.
fn pick(rng: &mut Rng, values: &[Value]) -> Value {
    values[rng.below(values.len())]
}

/// A random function of two to nine slots.
pub fn random_function(seed: u64) -> Function {
    let mut rng = Rng::new(seed);
    let count = 2 + rng.below(8);
    let gone: Vec<bool> = (0..count).map(|slot| slot != 0 && rng.chance(15)).collect();
    let live: Vec<usize> = (0..count).filter(|&slot| !gone[slot]).collect();

    let mut function = Function::new();
    let blocks: Vec<Block> = (0..count)
        .map(|slot| {
            if slot == 0 {
                function.entry()
            } else {
                function.add_block()
            }
        })
        .collect();
    for &block in &blocks {
        for _ in 0..rng.below(3) {
            function.add_param(block, Ty::I32);
        }
    }
    // Values of the entry block are visible in every block that it reaches.
    let entry = blocks[0];
    let mut shared: Vec<Value> = function_params(&function, entry);
    for value in [3, 7] {
        shared.push(function.add_inst(entry, Op::Const(value), &[], &[Ty::I32])[0]);
    }

    for (slot, &block) in blocks.iter().enumerate() {
        let mut available = shared.clone();
        if slot != 0 {
            available.extend(function_params(&function, block));
        }
        for _ in 0..rng.below(3) {
            let (left, right) = (pick(&mut rng, &available), pick(&mut rng, &available));
            available.push(function.add_inst(block, Op::Add, &[left, right], &[Ty::I32])[0]);
        }
        terminate(&mut function, &mut rng, block, &available, &blocks, &live);
    }
    for slot in 0..count {
        if gone[slot] {
            function.remove_block(blocks[slot]);
        }
    }
    function
}

fn function_params(function: &Function, block: Block) -> Vec<Value> {
    use cfglib::ir::mlil::FunctionView;
    function.params(block).to_vec()
}

/// Ends `block` with a terminator whose kind follows its out-degree.
fn terminate(
    function: &mut Function,
    rng: &mut Rng,
    block: Block,
    available: &[Value],
    blocks: &[Block],
    live: &[usize],
) {
    let degree = rng.below(4);
    let targets: Vec<Block> = (0..degree)
        .map(|_| blocks[live[rng.below(live.len())]])
        .collect();
    let args_for = |function: &Function, rng: &mut Rng, target: Block| -> Vec<Value> {
        function_params(function, target)
            .iter()
            .map(|_| pick(rng, available))
            .collect()
    };
    match targets.as_slice() {
        [] => {
            if rng.chance(25) {
                function.terminate(block, Op::Discard, &[], Vec::new());
            } else {
                let value = pick(rng, available);
                function.terminate(block, Op::Return, &[value], Vec::new());
            }
        }
        [only] => {
            let args = args_for(function, rng, *only);
            function.terminate(block, Op::Jump, &[], vec![(*only, args, Arm::Always)]);
        }
        [taken, not_taken] => {
            let (left, right) = (pick(rng, available), pick(rng, available));
            let condition = function.add_inst(block, Op::Lt, &[left, right], &[Ty::Bool])[0];
            let taken_args = args_for(function, rng, *taken);
            let not_taken_args = args_for(function, rng, *not_taken);
            function.terminate(
                block,
                Op::Branch,
                &[condition],
                vec![
                    (*taken, taken_args, Arm::Taken),
                    (*not_taken, not_taken_args, Arm::NotTaken),
                ],
            );
        }
        many => {
            let scrutinee = pick(rng, available);
            let last = many.len() - 1;
            let records = many
                .iter()
                .enumerate()
                .map(|(index, &target)| {
                    let edge = if index == last {
                        Arm::Default
                    } else {
                        Arm::Case(i32::try_from(index).expect("few arms"))
                    };
                    (target, args_for(function, rng, target), edge)
                })
                .collect();
            function.terminate(block, Op::Switch, &[scrutinee], records);
        }
    }
}
