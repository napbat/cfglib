//! Valid Glint functions: loops with block parameters, a diamond, a switch
//! with parallel records, and the awkward shapes the algorithms must accept.

use crate::glint::{Arm, Block, Function, Op, Ty, Value};

pub fn jump(function: &mut Function, from: Block, to: Block, args: &[Value]) {
    function.terminate(from, Op::Jump, &[], vec![(to, args.to_vec(), Arm::Always)]);
}

pub fn branch(
    function: &mut Function,
    from: Block,
    condition: Value,
    taken: (Block, &[Value]),
    not_taken: (Block, &[Value]),
) {
    function.terminate(
        from,
        Op::Branch,
        &[condition],
        vec![
            (taken.0, taken.1.to_vec(), Arm::Taken),
            (not_taken.0, not_taken.1.to_vec(), Arm::NotTaken),
        ],
    );
}

pub fn ret(function: &mut Function, from: Block, value: Value) {
    function.terminate(from, Op::Return, &[value], Vec::new());
}

fn constant(function: &mut Function, block: Block, value: i32) -> Value {
    function.add_inst(block, Op::Const(value), &[], &[Ty::I32])[0]
}

fn binary(function: &mut Function, block: Block, op: Op, left: Value, right: Value) -> Value {
    function.add_inst(block, op, &[left, right], &[Ty::I32])[0]
}

fn less(function: &mut Function, block: Block, left: Value, right: Value) -> Value {
    function.add_inst(block, Op::Lt, &[left, right], &[Ty::Bool])[0]
}

/// `sum of 0..n`: a counting loop whose header takes the counter and the
/// accumulator as block parameters.
pub fn counting_sum() -> Function {
    let mut f = Function::new();
    let entry = f.entry();
    let n = f.add_param(entry, Ty::I32);
    let zero = constant(&mut f, entry, 0);
    let one = constant(&mut f, entry, 1);
    let (header, body, exit) = (f.add_block(), f.add_block(), f.add_block());
    let i = f.add_param(header, Ty::I32);
    let acc = f.add_param(header, Ty::I32);
    let result = f.add_param(exit, Ty::I32);
    jump(&mut f, entry, header, &[zero, zero]);
    let below = less(&mut f, header, i, n);
    branch(&mut f, header, below, (body, &[]), (exit, &[acc]));
    let acc_next = binary(&mut f, body, Op::Add, acc, i);
    let i_next = binary(&mut f, body, Op::Add, i, one);
    jump(&mut f, body, header, &[i_next, acc_next]);
    ret(&mut f, exit, result);
    f
}

/// The `n`th Fibonacci number: the loop passes `(newer, older + newer)` to
/// `(older, newer)`, so the arguments of one record read the parameters they
/// replace.
pub fn fibonacci() -> Function {
    let mut f = Function::new();
    let entry = f.entry();
    let n = f.add_param(entry, Ty::I32);
    let zero = constant(&mut f, entry, 0);
    let one = constant(&mut f, entry, 1);
    let (head, step, done) = (f.add_block(), f.add_block(), f.add_block());
    let older = f.add_param(head, Ty::I32);
    let newer = f.add_param(head, Ty::I32);
    let remaining = f.add_param(head, Ty::I32);
    let result = f.add_param(done, Ty::I32);
    jump(&mut f, entry, head, &[zero, one, n]);
    let positive = less(&mut f, head, zero, remaining);
    branch(&mut f, head, positive, (step, &[]), (done, &[older]));
    let sum = binary(&mut f, step, Op::Add, older, newer);
    let remaining_next = binary(&mut f, step, Op::Sub, remaining, one);
    jump(&mut f, step, head, &[newer, sum, remaining_next]);
    ret(&mut f, done, result);
    f
}

/// `abs(x)` with a join parameter fed by both arms.
pub fn absolute_value() -> Function {
    let mut f = Function::new();
    let entry = f.entry();
    let x = f.add_param(entry, Ty::I32);
    let zero = constant(&mut f, entry, 0);
    let (negative, positive, join) = (f.add_block(), f.add_block(), f.add_block());
    let result = f.add_param(join, Ty::I32);
    let is_negative = less(&mut f, entry, x, zero);
    branch(&mut f, entry, is_negative, (negative, &[]), (positive, &[]));
    let flipped = binary(&mut f, negative, Op::Sub, zero, x);
    jump(&mut f, negative, join, &[flipped]);
    jump(&mut f, positive, join, &[x]);
    ret(&mut f, join, result);
    f
}

/// A switch whose cases 1 and 2 name the same block, so the entry has two
/// parallel successor records. It returns 10, 10, 20, or 30.
pub fn switch_with_parallel_records() -> Function {
    let mut f = Function::new();
    let entry = f.entry();
    let x = f.add_param(entry, Ty::I32);
    let (shared, other, fallback, join) =
        (f.add_block(), f.add_block(), f.add_block(), f.add_block());
    let result = f.add_param(join, Ty::I32);
    f.terminate(
        entry,
        Op::Switch,
        &[x],
        vec![
            (shared, vec![], Arm::Case(1)),
            (shared, vec![], Arm::Case(2)),
            (other, vec![], Arm::Case(3)),
            (fallback, vec![], Arm::Default),
        ],
    );
    for (block, value) in [(shared, 10), (other, 20), (fallback, 30)] {
        let produced = constant(&mut f, block, value);
        jump(&mut f, block, join, &[produced]);
    }
    ret(&mut f, join, result);
    f
}

/// A block the entry cannot reach, with an edge into a reachable block.
pub fn with_an_unreachable_block() -> Function {
    let mut f = Function::new();
    let entry = f.entry();
    let (exit, orphan) = (f.add_block(), f.add_block());
    let result = f.add_param(exit, Ty::I32);
    let seed = constant(&mut f, entry, 4);
    jump(&mut f, entry, exit, &[seed]);
    let local = constant(&mut f, orphan, 9);
    jump(&mut f, orphan, exit, &[local]);
    ret(&mut f, exit, result);
    f
}

/// A removed slot between two live blocks: the slot, its instructions, and
/// its values stay below the bounds.
pub fn with_a_removed_block() -> Function {
    let mut f = Function::new();
    let entry = f.entry();
    let (dead, live) = (f.add_block(), f.add_block());
    let stale = constant(&mut f, dead, 77);
    ret(&mut f, dead, stale);
    let value = constant(&mut f, entry, 5);
    jump(&mut f, entry, live, &[]);
    ret(&mut f, live, value);
    f.remove_block(dead);
    f
}

/// A branch with one arm that spins forever: a block with no path to an exit.
pub fn with_an_endless_loop() -> Function {
    let mut f = Function::new();
    let entry = f.entry();
    let flag = f.add_param(entry, Ty::Bool);
    let (spin, out) = (f.add_block(), f.add_block());
    branch(&mut f, entry, flag, (spin, &[]), (out, &[]));
    jump(&mut f, spin, spin, &[]);
    let done = constant(&mut f, out, 1);
    ret(&mut f, out, done);
    f
}

/// Two nested loops and an exit from each level.
pub fn nested_loops() -> Function {
    let mut f = Function::new();
    let entry = f.entry();
    let x = f.add_param(entry, Ty::I32);
    let blocks: Vec<Block> = (0..6).map(|_| f.add_block()).collect();
    let [outer, inner, inner_body, inner_exit, early, late] = blocks[..] else {
        unreachable!("six blocks")
    };
    jump(&mut f, entry, outer, &[]);
    let go = less(&mut f, outer, x, x);
    branch(&mut f, outer, go, (inner, &[]), (early, &[]));
    let again = less(&mut f, inner, x, x);
    branch(&mut f, inner, again, (inner_body, &[]), (inner_exit, &[]));
    jump(&mut f, inner_body, inner, &[]);
    let more = less(&mut f, inner_exit, x, x);
    branch(&mut f, inner_exit, more, (outer, &[]), (late, &[]));
    ret(&mut f, early, x);
    ret(&mut f, late, x);
    f
}

/// A cycle with two entries, so neither block of the cycle dominates the
/// other.
pub fn irreducible() -> Function {
    let mut f = Function::new();
    let entry = f.entry();
    let flag = f.add_param(entry, Ty::Bool);
    let (left, right, exit) = (f.add_block(), f.add_block(), f.add_block());
    branch(&mut f, entry, flag, (left, &[]), (right, &[]));
    branch(&mut f, left, flag, (right, &[]), (exit, &[]));
    branch(&mut f, right, flag, (left, &[]), (exit, &[]));
    let done = constant(&mut f, exit, 0);
    ret(&mut f, exit, done);
    f
}

/// A block that branches to itself, passing its own parameter back.
pub fn self_loop() -> Function {
    let mut f = Function::new();
    let entry = f.entry();
    let start = f.add_param(entry, Ty::I32);
    let (spin, exit) = (f.add_block(), f.add_block());
    let counter = f.add_param(spin, Ty::I32);
    jump(&mut f, entry, spin, &[start]);
    let go = less(&mut f, spin, counter, counter);
    branch(&mut f, spin, go, (spin, &[counter]), (exit, &[]));
    ret(&mut f, exit, start);
    f
}

/// `max(a, b)` squared, without a branch: one block with a `Select`.
pub fn branchless_square_of_max() -> Function {
    let mut f = Function::new();
    let entry = f.entry();
    let a = f.add_param(entry, Ty::I32);
    let b = f.add_param(entry, Ty::I32);
    let a_is_less = less(&mut f, entry, a, b);
    let larger = f.add_inst(entry, Op::Select, &[a_is_less, b, a], &[Ty::I32])[0];
    let squared = binary(&mut f, entry, Op::Mul, larger, larger);
    ret(&mut f, entry, squared);
    f
}

/// A function of one block that returns a constant.
pub fn single_block() -> Function {
    let mut f = Function::new();
    let entry = f.entry();
    let value = constant(&mut f, entry, 42);
    ret(&mut f, entry, value);
    f
}

/// A block that ends in `Discard`, which has no successor and no operand.
pub fn discarding() -> Function {
    let mut f = Function::new();
    let entry = f.entry();
    let flag = f.add_param(entry, Ty::Bool);
    let (stop, go) = (f.add_block(), f.add_block());
    branch(&mut f, entry, flag, (stop, &[]), (go, &[]));
    f.terminate(stop, Op::Discard, &[], Vec::new());
    let value = constant(&mut f, go, 3);
    ret(&mut f, go, value);
    f
}

/// Every named fixture.
pub fn all() -> Vec<(&'static str, Function)> {
    vec![
        ("counting_sum", counting_sum()),
        ("fibonacci", fibonacci()),
        ("absolute_value", absolute_value()),
        (
            "switch_with_parallel_records",
            switch_with_parallel_records(),
        ),
        ("with_an_unreachable_block", with_an_unreachable_block()),
        ("with_a_removed_block", with_a_removed_block()),
        ("with_an_endless_loop", with_an_endless_loop()),
        ("nested_loops", nested_loops()),
        ("irreducible", irreducible()),
        ("self_loop", self_loop()),
        ("single_block", single_block()),
        ("branchless_square_of_max", branchless_square_of_max()),
        ("discarding", discarding()),
    ]
}
