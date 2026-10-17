//! Functions that several test modules build.

use crate::ir::hlil::{FunctionBuilder, Handler, HandlerKind, SwitchArm};

use super::{Kit, Prim, StmtId, ToyHlil, ToyHlilBuilder, Ty};

/// `while (i < n) { if (i < 100) { break; } i = i + 1; } return i;` and its
/// loop, guard, break, step, and return statements.
pub(crate) fn structured_counting_loop<D: Kit>() -> (ToyHlil<D>, [StmtId; 5]) {
    let mut builder = ToyHlilBuilder::<D>::new();
    let i = builder.int_variable(0);
    let n = builder.int_variable(1);
    builder.set_signature(&[i, n], &[D::ty(Ty::Integer)]);

    let read_i = builder.read(i);
    let limit = builder.number(100);
    let below_limit = builder.call(Prim::LessThan, &[read_i, limit], Ty::Boolean);
    let leave = builder.break_(None);
    let guard = builder.if_(below_limit, &[leave], &[]);

    let read_i = builder.read(i);
    let one = builder.number(1);
    let sum = builder.call(Prim::Add, &[read_i, one], Ty::Integer);
    let step = builder.store(i, sum);

    let read_i = builder.read(i);
    let read_n = builder.read(n);
    let condition = builder.call(Prim::LessThan, &[read_i, read_n], Ty::Boolean);
    let looping = builder.while_(condition, &[guard, step]);
    let result = builder.read(i);
    let done = builder.return_(&[result]);
    let function = builder
        .finish(&[looping, done])
        .expect("the function is valid");
    (function, [looping, guard, leave, step, done])
}

/// `if (v0 < 10) { v0 = v0 + 1; } return v0;`
pub(crate) fn bump<D: Kit>() -> ToyHlil<D> {
    let mut builder = ToyHlilBuilder::<D>::new();
    let counter = builder.int_variable(0);
    let read = builder.read(counter);
    let ten = builder.number(10);
    let compare = builder.call(Prim::LessThan, &[read, ten], Ty::Boolean);
    let current = builder.read(counter);
    let one = builder.number(1);
    let sum = builder.call(Prim::Add, &[current, one], Ty::Integer);
    let assign = builder.store(counter, sum);
    let conditional = builder.if_(compare, &[assign], &[]);
    let result = builder.read(counter);
    let ret = builder.return_(&[result]);
    builder.set_signature(&[counter], &[D::ty(Ty::Integer)]);
    builder.finish(&[conditional, ret]).unwrap()
}

/// One of every statement shape, in an order that a golden reads well. No
/// shape of it is one that structure recovery rewrites.
pub(crate) fn every_statement_shape<D: Kit>() -> ToyHlil<D> {
    let mut builder = ToyHlilBuilder::<D>::new();
    let index = builder.int_variable(0);
    let bound = builder.int_variable(1);
    let caught = builder.int_variable(2);
    builder.set_signature(&[index, bound], &[D::ty(Ty::Integer)]);

    // for (v0 = 0; lt(v0, v1); v0 = add(v0, 1)) { call(v0); }
    let zero = builder.number(0);
    let initializer = builder.store(index, zero);
    let left = builder.read(index);
    let right = builder.read(bound);
    let test = builder.call(Prim::LessThan, &[left, right], Ty::Boolean);
    let current = builder.read(index);
    let one = builder.number(1);
    let sum = builder.call(Prim::Add, &[current, one], Ty::Integer);
    let update = builder.store(index, sum);
    let argument = builder.read(index);
    let work = builder.effect(Prim::Call, &[argument]);
    let counted = builder.for_(&[initializer], Some(test), &[update], &[work]);

    // for (; ; ) { break; }
    let leave = builder.break_(None);
    let unconditional = builder.for_(&[], None, &[], &[leave]);

    // outer: { inner: do { continue inner; } while (not(v1)); loop { break outer; } }
    let again = builder.continue_(Some("inner"));
    let flag = builder.read(bound);
    let negated = builder.call(Prim::Not, &[flag], Ty::Boolean);
    let tested_last = builder.do_while(&[again], negated);
    let inner = builder.labeled("inner", &[tested_last]);
    let escape = builder.break_(Some("outer"));
    let spinning = builder.loop_(&[escape]);
    let outer = builder.labeled("outer", &[inner, spinning]);

    // switch (v0) { case 1, 2: call(v0); case 3: ; default: goto done; }
    let scrutinee = builder.read(index);
    let seen = builder.read(index);
    let first_arm = builder.effect(Prim::Call, &[seen]);
    let jump = builder.goto("done");
    let one_two = [D::constant(1), D::constant(2)];
    let three = [D::constant(3)];
    let switch = builder.switch(
        scrutinee,
        &[
            SwitchArm {
                values: &one_two,
                body: &[first_arm],
            },
            SwitchArm {
                values: &three,
                body: &[],
            },
        ],
        &[jump],
    );

    // try { call(); } catch () v2 { ... } fault () { ... }
    // filter () { when { ... } ... } finally { ... }
    let attempt = builder.effect(Prim::Call, &[]);
    let handled = builder.use_of(caught);
    let cleanup = builder.noop();
    let predicate = builder.use_of(index);
    let filtered = builder.noop();
    let finalizer = builder.effect(Prim::Release, &[]);
    let handlers = [
        Handler {
            kind: HandlerKind::Catch,
            binding: Some(caught),
            caught_types: &[],
            body: &[handled],
        },
        Handler {
            kind: HandlerKind::Fault,
            binding: None,
            caught_types: &[],
            body: &[cleanup],
        },
        Handler {
            kind: HandlerKind::Filter {
                filter_body: &[predicate],
            },
            binding: None,
            caught_types: &[],
            body: &[filtered],
        },
    ];
    let guarded = builder.try_(&[attempt], &handlers, &[finalizer]);

    // @acquire(v1) { if (v0) { return; } else { return v0, v1; } }
    let condition = builder.read(index);
    let early = builder.return_(&[]);
    let first = builder.read(index);
    let second = builder.read(bound);
    let both = builder.return_(&[first, second]);
    let branch = builder.if_(condition, &[early], &[both]);
    let subject = builder.read(bound);
    let region = builder.region(&D::op(Prim::Acquire), &[subject], &[branch]);

    // done: { }
    let done = builder.labeled("done", &[]);
    builder
        .finish(&[counted, unconditional, outer, switch, guarded, region, done])
        .unwrap()
}
