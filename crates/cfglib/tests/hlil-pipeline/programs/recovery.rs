//! Programs with the shapes that structure recovery looks for, and the
//! plain bodies that it must leave alone.

use super::{Case, case, zero};
use crate::dialect::Op;
use crate::source::{
    Program, S, add, apply, catch_all, caught, cell, div, emit, int, lt, mul, ne, pick, raise, ret,
    set, sub, var,
};

/// A diamond that assigns one variable on both arms.
pub(crate) fn select_assign() -> Case {
    let body = vec![
        S::If(
            lt(var("a"), var("b")),
            vec![set("m", var("a"))],
            vec![set("m", var("b"))],
        ),
        ret(var("m")),
    ];
    case(
        Program::new("select_assign", &["a", "b"], &["m"], body),
        &[&[1, 2], &[2, 1], &[4, 4]],
    )
}

/// A diamond whose arms both return.
pub(crate) fn select_return() -> Case {
    let body = vec![S::If(
        lt(var("a"), zero()),
        vec![ret(zero())],
        vec![ret(var("a"))],
    )];
    case(
        Program::new("select_return", &["a"], &[], body),
        &[&[-1], &[0], &[9]],
    )
}

/// A diamond whose taken arm would throw if the other arm ran: a selection
/// must evaluate only the arm it takes.
pub(crate) fn select_guarding_division() -> Case {
    let body = vec![
        S::If(
            ne(var("b"), zero()),
            vec![set("q", div(var("a"), var("b")))],
            vec![set("q", sub(zero(), int(1)))],
        ),
        ret(var("q")),
    ];
    case(
        Program::new("select_guarding_division", &["a", "b"], &["q"], body),
        &[&[9, 3], &[9, 0]],
    )
}

/// `acquire`, a protected body, and a cleanup that releases and rethrows:
/// the lowered shape of a region.
pub(crate) fn acquire_with_cleanup() -> Case {
    let body = vec![
        S::Do(apply(Op::Acquire, vec![var("x")])),
        S::Try(
            vec![
                emit(div(int(10), var("x"))),
                S::Do(apply(Op::Release, vec![var("x")])),
            ],
            vec![catch_all(
                None,
                vec![S::Do(apply(Op::Release, vec![var("x")])), raise(caught())],
            )],
            vec![],
        ),
        ret(zero()),
    ];
    case(
        Program::new("acquire_with_cleanup", &["x"], &[], body),
        &[&[2], &[5], &[0]],
    )
}

/// A region statement over a variable that its body leaves alone.
pub(crate) fn guarded_body() -> Case {
    let body = vec![
        S::Region(
            Op::Guard,
            vec![var("x")],
            vec![emit(var("x")), set("y", add(var("x"), int(1)))],
        ),
        ret(var("y")),
    ];
    case(
        Program::new("guarded_body", &["x"], &["y"], body),
        &[&[3], &[-1]],
    )
}

/// A body with no `if`, `while`, or `try`.
pub(crate) fn straight_line() -> Case {
    let body = vec![
        emit(add(var("a"), var("b"))),
        S::Store(zero(), mul(var("a"), var("b"))),
        ret(cell(zero())),
    ];
    case(
        Program::new("straight_line", &["a", "b"], &[], body),
        &[&[2, 3], &[0, 0], &[-4, 5]],
    )
}

/// A counting loop in the `init; while { ...; update }` shape that
/// structure recovery turns into a `for`.
pub(crate) fn counting_while() -> Case {
    let body = vec![
        set("i", zero()),
        S::While(
            lt(var("i"), var("n")),
            vec![emit(var("i")), set("i", add(var("i"), int(1)))],
        ),
        ret(var("i")),
    ];
    case(
        Program::new("counting_while", &["n"], &["i"], body),
        &[&[0], &[1], &[4]],
    )
}

/// `pick` is a source operation too: a program may write it directly.
pub(crate) fn explicit_pick() -> Case {
    let body = vec![ret(pick(lt(var("a"), var("b")), var("b"), var("a")))];
    case(
        Program::new("explicit_pick", &["a", "b"], &[], body),
        &[&[1, 2], &[2, 1]],
    )
}
