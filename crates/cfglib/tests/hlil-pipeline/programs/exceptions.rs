//! Programs with exceptions: handlers that bind or read the exception,
//! `finally`, nested regions, and transfers out of a `try`.

use super::{Case, case, zero};
use crate::source::{
    Program, S, add, catch, catch_all, caught, div, emit, eq, if_then, int, lt, rem, ret, set, sub,
    var,
};

/// A catch that binds the exception.
pub(crate) fn division_guard() -> Case {
    let body = vec![
        set("r", zero()),
        S::Try(
            vec![emit(div(var("a"), var("b"))), set("r", int(1))],
            vec![catch_all(Some("e"), vec![emit(var("e")), set("r", int(2))])],
            vec![],
        ),
        ret(var("r")),
    ];
    case(
        Program::new("division_guard", &["a", "b"], &["r", "e"], body),
        &[&[10, 2], &[10, 0], &[0, 5]],
    )
}

/// A typed catch that reads the exception without a binding.
pub(crate) fn typed_catch() -> Case {
    let body = vec![
        set("r", zero()),
        S::Try(
            vec![set("r", rem(var("a"), var("b")))],
            vec![catch(
                None,
                vec![emit(caught()), set("r", sub(zero(), int(1)))],
            )],
            vec![],
        ),
        ret(var("r")),
    ];
    case(
        Program::new("typed_catch", &["a", "b"], &["r"], body),
        &[&[10, 3], &[10, 0]],
    )
}

/// `try`, `catch`, and `finally` together.
pub(crate) fn catch_and_finally() -> Case {
    let body = vec![
        set("r", zero()),
        S::Try(
            vec![
                emit(int(1)),
                set("r", div(var("a"), var("b"))),
                emit(int(2)),
            ],
            vec![catch_all(
                Some("e"),
                vec![emit(var("e")), set("r", int(99))],
            )],
            vec![emit(int(7))],
        ),
        emit(int(8)),
        ret(var("r")),
    ];
    case(
        Program::new("catch_and_finally", &["a", "b"], &["r", "e"], body),
        &[&[6, 3], &[6, 0]],
    )
}

/// An inner handler that throws again into the outer one.
pub(crate) fn nested_try() -> Case {
    let inner = S::Try(
        vec![set("r", div(var("a"), var("b")))],
        vec![catch_all(
            None,
            vec![emit(int(100)), set("r", div(var("a"), var("c")))],
        )],
        vec![],
    );
    let body = vec![
        set("r", zero()),
        S::Try(
            vec![inner, emit(var("r"))],
            vec![catch_all(
                Some("e"),
                vec![emit(var("e")), set("r", sub(zero(), int(5)))],
            )],
            vec![],
        ),
        ret(var("r")),
    ];
    case(
        Program::new("nested_try", &["a", "b", "c"], &["r", "e"], body),
        &[&[8, 2, 1], &[8, 0, 4], &[8, 0, 0]],
    )
}

/// A throw in a loop body that the loop survives.
pub(crate) fn catch_in_loop() -> Case {
    let body = vec![
        set("i", zero()),
        set("s", zero()),
        S::While(
            lt(var("i"), var("n")),
            vec![
                S::Try(
                    vec![set(
                        "s",
                        add(var("s"), div(int(100), sub(int(3), var("i")))),
                    )],
                    vec![catch_all(None, vec![emit(var("i"))])],
                    vec![],
                ),
                set("i", add(var("i"), int(1))),
            ],
        ),
        ret(var("s")),
    ];
    case(
        Program::new("catch_in_loop", &["n"], &["i", "s"], body),
        &[&[2], &[3], &[6]],
    )
}

/// Returns from inside a `try` and from its handler.
pub(crate) fn return_in_try() -> Case {
    let body = vec![
        S::Try(
            vec![
                if_then(lt(var("a"), zero()), vec![ret(zero())]),
                emit(div(int(10), var("a"))),
            ],
            vec![catch_all(None, vec![ret(int(7))])],
            vec![],
        ),
        ret(int(1)),
    ];
    case(
        Program::new("return_in_try", &["a"], &[], body),
        &[&[-3], &[0], &[5]],
    )
}

/// `break` and `continue` that leave a `try` inside a loop.
pub(crate) fn transfers_out_of_try() -> Case {
    let guarded = S::Try(
        vec![
            if_then(eq(var("i"), int(2)), vec![S::Continue(None)]),
            if_then(eq(var("i"), int(4)), vec![S::Break(None)]),
            emit(div(int(8), sub(var("i"), int(3)))),
        ],
        vec![catch_all(None, vec![emit(sub(zero(), var("i")))])],
        vec![],
    );
    let body = vec![
        set("i", zero()),
        S::While(
            lt(var("i"), var("n")),
            vec![set("i", add(var("i"), int(1))), guarded],
        ),
        ret(var("i")),
    ];
    case(
        Program::new("transfers_out_of_try", &["n"], &["i"], body),
        &[&[1], &[3], &[9]],
    )
}

/// Two throwing operations in one expression: the first thrown wins.
pub(crate) fn throw_order() -> Case {
    let body = vec![
        set("x", zero()),
        S::Try(
            vec![set(
                "x",
                add(div(var("a"), var("b")), rem(var("a"), var("c"))),
            )],
            vec![catch_all(
                Some("e"),
                vec![emit(var("e")), set("x", sub(zero(), int(9)))],
            )],
            vec![],
        ),
        ret(var("x")),
    ];
    case(
        Program::new("throw_order", &["a", "b", "c"], &["x", "e"], body),
        &[&[9, 3, 2], &[9, 0, 2], &[9, 3, 0], &[9, 0, 0]],
    )
}
