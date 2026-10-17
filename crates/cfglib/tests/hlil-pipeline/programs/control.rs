//! Programs whose interest is ordinary control flow: branches, every loop
//! form, switches, early returns, jumps, and memory.

use super::{Case, case, zero};
use crate::source::{
    Program, S, add, cell, div, emit, eq, ge, if_then, int, lt, mul, rem, ret, set, sub, var,
};

/// An `if` and an `else` that both emit and assign.
pub(crate) fn branches() -> Case {
    let body = vec![
        S::If(
            lt(var("a"), var("b")),
            vec![emit(var("a")), set("r", sub(var("b"), var("a")))],
            vec![emit(var("b")), set("r", sub(var("a"), var("b")))],
        ),
        ret(var("r")),
    ];
    case(
        Program::new("branches", &["a", "b"], &["r"], body),
        &[&[1, 5], &[5, 1], &[3, 3], &[-2, 4]],
    )
}

/// A pre-tested loop.
pub(crate) fn sum_while() -> Case {
    let body = vec![
        set("s", zero()),
        set("i", zero()),
        S::While(
            lt(var("i"), var("n")),
            vec![
                set("s", add(var("s"), var("i"))),
                set("i", add(var("i"), int(1))),
            ],
        ),
        ret(var("s")),
    ];
    case(
        Program::new("sum_while", &["n"], &["s", "i"], body),
        &[&[0], &[1], &[5], &[10]],
    )
}

/// A post-tested loop that runs its body at least once.
pub(crate) fn do_while_steps() -> Case {
    let body = vec![
        set("i", zero()),
        S::DoWhile(
            vec![emit(var("i")), set("i", add(var("i"), int(3)))],
            lt(var("i"), var("n")),
        ),
        ret(var("i")),
    ];
    case(
        Program::new("do_while_steps", &["n"], &["i"], body),
        &[&[0], &[1], &[10]],
    )
}

/// A counted loop written as a `for`, filling memory.
pub(crate) fn counted_for() -> Case {
    let body = vec![
        S::For(
            vec![set("i", zero())],
            Some(lt(var("i"), var("n"))),
            vec![set("i", add(var("i"), int(1)))],
            vec![S::Store(var("i"), mul(var("i"), var("i")))],
        ),
        ret(cell(sub(var("n"), int(1)))),
    ];
    case(
        Program::new("counted_for", &["n"], &["i"], body),
        &[&[0], &[1], &[6]],
    )
}

/// Two nested pre-tested loops.
pub(crate) fn nested_loops() -> Case {
    let inner = S::While(
        lt(var("j"), var("m")),
        vec![
            set("s", add(var("s"), mul(var("i"), var("j")))),
            set("j", add(var("j"), int(1))),
        ],
    );
    let outer = S::While(
        lt(var("i"), var("n")),
        vec![
            set("j", zero()),
            inner,
            emit(var("s")),
            set("i", add(var("i"), int(1))),
        ],
    );
    let body = vec![set("s", zero()), set("i", zero()), outer, ret(var("s"))];
    case(
        Program::new("nested_loops", &["n", "m"], &["s", "i", "j"], body),
        &[&[0, 3], &[3, 0], &[3, 4], &[1, 1]],
    )
}

/// An endless loop that ends through `break` and skips through `continue`.
pub(crate) fn break_continue() -> Case {
    let body = vec![
        set("i", zero()),
        set("s", zero()),
        S::Loop(vec![
            set("i", add(var("i"), int(1))),
            if_then(ge(var("i"), var("n")), vec![S::Break(None)]),
            if_then(eq(rem(var("i"), int(2)), zero()), vec![S::Continue(None)]),
            emit(var("i")),
            set("s", add(var("s"), var("i"))),
        ]),
        ret(var("s")),
    ];
    case(
        Program::new("break_continue", &["n"], &["i", "s"], body),
        &[&[0], &[1], &[2], &[7], &[10]],
    )
}

/// `continue` and `break` inside a `while`, with the counter advanced first.
pub(crate) fn while_break_continue() -> Case {
    let body = vec![
        set("i", zero()),
        set("s", zero()),
        S::While(
            lt(var("i"), int(10)),
            vec![
                set("i", add(var("i"), int(1))),
                S::If(
                    lt(var("i"), var("n")),
                    vec![
                        if_then(eq(rem(var("i"), int(3)), zero()), vec![S::Continue(None)]),
                        set("s", add(var("s"), var("i"))),
                    ],
                    vec![S::Break(None)],
                ),
            ],
        ),
        ret(var("s")),
    ];
    case(
        Program::new("while_break_continue", &["n"], &["i", "s"], body),
        &[&[0], &[4], &[8], &[20]],
    )
}

/// A labeled `break` out of two loops.
pub(crate) fn labeled_break() -> Case {
    let inner = S::While(
        lt(var("j"), var("n")),
        vec![
            if_then(
                eq(mul(var("i"), var("j")), int(6)),
                vec![
                    set("found", add(mul(var("i"), int(10)), var("j"))),
                    S::Break(Some("outer")),
                ],
            ),
            set("j", add(var("j"), int(1))),
        ],
    );
    let outer = S::While(
        lt(var("i"), var("n")),
        vec![set("j", zero()), inner, set("i", add(var("i"), int(1)))],
    );
    let body = vec![
        set("i", zero()),
        set("found", sub(zero(), int(1))),
        S::Label("outer", vec![outer]),
        ret(var("found")),
    ];
    case(
        Program::new("labeled_break", &["n"], &["i", "j", "found"], body),
        &[&[0], &[3], &[4], &[7]],
    )
}

/// A labeled `continue` of the outer loop from the inner one.
pub(crate) fn labeled_continue() -> Case {
    let inner = S::While(
        lt(var("j"), var("n")),
        vec![
            set("j", add(var("j"), int(1))),
            if_then(eq(var("j"), var("i")), vec![S::Continue(Some("outer"))]),
            set("s", add(var("s"), var("j"))),
        ],
    );
    let outer = S::While(
        lt(var("i"), var("n")),
        vec![set("i", add(var("i"), int(1))), set("j", zero()), inner],
    );
    let body = vec![
        set("s", zero()),
        S::Label("outer", vec![outer]),
        ret(var("s")),
    ];
    case(
        Program::new("labeled_continue", &["n"], &["i", "j", "s"], body),
        &[&[0], &[1], &[3], &[5]],
    )
}

/// A switch with a shared arm and a default.
pub(crate) fn switch_dispatch() -> Case {
    let body = vec![
        S::Switch(
            var("x"),
            vec![
                (vec![1, 2], vec![emit(int(10)), set("r", int(1))]),
                (vec![3], vec![emit(int(30)), set("r", int(3))]),
            ],
            vec![set("r", sub(zero(), int(1)))],
        ),
        ret(var("r")),
    ];
    case(
        Program::new("switch_dispatch", &["x"], &["r"], body),
        &[&[1], &[2], &[3], &[4], &[-7]],
    )
}

/// A switch inside a loop: `continue` reaches the loop, `break` the switch.
pub(crate) fn switch_in_loop() -> Case {
    let dispatch = S::Switch(
        rem(var("i"), int(3)),
        vec![
            (vec![0], vec![S::Continue(None)]),
            (
                vec![1],
                vec![set("s", add(var("s"), int(10))), S::Break(None)],
            ),
        ],
        vec![set("s", add(var("s"), int(1)))],
    );
    let body = vec![
        set("i", zero()),
        set("s", zero()),
        S::While(
            lt(var("i"), var("n")),
            vec![
                set("i", add(var("i"), int(1))),
                dispatch,
                set("s", add(var("s"), int(100))),
            ],
        ),
        ret(var("s")),
    ];
    case(
        Program::new("switch_in_loop", &["n"], &["i", "s"], body),
        &[&[0], &[2], &[7]],
    )
}

/// Returns from the middle of the function and from inside a loop.
pub(crate) fn early_returns() -> Case {
    let body = vec![
        if_then(lt(var("x"), zero()), vec![ret(sub(zero(), var("x")))]),
        if_then(eq(var("x"), zero()), vec![ret(int(100))]),
        set("r", var("x")),
        S::While(
            ge(var("r"), int(10)),
            vec![
                set("r", div(var("r"), int(2))),
                if_then(eq(var("r"), int(13)), vec![ret(int(13))]),
            ],
        ),
        ret(var("r")),
    ];
    case(
        Program::new("early_returns", &["x"], &["r"], body),
        &[&[-4], &[0], &[7], &[26], &[100], &[1000]],
    )
}

/// A backward `goto` that makes a loop.
pub(crate) fn goto_backward() -> Case {
    let body = vec![
        set("i", zero()),
        set("s", zero()),
        S::Label(
            "top",
            vec![S::If(
                lt(var("i"), var("n")),
                vec![
                    set("s", add(var("s"), var("i"))),
                    set("i", add(var("i"), int(1))),
                    S::Goto("top"),
                ],
                vec![],
            )],
        ),
        ret(var("s")),
    ];
    case(
        Program::new("goto_backward", &["n"], &["i", "s"], body),
        &[&[0], &[1], &[6]],
    )
}

/// A forward `goto` that skips a statement.
pub(crate) fn goto_forward() -> Case {
    let body = vec![
        set("r", zero()),
        if_then(lt(var("x"), zero()), vec![S::Goto("done")]),
        set("r", mul(var("x"), int(2))),
        S::Label("done", vec![emit(var("r"))]),
        ret(var("r")),
    ];
    case(
        Program::new("goto_forward", &["x"], &["r"], body),
        &[&[-3], &[0], &[4]],
    )
}

/// A `goto` into the middle of a loop body: the loop has two entries, so the
/// control-flow graph is irreducible.
pub(crate) fn irreducible_loop() -> Case {
    let body = vec![
        set("i", zero()),
        set("s", zero()),
        if_then(lt(var("a"), zero()), vec![S::Goto("middle")]),
        S::Loop(vec![
            set("s", add(var("s"), int(1))),
            S::Label("middle", vec![set("s", add(var("s"), int(10)))]),
            set("i", add(var("i"), int(1))),
            if_then(ge(var("i"), var("n")), vec![S::Break(None)]),
        ]),
        ret(var("s")),
    ];
    case(
        Program::new("irreducible_loop", &["a", "n"], &["i", "s"], body),
        &[&[1, 3], &[-1, 3], &[0, 1], &[-1, 0]],
    )
}

/// Stores and loads through memory cells.
pub(crate) fn memory_walk() -> Case {
    let fill = S::While(
        lt(var("i"), var("n")),
        vec![
            S::Store(var("i"), mul(var("i"), int(3))),
            set("i", add(var("i"), int(1))),
        ],
    );
    let sum = S::While(
        lt(var("i"), var("n")),
        vec![
            set("s", add(var("s"), cell(var("i")))),
            set("i", add(var("i"), int(1))),
        ],
    );
    let body = vec![
        set("i", zero()),
        fill,
        set("s", zero()),
        set("i", zero()),
        sum,
        ret(var("s")),
    ];
    case(
        Program::new("memory_walk", &["n"], &["i", "s"], body),
        &[&[0], &[1], &[5]],
    )
}

/// A load that must stay before the store that follows it.
pub(crate) fn load_store_order() -> Case {
    let body = vec![
        set("t", cell(zero())),
        S::Store(zero(), var("a")),
        set("r", add(var("t"), int(1))),
        ret(add(var("r"), cell(zero()))),
    ];
    case(
        Program::new("load_store_order", &["a"], &["t", "r"], body),
        &[&[5], &[9], &[0]],
    )
}
