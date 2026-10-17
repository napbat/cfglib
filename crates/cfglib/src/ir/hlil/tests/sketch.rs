//! Compact descriptions of structured functions.
//!
//! A [`Sketch`] names variables by number and nests statements and
//! expressions as plain data, so a test that lowers many shapes states each
//! one in a few lines. [`Sketch::build`] turns it into a [`ToyHlil`] of
//! either toy dialect, and [`cases`] lists the shapes that lowering must
//! translate: every statement form, every join, and every kind of transfer.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::{Prim, Ty};

/// An expression.
pub(crate) enum Ex {
    /// A read of the variable with this number.
    Var(u8),
    /// An integer literal.
    Num(i64),
    /// An operation producing a value of the given type.
    Apply(Prim, Ty, Vec<Ex>),
}

/// A handler of a `try`.
pub(crate) enum Class {
    Catch,
    CatchAll,
    Fault,
    /// A filter handler with its filter body.
    Filter(Vec<Sn>),
}

/// One handler: its class, the variable it binds, and its body.
pub(crate) struct Hn {
    pub(crate) class: Class,
    pub(crate) binding: Option<u8>,
    pub(crate) body: Vec<Sn>,
}

/// A statement.
pub(crate) enum Sn {
    Eval(Ex),
    /// `variable = value`.
    Assign(u8, Ex),
    /// `place(operands) = value`.
    Put(Prim, Vec<Ex>, Ex),
    If(Ex, Vec<Sn>, Vec<Sn>),
    While(Ex, Vec<Sn>),
    DoWhile(Vec<Sn>, Ex),
    Loop(Vec<Sn>),
    For(Vec<Sn>, Option<Ex>, Vec<Sn>, Vec<Sn>),
    /// A scrutinee, the arms with their case values, and the default body.
    Switch(Ex, Vec<(Vec<i64>, Vec<Sn>)>, Vec<Sn>),
    Break(Option<&'static str>),
    Continue(Option<&'static str>),
    Return(Vec<Ex>),
    Labeled(&'static str, Vec<Sn>),
    Goto(&'static str),
    Try(Vec<Sn>, Vec<Hn>, Vec<Sn>),
    Region(Prim, Vec<Ex>, Vec<Sn>),
}

/// A structured function: integer variables numbered from zero, the numbers
/// of the parameters, how many integers it returns, and its body.
pub(crate) struct Sketch {
    pub(crate) variables: u8,
    /// Variables declared without a type.
    pub(crate) untyped: Vec<u8>,
    pub(crate) parameters: Vec<u8>,
    pub(crate) returns: usize,
    pub(crate) body: Vec<Sn>,
}

/// A named sketch.
type Case = (&'static str, Sketch);

fn var(number: u8) -> Ex {
    Ex::Var(number)
}

fn num(value: i64) -> Ex {
    Ex::Num(value)
}

fn add(left: Ex, right: Ex) -> Ex {
    Ex::Apply(Prim::Add, Ty::Integer, vec![left, right])
}

fn less(left: Ex, right: Ex) -> Ex {
    Ex::Apply(Prim::LessThan, Ty::Boolean, vec![left, right])
}

fn call(operands: Vec<Ex>) -> Ex {
    Ex::Apply(Prim::Call, Ty::Integer, operands)
}

fn effect(operands: Vec<Ex>) -> Sn {
    Sn::Eval(Ex::Apply(Prim::Call, Ty::Void, operands))
}

fn when(condition: Ex, then_body: Vec<Sn>) -> Sn {
    Sn::If(condition, then_body, Vec::new())
}

fn function(variables: u8, parameters: &[u8], body: Vec<Sn>) -> Sketch {
    Sketch {
        variables,
        untyped: Vec::new(),
        parameters: parameters.to_vec(),
        returns: 1,
        body,
    }
}

/// A function that runs `statement` and returns variable zero.
fn then_return(variables: u8, parameters: &[u8], statement: Sn) -> Sketch {
    function(
        variables,
        parameters,
        vec![statement, Sn::Return(vec![var(0)])],
    )
}

/// The shapes that lowering must translate, each under the name of its
/// golden file.
pub(crate) fn cases() -> Vec<Case> {
    let mut cases = loops();
    cases.extend(loop_transfers());
    cases.extend(branches());
    cases.extend(blocks());
    cases.extend(exceptions());
    cases
}

/// Every loop form.
fn loops() -> Vec<Case> {
    vec![
        (
            "counting-loop",
            then_return(
                2,
                &[0, 1],
                Sn::While(
                    less(var(0), var(1)),
                    vec![
                        when(less(var(0), num(100)), vec![Sn::Break(None)]),
                        Sn::Assign(0, add(var(0), num(1))),
                    ],
                ),
            ),
        ),
        (
            "do-while",
            then_return(
                1,
                &[0],
                Sn::DoWhile(
                    vec![
                        Sn::Assign(0, add(var(0), num(1))),
                        when(less(var(0), num(5)), vec![Sn::Continue(None)]),
                        effect(vec![var(0)]),
                    ],
                    less(var(0), num(10)),
                ),
            ),
        ),
        (
            "nested-do-while",
            then_return(
                1,
                &[0],
                Sn::DoWhile(
                    vec![Sn::DoWhile(
                        vec![Sn::Assign(0, add(var(0), num(1)))],
                        less(var(0), num(3)),
                    )],
                    less(var(0), num(9)),
                ),
            ),
        ),
        (
            "for-loop",
            then_return(
                2,
                &[1],
                Sn::For(
                    vec![Sn::Assign(0, num(0))],
                    Some(less(var(0), var(1))),
                    vec![Sn::Assign(0, add(var(0), num(1)))],
                    vec![
                        when(less(var(0), num(2)), vec![Sn::Continue(None)]),
                        effect(vec![var(0)]),
                        when(less(num(7), var(0)), vec![Sn::Break(None)]),
                    ],
                ),
            ),
        ),
        (
            "for-without-condition-or-update",
            then_return(
                1,
                &[0],
                Sn::For(
                    Vec::new(),
                    None,
                    Vec::new(),
                    vec![
                        when(less(num(3), var(0)), vec![Sn::Break(None)]),
                        Sn::Assign(0, add(var(0), num(1))),
                    ],
                ),
            ),
        ),
        (
            "endless-loop",
            then_return(
                1,
                &[0],
                Sn::Loop(vec![
                    when(less(num(9), var(0)), vec![Sn::Break(None)]),
                    Sn::Assign(0, add(var(0), num(1))),
                ]),
            ),
        ),
    ]
}

/// The labels that name loops, and the transfers inside nested loops and
/// switches.
fn loop_transfers() -> Vec<Case> {
    vec![
        (
            "labeled-loops",
            then_return(
                2,
                &[0, 1],
                Sn::Labeled(
                    "outer",
                    vec![Sn::While(
                        less(var(0), var(1)),
                        vec![Sn::Labeled(
                            "inner",
                            vec![Sn::While(
                                less(var(1), num(50)),
                                vec![
                                    when(less(var(0), num(1)), vec![Sn::Break(Some("outer"))]),
                                    when(less(var(0), num(2)), vec![Sn::Continue(Some("outer"))]),
                                    when(less(var(0), num(3)), vec![Sn::Break(Some("inner"))]),
                                    Sn::Assign(1, add(var(1), num(1))),
                                ],
                            )],
                        )],
                    )],
                ),
            ),
        ),
        (
            "switch-in-loop",
            function(
                2,
                &[0],
                vec![
                    Sn::While(
                        less(var(1), num(10)),
                        vec![Sn::Switch(
                            var(0),
                            vec![
                                (vec![1], vec![Sn::Assign(1, num(5)), Sn::Break(None)]),
                                (vec![2], vec![Sn::Continue(None)]),
                                (vec![3, 4], vec![effect(vec![var(1)])]),
                            ],
                            vec![Sn::Assign(1, add(var(1), num(1)))],
                        )],
                    ),
                    Sn::Return(vec![var(1)]),
                ],
            ),
        ),
    ]
}

/// Branches and joins, and gotos in both directions.
fn branches() -> Vec<Case> {
    vec![
        (
            "redispatch",
            function(
                2,
                &[0],
                vec![
                    Sn::Switch(
                        var(0),
                        vec![
                            (vec![1, 2], vec![Sn::Assign(1, num(10))]),
                            (vec![3], vec![Sn::Assign(1, num(20))]),
                        ],
                        vec![Sn::Assign(1, num(30))],
                    ),
                    Sn::Return(vec![var(1)]),
                ],
            ),
        ),
        (
            "if-else-join",
            function(
                3,
                &[0, 1],
                vec![
                    Sn::If(
                        less(var(0), var(1)),
                        vec![Sn::Assign(2, add(var(0), var(1)))],
                        vec![Sn::Assign(2, add(var(1), var(0)))],
                    ),
                    Sn::Return(vec![var(2)]),
                ],
            ),
        ),
        (
            "early-return",
            function(
                1,
                &[0],
                vec![
                    when(less(var(0), num(0)), vec![Sn::Return(vec![num(1)])]),
                    when(less(var(0), num(9)), Vec::new()),
                    Sn::Return(vec![num(2)]),
                ],
            ),
        ),
        (
            "goto-both-ways",
            function(
                1,
                &[0],
                vec![
                    Sn::Labeled(
                        "top",
                        vec![
                            Sn::Assign(0, add(var(0), num(1))),
                            when(less(var(0), num(4)), vec![Sn::Goto("top")]),
                            when(less(var(0), num(8)), vec![Sn::Goto("end")]),
                            effect(vec![var(0)]),
                        ],
                    ),
                    Sn::Assign(0, num(0)),
                    Sn::Labeled("end", vec![Sn::Return(vec![var(0)])]),
                ],
            ),
        ),
    ]
}

/// Labeled blocks, the forms of straight-line code, and a region that wraps a
/// body.
fn blocks() -> Vec<Case> {
    vec![
        (
            "labeled-block-break",
            then_return(
                1,
                &[0],
                Sn::Labeled(
                    "block",
                    vec![
                        when(less(var(0), num(1)), vec![Sn::Break(Some("block"))]),
                        Sn::Assign(0, add(var(0), num(1))),
                    ],
                ),
            ),
        ),
        (
            "expression-forms",
            function(
                2,
                &[0],
                vec![
                    Sn::Eval(Ex::Apply(
                        Prim::Call,
                        Ty::Void,
                        vec![add(var(0), num(4)), num(2)],
                    )),
                    Sn::Assign(1, var(0)),
                    Sn::Assign(1, num(3)),
                    Sn::Put(Prim::Add, vec![var(0), num(8)], call(vec![var(1)])),
                    Sn::Eval(var(0)),
                    Sn::Eval(num(1)),
                    Sn::Return(vec![var(0), num(5)]),
                ],
            ),
        ),
        (
            "unreachable-after-return",
            function(
                1,
                &[0],
                vec![
                    Sn::Return(vec![var(0)]),
                    Sn::Assign(0, num(1)),
                    Sn::Return(vec![var(0)]),
                ],
            ),
        ),
        (
            "acquire-region",
            then_return(
                1,
                &[0],
                Sn::Region(
                    Prim::Acquire,
                    vec![var(0)],
                    vec![effect(vec![var(0)]), Sn::Assign(0, add(var(0), num(1)))],
                ),
            ),
        ),
    ]
}

/// Protected regions: handlers of every class, a finally clause, nesting, and
/// a filter funclet.
fn exceptions() -> Vec<Case> {
    let catch_all = |body: Vec<Sn>| Hn {
        class: Class::CatchAll,
        binding: None,
        body,
    };
    vec![
        (
            "guarded-call",
            function(
                1,
                &[],
                vec![
                    Sn::Try(
                        vec![Sn::Assign(0, call(Vec::new()))],
                        vec![catch_all(vec![Sn::Return(vec![num(7)])])],
                        Vec::new(),
                    ),
                    Sn::Return(vec![var(0)]),
                ],
            ),
        ),
        (
            "try-handlers-and-finally",
            function(
                3,
                &[0],
                vec![
                    Sn::Try(
                        vec![Sn::Assign(1, call(vec![var(0)]))],
                        vec![
                            Hn {
                                class: Class::Catch,
                                binding: Some(2),
                                body: vec![Sn::Assign(1, var(2))],
                            },
                            Hn {
                                class: Class::Fault,
                                binding: None,
                                body: vec![Sn::Assign(1, num(0))],
                            },
                            catch_all(vec![Sn::Return(vec![num(5)])]),
                        ],
                        vec![Sn::Assign(1, add(var(1), num(1)))],
                    ),
                    Sn::Return(vec![var(1)]),
                ],
            ),
        ),
        (
            "nested-try",
            function(
                2,
                &[0],
                vec![
                    Sn::Try(
                        vec![
                            effect(vec![var(0)]),
                            Sn::Try(
                                vec![Sn::Assign(1, call(vec![var(0)]))],
                                vec![catch_all(vec![Sn::Assign(1, num(1))])],
                                Vec::new(),
                            ),
                        ],
                        vec![catch_all(vec![Sn::Return(vec![num(2)])])],
                        Vec::new(),
                    ),
                    Sn::Return(vec![var(1)]),
                ],
            ),
        ),
        (
            "filter-handler",
            function(
                1,
                &[0],
                vec![
                    Sn::Try(
                        vec![effect(vec![var(0)])],
                        vec![Hn {
                            class: Class::Filter(vec![Sn::Return(vec![num(1)])]),
                            binding: None,
                            body: vec![Sn::Assign(0, num(9))],
                        }],
                        Vec::new(),
                    ),
                    Sn::Return(vec![var(0)]),
                ],
            ),
        ),
    ]
}
