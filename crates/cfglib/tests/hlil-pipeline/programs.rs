//! The programs that the pipeline runs, each with the input vectors that
//! exercise its paths.
//!
//! Every program meets the contract of `lower`: each path ends in a return
//! or iterates forever, and no `break` or `return` leaves a `try` that has a
//! `finally` (lowering does not route those through it). A throwing `try`
//! always has a catch handler, because a `finally` that nothing catches
//! continues after itself in flat form.

mod control;
mod exceptions;
mod recovery;

pub(crate) use control::*;
pub(crate) use exceptions::*;
pub(crate) use recovery::*;

use crate::source::{E, Program, int};

/// A program and the inputs to run it on.
pub(crate) struct Case {
    pub(crate) program: Program,
    pub(crate) inputs: Vec<Vec<i64>>,
}

fn case(program: Program, inputs: &[&[i64]]) -> Case {
    Case {
        program,
        inputs: inputs.iter().map(|input| input.to_vec()).collect(),
    }
}

fn zero() -> E {
    int(0)
}

/// Every program that lowers and lifts without a documented exception.
pub(crate) fn all() -> Vec<Case> {
    vec![
        branches(),
        sum_while(),
        do_while_steps(),
        counted_for(),
        nested_loops(),
        break_continue(),
        while_break_continue(),
        labeled_break(),
        labeled_continue(),
        switch_dispatch(),
        switch_in_loop(),
        early_returns(),
        division_guard(),
        typed_catch(),
        catch_and_finally(),
        nested_try(),
        catch_in_loop(),
        return_in_try(),
        transfers_out_of_try(),
        goto_backward(),
        goto_forward(),
        irreducible_loop(),
        memory_walk(),
        load_store_order(),
        throw_order(),
        select_assign(),
        select_return(),
        select_guarding_division(),
        acquire_with_cleanup(),
        guarded_body(),
        straight_line(),
        counting_while(),
        explicit_pick(),
    ]
}
