//! Running Glint functions through the view alone.
//!
//! The interpreter follows the documented meaning of the view: the successor
//! arguments become the parameters of the target in order and all at once, and
//! the parameters of the entry block are the parameters of the function. The
//! expected results come from plain integer arithmetic.

use crate::fixtures;
use crate::interpreter::{Outcome, run};

#[test]
fn the_counting_loop_sums_the_counter_through_its_block_parameters() {
    let function = fixtures::counting_sum();
    for n in -2..30 {
        let expected: i32 = (0..n).sum();
        assert_eq!(run(&function, &[n]), Outcome::Returned(expected), "n = {n}");
    }
}

#[test]
fn successor_arguments_are_copied_in_parallel_so_the_loop_computes_fibonacci() {
    let function = fixtures::fibonacci();
    let (mut older, mut newer) = (0_i32, 1_i32);
    for n in 0..=40 {
        assert_eq!(run(&function, &[n]), Outcome::Returned(older), "n = {n}");
        (older, newer) = (newer, older + newer);
    }
    // A loop that does not start never reads the loop parameters again.
    assert_eq!(run(&function, &[-5]), Outcome::Returned(0));
}

#[test]
fn a_join_parameter_takes_the_value_of_the_arm_that_ran() {
    let function = fixtures::absolute_value();
    for x in -5..=5_i32 {
        assert_eq!(run(&function, &[x]), Outcome::Returned(x.abs()), "x = {x}");
    }
}

#[test]
fn a_switch_takes_its_case_record_or_the_default() {
    let function = fixtures::switch_with_parallel_records();
    for (x, expected) in [(1, 10), (2, 10), (3, 20), (0, 30), (7, 30), (-1, 30)] {
        assert_eq!(run(&function, &[x]), Outcome::Returned(expected), "x = {x}");
    }
}

#[test]
fn a_terminator_without_successors_ends_the_run() {
    let function = fixtures::discarding();
    assert_eq!(run(&function, &[1]), Outcome::Discarded);
    assert_eq!(run(&function, &[0]), Outcome::Returned(3));
    assert_eq!(run(&fixtures::single_block(), &[]), Outcome::Returned(42));
}

#[test]
fn a_block_that_loops_forever_never_ends_the_run() {
    let function = fixtures::with_an_endless_loop();
    assert_eq!(run(&function, &[1]), Outcome::OutOfSteps);
    assert_eq!(run(&function, &[0]), Outcome::Returned(1));
}

#[test]
fn unreachable_and_removed_blocks_take_no_part_in_a_run() {
    assert_eq!(
        run(&fixtures::with_an_unreachable_block(), &[]),
        Outcome::Returned(4)
    );
    assert_eq!(
        run(&fixtures::with_a_removed_block(), &[]),
        Outcome::Returned(5)
    );
}

#[test]
fn a_block_may_pass_its_own_parameter_back_to_itself() {
    assert_eq!(run(&fixtures::self_loop(), &[9]), Outcome::Returned(9));
}

#[test]
fn a_selection_and_a_product_run_without_a_branch() {
    let function = fixtures::branchless_square_of_max();
    for (a, b) in [(3, 5), (5, 3), (-4, 2), (-4, -9), (7, 7)] {
        let larger: i32 = std::cmp::max(a, b);
        assert_eq!(
            run(&function, &[a, b]),
            Outcome::Returned(larger * larger),
            "a = {a}, b = {b}"
        );
    }
}
