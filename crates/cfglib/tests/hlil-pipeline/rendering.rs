//! `pseudocode` and `write_pseudocode`: the indented text that the pipeline
//! ends in, as goldens for the constructs a reader checks by eye.

use std::fmt;

use cfglib::ir::hlil::write_pseudocode;

use crate::dialect::Calc;
use crate::golden::assert_golden;
use crate::pipeline::{lifted, lowered, recovered, structured, text};
use crate::programs::{self, Case};

/// The text of the function that `lower`, `lift`, and `recover_structure`
/// hand a consumer, for the program of `case`.
fn pipeline_text(case: &Case) -> String {
    let code = lowered(&structured(&case.program)).function;
    text(&recovered(&lifted(&code).function).function)
}

#[test]
fn a_counting_loop_is_printed_as_a_for_with_its_clauses() {
    assert_golden(
        "hlil-pipeline/counting-loop-recovered-as-for.pseudo",
        &pipeline_text(&programs::sum_while()),
    );
}

#[test]
fn nested_loops_print_a_labeled_block_and_a_labeled_break() {
    assert_golden(
        "hlil-pipeline/labeled-break-out-of-nested-loops.pseudo",
        &pipeline_text(&programs::labeled_break()),
    );
}

#[test]
fn a_switch_prints_its_arms_with_every_case_value() {
    assert_golden(
        "hlil-pipeline/switch-with-shared-arm.pseudo",
        &pipeline_text(&programs::switch_dispatch()),
    );
}

#[test]
fn a_try_prints_its_catch_with_the_caught_exception_inlined() {
    assert_golden(
        "hlil-pipeline/try-catch-division.pseudo",
        &pipeline_text(&programs::division_guard()),
    );
}

#[test]
fn a_recovered_region_prints_as_a_wrapped_body() {
    assert_golden(
        "hlil-pipeline/acquire-recovered-as-region.pseudo",
        &pipeline_text(&programs::acquire_with_cleanup()),
    );
}

#[test]
fn an_irreducible_loop_prints_its_goto_and_the_label_it_needs() {
    assert_golden(
        "hlil-pipeline/irreducible-loop-with-goto.pseudo",
        &pipeline_text(&programs::irreducible_loop()),
    );
}

#[test]
fn a_recovered_selection_prints_as_one_pick() {
    assert_golden(
        "hlil-pipeline/select-recovered-as-pick.pseudo",
        &pipeline_text(&programs::select_assign()),
    );
}

#[test]
fn writing_into_a_sink_gives_the_text_of_pseudocode() {
    for case in programs::all() {
        let function = structured(&case.program);
        let mut sink = String::new();
        write_pseudocode::<Calc, _>(&function, &mut sink).expect("a string sink accepts text");
        assert_eq!(sink, text(&function), "{}", case.program.name);
    }
}

/// A sink that accepts `capacity` bytes and then fails.
struct Overflowing {
    capacity: usize,
}

impl fmt::Write for Overflowing {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.capacity = self.capacity.checked_sub(text.len()).ok_or(fmt::Error)?;
        Ok(())
    }
}

#[test]
fn a_failing_sink_ends_the_write_with_its_error() {
    let function = structured(&programs::nested_loops().program);
    let full = text(&function).len();
    assert!(write_pseudocode::<Calc, _>(&function, &mut Overflowing { capacity: full }).is_ok());
    assert!(
        write_pseudocode::<Calc, _>(&function, &mut Overflowing { capacity: full / 2 }).is_err()
    );
}

#[test]
fn variables_print_as_v_and_their_index_and_operations_as_mnemonic_calls() {
    let function = structured(&programs::straight_line().program);
    let printed = text(&function);
    assert!(printed.contains("emit(add(v0, v1));"), "{printed}");
    assert!(printed.contains("cell(0) = mul(v0, v1);"), "{printed}");
    assert!(printed.contains("return cell(0);"), "{printed}");
}

#[test]
fn a_function_with_no_statement_prints_nothing() {
    let function = crate::structured::Builder::new().seal(&[]);
    assert_eq!(text(&function), "");
}
