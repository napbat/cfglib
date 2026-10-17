//! The execution oracle: a program, its lowered bytecode, the function that
//! `lift` rebuilds from the bytecode, and the function that
//! `recover_structure` rebuilds from that, observe the same trace.

use cfglib::ir::hlil::has_recovery_candidates;

use crate::flat_run::run_flat;
use crate::machine::{End, Outcome};
use crate::pipeline::{lifted, lowered, recovered, structured};
use crate::programs::{self, Case};
use crate::structured_run::run_structured;

/// Runs `check` on every input of every program with the reference outcome.
fn for_every_run(mut check: impl FnMut(&Case, &[i64], &Outcome)) {
    for case in programs::all() {
        let function = structured(&case.program);
        for input in &case.inputs {
            let reference = run_structured(&function, input);
            assert!(
                matches!(reference.end, End::Returned(_) | End::Uncaught(_)),
                "{} on {input:?} ends as {:?}, which no comparison can use",
                case.program.name,
                reference.end
            );
            check(&case, input, &reference);
        }
    }
}

#[test]
fn the_inputs_reach_returns_uncaught_exceptions_events_and_memory() {
    let (mut returned, mut uncaught, mut with_events, mut with_memory) = (0, 0, 0, 0);
    for_every_run(|_, _, reference| {
        match reference.end {
            End::Returned(_) => returned += 1,
            End::Uncaught(_) => uncaught += 1,
            _ => unreachable!("for_every_run admits only these two"),
        }
        with_events += usize::from(!reference.events.is_empty());
        with_memory += usize::from(!reference.memory.is_empty());
    });
    assert!(returned > 40, "only {returned} runs return");
    assert!(uncaught >= 1, "no run ends in an uncaught exception");
    assert!(with_events > 30, "only {with_events} runs emit events");
    assert!(with_memory >= 8, "only {with_memory} runs write memory");
}

#[test]
fn lowered_bytecode_observes_what_the_structured_program_observes() {
    let mut cached = None;
    for_every_run(|case, input, reference| {
        if cached.as_ref().map(|(name, _)| *name) != Some(case.program.name) {
            let code = lowered(&structured(&case.program)).function;
            cached = Some((case.program.name, code));
        }
        let (_, code) = cached.as_ref().expect("the bytecode was just lowered");
        assert_eq!(
            &run_flat(code, input),
            reference,
            "lowering {} changed the run on {input:?}",
            case.program.name
        );
    });
}

#[test]
fn lifting_the_lowered_bytecode_observes_what_the_program_observes() {
    let mut cached = None;
    for_every_run(|case, input, reference| {
        if cached.as_ref().map(|(name, _)| *name) != Some(case.program.name) {
            let code = lowered(&structured(&case.program)).function;
            cached = Some((case.program.name, lifted(&code).function));
        }
        let (_, function) = cached.as_ref().expect("the function was just lifted");
        assert_eq!(
            &run_structured(function, input),
            reference,
            "lift(lower({})) changed the run on {input:?}",
            case.program.name
        );
    });
}

#[test]
fn recovering_the_lifted_function_observes_what_the_program_observes() {
    let mut cached = None;
    for_every_run(|case, input, reference| {
        if cached.as_ref().map(|(name, _)| *name) != Some(case.program.name) {
            let code = lowered(&structured(&case.program)).function;
            let lifted = lifted(&code).function;
            let recovered = recovered(&lifted).function;
            cached = Some((case.program.name, recovered));
        }
        let (_, function) = cached.as_ref().expect("the function was just recovered");
        assert_eq!(
            &run_structured(function, input),
            reference,
            "recover(lift(lower({}))) changed the run on {input:?}",
            case.program.name
        );
    });
}

#[test]
fn recovering_the_original_program_observes_what_the_program_observes() {
    let mut cached = None;
    for_every_run(|case, input, reference| {
        if cached.as_ref().map(|(name, _)| *name) != Some(case.program.name) {
            let function = structured(&case.program);
            assert_eq!(
                has_recovery_candidates::<crate::dialect::Calc, _>(&function),
                function.stmts.iter().any(|statement| matches!(
                    statement,
                    crate::structured::StmtNode::If { .. }
                        | crate::structured::StmtNode::While { .. }
                        | crate::structured::StmtNode::Try { .. }
                )),
                "{}",
                case.program.name
            );
            cached = Some((case.program.name, recovered(&function).function));
        }
        let (_, function) = cached.as_ref().expect("the function was just recovered");
        assert_eq!(
            &run_structured(function, input),
            reference,
            "recovering {} changed the run on {input:?}",
            case.program.name
        );
    });
}
