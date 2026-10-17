//! Vectored handlers and registered SEH frames composed into one dispatch
//! order, with `ExceptionDisposition` as each handler's answer.
//!
//! The visit order is read straight from the two models: vectored exception
//! handlers in call order, then registered frames newest first; a handler that
//! continues execution makes the dispatcher consult the continue list. The
//! expected visits are written out by hand for each scenario.

use cfglib::{
    ExceptionDisposition, SehRegistration, SehRegistrationChain, VectoredHandlerKind,
    VectoredHandlerOrder, VehModel,
};

type Frames = SehRegistrationChain<&'static str, u32>;

/// The handlers one exception visits, in order, and how dispatch ended.
fn dispatch(
    vectored: &VehModel<&'static str>,
    frames: &Frames,
    answer: impl Fn(&str) -> ExceptionDisposition,
) -> (Vec<String>, ExceptionDisposition) {
    let candidates = vectored
        .handlers(VectoredHandlerKind::Exception)
        .map(|registration| format!("veh:{}", registration.handler))
        .chain(
            frames
                .dispatch_order()
                .map(|registration| format!("seh:{}", registration.frame)),
        );
    let mut visited = Vec::new();
    let mut outcome = ExceptionDisposition::ContinueSearch;
    for name in candidates {
        outcome = answer(&name);
        visited.push(name);
        if outcome != ExceptionDisposition::ContinueSearch {
            break;
        }
    }
    if outcome == ExceptionDisposition::ContinueExecution {
        visited.extend(
            vectored
                .handlers(VectoredHandlerKind::Continue)
                .map(|registration| format!("continue:{}", registration.handler)),
        );
    }
    (visited, outcome)
}

fn process() -> (VehModel<&'static str>, Frames) {
    let mut vectored = VehModel::new();
    for (kind, order, name) in [
        (
            VectoredHandlerKind::Exception,
            VectoredHandlerOrder::Last,
            "tail",
        ),
        (
            VectoredHandlerKind::Exception,
            VectoredHandlerOrder::First,
            "first-a",
        ),
        (
            VectoredHandlerKind::Exception,
            VectoredHandlerOrder::First,
            "first-b",
        ),
        (
            VectoredHandlerKind::Continue,
            VectoredHandlerOrder::Last,
            "watch-1",
        ),
        (
            VectoredHandlerKind::Continue,
            VectoredHandlerOrder::First,
            "watch-0",
        ),
    ] {
        vectored.register(kind, order, name);
    }
    let mut frames = Frames::new();
    for (frame, handler) in [("outer", 0x0040_1000), ("inner", 0x0040_2000)] {
        frames.push(SehRegistration {
            frame,
            handler,
            regions: Vec::new(),
        });
    }
    (vectored, frames)
}

#[test]
fn a_selected_handler_ends_the_search_after_every_vectored_handler_declined() {
    let (vectored, frames) = process();
    let (visited, outcome) = dispatch(&vectored, &frames, |name| match name {
        "seh:inner" => ExceptionDisposition::ExecuteHandler,
        _ => ExceptionDisposition::ContinueSearch,
    });
    assert_eq!(
        visited,
        ["veh:first-b", "veh:first-a", "veh:tail", "seh:inner"]
    );
    assert_eq!(outcome, ExceptionDisposition::ExecuteHandler);
}

#[test]
fn a_vectored_handler_that_continues_execution_skips_the_frames_and_wakes_the_continue_list() {
    let (vectored, frames) = process();
    let (visited, outcome) = dispatch(&vectored, &frames, |name| match name {
        "veh:first-a" => ExceptionDisposition::ContinueExecution,
        _ => ExceptionDisposition::ContinueSearch,
    });
    assert_eq!(
        visited,
        [
            "veh:first-b",
            "veh:first-a",
            "continue:watch-0",
            "continue:watch-1"
        ]
    );
    assert_eq!(outcome, ExceptionDisposition::ContinueExecution);
}

#[test]
fn an_exception_nobody_handles_visits_every_handler_newest_frame_first() {
    let (vectored, frames) = process();
    let (visited, outcome) = dispatch(&vectored, &frames, |_| ExceptionDisposition::ContinueSearch);
    assert_eq!(
        visited,
        [
            "veh:first-b",
            "veh:first-a",
            "veh:tail",
            "seh:inner",
            "seh:outer"
        ]
    );
    assert_eq!(outcome, ExceptionDisposition::ContinueSearch);
}

#[test]
fn unregistering_a_frame_or_a_vectored_handler_changes_the_next_dispatch() {
    let (mut vectored, mut frames) = process();
    let popped = frames.pop().expect("the inner frame is active");
    assert_eq!(popped.frame, "inner");
    let first_b = vectored
        .handlers(VectoredHandlerKind::Exception)
        .next()
        .expect("a handler leads the list")
        .id;
    let removed = vectored.remove(first_b).expect("the handler is registered");
    assert_eq!(removed.handler, "first-b");

    let (visited, _) = dispatch(&vectored, &frames, |_| ExceptionDisposition::ContinueSearch);
    assert_eq!(visited, ["veh:first-a", "veh:tail", "seh:outer"]);
}
