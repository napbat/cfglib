//! Tail-call detection: the heuristic (a call straight into a return) and the
//! explicit markers a frontend sets.

use cfglib::{BlockId, Cfg, EdgeKind, TailCall, detect_explicit_tail_calls, detect_tail_calls};

use crate::stmt::{Route, Stmt, block, link};

/// The instruction index of a detected call, whichever way the analysis
/// spells "known".
fn index_of(call: &TailCall) -> Option<usize> {
    fn known(index: impl Into<Option<usize>>) -> Option<usize> {
        index.into()
    }
    known(call.inst_idx)
}

fn summary(calls: &[TailCall]) -> Vec<(BlockId, Option<usize>, bool)> {
    calls
        .iter()
        .map(|call| (call.block, index_of(call), call.explicit))
        .collect()
}

#[test]
fn a_block_ending_in_a_call_whose_only_successor_returns_is_a_tail_call_candidate() {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(Stmt::add("x", "a", "b"));
    cfg.block_mut(entry).push(Stmt::call("callee", &["r"], &[]));
    let exit = block(&mut cfg, [Stmt::ret()]);
    link(&mut cfg, entry, exit, EdgeKind::CallReturn);

    assert_eq!(summary(&detect_tail_calls(&cfg)), [(entry, Some(1), false)]);
}

#[test]
fn conditional_calls_count_as_calls() {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(Stmt::conditional_call("maybe"));
    let exit = block(&mut cfg, []);
    cfg.add_edge(entry, exit, EdgeKind::Fallthrough);

    assert_eq!(summary(&detect_tail_calls(&cfg)), [(entry, Some(0), false)]);
}

#[test]
fn work_after_the_call_or_a_second_way_out_rules_a_block_out() {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let call_then_work = cfg.entry();
    cfg.block_mut(call_then_work)
        .push(Stmt::call("f", &["r"], &[]));
    cfg.block_mut(call_then_work).push(Stmt::add("s", "r", "r"));
    let two_ways = block(&mut cfg, [Stmt::call("g", &[], &[])]);
    let middle = block(&mut cfg, [Stmt::call("h", &[], &[])]);
    let exit_a = block(&mut cfg, [Stmt::ret()]);
    let exit_b = block(&mut cfg, [Stmt::ret()]);
    let not_exit = block(&mut cfg, [Stmt::plain()]);
    link(&mut cfg, call_then_work, exit_a, EdgeKind::Fallthrough);
    link(&mut cfg, two_ways, exit_a, EdgeKind::ConditionalTrue);
    link(&mut cfg, two_ways, exit_b, EdgeKind::ConditionalFalse);
    link(&mut cfg, middle, not_exit, EdgeKind::Fallthrough);
    link(&mut cfg, not_exit, exit_b, EdgeKind::Fallthrough);

    assert_eq!(
        detect_tail_calls(&cfg),
        [],
        "the call is not last, the block has two exits, or the successor still has somewhere to go"
    );
}

#[test]
fn a_call_in_an_exit_block_has_no_return_after_it_to_fold_into() {
    let mut cfg = Cfg::<Stmt>::new();
    cfg.block_mut(cfg.entry())
        .push(Stmt::call("noreturn", &[], &[]));

    assert_eq!(detect_tail_calls(&cfg), []);
}

#[test]
fn a_call_followed_by_more_work_before_the_return_is_not_a_tail_call() {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(Stmt::call("f", &["r"], &[]));
    let epilogue = block(&mut cfg, [Stmt::add("s", "r", "r"), Stmt::ret()]);
    cfg.add_edge(entry, epilogue, EdgeKind::Fallthrough);

    assert_eq!(
        detect_tail_calls(&cfg),
        [],
        "the result is used after the call returns, so the call is not immediately followed by a return"
    );
}

#[test]
fn explicit_markers_are_reported_wherever_they_are_including_indirect_calls() {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    cfg.block_mut(entry)
        .push(Stmt::call("not_marked", &[], &[]));
    cfg.block_mut(entry).push(Stmt::tail_call("direct"));
    cfg.block_mut(entry).push(Stmt::indirect_tail_call());
    let later = block(
        &mut cfg,
        [Stmt::add("x", "a", "b"), Stmt::tail_call("again")],
    );
    let unreachable = block(&mut cfg, [Stmt::tail_call("dead")]);
    link(&mut cfg, entry, later, EdgeKind::Fallthrough);

    assert_eq!(
        summary(&detect_explicit_tail_calls(&cfg)),
        [
            (entry, Some(1), true),
            (entry, Some(2), true),
            (later, Some(1), true),
            (unreachable, Some(0), true),
        ],
        "every marked instruction, in block then instruction order, reachable or not"
    );

    cfg.remove_block(unreachable);
    assert_eq!(
        detect_explicit_tail_calls(&cfg).len(),
        3,
        "a removed block reports nothing"
    );
}

#[test]
fn the_two_detectors_are_independent_of_each_other() {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(Stmt::tail_call("marked"));
    let exit = block(&mut cfg, [Stmt::ret()]);
    cfg.add_edge(entry, exit, EdgeKind::Fallthrough);
    let plain = block(&mut cfg, [Stmt::call("plain", &[], &[])]);
    let exit_two = block(&mut cfg, [Stmt::ret()]);
    cfg.add_edge(entry, plain, EdgeKind::ConditionalTrue);
    cfg.add_edge(plain, exit_two, EdgeKind::Fallthrough);

    assert_eq!(summary(&detect_tail_calls(&cfg)), [(plain, Some(0), false)]);
    assert_eq!(
        summary(&detect_explicit_tail_calls(&cfg)),
        [(entry, Some(0), true)]
    );
}
