//! The corpus the pipeline tests run on: structured programs from the
//! builder (diamonds, nested diamonds, loops with break and continue) and
//! hand-wired graphs for what a structured stream cannot say (unreachable
//! components, a removed block slot, an unwind edge).

use cfglib::{Cfg, CfgBuilder, EdgeKind, FlowEffect};

use crate::toy::{Inst, Var};

/// One named program and the variables an execution starts with.
pub struct Case {
    /// Names the program in assertion messages.
    pub name: &'static str,
    /// The control-flow graph.
    pub cfg: Cfg<Inst<Var>>,
    /// The live-in variables; the interpreter tries many values for each.
    pub inputs: &'static [&'static str],
}

fn built(stream: Vec<Inst<Var>>) -> Cfg<Inst<Var>> {
    CfgBuilder::build(stream).expect("a well-formed structured stream")
}

fn otherwise() -> Inst<Var> {
    Inst::marker(FlowEffect::ConditionalAlternate)
}

fn end_if() -> Inst<Var> {
    Inst::marker(FlowEffect::ConditionalClose)
}

fn loop_open() -> Inst<Var> {
    Inst::marker(FlowEffect::LoopOpen)
}

fn loop_close() -> Inst<Var> {
    Inst::marker(FlowEffect::LoopClose)
}

fn break_if(condition: &'static str) -> Inst<Var> {
    Inst::branch(condition).with_flow(FlowEffect::ConditionalBreak)
}

fn continue_if(condition: &'static str) -> Inst<Var> {
    Inst::branch(condition).with_flow(FlowEffect::ConditionalContinue)
}

fn case(name: &'static str, cfg: Cfg<Inst<Var>>, inputs: &'static [&'static str]) -> Case {
    Case { name, cfg, inputs }
}

/// A copy chain, a dead chain, and straight-line arithmetic.
fn copy_chain() -> Case {
    case(
        "copy_chain",
        built(vec![
            Inst::add("t", "a", "b"),
            Inst::copy("u", "t"),
            Inst::copy("w", "u"),
            Inst::copy("v", "w"),
            Inst::mul("junk", "a", "b"),
            Inst::add("more_junk", "junk", "a"),
            Inst::print("v"),
            Inst::ret("v"),
        ]),
        &["a", "b"],
    )
}

/// A branch every execution resolves the same way.
fn constant_condition_diamond() -> Case {
    case(
        "constant_condition_diamond",
        built(vec![
            Inst::konst("x", 3),
            Inst::konst("y", 4),
            Inst::less("c", "x", "y"),
            Inst::branch("c"),
            Inst::add("z", "x", "y"),
            otherwise(),
            Inst::mul("z", "x", "y"),
            end_if(),
            Inst::print("z"),
            Inst::ret("z"),
        ]),
        &[],
    )
}

/// A diamond whose arms copy different inputs.
fn minimum_diamond() -> Case {
    case(
        "minimum_diamond",
        built(vec![
            Inst::less("c", "a", "b"),
            Inst::branch("c"),
            Inst::copy("m", "a"),
            otherwise(),
            Inst::copy("m", "b"),
            end_if(),
            Inst::print("m"),
            Inst::ret("m"),
        ]),
        &["a", "b"],
    )
}

/// Diamonds nested in both arms of a diamond.
fn nested_diamonds() -> Case {
    case(
        "nested_diamonds",
        built(vec![
            Inst::less("x", "a", "b"),
            Inst::branch("x"),
            Inst::less("y", "b", "c"),
            Inst::branch("y"),
            Inst::copy("r", "c"),
            otherwise(),
            Inst::copy("r", "b"),
            end_if(),
            otherwise(),
            Inst::less("z", "a", "c"),
            Inst::branch("z"),
            Inst::copy("r", "c"),
            otherwise(),
            Inst::copy("r", "a"),
            end_if(),
            end_if(),
            Inst::print("r"),
            Inst::ret("r"),
        ]),
        &["a", "b", "c"],
    )
}

/// Sums `0..=n` with a conditional break at the top of the loop.
fn counting_loop() -> Case {
    case(
        "counting_loop",
        built(vec![
            Inst::konst("s", 0),
            Inst::konst("i", 0),
            Inst::konst("one", 1),
            loop_open(),
            Inst::less("stop", "n", "i"),
            break_if("stop"),
            Inst::add("s", "s", "i"),
            Inst::add("i", "i", "one"),
            loop_close(),
            Inst::print("s"),
            Inst::ret("s"),
        ]),
        &["n"],
    )
}

/// A loop with a conditional break and a conditional continue.
fn loop_with_continue() -> Case {
    case(
        "loop_with_continue",
        built(vec![
            Inst::konst("s", 0),
            Inst::konst("i", 0),
            Inst::konst("one", 1),
            Inst::konst("two", 2),
            loop_open(),
            Inst::add("i", "i", "one"),
            Inst::less("stop", "n", "i"),
            break_if("stop"),
            Inst::less("skip", "i", "two"),
            continue_if("skip"),
            Inst::add("s", "s", "i"),
            loop_close(),
            Inst::print("s"),
            Inst::ret("s"),
        ]),
        &["n"],
    )
}

/// A loop nested in a loop, with a product accumulated in the inner one.
fn nested_loops() -> Case {
    case(
        "nested_loops",
        built(vec![
            Inst::konst("s", 0),
            Inst::konst("i", 0),
            Inst::konst("one", 1),
            loop_open(),
            Inst::less("stop_outer", "n", "i"),
            break_if("stop_outer"),
            Inst::konst("j", 0),
            loop_open(),
            Inst::less("stop_inner", "m", "j"),
            break_if("stop_inner"),
            Inst::mul("t", "i", "j"),
            Inst::add("s", "s", "t"),
            Inst::add("j", "j", "one"),
            loop_close(),
            Inst::add("i", "i", "one"),
            loop_close(),
            Inst::print("s"),
            Inst::ret("s"),
        ]),
        &["n", "m"],
    )
}

/// Code after a `return` that no path reaches.
fn code_after_return() -> Case {
    case(
        "code_after_return",
        built(vec![
            Inst::print("a"),
            Inst::ret("a"),
            Inst::konst("dead", 9),
            Inst::print("dead"),
        ]),
        &["a"],
    )
}

/// A copy of an input used on both arms of a diamond and after it.
fn copies_across_a_diamond() -> Case {
    case(
        "copies_across_a_diamond",
        built(vec![
            Inst::add("u", "a", "b"),
            Inst::copy("v", "u"),
            Inst::less("c", "v", "a"),
            Inst::branch("c"),
            Inst::add("w", "v", "a"),
            otherwise(),
            Inst::mul("w", "v", "u"),
            end_if(),
            Inst::print("w"),
            Inst::ret("v"),
        ]),
        &["a", "b"],
    )
}

/// A copy of an input made once before a loop that reads it every iteration.
fn input_copy_read_in_loop() -> Case {
    case(
        "input_copy_read_in_loop",
        built(vec![
            Inst::copy("k", "n"),
            Inst::konst("s", 0),
            Inst::konst("i", 0),
            Inst::konst("one", 1),
            loop_open(),
            Inst::less("stop", "k", "i"),
            break_if("stop"),
            Inst::add("s", "s", "i"),
            Inst::add("i", "i", "one"),
            loop_close(),
            Inst::print("s"),
            Inst::ret("s"),
        ]),
        &["n"],
    )
}

/// Every swap of a pair through a loop: the loop-carried parallel copy.
fn swap_loop() -> Case {
    case(
        "swap_loop",
        built(vec![
            Inst::copy("x", "a"),
            Inst::copy("y", "b"),
            Inst::konst("i", 0),
            Inst::konst("one", 1),
            loop_open(),
            Inst::less("stop", "n", "i"),
            break_if("stop"),
            Inst::parallel_copy(vec![Var("x"), Var("y")], vec![Var("y"), Var("x")]),
            Inst::add("i", "i", "one"),
            loop_close(),
            Inst::print("x"),
            Inst::print("y"),
            Inst::ret("x"),
        ]),
        &["a", "b", "n"],
    )
}

/// A parallel copy that is the sole definition of each destination.
fn single_definition_parallel_copy() -> Case {
    case(
        "single_definition_parallel_copy",
        built(vec![
            Inst::parallel_copy(vec![Var("p"), Var("q")], vec![Var("a"), Var("b")]),
            Inst::add("s", "p", "q"),
            Inst::print("s"),
            Inst::ret("s"),
        ]),
        &["a", "b"],
    )
}

/// A copy inside a loop whose source is redefined by the loop.
fn loop_carried_copy() -> Case {
    case(
        "loop_carried_copy",
        built(vec![
            Inst::copy("x", "a"),
            Inst::konst("i", 0),
            Inst::konst("one", 1),
            loop_open(),
            Inst::less("stop", "n", "i"),
            break_if("stop"),
            Inst::copy("y", "x"),
            Inst::add("x", "y", "one"),
            Inst::add("i", "i", "one"),
            loop_close(),
            Inst::print("x"),
            Inst::ret("x"),
        ]),
        &["a", "n"],
    )
}

/// A scratch block between the entry and the diamond was removed before any
/// analysis, so block identities have a hole.
fn removed_block_slot() -> Case {
    let mut cfg = Cfg::new();
    let entry = cfg.entry();
    let scratch = cfg.new_block();
    let left = cfg.new_block();
    let right = cfg.new_block();
    let join = cfg.new_block();
    cfg.block_mut(entry).push(Inst::less("c", "a", "b"));
    cfg.block_mut(entry).push(Inst::branch("c"));
    cfg.block_mut(scratch).push(Inst::konst("never", 0));
    cfg.block_mut(left).push(Inst::add("x", "a", "b"));
    cfg.block_mut(right).push(Inst::sub("x", "a", "b"));
    cfg.block_mut(join).push(Inst::print("x"));
    cfg.block_mut(join).push(Inst::ret("x"));
    cfg.add_edge(entry, scratch, EdgeKind::Jump);
    cfg.add_edge(scratch, join, EdgeKind::Fallthrough);
    cfg.add_edge(entry, left, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, right, EdgeKind::ConditionalFalse);
    cfg.add_edge(left, join, EdgeKind::Fallthrough);
    cfg.add_edge(right, join, EdgeKind::Fallthrough);
    assert!(cfg.remove_block(scratch));
    case("removed_block_slot", cfg, &["a", "b"])
}

/// A reachable return and, beside it, an unreachable two-block loop that
/// defines and reads its own values.
fn unreachable_loop_component() -> Case {
    let mut cfg = Cfg::new();
    let entry = cfg.entry();
    let head = cfg.new_block();
    let body = cfg.new_block();
    cfg.block_mut(entry).push(Inst::print("a"));
    cfg.block_mut(entry).push(Inst::ret("a"));
    cfg.block_mut(head).push(Inst::add("x", "a", "a"));
    cfg.block_mut(body).push(Inst::add("a", "x", "x"));
    cfg.block_mut(body).push(Inst::print("a"));
    cfg.add_edge(head, body, EdgeKind::Fallthrough);
    cfg.add_edge(body, head, EdgeKind::Back);
    case("unreachable_loop_component", cfg, &["a"])
}

/// A loop whose header is the entry block, as a CFG recovered from a
/// function whose first instruction is a branch target has. The entry prints
/// the carried value `k` before the body replaces it with a constant, so the
/// first iteration prints the caller's value and every later one prints 7:
/// the entry's phi for `k` merges the value the function starts with and
/// the loop's constant.
fn entry_headed_loop() -> Case {
    let mut cfg = Cfg::new();
    let entry = cfg.entry();
    let body = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(entry).push(Inst::print("k"));
    cfg.block_mut(entry).push(Inst::less("stop", "n", "i"));
    cfg.block_mut(entry).push(Inst::branch("stop"));
    cfg.block_mut(body).push(Inst::konst("k", 7));
    cfg.block_mut(body).push(Inst::konst("one", 1));
    cfg.block_mut(body).push(Inst::add("i", "i", "one"));
    cfg.block_mut(exit).push(Inst::print("k"));
    cfg.block_mut(exit).push(Inst::ret("k"));
    cfg.add_edge(entry, exit, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, body, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, entry, EdgeKind::Back);
    case("entry_headed_loop", cfg, &["n", "i", "k"])
}

/// A loop whose header is the entry block and holds a copy `c = x`, with a
/// read of `c` before it. The body then increments `x`, so on every later
/// iteration the read at the top of the entry sees the copy from the
/// iteration before, which no longer equals `x`: the copy does not reach
/// that read, and a propagation that substituted `x` there would print the
/// wrong value.
fn entry_loop_with_copy() -> Case {
    let mut cfg = Cfg::new();
    let entry = cfg.entry();
    let body = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(entry).push(Inst::print("c"));
    cfg.block_mut(entry).push(Inst::copy("c", "x"));
    cfg.block_mut(entry).push(Inst::print("c"));
    cfg.block_mut(entry).push(Inst::less("stop", "n", "i"));
    cfg.block_mut(entry).push(Inst::branch("stop"));
    cfg.block_mut(body).push(Inst::konst("one", 1));
    cfg.block_mut(body).push(Inst::add("x", "x", "one"));
    cfg.block_mut(body).push(Inst::add("i", "i", "one"));
    cfg.block_mut(exit).push(Inst::ret("c"));
    cfg.add_edge(entry, exit, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, body, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, entry, EdgeKind::Back);
    case("entry_loop_with_copy", cfg, &["n", "i", "x", "c"])
}

/// Two throwing divisions in one block; the handler prints the variable the
/// block redefines between them, so it must merge both unwind points.
fn division_with_handler() -> Case {
    let mut cfg = Cfg::new();
    let entry = cfg.entry();
    let done = cfg.new_block();
    let handler = cfg.new_block();
    for inst in [
        Inst::copy("t", "a"),
        Inst::divide("d", "t", "b"),
        Inst::copy("t", "d"),
        Inst::divide("e", "t", "c"),
        Inst::add("r", "d", "e"),
    ] {
        cfg.block_mut(entry).push(inst);
    }
    cfg.block_mut(done).push(Inst::print("r"));
    cfg.block_mut(done).push(Inst::ret("r"));
    cfg.block_mut(handler).push(Inst::print("t"));
    cfg.block_mut(handler).push(Inst::ret("t"));
    cfg.add_edge(entry, done, EdgeKind::Fallthrough);
    cfg.add_edge(entry, handler, EdgeKind::ExceptionUnwind);
    case("division_with_handler", cfg, &["a", "b", "c"])
}

/// A division by a value constant propagation can see is zero.
fn constant_zero_divisor() -> Case {
    let mut cfg = Cfg::new();
    let entry = cfg.entry();
    let done = cfg.new_block();
    let handler = cfg.new_block();
    cfg.block_mut(entry).push(Inst::konst("zero", 0));
    cfg.block_mut(entry).push(Inst::divide("q", "a", "zero"));
    cfg.block_mut(done).push(Inst::print("q"));
    cfg.block_mut(done).push(Inst::ret("q"));
    cfg.block_mut(handler).push(Inst::konst("recovered", -1));
    cfg.block_mut(handler).push(Inst::print("recovered"));
    cfg.block_mut(handler).push(Inst::ret("recovered"));
    cfg.add_edge(entry, done, EdgeKind::Fallthrough);
    cfg.add_edge(entry, handler, EdgeKind::ExceptionUnwind);
    case("constant_zero_divisor", cfg, &["a"])
}

/// Every program, structured ones first.
pub fn corpus() -> Vec<Case> {
    vec![
        copy_chain(),
        copies_across_a_diamond(),
        input_copy_read_in_loop(),
        constant_condition_diamond(),
        minimum_diamond(),
        nested_diamonds(),
        counting_loop(),
        loop_with_continue(),
        nested_loops(),
        code_after_return(),
        swap_loop(),
        single_definition_parallel_copy(),
        loop_carried_copy(),
        removed_block_slot(),
        unreachable_loop_component(),
        division_with_handler(),
        constant_zero_divisor(),
        entry_headed_loop(),
        entry_loop_with_copy(),
    ]
}

/// The program called `name`.
///
/// # Panics
///
/// Panics when the corpus has no such program.
pub fn program(name: &str) -> Case {
    corpus()
        .into_iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("no program named {name}"))
}
