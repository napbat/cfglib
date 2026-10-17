//! The programs the destruction tests lower, and the SSA analysis they share.

use cfglib::{BlockId, Cfg, DominatorTree, EdgeKind, ProgramPoint, SsaForm};

use crate::machine::{Insn, Reg};

pub struct Program {
    pub cfg: Cfg<Insn<Reg>>,
    pub inputs: &'static [&'static str],
}

pub fn wire(cfg: &mut Cfg<Insn<Reg>>, block: BlockId, insns: Vec<Insn<Reg>>) {
    cfg.block_mut(block).instructions_mut().extend(insns);
}

/// A diamond that merges one register.
pub fn diamond() -> Program {
    let mut cfg = Cfg::new();
    let (entry, left, right, merge) = (
        cfg.entry(),
        cfg.new_block(),
        cfg.new_block(),
        cfg.new_block(),
    );
    wire(
        &mut cfg,
        entry,
        vec![Insn::slt("c", "a", "b"), Insn::bnez("c")],
    );
    wire(&mut cfg, left, vec![Insn::add("x", "a", "a")]);
    wire(&mut cfg, right, vec![Insn::add("x", "b", "b")]);
    wire(&mut cfg, merge, vec![Insn::out("x"), Insn::ret("x")]);
    cfg.add_edge(entry, left, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, right, EdgeKind::ConditionalFalse);
    cfg.add_edge(left, merge, EdgeKind::Fallthrough);
    cfg.add_edge(right, merge, EdgeKind::Fallthrough);
    Program {
        cfg,
        inputs: &["a", "b"],
    }
}

/// A diamond whose join merges two registers, with the arms swapping them.
pub fn two_phi_diamond() -> Program {
    let mut cfg = Cfg::new();
    let (entry, left, right, merge) = (
        cfg.entry(),
        cfg.new_block(),
        cfg.new_block(),
        cfg.new_block(),
    );
    wire(
        &mut cfg,
        entry,
        vec![Insn::slt("c", "a", "b"), Insn::bnez("c")],
    );
    wire(
        &mut cfg,
        left,
        vec![Insn::add("x", "a", "a"), Insn::add("y", "b", "b")],
    );
    wire(
        &mut cfg,
        right,
        vec![Insn::add("x", "b", "b"), Insn::add("y", "a", "a")],
    );
    wire(
        &mut cfg,
        merge,
        vec![Insn::out("x"), Insn::out("y"), Insn::ret("x")],
    );
    cfg.add_edge(entry, left, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, right, EdgeKind::ConditionalFalse);
    cfg.add_edge(left, merge, EdgeKind::Fallthrough);
    cfg.add_edge(right, merge, EdgeKind::Fallthrough);
    Program {
        cfg,
        inputs: &["a", "b"],
    }
}

/// A diamond that defines `dead` on both arms and never reads it.
pub fn dead_phi_diamond() -> Program {
    let mut cfg = Cfg::new();
    let (entry, left, right, merge) = (
        cfg.entry(),
        cfg.new_block(),
        cfg.new_block(),
        cfg.new_block(),
    );
    wire(
        &mut cfg,
        entry,
        vec![Insn::slt("c", "a", "b"), Insn::bnez("c")],
    );
    wire(
        &mut cfg,
        left,
        vec![Insn::add("dead", "a", "a"), Insn::add("x", "a", "b")],
    );
    wire(
        &mut cfg,
        right,
        vec![Insn::add("dead", "b", "b"), Insn::add("x", "b", "a")],
    );
    wire(&mut cfg, merge, vec![Insn::out("x"), Insn::ret("x")]);
    cfg.add_edge(entry, left, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, right, EdgeKind::ConditionalFalse);
    cfg.add_edge(left, merge, EdgeKind::Fallthrough);
    cfg.add_edge(right, merge, EdgeKind::Fallthrough);
    Program {
        cfg,
        inputs: &["a", "b"],
    }
}

/// Swaps two registers `n + 1` times through a loop:
/// `t = x; x = y; y = t`.
pub fn swap_loop() -> Program {
    let mut cfg = Cfg::new();
    let (entry, header, body, exit) = (
        cfg.entry(),
        cfg.new_block(),
        cfg.new_block(),
        cfg.new_block(),
    );
    wire(
        &mut cfg,
        entry,
        vec![
            Insn::mov(Reg("x"), Reg("a")),
            Insn::mov(Reg("y"), Reg("b")),
            Insn::li("i", 0),
            Insn::li("one", 1),
        ],
    );
    wire(
        &mut cfg,
        header,
        vec![Insn::slt("stop", "n", "i"), Insn::bnez("stop")],
    );
    wire(
        &mut cfg,
        body,
        vec![
            Insn::mov(Reg("t"), Reg("x")),
            Insn::mov(Reg("x"), Reg("y")),
            Insn::mov(Reg("y"), Reg("t")),
            Insn::add("i", "i", "one"),
        ],
    );
    wire(
        &mut cfg,
        exit,
        vec![Insn::out("x"), Insn::out("y"), Insn::ret("x")],
    );
    cfg.add_edge(entry, header, EdgeKind::Fallthrough);
    cfg.add_edge(header, exit, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, body, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, header, EdgeKind::Back);
    Program {
        cfg,
        inputs: &["a", "b", "n"],
    }
}

/// A one-block loop whose exit reads the value from before the last
/// increment: `p = x; x = x + 1; if x < n goto loop`, then `return p`.
pub fn lost_copy_loop() -> Program {
    let mut cfg = Cfg::new();
    let (entry, body, exit) = (cfg.entry(), cfg.new_block(), cfg.new_block());
    wire(&mut cfg, entry, vec![Insn::li("x", 0), Insn::li("one", 1)]);
    wire(
        &mut cfg,
        body,
        vec![
            Insn::mov(Reg("p"), Reg("x")),
            Insn::add("x", "x", "one"),
            Insn::slt("again", "x", "n"),
            Insn::bnez("again"),
        ],
    );
    wire(&mut cfg, exit, vec![Insn::out("p"), Insn::ret("p")]);
    cfg.add_edge(entry, body, EdgeKind::Fallthrough);
    cfg.add_edge(body, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(body, exit, EdgeKind::ConditionalFalse);
    Program {
        cfg,
        inputs: &["n"],
    }
}

/// A loop whose body skips the rest of an iteration with a conditional
/// jump straight back to the header: a critical edge.
pub fn loop_with_early_continue() -> Program {
    let mut cfg = Cfg::new();
    let (entry, header, body, tail, exit) = (
        cfg.entry(),
        cfg.new_block(),
        cfg.new_block(),
        cfg.new_block(),
        cfg.new_block(),
    );
    wire(
        &mut cfg,
        entry,
        vec![
            Insn::li("s", 0),
            Insn::li("i", 0),
            Insn::li("one", 1),
            Insn::li("two", 2),
        ],
    );
    wire(
        &mut cfg,
        header,
        vec![Insn::slt("stop", "n", "i"), Insn::bnez("stop")],
    );
    wire(
        &mut cfg,
        body,
        vec![
            Insn::add("i", "i", "one"),
            Insn::slt("skip", "i", "two"),
            Insn::bnez("skip"),
        ],
    );
    wire(&mut cfg, tail, vec![Insn::add("s", "s", "i")]);
    wire(&mut cfg, exit, vec![Insn::out("s"), Insn::ret("s")]);
    cfg.add_edge(entry, header, EdgeKind::Fallthrough);
    cfg.add_edge(header, exit, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, body, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, header, EdgeKind::ConditionalTrue);
    cfg.add_edge(body, tail, EdgeKind::ConditionalFalse);
    cfg.add_edge(tail, header, EdgeKind::Back);
    Program {
        cfg,
        inputs: &["n"],
    }
}

/// Two divisions in one block redefine `t` between them; the handler reads
/// `t`, so its phi merges the value before each division.
pub fn handler_program() -> Program {
    let mut cfg = Cfg::new();
    let (entry, done, handler) = (cfg.entry(), cfg.new_block(), cfg.new_block());
    wire(
        &mut cfg,
        entry,
        vec![
            Insn::mov(Reg("t"), Reg("a")),
            Insn::div("d", "t", "b"),
            Insn::mov(Reg("t"), Reg("d")),
            Insn::div("e", "t", "c"),
        ],
    );
    wire(&mut cfg, done, vec![Insn::out("e"), Insn::ret("e")]);
    wire(&mut cfg, handler, vec![Insn::out("t"), Insn::ret("t")]);
    cfg.add_edge(entry, done, EdgeKind::Fallthrough);
    cfg.add_edge(entry, handler, EdgeKind::ExceptionUnwind);
    Program {
        cfg,
        inputs: &["a", "b", "c"],
    }
}

/// Loops nested in loops, so phis feed phis.
pub fn nested_loops() -> Program {
    let mut cfg = Cfg::new();
    let (entry, outer, inner, inner_body, after_inner, exit) = (
        cfg.entry(),
        cfg.new_block(),
        cfg.new_block(),
        cfg.new_block(),
        cfg.new_block(),
        cfg.new_block(),
    );
    wire(
        &mut cfg,
        entry,
        vec![Insn::li("s", 0), Insn::li("i", 0), Insn::li("one", 1)],
    );
    wire(
        &mut cfg,
        outer,
        vec![
            Insn::slt("stop_i", "n", "i"),
            Insn::li("j", 0),
            Insn::bnez("stop_i"),
        ],
    );
    wire(
        &mut cfg,
        inner,
        vec![Insn::slt("stop_j", "m", "j"), Insn::bnez("stop_j")],
    );
    wire(
        &mut cfg,
        inner_body,
        vec![Insn::add("s", "s", "j"), Insn::add("j", "j", "one")],
    );
    wire(&mut cfg, after_inner, vec![Insn::add("i", "i", "one")]);
    wire(&mut cfg, exit, vec![Insn::out("s"), Insn::ret("s")]);
    cfg.add_edge(entry, outer, EdgeKind::Fallthrough);
    cfg.add_edge(outer, exit, EdgeKind::ConditionalTrue);
    cfg.add_edge(outer, inner, EdgeKind::ConditionalFalse);
    cfg.add_edge(inner, after_inner, EdgeKind::ConditionalTrue);
    cfg.add_edge(inner, inner_body, EdgeKind::ConditionalFalse);
    cfg.add_edge(inner_body, inner, EdgeKind::Back);
    cfg.add_edge(after_inner, outer, EdgeKind::Back);
    Program {
        cfg,
        inputs: &["n", "m"],
    }
}

pub fn analyze(cfg: &Cfg<Insn<Reg>>) -> SsaForm<Reg> {
    SsaForm::compute(cfg, &DominatorTree::compute(cfg))
}

pub fn departure(block: BlockId, inst_idx: usize) -> ProgramPoint {
    ProgramPoint { block, inst_idx }
}

pub fn corpus() -> Vec<(&'static str, Program)> {
    vec![
        ("diamond", diamond()),
        ("two_phi_diamond", two_phi_diamond()),
        ("dead_phi_diamond", dead_phi_diamond()),
        ("swap_loop", swap_loop()),
        ("lost_copy_loop", lost_copy_loop()),
        ("loop_with_early_continue", loop_with_early_continue()),
        ("handler_program", handler_program()),
        ("nested_loops", nested_loops()),
    ]
}
