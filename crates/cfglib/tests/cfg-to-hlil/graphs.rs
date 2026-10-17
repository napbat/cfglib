//! The control-flow graphs that the pipeline structures: one per shape.
//!
//! Each is written the way a frontend would build it, block by block, with
//! an edge payload that says why control moves.

use std::collections::BTreeSet;

use cfglib::{BlockId, Cfg, Handler, HandlerBody, HandlerKind, KindedEdge, Region, RegionId};

use crate::dialect::{Link, Prim, Source, Stmt, act, dispatch, konst, ret, set, test};

/// A CFG under construction over named variables. The first `parameters`
/// names are the parameters.
pub(crate) struct Graph {
    cfg: Cfg<Stmt, Link>,
    names: Vec<&'static str>,
    parameters: usize,
}

impl Graph {
    pub(crate) fn new(names: &[&'static str], parameters: usize) -> Self {
        Self {
            cfg: Cfg::with_edge_payload(),
            names: names.to_vec(),
            parameters,
        }
    }

    /// The entry block, which holds `statements`.
    pub(crate) fn entry(&mut self, statements: Vec<Stmt>) -> BlockId {
        let entry = self.cfg.entry();
        self.cfg
            .block_mut(entry)
            .instructions_mut()
            .extend(statements);
        entry
    }

    /// A new block that holds `statements`.
    pub(crate) fn block(&mut self, statements: Vec<Stmt>) -> BlockId {
        let block = self.cfg.new_block();
        self.cfg
            .block_mut(block)
            .instructions_mut()
            .extend(statements);
        block
    }

    pub(crate) fn edge(&mut self, from: BlockId, to: BlockId, link: Link) {
        let kind = link.kind();
        self.cfg.add_edge_with_payload(from, to, kind, link);
    }

    /// Registers a try region over the `protected` blocks. Each handler is
    /// its entry block, its kind, and the blocks of its body.
    pub(crate) fn region(
        &mut self,
        protected: &[BlockId],
        handlers: &[(BlockId, HandlerKind, &[BlockId])],
    ) {
        self.cfg.add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks: protected.iter().copied().collect(),
            handlers: handlers
                .iter()
                .map(|&(entry, kind, body)| Handler {
                    entry,
                    body: HandlerBody::Known(body.iter().copied().collect::<BTreeSet<_>>()),
                    kind,
                })
                .collect(),
            parent: None,
        });
    }

    pub(crate) fn finish(self) -> Source {
        Source {
            cfg: self.cfg,
            parameters: (0..self.parameters).collect(),
            names: self.names,
        }
    }
}

/// A fixture and the inputs that exercise its paths.
pub(crate) struct Fixture {
    pub(crate) name: &'static str,
    pub(crate) source: Source,
    pub(crate) inputs: Vec<Vec<i64>>,
}

fn inputs(rows: &[&[i64]]) -> Vec<Vec<i64>> {
    rows.iter().map(|row| row.to_vec()).collect()
}

/// `if (a < b) { show(a); r = a + b } else { show(b); r = b + b }; return r`
pub(crate) fn branching() -> Fixture {
    let mut graph = Graph::new(&["a", "b", "c", "r"], 2);
    let head = graph.entry(vec![set(2, Prim::Less, &[0, 1]), test(2)]);
    let then_block = graph.block(vec![act(Prim::Show, &[0]), set(3, Prim::Plus, &[0, 1])]);
    let else_block = graph.block(vec![act(Prim::Show, &[1]), set(3, Prim::Plus, &[1, 1])]);
    let join = graph.block(vec![ret(3)]);
    graph.edge(head, then_block, Link::True);
    graph.edge(head, else_block, Link::False);
    graph.edge(then_block, join, Link::Next);
    graph.edge(else_block, join, Link::Next);
    Fixture {
        name: "branching",
        source: graph.finish(),
        inputs: inputs(&[&[1, 2], &[2, 1], &[3, 3]]),
    }
}

/// `i = 0; s = 0; while (i < n) { s = s + i; i = i + 1; show(s) } return s`
pub(crate) fn counting_loop() -> Fixture {
    let mut graph = Graph::new(&["n", "i", "s", "c", "one"], 1);
    let init = graph.entry(vec![konst(1, 0), konst(2, 0), konst(4, 1)]);
    let header = graph.block(vec![set(3, Prim::Less, &[1, 0]), test(3)]);
    let body = graph.block(vec![
        set(2, Prim::Plus, &[2, 1]),
        set(1, Prim::Plus, &[1, 4]),
        act(Prim::Show, &[2]),
    ]);
    let exit = graph.block(vec![ret(2)]);
    graph.edge(init, header, Link::Next);
    graph.edge(header, body, Link::True);
    graph.edge(header, exit, Link::False);
    graph.edge(body, header, Link::Jump);
    Fixture {
        name: "counting_loop",
        source: graph.finish(),
        inputs: inputs(&[&[0], &[1], &[5]]),
    }
}

/// A loop that shows the counter only while it is below three.
pub(crate) fn loop_with_a_branch() -> Fixture {
    let mut graph = Graph::new(&["n", "i", "one", "three", "c", "d"], 1);
    let init = graph.entry(vec![konst(1, 0), konst(2, 1), konst(3, 3)]);
    let header = graph.block(vec![set(4, Prim::Less, &[1, 0]), test(4)]);
    let choice = graph.block(vec![set(5, Prim::Less, &[1, 3]), test(5)]);
    let shown = graph.block(vec![act(Prim::Show, &[1])]);
    let step = graph.block(vec![set(1, Prim::Plus, &[1, 2])]);
    let exit = graph.block(vec![ret(1)]);
    graph.edge(init, header, Link::Next);
    graph.edge(header, choice, Link::True);
    graph.edge(header, exit, Link::False);
    graph.edge(choice, shown, Link::True);
    graph.edge(choice, step, Link::False);
    graph.edge(shown, step, Link::Next);
    graph.edge(step, header, Link::Jump);
    Fixture {
        name: "loop_with_a_branch",
        source: graph.finish(),
        inputs: inputs(&[&[0], &[2], &[6]]),
    }
}

/// A dispatch on `x` with a shared arm and a default.
pub(crate) fn dispatching() -> Fixture {
    let mut graph = Graph::new(&["x", "r"], 1);
    let head = graph.entry(vec![dispatch(0)]);
    let shared = graph.block(vec![act(Prim::Show, &[0]), konst(1, 10)]);
    let single = graph.block(vec![konst(1, 30)]);
    let other = graph.block(vec![konst(1, 0)]);
    let join = graph.block(vec![ret(1)]);
    graph.edge(head, shared, Link::Case(vec![1, 2]));
    graph.edge(head, single, Link::Case(vec![3]));
    graph.edge(head, other, Link::Default);
    for arm in [shared, single, other] {
        graph.edge(arm, join, Link::Next);
    }
    Fixture {
        name: "dispatching",
        source: graph.finish(),
        inputs: inputs(&[&[1], &[2], &[3], &[4], &[-1]]),
    }
}

/// `try { q = a / b; show(q) } catch { show(9) } return q`
pub(crate) fn guarded_division() -> Fixture {
    let mut graph = Graph::new(&["a", "b", "q", "nine"], 2);
    let start = graph.entry(vec![konst(3, 9)]);
    let protected = graph.block(vec![set(2, Prim::Quot, &[0, 1]), act(Prim::Show, &[2])]);
    let handler = graph.block(vec![act(Prim::Show, &[3])]);
    let join = graph.block(vec![ret(2)]);
    graph.edge(start, protected, Link::Next);
    graph.edge(protected, join, Link::Next);
    graph.edge(protected, handler, Link::Unwind);
    graph.edge(handler, join, Link::Next);
    graph.region(
        &[protected],
        &[(handler, HandlerKind::CatchAll, &[handler])],
    );
    Fixture {
        name: "guarded_division",
        source: graph.finish(),
        inputs: inputs(&[&[8, 2], &[8, 0]]),
    }
}

/// A loop with two entries: `a < 0` enters in the middle, otherwise at the
/// head.
pub(crate) fn irreducible_loop() -> Fixture {
    let mut graph = Graph::new(&["a", "n", "i", "one", "c", "d"], 2);
    let start = graph.entry(vec![
        konst(2, 0),
        konst(3, 1),
        konst(4, 0),
        set(4, Prim::Less, &[0, 4]),
        test(4),
    ]);
    let head = graph.block(vec![act(Prim::Show, &[2])]);
    let middle = graph.block(vec![
        set(2, Prim::Plus, &[2, 3]),
        set(5, Prim::Less, &[2, 1]),
        test(5),
    ]);
    let exit = graph.block(vec![ret(2)]);
    graph.edge(start, middle, Link::True);
    graph.edge(start, head, Link::False);
    graph.edge(head, middle, Link::Next);
    graph.edge(middle, head, Link::True);
    graph.edge(middle, exit, Link::False);
    Fixture {
        name: "irreducible_loop",
        source: graph.finish(),
        inputs: inputs(&[&[1, 3], &[-1, 3], &[0, 1], &[-1, 0]]),
    }
}

/// An irreducible loop whose head holds no statement, so a jump back to it
/// has no node to carry its label.
pub(crate) fn irreducible_loop_with_an_empty_head() -> Fixture {
    let mut graph = Graph::new(&["a", "n", "i", "one", "c", "d"], 2);
    let start = graph.entry(vec![
        konst(2, 0),
        konst(3, 1),
        konst(4, 0),
        set(4, Prim::Less, &[0, 4]),
        test(4),
    ]);
    let head = graph.block(vec![]);
    let middle = graph.block(vec![
        set(2, Prim::Plus, &[2, 3]),
        set(5, Prim::Less, &[2, 1]),
        test(5),
    ]);
    let exit = graph.block(vec![ret(2)]);
    graph.edge(start, middle, Link::True);
    graph.edge(start, head, Link::False);
    graph.edge(head, middle, Link::Next);
    graph.edge(middle, head, Link::True);
    graph.edge(middle, exit, Link::False);
    Fixture {
        name: "irreducible_loop_with_an_empty_head",
        source: graph.finish(),
        inputs: inputs(&[&[1, 3], &[-1, 3], &[0, 1]]),
    }
}

/// `i = 0; do { show(i); i = i + 1 } while (i < n); return i`
pub(crate) fn do_while_loop() -> Fixture {
    let mut graph = Graph::new(&["n", "i", "one", "c"], 1);
    let init = graph.entry(vec![konst(1, 0), konst(2, 1)]);
    let body = graph.block(vec![act(Prim::Show, &[1]), set(1, Prim::Plus, &[1, 2])]);
    let latch = graph.block(vec![set(3, Prim::Less, &[1, 0]), test(3)]);
    let exit = graph.block(vec![ret(1)]);
    graph.edge(init, body, Link::Next);
    graph.edge(body, latch, Link::Next);
    graph.edge(latch, body, Link::True);
    graph.edge(latch, exit, Link::False);
    Fixture {
        name: "do_while_loop",
        source: graph.finish(),
        inputs: inputs(&[&[0], &[1], &[4]]),
    }
}

/// A loop whose test leaves on true: `Tiny` has no negation to state it as
/// a `while`.
pub(crate) fn loop_leaving_on_true() -> Fixture {
    let mut graph = Graph::new(&["n", "i", "one", "c"], 1);
    let init = graph.entry(vec![konst(1, 0), konst(2, 1)]);
    let header = graph.block(vec![set(3, Prim::Less, &[0, 1]), test(3)]);
    let body = graph.block(vec![act(Prim::Show, &[1]), set(1, Prim::Plus, &[1, 2])]);
    let exit = graph.block(vec![ret(1)]);
    graph.edge(init, header, Link::Next);
    graph.edge(header, exit, Link::True);
    graph.edge(header, body, Link::False);
    graph.edge(body, header, Link::Jump);
    Fixture {
        name: "loop_leaving_on_true",
        source: graph.finish(),
        inputs: inputs(&[&[0], &[3]]),
    }
}

/// `while (i < n) { i = i + 1; if (i < 2) continue; if (!(i < 4)) break;
/// show(i) } return i`
pub(crate) fn loop_with_break_and_continue() -> Fixture {
    let mut graph = Graph::new(&["n", "i", "one", "two", "four", "c", "d", "e"], 1);
    let init = graph.entry(vec![konst(1, 0), konst(2, 1), konst(3, 2), konst(4, 4)]);
    let head = graph.block(vec![set(5, Prim::Less, &[1, 0]), test(5)]);
    let first = graph.block(vec![
        set(1, Prim::Plus, &[1, 2]),
        set(6, Prim::Less, &[1, 3]),
        test(6),
    ]);
    let second = graph.block(vec![set(7, Prim::Less, &[1, 4]), test(7)]);
    let shown = graph.block(vec![act(Prim::Show, &[1])]);
    let exit = graph.block(vec![ret(1)]);
    graph.edge(init, head, Link::Next);
    graph.edge(head, first, Link::True);
    graph.edge(head, exit, Link::False);
    graph.edge(first, head, Link::True);
    graph.edge(first, second, Link::False);
    graph.edge(second, shown, Link::True);
    graph.edge(second, exit, Link::False);
    graph.edge(shown, head, Link::Jump);
    Fixture {
        name: "loop_with_break_and_continue",
        source: graph.finish(),
        inputs: inputs(&[&[0], &[1], &[3], &[9]]),
    }
}

/// `try { q = a / b; show(q) } catch { show(9) } finally { show(7) }`, then
/// `return q`: the finally body is where the normal and the exceptional
/// paths meet.
pub(crate) fn guarded_division_with_finally() -> Fixture {
    let mut graph = Graph::new(&["a", "b", "q", "nine", "seven"], 2);
    let start = graph.entry(vec![konst(3, 9), konst(4, 7)]);
    let protected = graph.block(vec![set(2, Prim::Quot, &[0, 1]), act(Prim::Show, &[2])]);
    let catcher = graph.block(vec![act(Prim::Show, &[3])]);
    let cleanup = graph.block(vec![act(Prim::Show, &[4])]);
    let join = graph.block(vec![ret(2)]);
    graph.edge(start, protected, Link::Next);
    graph.edge(protected, cleanup, Link::Next);
    graph.edge(protected, catcher, Link::Unwind);
    graph.edge(protected, cleanup, Link::Unwind);
    graph.edge(catcher, cleanup, Link::Next);
    graph.edge(cleanup, join, Link::Next);
    graph.region(
        &[protected],
        &[
            (catcher, HandlerKind::CatchAll, &[catcher]),
            (cleanup, HandlerKind::Finally, &[cleanup]),
        ],
    );
    Fixture {
        name: "guarded_division_with_finally",
        source: graph.finish(),
        inputs: inputs(&[&[8, 2], &[8, 0]]),
    }
}

/// The fixtures that structure without a `goto`.
pub(crate) fn structured() -> Vec<Fixture> {
    vec![
        branching(),
        counting_loop(),
        loop_with_a_branch(),
        dispatching(),
        guarded_division(),
        do_while_loop(),
        loop_leaving_on_true(),
        loop_with_break_and_continue(),
    ]
}
