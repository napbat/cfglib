//! The toy source language the tests structure: every block holds source-like
//! statements, so a tree and a pseudocode golden read like the program a
//! decompiler would print.

use cfglib::{
    BlockId, Cfg, ControlTree, EdgeId, EdgeKind, Handler, HandlerBody, HandlerKind, Region,
    RegionId,
};

/// A graph under construction: `Cfg<&'static str>` whose blocks carry
/// statements, with the entry block named up front.
pub struct Flow {
    pub cfg: Cfg<&'static str>,
}

impl Flow {
    /// A graph whose entry block holds `statement`.
    pub fn starting_with(statement: &'static str) -> Self {
        let mut cfg = Cfg::new();
        cfg.block_mut(cfg.entry()).push(statement);
        Self { cfg }
    }

    pub fn entry(&self) -> BlockId {
        self.cfg.entry()
    }

    /// Append `statement` to `block`.
    pub fn push(&mut self, block: BlockId, statement: &'static str) {
        self.cfg.block_mut(block).push(statement);
    }

    /// A new block holding one statement.
    pub fn stmt(&mut self, statement: &'static str) -> BlockId {
        let block = self.cfg.new_block();
        self.cfg.block_mut(block).push(statement);
        block
    }

    /// A new block holding every statement, in order.
    pub fn stmts(&mut self, statements: &[&'static str]) -> BlockId {
        let block = self.cfg.new_block();
        for &statement in statements {
            self.cfg.block_mut(block).push(statement);
        }
        block
    }

    /// A new block without a statement: a join point or an empty arm.
    pub fn empty(&mut self) -> BlockId {
        self.cfg.new_block()
    }

    pub fn edge(&mut self, from: BlockId, to: BlockId, kind: EdgeKind) -> EdgeId {
        self.cfg.add_edge(from, to, kind)
    }

    /// The two arms of a conditional branch at `from`.
    pub fn branch(&mut self, from: BlockId, if_true: BlockId, if_false: BlockId) {
        self.edge(from, if_true, EdgeKind::ConditionalTrue);
        self.edge(from, if_false, EdgeKind::ConditionalFalse);
    }

    pub fn fall(&mut self, from: BlockId, to: BlockId) {
        self.edge(from, to, EdgeKind::Fallthrough);
    }

    pub fn jump(&mut self, from: BlockId, to: BlockId) {
        self.edge(from, to, EdgeKind::Jump);
    }

    /// Register a region protecting `protected`, with `handlers` attached.
    pub fn region(&mut self, protected: &[BlockId], handlers: Vec<Handler>) -> RegionId {
        self.cfg.add_region(Region {
            id: RegionId::from_raw(0), // overwritten by `add_region`
            protected_blocks: protected.iter().copied().collect(),
            handlers,
            parent: None,
        })
    }

    /// One dispatch edge of a `switch` at `from`.
    pub fn case(&mut self, from: BlockId, to: BlockId) -> EdgeId {
        self.edge(from, to, EdgeKind::SwitchCase)
    }
}

/// The statements of one program, as the pseudocode printer sees them: the
/// instruction a [`cfglib::ProgramPoint`] names, read back from the graph.
pub fn pseudocode(cfg: &Cfg<&'static str>, tree: &ControlTree) -> String {
    tree.to_pseudocode(|sink, point| {
        sink.write_str(cfg.block(point.block).instructions()[point.inst_idx])
    })
    .expect("writing to a String cannot fail")
}

/// A handler whose whole body is the one block `entry`.
pub fn handler(entry: BlockId, kind: HandlerKind) -> Handler {
    handler_over(entry, kind, &[entry])
}

/// A handler whose body is exactly the `extent` blocks.
pub fn handler_over(entry: BlockId, kind: HandlerKind, extent: &[BlockId]) -> Handler {
    Handler {
        entry,
        body: HandlerBody::known(extent.iter().copied()),
        kind,
    }
}

/// `for i { for j { if found(i, j) { break outer } } }`: the inner `if` leaves
/// both loops. Returns the graph and the outer loop's header.
pub fn labeled_break_loops() -> (Flow, BlockId) {
    let mut flow = Flow::starting_with("i = 0");
    let outer = flow.stmt("i < n");
    let inner_init = flow.stmt("j = 0");
    let inner = flow.stmt("j < m");
    let found = flow.stmt("found(i, j)");
    let step_j = flow.stmt("j += 1");
    let step_i = flow.stmt("i += 1");
    let exit = flow.stmt("return");
    flow.fall(flow.entry(), outer);
    flow.branch(outer, inner_init, exit);
    flow.fall(inner_init, inner);
    flow.branch(inner, found, step_i);
    flow.branch(found, exit, step_j);
    flow.jump(step_j, inner);
    flow.jump(step_i, outer);
    (flow, outer)
}

/// `open(); try { work() } catch { recover() } finally { close() }; return`,
/// with both handlers entered through exception edges only.
pub fn try_catch_finally() -> Flow {
    let mut flow = Flow::starting_with("open()");
    let body = flow.stmt("work()");
    let pad = flow.stmt("recover()");
    let cleanup = flow.stmt("close()");
    let after = flow.stmt("return");
    flow.fall(flow.entry(), body);
    flow.fall(body, after);
    flow.edge(body, pad, EdgeKind::ExceptionHandler);
    flow.edge(body, cleanup, EdgeKind::ExceptionUnwind);
    flow.fall(pad, after);
    flow.fall(cleanup, after);
    flow.region(
        &[body],
        vec![
            handler(pad, HandlerKind::Catch),
            handler(cleanup, HandlerKind::Finally),
        ],
    );
    flow
}

/// `switch (k) { case 0: a(); case 1: b(); default: d() }` with every arm
/// breaking to the join.
pub fn dispatch() -> Flow {
    let mut flow = Flow::starting_with("switch k");
    let dispatch = flow.entry();
    let first = flow.stmt("a()");
    let second = flow.stmt("b()");
    let default_arm = flow.stmt("d()");
    let join = flow.stmt("return");
    flow.case(dispatch, first);
    flow.case(dispatch, second);
    flow.fall(dispatch, default_arm);
    for arm in [first, second, default_arm] {
        flow.fall(arm, join);
    }
    flow
}

/// The classic irreducible cycle: two blocks that reach each other, each
/// entered from outside the cycle. Returns the graph and the two blocks.
pub fn irreducible_pair() -> (Flow, BlockId, BlockId) {
    let mut flow = Flow::starting_with("start");
    let a = flow.stmt("A");
    let b = flow.stmt("B");
    let exit = flow.stmt("return");
    flow.branch(flow.entry(), a, b);
    flow.branch(a, b, exit);
    flow.branch(b, a, exit);
    (flow, a, b)
}
