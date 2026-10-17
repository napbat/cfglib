//! The toy instruction and the try/catch graphs the tests share.

use std::borrow::Cow;

use cfglib::{
    BlockId, Cfg, DisplayInstr, EdgeKind, EffectInfo, Handler, HandlerBody, HandlerKind, InstrInfo,
    Region, RegionId,
};

/// A source-like statement with the def, use, and effect sets the dataflow
/// traits ask for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stmt {
    text: String,
    defs: Vec<&'static str>,
    uses: Vec<&'static str>,
    effects: Vec<&'static str>,
}

impl Stmt {
    pub fn text(text: &str) -> Self {
        Self {
            text: text.to_owned(),
            defs: Vec::new(),
            uses: Vec::new(),
            effects: Vec::new(),
        }
    }

    pub fn defining(mut self, variable: &'static str) -> Self {
        self.defs.push(variable);
        self
    }

    pub fn with_effect(mut self, effect: &'static str) -> Self {
        self.effects.push(effect);
        self
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }
}

impl InstrInfo for Stmt {
    type Variable = &'static str;

    fn uses(&self) -> &[&'static str] {
        &self.uses
    }

    fn defs(&self) -> &[&'static str] {
        &self.defs
    }
}

impl EffectInfo for Stmt {
    type Effect = &'static str;

    fn effects(&self) -> &[&'static str] {
        &self.effects
    }
}

impl DisplayInstr for Stmt {
    fn mnemonic(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.text)
    }
}

/// A new block holding `statements`, in order.
pub fn block_of(cfg: &mut Cfg<Stmt>, statements: &[&str]) -> BlockId {
    let block = cfg.new_block();
    for statement in statements {
        cfg.block_mut(block).push(Stmt::text(statement));
    }
    block
}

/// A handler whose whole body is the one block `entry`.
pub fn catch(entry: BlockId) -> Handler {
    Handler {
        entry,
        body: HandlerBody::known([entry]),
        kind: HandlerKind::Catch,
    }
}

/// A catch whose entry is all a table-driven format tells us: its extent is
/// not known.
pub fn unknown_catch(entry: BlockId) -> Handler {
    Handler {
        entry,
        body: HandlerBody::unknown(),
        kind: HandlerKind::Catch,
    }
}

/// A handler of `kind` whose body is exactly `body`.
pub fn handler(entry: BlockId, kind: HandlerKind, body: &[BlockId]) -> Handler {
    Handler {
        entry,
        body: HandlerBody::known(body.iter().copied()),
        kind,
    }
}

/// Register a region protecting `protected`.
pub fn protect(cfg: &mut Cfg<Stmt>, protected: &[BlockId], handlers: Vec<Handler>) -> RegionId {
    cfg.add_region(Region {
        id: RegionId::from_raw(0), // overwritten by `add_region`
        protected_blocks: protected.iter().copied().collect(),
        handlers,
        parent: None,
    })
}

/// `setup(); try { risky() } catch { <pad> }; return`: the body falls through
/// to the join and raises into the pad through an unwind edge.
pub struct Try {
    pub cfg: Cfg<Stmt>,
    pub body: BlockId,
    pub pad: BlockId,
    pub join: BlockId,
    pub region: RegionId,
}

/// A try/catch whose pad holds `pad_statements`; none makes the pad an empty
/// block with a sole fallthrough.
pub fn try_catch(pad_statements: &[&str]) -> Try {
    try_catch_with(pad_statements.iter().map(|text| Stmt::text(text)).collect())
}

/// [`try_catch`] with a pad of full statements, and effects on the code the
/// pad does not own, so dead-code elimination keeps it.
pub fn try_catch_with(pad: Vec<Stmt>) -> Try {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry)
        .push(Stmt::text("setup()").with_effect("io"));
    let body = cfg.new_block();
    cfg.block_mut(body)
        .push(Stmt::text("risky()").with_effect("io"));
    let pad_block = cfg.new_block();
    cfg.block_mut(pad_block).instructions_mut().extend(pad);
    let join = cfg.new_block();
    cfg.block_mut(join)
        .push(Stmt::text("return").with_effect("exit"));
    cfg.add_edge(entry, body, EdgeKind::Fallthrough);
    cfg.add_edge(body, join, EdgeKind::Fallthrough);
    cfg.add_edge(body, pad_block, EdgeKind::ExceptionUnwind);
    cfg.add_edge(pad_block, join, EdgeKind::Fallthrough);
    let region = protect(&mut cfg, &[body], vec![catch(pad_block)]);
    Try {
        cfg,
        body,
        pad: pad_block,
        join,
        region,
    }
}

/// `try { throw e } catch { log(e) }`: the body's only edge is the one that
/// raises into the pad, so the join is reached through the pad alone.
pub fn always_throwing_try() -> Try {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(Stmt::text("setup()"));
    let body = block_of(&mut cfg, &["throw e"]);
    let pad = block_of(&mut cfg, &["log(e)"]);
    let join = block_of(&mut cfg, &["return"]);
    cfg.add_edge(entry, body, EdgeKind::Fallthrough);
    cfg.add_edge(body, pad, EdgeKind::ExceptionUnwind);
    cfg.add_edge(pad, join, EdgeKind::Fallthrough);
    let region = protect(&mut cfg, &[body], vec![catch(pad)]);
    Try {
        cfg,
        body,
        pad,
        join,
        region,
    }
}

/// `try { throw e } catch { log(e) }` where the throwing block is the entry
/// itself, so nothing can merge it into a predecessor.
pub fn throwing_entry_try() -> Try {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(Stmt::text("throw e"));
    let pad = block_of(&mut cfg, &["log(e)"]);
    let join = block_of(&mut cfg, &["return"]);
    cfg.add_edge(entry, pad, EdgeKind::ExceptionUnwind);
    cfg.add_edge(pad, join, EdgeKind::Fallthrough);
    let region = protect(&mut cfg, &[entry], vec![catch(pad)]);
    Try {
        cfg,
        body: entry,
        pad,
        join,
        region,
    }
}

/// A try with two catch handlers in one region: the first can never be
/// entered (no edge raises into it), the second can.
pub struct TwoHandlers {
    pub cfg: Cfg<Stmt>,
    pub region: RegionId,
}

pub fn dead_and_live_handlers() -> TwoHandlers {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(Stmt::text("setup()"));
    let body = block_of(&mut cfg, &["risky()"]);
    let dead = block_of(&mut cfg, &["catch dead"]);
    let live = block_of(&mut cfg, &["catch live"]);
    let join = block_of(&mut cfg, &["return"]);
    cfg.add_edge(entry, body, EdgeKind::Fallthrough);
    cfg.add_edge(body, join, EdgeKind::Fallthrough);
    cfg.add_edge(body, live, EdgeKind::ExceptionUnwind);
    cfg.add_edge(dead, join, EdgeKind::Fallthrough);
    cfg.add_edge(live, join, EdgeKind::Fallthrough);
    let region = protect(&mut cfg, &[body], vec![catch(dead), catch(live)]);
    TwoHandlers { cfg, region }
}

/// The statements executed from `start` by following the one normal
/// successor of each block until the flow ends.
///
/// Used where a transform may move blocks about but must not change what a
/// handler does: the trace from a handler's entry is the handler's behavior.
pub fn trace_from(cfg: &Cfg<Stmt>, start: BlockId) -> Vec<String> {
    let mut trace = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    let mut current = Some(start);
    while let Some(block) = current {
        assert!(seen.insert(block), "a loop in a straight-line trace");
        trace.extend(
            cfg.block(block)
                .instructions()
                .iter()
                .map(|stmt| stmt.as_str().to_owned()),
        );
        let mut normal = cfg
            .outgoing(block)
            .map(|edge| cfg.edge(edge))
            .filter(|edge| !edge.kind().is_exceptional());
        current = normal.next().map(|edge| edge.target());
        assert!(normal.next().is_none(), "the trace is straight-line");
    }
    trace
}
