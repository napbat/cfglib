//! A consumer's instruction and edge-payload types, and the CFGs built from
//! them: a persisted CFG is only useful if the consumer's own types survive.

use std::borrow::Cow;

use cfglib::{
    BlockId, Cfg, CompletionReason, Continuation, DisplayInstr, EdgeId, EdgeKind, Handler,
    HandlerBody, HandlerKind, HandlerRef, Region, RegionId,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Op {
    Assign { target: String, value: i64 },
    Branch(String),
    Call(String),
    Return,
}

impl DisplayInstr for Op {
    fn mnemonic(&self) -> Cow<'_, str> {
        Cow::Owned(match self {
            Self::Assign { target, value } => format!("{target} = {value}"),
            Self::Branch(condition) => format!("branch {condition}"),
            Self::Call(callee) => format!("call {callee}"),
            Self::Return => "return".to_owned(),
        })
    }
}

/// Edge provenance a frontend keeps beside the control-flow kind.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Route {
    pub case: Option<i64>,
    pub provenance: Vec<String>,
}

impl Route {
    pub fn case(value: i64) -> Self {
        Self {
            case: Some(value),
            provenance: Vec::new(),
        }
    }

    pub fn from_source(source: &str) -> Self {
        Self {
            case: None,
            provenance: vec![source.to_owned()],
        }
    }
}

pub fn region(protected: &[BlockId], handlers: Vec<Handler>, parent: Option<RegionId>) -> Region {
    Region {
        id: RegionId::from_raw(0),
        protected_blocks: protected.iter().copied().collect(),
        handlers,
        parent,
    }
}

pub fn handler(entry: BlockId, kind: HandlerKind, body: HandlerBody) -> Handler {
    Handler { entry, body, kind }
}

/// The blocks of the base graph that [`rich_cfg`] wires up and annotates in stages.
struct BaseBlocks {
    entry: BlockId,
    header: BlockId,
    body: BlockId,
    scratch: BlockId,
    catch: BlockId,
    filter: BlockId,
    cleanup: BlockId,
    after: BlockId,
}

/// The base blocks, created in the order of their ids, with the instructions and labels they
/// carry.
fn base_blocks(cfg: &mut Cfg<Op, Route>) -> BaseBlocks {
    let entry = cfg.entry();
    let header = cfg.new_block();
    let body = cfg.new_block();
    let scratch = cfg.new_block();
    let catch = cfg.new_block();
    let filter = cfg.new_block();
    let cleanup = cfg.new_block();
    let after = cfg.new_block();

    cfg.block_mut(entry).push(Op::Assign {
        target: "i".to_owned(),
        value: 0,
    });
    cfg.block_mut(header).set_label("loop \"header\"\n(hot)");
    cfg.block_mut(header).push(Op::Branch("i < n".to_owned()));
    cfg.block_mut(body).push(Op::Call("step".to_owned()));
    cfg.block_mut(body).push(Op::Assign {
        target: "i".to_owned(),
        value: -1,
    });
    cfg.block_mut(catch).set_label("catch");
    cfg.block_mut(catch).push(Op::Call("log".to_owned()));
    cfg.block_mut(filter)
        .push(Op::Branch("is_overflow".to_owned()));
    cfg.block_mut(cleanup).set_label("finally");
    cfg.block_mut(cleanup).push(Op::Call("close".to_owned()));
    cfg.block_mut(after).set_label("after");
    cfg.block_mut(after).push(Op::Return);
    BaseBlocks {
        entry,
        header,
        body,
        scratch,
        catch,
        filter,
        cleanup,
        after,
    }
}

/// The base edges: weighted, parallel, and exceptional ones, with payloads of the consumer's own.
/// Returns the edge that is removed before compaction.
fn connect_base_blocks(cfg: &mut Cfg<Op, Route>, blocks: &BaseBlocks) -> EdgeId {
    let BaseBlocks {
        entry,
        header,
        body,
        scratch,
        catch,
        cleanup,
        after,
        ..
    } = *blocks;
    cfg.add_edge(entry, header, EdgeKind::Fallthrough);
    cfg.add_weighted_edge_with_payload(
        header,
        body,
        EdgeKind::ConditionalTrue,
        0.75,
        Route::from_source("loop condition"),
    );
    cfg.add_weighted_edge(header, after, EdgeKind::ConditionalFalse, 0.25);
    cfg.add_edge(body, header, EdgeKind::Back);
    cfg.add_edge_with_payload(header, body, EdgeKind::SwitchCase, Route::case(3));
    cfg.add_edge(body, catch, EdgeKind::ExceptionUnwind);
    cfg.add_edge(body, cleanup, EdgeKind::ExceptionLeave);
    cfg.add_edge(catch, cleanup, EdgeKind::Fallthrough);
    cfg.add_edge(cleanup, after, EdgeKind::Fallthrough);
    cfg.add_edge(scratch, after, EdgeKind::Jump);
    cfg.add_edge(entry, after, EdgeKind::Jump)
}

/// An outer region with every handler kind and body shape, a nested region, and a region that is
/// removed. Returns the outer region.
fn add_regions(cfg: &mut Cfg<Op, Route>, blocks: &BaseBlocks) -> RegionId {
    let BaseBlocks {
        header,
        body,
        catch,
        filter,
        cleanup,
        ..
    } = *blocks;
    let outer = cfg.add_region(region(
        &[header, body],
        vec![
            handler(catch, HandlerKind::Catch, HandlerBody::known([catch])),
            handler(catch, HandlerKind::CatchAll, HandlerBody::known([])),
            handler(
                catch,
                HandlerKind::Filter {
                    filter_block: filter,
                },
                HandlerBody::Unknown,
            ),
            handler(cleanup, HandlerKind::Finally, HandlerBody::known([cleanup])),
            handler(cleanup, HandlerKind::Fault, HandlerBody::Unknown),
        ],
        None,
    ));
    cfg.add_region(region(
        &[body],
        vec![handler(
            catch,
            HandlerKind::Catch,
            HandlerBody::known([catch]),
        )],
        Some(outer),
    ));
    let vanished = cfg.add_region(region(&[header], Vec::new(), None));
    cfg.remove_region(vanished);
    outer
}

/// Cleanup records with continuations for the outer region's finally and fault handlers.
fn record_cleanups(cfg: &mut Cfg<Op, Route>, blocks: &BaseBlocks, outer: RegionId) {
    let BaseBlocks {
        header,
        catch,
        cleanup,
        after,
        ..
    } = *blocks;
    let finally = HandlerRef::new(outer, 3);
    cfg.set_cleanup_resume(finally, cleanup);
    for (reason, resume) in [
        (CompletionReason::Normal, after),
        (CompletionReason::Transfer, header),
        (CompletionReason::Return, after),
        (CompletionReason::Unwind, catch),
    ] {
        cfg.add_continuation(finally, Continuation { reason, resume });
    }
    let fault = HandlerRef::new(outer, 4);
    cfg.set_cleanup_resume(fault, cleanup);
    cfg.add_continuation(
        fault,
        Continuation {
            reason: CompletionReason::Unwind,
            resume: catch,
        },
    );
}

/// The delta after the compacted base: a new block with a weighted edge, a removed block and
/// edge, and an edge whose source was redirected. The blocks passed in are the base blocks as
/// numbered after compaction.
fn add_delta(cfg: &mut Cfg<Op, Route>, header: BlockId, after: BlockId, cleanup: BlockId) {
    let late = cfg.new_block();
    let doomed = cfg.new_block();
    cfg.block_mut(late).set_label("late");
    cfg.block_mut(late).push(Op::Return);
    cfg.add_weighted_edge(after, late, EdgeKind::Fallthrough, 0.5);
    let moved = cfg.add_edge(cleanup, late, EdgeKind::IndirectJump);
    cfg.add_edge(after, doomed, EdgeKind::Jump);
    cfg.add_edge(doomed, late, EdgeKind::Jump);
    let withdrawn = cfg.add_edge(late, header, EdgeKind::Back);
    cfg.remove_block(doomed);
    cfg.remove_edge(withdrawn);
    cfg.redirect_edge_source(moved, after);
}

/// A CFG that exercises every piece of state a persisted CFG carries: payload
/// types of the consumer's own, weighted and parallel edges, nested and
/// tombstoned regions with every handler kind and body shape, cleanup records
/// with continuations, a compacted base with removed blocks and edges before
/// it, and after it a delta with a removed block, a removed edge, and an edge
/// whose endpoint was redirected.
pub fn rich_cfg() -> Cfg<Op, Route> {
    let mut cfg = Cfg::<Op, Route>::with_edge_payload();
    let blocks = base_blocks(&mut cfg);
    let dropped = connect_base_blocks(&mut cfg, &blocks);
    let outer = add_regions(&mut cfg, &blocks);
    record_cleanups(&mut cfg, &blocks, outer);

    cfg.remove_block(blocks.scratch);
    cfg.remove_edge(dropped);
    let renumbering = cfg.compact();
    let header = renumbering
        .node(blocks.header)
        .expect("the header survived");
    let after = renumbering.node(blocks.after).expect("the exit survived");
    let cleanup = renumbering
        .node(blocks.cleanup)
        .expect("the cleanup survived");

    add_delta(&mut cfg, header, after, cleanup);
    cfg
}

/// A CFG small enough to read its document: two blocks, one edge, one region,
/// nothing compacted, and a cleanup record for the region's handler when
/// `record_cleanup` is set.
pub fn small_cfg(record_cleanup: bool) -> Cfg<String> {
    let mut cfg = Cfg::<String>::new();
    let entry = cfg.entry();
    let handler_block = cfg.new_block();
    cfg.block_mut(entry).push("x = f()".to_owned());
    cfg.block_mut(handler_block).set_label("cleanup");
    cfg.add_edge(entry, handler_block, EdgeKind::ExceptionUnwind);
    let region = cfg.add_region(region(
        &[entry],
        vec![handler(
            handler_block,
            HandlerKind::Finally,
            HandlerBody::known([handler_block]),
        )],
        None,
    ));
    if record_cleanup {
        let finally = HandlerRef::new(region, 0);
        cfg.set_cleanup_resume(finally, handler_block);
        cfg.add_continuation(
            finally,
            Continuation {
                reason: CompletionReason::Normal,
                resume: handler_block,
            },
        );
    }
    cfg
}
