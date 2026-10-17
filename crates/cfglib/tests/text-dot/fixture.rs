//! Control-flow graphs the text and DOT tests share: one realistic lowering
//! for goldens and one that uses every feature the output formats carry.

use cfglib::{
    BlockId, Cfg, CompletionReason, Continuation, EdgeKind, Handler, HandlerBody, HandlerKind,
    HandlerRef, Region, RegionId,
};

use crate::instruction::Line;

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

/// `try { use(x) } catch (e) { log(e) } finally { close(x) }` as a source
/// lowering leaves it: the protected body, a catch pad, one shared cleanup
/// block with two recorded continuations, and a removed scratch block.
pub fn try_finally() -> Cfg<Line> {
    let mut cfg = Cfg::<Line>::new();
    let body = cfg.new_block();
    let scratch = cfg.new_block();
    let catch = cfg.new_block();
    let cleanup = cfg.new_block();
    let after = cfg.new_block();
    let exit = cfg.new_block();

    cfg.block_mut(cfg.entry()).push(Line::new("x = open()"));
    cfg.block_mut(body).set_label("try body");
    cfg.block_mut(body).push(Line::new("use(x)"));
    cfg.block_mut(catch).set_label("catch");
    cfg.block_mut(catch).push(Line::new("log(e)"));
    cfg.block_mut(cleanup).set_label("finally");
    cfg.block_mut(cleanup).push(Line::new("close(x)"));
    cfg.block_mut(after).set_label("after");
    cfg.block_mut(after).push(Line::new("result = x"));
    cfg.block_mut(exit).set_label("exit");
    cfg.block_mut(exit).push(Line::new("return"));
    cfg.remove_block(scratch);

    cfg.add_edge(cfg.entry(), body, EdgeKind::Fallthrough);
    cfg.add_edge(body, cleanup, EdgeKind::ExceptionLeave);
    cfg.add_edge(body, catch, EdgeKind::ExceptionUnwind);
    cfg.add_edge(catch, cleanup, EdgeKind::Fallthrough);
    cfg.add_edge(cleanup, after, EdgeKind::Fallthrough);
    cfg.add_edge(cleanup, exit, EdgeKind::Jump);

    let region = cfg.add_region(region(
        &[body],
        vec![
            handler(catch, HandlerKind::Catch, HandlerBody::known([catch])),
            handler(cleanup, HandlerKind::Finally, HandlerBody::known([cleanup])),
        ],
        None,
    ));
    let finally = HandlerRef::new(region, 1);
    cfg.set_cleanup_resume(finally, cleanup);
    cfg.add_continuation(
        finally,
        Continuation {
            reason: CompletionReason::Normal,
            resume: after,
        },
    );
    cfg.add_continuation(
        finally,
        Continuation {
            reason: CompletionReason::Return,
            resume: exit,
        },
    );
    cfg
}

/// The blocks of [`rich_cfg`] by role; `hole_a` and `hole_b` are removed once
/// the others hold their text.
struct RichBlocks {
    entry: BlockId,
    loop_header: BlockId,
    empty: BlockId,
    hole_a: BlockId,
    catch: BlockId,
    filter: BlockId,
    cleanup: BlockId,
    tail: BlockId,
    hole_b: BlockId,
    exit: BlockId,
}

impl RichBlocks {
    /// Allocate the blocks of `cfg` in the order `rich_cfg` numbers them.
    fn allocate(cfg: &mut Cfg<Line>) -> Self {
        Self {
            entry: cfg.entry(),
            loop_header: cfg.new_block(),
            empty: cfg.new_block(),
            hole_a: cfg.new_block(),
            catch: cfg.new_block(),
            filter: cfg.new_block(),
            cleanup: cfg.new_block(),
            tail: cfg.new_block(),
            hole_b: cfg.new_block(),
            exit: cfg.new_block(),
        }
    }
}

/// Give the blocks their text and labels, including ones that need escaping,
/// then remove the two holes.
fn populate_rich_blocks(cfg: &mut Cfg<Line>, blocks: &RichBlocks) {
    let RichBlocks {
        entry,
        loop_header,
        hole_a,
        catch,
        filter,
        cleanup,
        tail,
        hole_b,
        exit,
        ..
    } = *blocks;
    for text in [
        "x = read()",
        "-> not an arrow",
        "path C:\\dir\\file",
        "first\nsecond",
    ] {
        cfg.block_mut(entry).push(Line::new(text));
    }
    cfg.block_mut(loop_header).set_label("loop header");
    cfg.block_mut(catch)
        .set_label("-dash \\ label\nwith a break");
    cfg.block_mut(catch).push(Line::new("catch e"));
    cfg.block_mut(filter).push(Line::new("e is Overflow"));
    cfg.block_mut(cleanup).set_label("finally");
    cfg.block_mut(cleanup).push(Line::new("close(x)"));
    cfg.block_mut(tail).set_label("tail");
    cfg.block_mut(exit).push(Line::new("return"));
    cfg.remove_block(hole_a);
    cfg.remove_block(hole_b);
}

/// One edge of every kind spread over the live blocks, parallel edges, and a
/// self-loop.
fn add_rich_edges(cfg: &mut Cfg<Line>, blocks: &RichBlocks) {
    let RichBlocks {
        entry,
        loop_header,
        empty,
        catch,
        filter,
        cleanup,
        tail,
        exit,
        ..
    } = *blocks;
    let live = [
        entry,
        loop_header,
        empty,
        catch,
        filter,
        cleanup,
        tail,
        exit,
    ];
    for (index, kind) in EdgeKind::ALL.into_iter().enumerate() {
        cfg.add_edge(
            live[index % live.len()],
            live[(index * 3 + 1) % live.len()],
            kind,
        );
    }
    cfg.add_edge(entry, loop_header, EdgeKind::Fallthrough);
    cfg.add_edge(entry, loop_header, EdgeKind::Fallthrough);
    cfg.add_edge(entry, loop_header, EdgeKind::ConditionalTrue);
    cfg.add_edge(tail, tail, EdgeKind::Back);
}

/// An outer region with a handler of every kind and body shape, an inner
/// region nested in it, a region removed right after it is added, and an empty
/// region nested in the inner one. Returns the outer region.
fn add_rich_regions(cfg: &mut Cfg<Line>, blocks: &RichBlocks) -> RegionId {
    let RichBlocks {
        entry,
        loop_header,
        empty,
        catch,
        filter,
        cleanup,
        tail,
        exit,
        ..
    } = *blocks;
    let outer = cfg.add_region(region(
        &[entry, loop_header, empty],
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
            handler(
                cleanup,
                HandlerKind::Finally,
                HandlerBody::known([cleanup, tail]),
            ),
            handler(cleanup, HandlerKind::Fault, HandlerBody::Unknown),
        ],
        None,
    ));
    let inner = cfg.add_region(region(
        &[loop_header],
        vec![handler(
            exit,
            HandlerKind::Catch,
            HandlerBody::known([exit]),
        )],
        Some(outer),
    ));
    let vanished = cfg.add_region(region(&[empty], Vec::new(), None));
    cfg.remove_region(vanished);
    cfg.add_region(region(&[empty], Vec::new(), Some(inner)));
    outer
}

/// The `finally` handler of `outer` resumes at the cleanup block and records
/// a normal and a return continuation.
fn add_rich_continuations(cfg: &mut Cfg<Line>, blocks: &RichBlocks, outer: RegionId) {
    let RichBlocks {
        cleanup,
        tail,
        exit,
        ..
    } = *blocks;
    let finally = HandlerRef::new(outer, 3);
    cfg.set_cleanup_resume(finally, cleanup);
    cfg.add_continuation(
        finally,
        Continuation {
            reason: CompletionReason::Normal,
            resume: tail,
        },
    );
    cfg.add_continuation(
        finally,
        Continuation {
            reason: CompletionReason::Return,
            resume: exit,
        },
    );
}

/// A CFG that uses every feature the output formats carry: instructions that
/// need escaping, labels that need escaping, empty and unlabeled blocks, two
/// removed blocks, every edge kind, parallel edges, a self-loop, and nested,
/// tombstoned and empty regions with every handler kind and body shape.
pub fn rich_cfg() -> Cfg<Line> {
    let mut cfg = Cfg::<Line>::new();
    let blocks = RichBlocks::allocate(&mut cfg);
    populate_rich_blocks(&mut cfg, &blocks);
    add_rich_edges(&mut cfg, &blocks);
    let outer = add_rich_regions(&mut cfg, &blocks);
    add_rich_continuations(&mut cfg, &blocks, outer);
    cfg
}
