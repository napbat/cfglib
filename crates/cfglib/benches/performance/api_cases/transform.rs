use std::collections::BTreeSet;
use std::convert::Infallible;

use cfglib::{
    BlockId, BlockOrder, Cfg, EdgeKind, Emitter, Flow, FlowEffect, Handler, HandlerBody,
    HandlerKind, Region, contract_edge_mapped, linearize, merge_blocks_mapped,
    promote_handler_extents, remove_empty_blocks_mapped, remove_unreachable,
    remove_unreachable_mapped, resolve_jump_edges, simplify, simplify_mapped, split_critical_edges,
    split_critical_edges_mapped, split_critical_edges_with, verify,
};
use cfglib::{
    DispatchBuilder, DispatchReport, DominatorTree, ExclusiveExtent, ExtentPromotionStatus,
    RegionId, RelaxError, RelaxMode, SpanItem, is_reducible, make_reducible,
    make_reducible_with_dispatch, promote_exclusive_extents, recover_exclusive_extents,
    relax_layout, relax_spans,
};

use super::BenchmarkSuite;
use super::fixtures::{ApiInst, dataflow_cfg};
use crate::fixtures::{empty_chain_cfg, fixture_u32, linear_cfg};
use crate::harness::benchmark_case;

const BLOCK_COUNT: usize = 256;

/// The complete digraph node splitting still fixes in a benchmark sample.
const SPLIT_COMPLETE_BLOCKS: usize = 10;

/// The complete digraph only dispatch fixes in a benchmark sample.
const DISPATCH_COMPLETE_BLOCKS: usize = 64;

/// Selector code that names its entry, and dispatch code that names its hub.
struct ApiDispatch;

impl DispatchBuilder<u32, ()> for ApiDispatch {
    fn select(&mut self, _hub: BlockId, index: usize, _entry: BlockId) -> Vec<u32> {
        vec![fixture_u32(index)]
    }

    fn dispatch(&mut self, hub: BlockId, _entries: &[BlockId]) -> Vec<u32> {
        vec![fixture_u32(hub.index())]
    }

    fn case(&mut self, _hub: BlockId, _index: usize) {}

    fn jump(&mut self, _selector: BlockId, _hub: BlockId) {}
}

/// `blocks` blocks that all jump to each other and are each entered from the
/// entry: an irreducible cycle at every level of the cycle forest, which node
/// splitting fixes only with exponentially many copies.
fn complete_cfg(blocks: usize) -> Cfg<u32> {
    let mut cfg = Cfg::new();
    let ids: Vec<BlockId> = (0..blocks).map(|_| cfg.new_block()).collect();
    for &to in &ids {
        cfg.add_edge(cfg.entry(), to, EdgeKind::Jump);
    }
    for &from in &ids {
        for &to in ids.iter().filter(|&&to| to != from) {
            cfg.add_edge(from, to, EdgeKind::Jump);
        }
    }
    cfg
}

/// The dispatched complete digraph of `blocks` blocks: reducible, one hub per
/// level of the forest, and at most one selector per edge and level.
fn assert_dispatched_complete(result: &Cfg<u32>, report: &DispatchReport, blocks: usize) {
    assert!(is_reducible(result, &DominatorTree::compute(result)));
    let levels = blocks - 1;
    assert_eq!(report.hubs.len(), levels);
    assert!(report.selectors.len() <= blocks * blocks * levels);
    assert_eq!(
        result.block_count(),
        blocks + 1 + report.hubs.len() + report.selectors.len()
    );
}

struct ApiEmitter;

/// The `u32` token of `block`, which both a jump target and a label carry.
fn block_token(block: BlockId) -> u32 {
    u32::try_from(block.index()).expect("benchmark block index must fit in u32")
}

impl Emitter<ApiInst> for ApiEmitter {
    fn emit_jump(&self, target: BlockId) -> ApiInst {
        ApiInst::control(FlowEffect::Flow(Flow::Jump {
            target: block_token(target),
        }))
    }

    fn emit_conditional_branch(&self, _condition: &ApiInst, target: BlockId) -> ApiInst {
        ApiInst::control(FlowEffect::Flow(Flow::Conditional {
            target: block_token(target),
        }))
    }

    fn emit_block_start(&self, block: BlockId) -> Option<ApiInst> {
        Some(ApiInst::control(FlowEffect::Label(block_token(block))))
    }
}

fn unreachable_cfg() -> Cfg<u32> {
    let mut cfg = Cfg::new();
    cfg.block_mut(cfg.entry()).push(0);
    let unreachable = cfg.new_block();
    cfg.block_mut(unreachable).push(1);
    cfg
}

fn critical_cfg() -> Cfg<ApiInst> {
    let mut cfg = Cfg::new();
    let side = cfg.new_block();
    let merge = cfg.new_block();
    cfg.add_edge(cfg.entry(), side, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), merge, EdgeKind::ConditionalFalse);
    cfg.add_edge(side, merge, EdgeKind::Fallthrough);
    cfg
}

fn jump_cfg() -> Cfg<ApiInst> {
    let mut cfg = Cfg::new();
    let target = cfg.new_block();
    cfg.block_mut(cfg.entry())
        .push(ApiInst::control(FlowEffect::Flow(Flow::Jump { target: 7 })));
    cfg.block_mut(target)
        .push(ApiInst::control(FlowEffect::Label(7)));
    cfg
}

fn register_cleanup(suite: &mut BenchmarkSuite<'_>) {
    let linear = linear_cfg(BLOCK_COUNT);
    benchmark_case!(
        suite,
        "api_merge_blocks_mapped",
        covers[merge_blocks_mapped],
        || {
            let mut candidate = linear.clone();
            let result = merge_blocks_mapped(&mut candidate);
            (candidate, result)
        },
        |(candidate, (merged, _))| {
            assert_eq!(*merged, BLOCK_COUNT - 1);
            assert!(verify(candidate).is_ok());
        }
    );

    let empty = empty_chain_cfg(BLOCK_COUNT);
    benchmark_case!(
        suite,
        "api_remove_empty_blocks_mapped",
        covers[remove_empty_blocks_mapped],
        || {
            let mut candidate = empty.clone();
            let result = remove_empty_blocks_mapped(&mut candidate);
            (candidate, result)
        },
        |(candidate, (removed, _))| {
            assert_eq!(*removed, BLOCK_COUNT - 2);
            assert!(verify(candidate).is_ok());
        }
    );

    let unreachable = unreachable_cfg();
    benchmark_case!(
        suite,
        "api_remove_unreachable",
        covers[remove_unreachable],
        || {
            let mut candidate = unreachable.clone();
            let removed = remove_unreachable(&mut candidate);
            (candidate, removed)
        },
        |(candidate, removed)| {
            assert_eq!(*removed, 1);
            assert!(verify(candidate).is_ok());
        }
    );
    benchmark_case!(
        suite,
        "api_remove_unreachable_mapped",
        covers[remove_unreachable_mapped],
        || {
            let mut candidate = unreachable.clone();
            let result = remove_unreachable_mapped(&mut candidate);
            (candidate, result)
        },
        |(candidate, (removed, _))| {
            assert_eq!(*removed, 1);
            assert!(verify(candidate).is_ok());
        }
    );
    benchmark_case!(
        suite,
        "api_simplify",
        covers[simplify],
        || {
            let mut candidate = unreachable.clone();
            let changed = simplify(&mut candidate);
            (candidate, changed)
        },
        |(candidate, changed)| {
            assert_eq!(*changed, 1);
            assert!(verify(candidate).is_ok());
        }
    );
    benchmark_case!(
        suite,
        "api_simplify_mapped",
        covers[simplify_mapped],
        || {
            let mut candidate = unreachable.clone();
            let result = simplify_mapped(&mut candidate);
            (candidate, result)
        },
        |(candidate, (changed, _))| {
            assert_eq!(*changed, 1);
            assert!(verify(candidate).is_ok());
        }
    );
}

fn register_node_edits(suite: &mut BenchmarkSuite<'_>) {
    let linear = linear_cfg(2);
    let source = linear.entry();
    let target = linear
        .successors(source)
        .next()
        .expect("linear fixture must have a successor");
    benchmark_case!(
        suite,
        "api_contract_edge_mapped",
        covers[contract_edge_mapped],
        || {
            let mut candidate = linear.clone();
            let mapping = contract_edge_mapped(&mut candidate, source, target);
            (candidate, mapping)
        },
        |(candidate, mapping)| {
            assert!(mapping.is_some());
            assert!(verify(candidate).is_ok());
        }
    );

    let split = dataflow_cfg(1, 8);
    benchmark_case!(
        suite,
        "cfg_split_block",
        || {
            let mut candidate = split.clone();
            let entry = candidate.entry();
            let new_block = candidate.split_block(entry, 4);
            (candidate, new_block)
        },
        |(candidate, _)| {
            assert_eq!(candidate.block_count(), 2);
            assert!(verify(candidate).is_ok());
        }
    );
    benchmark_case!(
        suite,
        "cfg_split_block_with_payload_mapped",
        || {
            let mut candidate = split.clone();
            let entry = candidate.entry();
            let result = candidate.split_block_with_payload_mapped(entry, 4, ());
            (candidate, result)
        },
        |(candidate, _)| {
            assert_eq!(candidate.block_count(), 2);
            assert!(verify(candidate).is_ok());
        }
    );
    benchmark_case!(
        suite,
        "cfg_split_block_at_points",
        || {
            let mut candidate = split.clone();
            let entry = candidate.entry();
            let result = candidate.split_block_at_points_with_payloads(entry, [(2, ()), (5, ())]);
            (candidate, result)
        },
        |(candidate, result)| {
            assert!(result.is_ok());
            assert_eq!(candidate.block_count(), 3);
            assert!(verify(candidate).is_ok());
        }
    );
}

fn register_critical_edges(suite: &mut BenchmarkSuite<'_>) {
    let cfg = critical_cfg();
    benchmark_case!(
        suite,
        "api_split_critical_edges",
        covers[split_critical_edges],
        || {
            let mut candidate = cfg.clone();
            let split = split_critical_edges(&mut candidate);
            (candidate, split)
        },
        |(candidate, split)| {
            assert_eq!(*split, 1);
            assert!(verify(candidate).is_ok());
        }
    );
    benchmark_case!(
        suite,
        "api_split_critical_edges_mapped",
        covers[split_critical_edges_mapped],
        || {
            let mut candidate = cfg.clone();
            let result = split_critical_edges_mapped(&mut candidate);
            (candidate, result)
        },
        |(candidate, (split, _))| {
            assert_eq!(*split, 1);
            assert!(verify(candidate).is_ok());
        }
    );
    benchmark_case!(
        suite,
        "api_split_critical_edges_with",
        covers[split_critical_edges_with],
        || {
            let mut candidate = cfg.clone();
            let result = split_critical_edges_with(&mut candidate, |_, _| ());
            (candidate, result)
        },
        |(candidate, (split, _))| {
            assert_eq!(*split, 1);
            assert!(verify(candidate).is_ok());
        }
    );
}

fn register_frontend_utilities(suite: &mut BenchmarkSuite<'_>) {
    let cfg = dataflow_cfg(64, 4);
    benchmark_case!(
        suite,
        "api_linearize",
        covers[linearize],
        || linearize(&cfg, BlockOrder::ReversePostorder, &ApiEmitter),
        |linearized: &Result<Vec<_>, _>| {
            let instructions = linearized
                .as_ref()
                .expect("the fixture CFG has an instruction in every block that branches");
            assert!(instructions.len() >= 64 * 4);
        }
    );

    let jumps = jump_cfg();
    benchmark_case!(
        suite,
        "api_resolve_jump_edges",
        covers[resolve_jump_edges],
        || {
            let mut candidate = jumps.clone();
            let resolution = resolve_jump_edges(&mut candidate);
            (candidate, resolution)
        },
        |(candidate, resolution)| {
            assert_eq!(resolution.resolved, 1);
            assert_eq!(
                resolution.unresolved,
                [] as [cfglib::UnresolvedTransfer<u32, cfglib::BlockId>; 0]
            );
            assert!(verify(candidate).is_ok());
        }
    );
}

/// A region protecting `protected_blocks` with one catch handler of unknown
/// extent entered at `handler`.
fn unknown_catch_region(protected_blocks: BTreeSet<BlockId>, handler: BlockId) -> Region {
    Region {
        protected_blocks,
        handlers: vec![Handler {
            entry: handler,
            body: HandlerBody::Unknown,
            kind: HandlerKind::Catch,
        }],
        parent: None,
    }
}

fn register_exception_regions(suite: &mut BenchmarkSuite<'_>) {
    let mut base = Cfg::<ApiInst>::new();
    let handler = base.new_block();
    let protected_blocks = BTreeSet::from([base.entry()]);
    benchmark_case!(
        suite,
        "api_promote_handler_extents",
        covers[promote_handler_extents],
        || {
            let mut candidate = base.clone();
            candidate
                .add_region(unknown_catch_region(protected_blocks.clone(), handler))
                .unwrap();
            promote_handler_extents(&mut candidate)
        },
        |decisions: &Vec<(RegionId, ExtentPromotionStatus)>| assert_eq!(decisions.len(), 1)
    );
}

/// One protected block, one catch handler, and a shared continuation — the
/// shape extent recovery has to reason about.
fn guarded_cfg() -> Cfg<ApiInst> {
    let mut cfg = Cfg::<ApiInst>::new();
    let entry = cfg.entry();
    let handler = cfg.new_block();
    let body = cfg.new_block();
    let join = cfg.new_block();
    cfg.add_edge(entry, body, EdgeKind::Fallthrough);
    cfg.add_edge(entry, handler, EdgeKind::ExceptionHandler);
    cfg.add_edge(body, join, EdgeKind::Fallthrough);
    cfg.add_edge(handler, join, EdgeKind::Fallthrough);
    cfg.add_region(unknown_catch_region(BTreeSet::from([entry]), handler))
        .unwrap();
    cfg
}

fn register_extents(suite: &mut BenchmarkSuite<'_>) {
    let guarded = guarded_cfg();

    benchmark_case!(
        suite,
        "api_recover_exclusive_extents",
        covers[recover_exclusive_extents],
        || recover_exclusive_extents(&guarded, |_, edge| !edge.kind().is_exceptional()),
        |extents: &Vec<ExclusiveExtent>| assert_eq!(extents.len(), 1)
    );
    benchmark_case!(
        suite,
        "api_promote_exclusive_extents",
        covers[promote_exclusive_extents],
        || {
            let mut candidate = guarded.clone();
            let extents =
                recover_exclusive_extents(&candidate, |_, edge| !edge.kind().is_exceptional());
            let decisions = promote_exclusive_extents(&mut candidate, &extents);
            (candidate, decisions)
        },
        |(candidate, decisions): &(Cfg<ApiInst>, Vec<(RegionId, ExtentPromotionStatus)>)| {
            assert!(verify(candidate).is_ok());
            assert_eq!(decisions.len(), 1);
        }
    );
}

fn register_layout(suite: &mut BenchmarkSuite<'_>) {
    // Every eighth item widens once, so the relaxation needs a second pass
    // before it reaches its fixed point.
    let items: Vec<usize> = (0..BLOCK_COUNT).map(|index| index % 4).collect();

    benchmark_case!(
        suite,
        "api_relax_layout",
        covers[relax_layout],
        || {
            let mut candidate = items.clone();
            relax_layout(
                &mut candidate,
                |item: &usize, _offset| Ok::<usize, ()>(item + 1),
                |offsets: &[usize], total| Ok::<usize, ()>(offsets.len() + total),
                |item: &mut usize, offset, _context| {
                    let widen = *item == 0 && offset % 8 == 0;
                    if widen {
                        *item = 1;
                    }
                    Ok::<bool, ()>(widen)
                },
            )
        },
        |result: &Result<(Vec<usize>, usize, usize), RelaxError<()>>| {
            let (offsets, total, _context) = result.as_ref().expect("the relaxation converges");
            assert_eq!(offsets.len(), BLOCK_COUNT);
            assert!(*total >= BLOCK_COUNT);
        }
    );

    let short_chain = cascade(10_000, RelaxMode::Grow);
    benchmark_case!(
        suite,
        "api_relax_spans_cascade_grow_10k",
        covers[relax_spans],
        || relax_cascade(&short_chain, RelaxMode::Grow),
        |result: &CascadeResult| check_cascade(result, 10_000, RelaxMode::Grow)
    );
    let large_short_chain = cascade(100_000, RelaxMode::Grow);
    benchmark_case!(
        suite,
        "api_relax_spans_cascade_grow_100k",
        || relax_cascade(&large_short_chain, RelaxMode::Grow),
        |result: &CascadeResult| check_cascade(result, 100_000, RelaxMode::Grow)
    );
    let long_chain = cascade(10_000, RelaxMode::Shrink);
    benchmark_case!(
        suite,
        "api_relax_spans_cascade_shrink_10k",
        || relax_cascade(&long_chain, RelaxMode::Shrink),
        |result: &CascadeResult| check_cascade(result, 10_000, RelaxMode::Shrink)
    );
    let large_long_chain = cascade(100_000, RelaxMode::Shrink);
    benchmark_case!(
        suite,
        "api_relax_spans_cascade_shrink_100k",
        || relax_cascade(&large_long_chain, RelaxMode::Shrink),
        |result: &CascadeResult| check_cascade(result, 100_000, RelaxMode::Shrink)
    );
}

type CascadeResult = Result<(Vec<usize>, usize), RelaxError<Infallible>>;

/// A branch to the item after next, 2 bytes short and 5 bytes long, whose
/// short form reaches 3 bytes past its own end; or a tail of padding.
#[derive(Clone)]
enum CascadeItem {
    Branch { target: usize, long: bool },
    Tail(usize),
}

impl SpanItem for CascadeItem {
    fn width(&self, _offset: usize) -> usize {
        match *self {
            Self::Branch { long: false, .. } => 2,
            Self::Branch { long: true, .. } => 5,
            Self::Tail(width) => width,
        }
    }

    fn span_target(&self) -> Option<usize> {
        match *self {
            Self::Branch { target, .. } => Some(target),
            Self::Tail(_) => None,
        }
    }

    fn is_aligned(&self) -> bool {
        false
    }

    fn refit(&mut self, displacement: isize, mode: RelaxMode) -> bool {
        let width = isize::try_from(self.width(0)).expect("an item width fits isize");
        let Self::Branch { long, .. } = self else {
            return false;
        };
        let reaches_short = displacement - width <= 3;
        let refit = match mode {
            RelaxMode::Grow => !*long && !reaches_short,
            RelaxMode::Shrink => *long && reaches_short,
        };
        *long ^= refit;
        refit
    }
}

/// `count` branches, each over the next one, so each needs its long form
/// exactly when the next one is long. The tail makes the last branch long
/// when growing and short when shrinking, and the change then runs back
/// through the whole chain one branch per round.
fn cascade(count: usize, mode: RelaxMode) -> Vec<CascadeItem> {
    let (long, tail) = match mode {
        RelaxMode::Grow => (false, 100),
        RelaxMode::Shrink => (true, 1),
    };
    let mut items: Vec<CascadeItem> = (0..count)
        .map(|index| CascadeItem::Branch {
            target: index + 2,
            long,
        })
        .collect();
    items.push(CascadeItem::Tail(tail));
    items
}

fn relax_cascade(items: &[CascadeItem], mode: RelaxMode) -> CascadeResult {
    let mut candidate = items.to_vec();
    relax_spans(&mut candidate, mode)
}

fn check_cascade(result: &CascadeResult, count: usize, mode: RelaxMode) {
    let (offsets, total) = result.as_ref().expect("the cascade fits the address space");
    assert_eq!(offsets.len(), count + 1);
    let expected = match mode {
        RelaxMode::Grow => 5 * count + 100,
        RelaxMode::Shrink => 2 * count + 1,
    };
    assert_eq!(*total, expected, "every branch changed form");
}

/// Node splitting against dispatch on the complete digraph: splitting copies
/// `2^n - n - 1` blocks there, so it gets the small one, and dispatch both.
fn register_reducibility(suite: &mut BenchmarkSuite<'_>) {
    let small = complete_cfg(SPLIT_COMPLETE_BLOCKS);
    let large = complete_cfg(DISPATCH_COMPLETE_BLOCKS);
    benchmark_case!(
        suite,
        "cfg_clone_make_reducible_complete_10",
        || {
            let mut cloned = small.clone();
            let copies = make_reducible(&mut cloned);
            (cloned, copies)
        },
        |(result, copies): &(Cfg<u32>, usize)| {
            assert!(*copies >= 1 << (SPLIT_COMPLETE_BLOCKS - 1));
            assert!(is_reducible(result, &DominatorTree::compute(result)));
        }
    );
    benchmark_case!(
        suite,
        "cfg_clone_make_reducible_with_dispatch_complete_10",
        || {
            let mut cloned = small.clone();
            let report = make_reducible_with_dispatch(&mut cloned, &mut ApiDispatch);
            (cloned, report)
        },
        |(result, report): &(Cfg<u32>, DispatchReport)| {
            assert_dispatched_complete(result, report, SPLIT_COMPLETE_BLOCKS);
        }
    );
    benchmark_case!(
        suite,
        "api_make_reducible_with_dispatch_complete_64",
        covers[make_reducible_with_dispatch],
        || {
            let mut cloned = large.clone();
            let report = make_reducible_with_dispatch(&mut cloned, &mut ApiDispatch);
            (cloned, report)
        },
        |(result, report): &(Cfg<u32>, DispatchReport)| {
            assert_dispatched_complete(result, report, DISPATCH_COMPLETE_BLOCKS);
        }
    );
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    register_cleanup(suite);
    register_extents(suite);
    register_layout(suite);
    register_node_edits(suite);
    register_critical_edges(suite);
    register_frontend_utilities(suite);
    register_exception_regions(suite);
    register_reducibility(suite);
}
