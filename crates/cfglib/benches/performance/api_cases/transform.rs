use std::collections::BTreeSet;

use cfglib::{
    BlockId, BlockOrder, Cfg, ClrExceptionRegion, ClrHandler, ClrHandlerKind, DominatorTree,
    EdgeKind, Emitter, FlowEffect, HandlerBody, HandlerTypes, Liveness, SehExceptionRegion,
    SehHandler, SehHandlerKind, color_graph, contract_edge_mapped, dead_code_elimination,
    detect_loops_tagged, duplicate_structuring_tails, duplicate_structuring_tails_with_structure,
    eliminate_pre, find_loop_invariants, install_clr_region, install_seh_region,
    interference_graph, linearize, merge_blocks_mapped, promote_handler_extents, remove_dead_code,
    remove_dead_code_mapped, remove_empty_blocks_mapped, remove_unreachable,
    remove_unreachable_mapped, resolve_jump_edges, rotate_loop, simplify, simplify_mapped,
    split_critical_edges, split_critical_edges_mapped, split_critical_edges_with, split_node,
    split_node_at_points, split_node_with_payload_mapped, verify,
};
use cfglib::{
    ExclusiveExtent, ExtentPromotionDecision, HandlerRef, RelaxError, promote_exclusive_extents,
    recover_exclusive_extents, relax_layout,
};

use super::BenchmarkSuite;
use super::fixtures::{ApiInst, dataflow_cfg};
use crate::fixtures::{empty_chain_cfg, linear_cfg};
use crate::harness::benchmark_case;

const BLOCK_COUNT: usize = 256;

struct ApiEmitter;

impl Emitter<ApiInst> for ApiEmitter {
    fn emit_jump(&self, target: BlockId) -> ApiInst {
        let mut instruction = ApiInst::control(FlowEffect::Jump);
        instruction.jump_target =
            Some(u32::try_from(target.index()).expect("benchmark block index must fit in u32"));
        instruction
    }

    fn emit_conditional_branch(&self, _condition: &ApiInst, target: BlockId) -> ApiInst {
        let mut instruction = ApiInst::control(FlowEffect::ConditionalJump);
        instruction.jump_target =
            Some(u32::try_from(target.index()).expect("benchmark block index must fit in u32"));
        instruction
    }

    fn emit_block_start(&self, block: BlockId) -> Option<ApiInst> {
        let mut instruction = ApiInst::control(FlowEffect::Label);
        instruction.label =
            Some(u32::try_from(block.index()).expect("benchmark block index must fit in u32"));
        Some(instruction)
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

fn loop_cfg() -> Cfg<ApiInst> {
    let mut cfg = Cfg::new();
    let header = cfg.new_block();
    let body = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ApiInst::constant(0, 1));
    cfg.block_mut(header).push(ApiInst::pure(1, vec![0]));
    cfg.block_mut(body).push(ApiInst::pure(2, vec![0]));
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, header, EdgeKind::Back);
    cfg
}

fn redundant_cfg() -> Cfg<ApiInst> {
    let mut cfg = Cfg::new();
    cfg.block_mut(cfg.entry()).push(ApiInst::constant(0, 1));
    cfg.block_mut(cfg.entry()).push(ApiInst::constant(1, 2));
    cfg.block_mut(cfg.entry())
        .push(ApiInst::pure(2, vec![0, 1]));
    cfg.block_mut(cfg.entry())
        .push(ApiInst::pure(3, vec![0, 1]));
    cfg
}

fn jump_cfg() -> Cfg<ApiInst> {
    let mut cfg = Cfg::new();
    let target = cfg.new_block();
    let mut jump = ApiInst::control(FlowEffect::Jump);
    jump.jump_target = Some(7);
    cfg.block_mut(cfg.entry()).push(jump);
    let mut label = ApiInst::control(FlowEffect::Label);
    label.label = Some(7);
    cfg.block_mut(target).push(label);
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
        "api_split_node",
        covers[split_node],
        || {
            let mut candidate = split.clone();
            let entry = candidate.entry();
            let new_block = split_node(&mut candidate, entry, 4);
            (candidate, new_block)
        },
        |(candidate, _)| {
            assert_eq!(candidate.block_count(), 2);
            assert!(verify(candidate).is_ok());
        }
    );
    benchmark_case!(
        suite,
        "api_split_node_with_payload_mapped",
        covers[split_node_with_payload_mapped],
        || {
            let mut candidate = split.clone();
            let entry = candidate.entry();
            let result = split_node_with_payload_mapped(&mut candidate, entry, 4, ());
            (candidate, result)
        },
        |(candidate, _)| {
            assert_eq!(candidate.block_count(), 2);
            assert!(verify(candidate).is_ok());
        }
    );
    benchmark_case!(
        suite,
        "api_split_node_at_points",
        covers[split_node_at_points],
        || {
            let mut candidate = split.clone();
            let entry = candidate.entry();
            let result = split_node_at_points(&mut candidate, entry, [(2, ()), (5, ())]);
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

fn register_dataflow_transforms(suite: &mut BenchmarkSuite<'_>) {
    let cfg = dataflow_cfg(64, 8);
    benchmark_case!(
        suite,
        "api_dead_code_elimination",
        covers[dead_code_elimination],
        || {
            let mut candidate = cfg.clone();
            let removed = dead_code_elimination(&mut candidate);
            (candidate, removed)
        },
        |(candidate, removed)| {
            assert!(*removed > 0);
            assert!(verify(candidate).is_ok());
        }
    );
    benchmark_case!(
        suite,
        "api_remove_dead_code",
        covers[remove_dead_code],
        || {
            let mut candidate = cfg.clone();
            let removed = remove_dead_code(&mut candidate);
            (candidate, removed)
        },
        |(candidate, removed)| {
            assert!(*removed > 0);
            assert!(verify(candidate).is_ok());
        }
    );
    benchmark_case!(
        suite,
        "api_remove_dead_code_mapped",
        covers[remove_dead_code_mapped],
        || {
            let mut candidate = cfg.clone();
            let result = remove_dead_code_mapped(&mut candidate);
            (candidate, result)
        },
        |(candidate, (removed, _))| {
            assert!(*removed > 0);
            assert!(verify(candidate).is_ok());
        }
    );

    let live = Liveness::compute(&cfg);
    let conflicts = interference_graph(&cfg, &live);
    benchmark_case!(
        suite,
        "api_interference_graph",
        covers[interference_graph],
        || interference_graph(&cfg, &live),
        |graph| assert!(graph.node_count() > 0)
    );
    benchmark_case!(
        suite,
        "api_color_graph",
        covers[color_graph],
        || color_graph(&conflicts),
        |colors| assert_eq!(colors.assignment.len(), conflicts.node_count())
    );

    let redundant = redundant_cfg();
    let dominators = DominatorTree::compute(&redundant);
    benchmark_case!(
        suite,
        "api_eliminate_pre",
        covers[eliminate_pre],
        || {
            let mut candidate = redundant.clone();
            let eliminated = eliminate_pre(&mut candidate, &dominators);
            (candidate, eliminated)
        },
        |(candidate, eliminated)| {
            assert!(*eliminated > 0);
            assert!(verify(candidate).is_ok());
        }
    );
}

fn register_loops(suite: &mut BenchmarkSuite<'_>) {
    let cfg = loop_cfg();
    let dominators = DominatorTree::compute(&cfg);
    let natural_loop = detect_loops_tagged(&cfg, &dominators)
        .into_iter()
        .next()
        .expect("loop benchmark fixture must contain a loop");
    benchmark_case!(
        suite,
        "api_find_loop_invariants",
        covers[find_loop_invariants],
        || find_loop_invariants(&cfg, &natural_loop),
        |invariants: &Vec<_>| assert!(!invariants.is_empty())
    );
    benchmark_case!(
        suite,
        "api_rotate_loop",
        covers[rotate_loop],
        || {
            let mut candidate = cfg.clone();
            let rotation = rotate_loop(&mut candidate, &natural_loop);
            (candidate, rotation)
        },
        |(candidate, rotation)| {
            assert!(rotation.is_some());
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
        |instructions: &Vec<_>| assert!(instructions.len() >= 64 * 4)
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
            assert!(resolution.unresolved.is_empty());
            assert!(verify(candidate).is_ok());
        }
    );
}

fn register_tail_duplication(suite: &mut BenchmarkSuite<'_>) {
    // A short-circuit shape: two conditionals sharing one small tail.
    let mut shared = Cfg::<ApiInst>::new();
    let second = shared.new_block();
    let tail = shared.new_block();
    let then = shared.new_block();
    let join = shared.new_block();
    shared.add_edge(shared.entry(), tail, EdgeKind::ConditionalTrue);
    shared.add_edge(shared.entry(), second, EdgeKind::ConditionalFalse);
    shared.add_edge(second, tail, EdgeKind::ConditionalTrue);
    shared.add_edge(second, then, EdgeKind::ConditionalFalse);
    shared.add_edge(tail, join, EdgeKind::Fallthrough);
    shared.add_edge(then, join, EdgeKind::Fallthrough);
    benchmark_case!(
        suite,
        "api_duplicate_structuring_tails",
        covers[duplicate_structuring_tails],
        || {
            let mut candidate = shared.clone();
            duplicate_structuring_tails(&mut candidate)
        },
        |duplicated: &usize| assert_eq!(*duplicated, 1)
    );
    benchmark_case!(
        suite,
        "api_duplicate_structuring_tails_with_structure",
        covers[duplicate_structuring_tails_with_structure],
        || {
            let mut candidate = shared.clone();
            duplicate_structuring_tails_with_structure(&mut candidate)
        },
        |duplication| {
            assert_eq!(duplication.blocks_materialized, 1);
            assert!(duplication.report.is_fully_structured());
            assert!(!duplication.ast.is_empty());
        }
    );
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
            let mut handler_types = HandlerTypes::new();
            install_clr_region(
                &mut candidate,
                &mut handler_types,
                ClrExceptionRegion {
                    protected_blocks: protected_blocks.clone(),
                    handlers: vec![ClrHandler {
                        entry: handler,
                        body: HandlerBody::Unknown,
                        kind: ClrHandlerKind::Catch { ty: 1_u32 },
                    }],
                    parent: None,
                },
            );
            promote_handler_extents(&mut candidate)
        },
        |promoted: &usize| assert!(*promoted <= 1)
    );
    benchmark_case!(
        suite,
        "api_install_clr_region",
        covers[install_clr_region],
        || {
            let mut candidate = base.clone();
            let mut handler_types = HandlerTypes::new();
            let id = install_clr_region(
                &mut candidate,
                &mut handler_types,
                ClrExceptionRegion {
                    protected_blocks: protected_blocks.clone(),
                    handlers: vec![ClrHandler {
                        entry: handler,
                        body: HandlerBody::Unknown,
                        kind: ClrHandlerKind::Catch { ty: 1_u32 },
                    }],
                    parent: None,
                },
            );
            (candidate, handler_types, id)
        },
        |(candidate, _, _)| assert_eq!(candidate.regions().len(), 1)
    );
    benchmark_case!(
        suite,
        "api_install_seh_region",
        covers[install_seh_region],
        || {
            let mut candidate = base.clone();
            let id = install_seh_region(
                &mut candidate,
                SehExceptionRegion {
                    protected_blocks: protected_blocks.clone(),
                    handlers: vec![SehHandler {
                        entry: handler,
                        body: HandlerBody::Unknown,
                        kind: SehHandlerKind::Finally,
                    }],
                    parent: None,
                },
            );
            (candidate, id)
        },
        |(candidate, _)| assert_eq!(candidate.regions().len(), 1)
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
    let mut handler_types = HandlerTypes::new();
    install_clr_region(
        &mut cfg,
        &mut handler_types,
        ClrExceptionRegion {
            protected_blocks: BTreeSet::from([entry]),
            handlers: vec![ClrHandler {
                entry: handler,
                body: HandlerBody::Unknown,
                kind: ClrHandlerKind::Catch { ty: 1_u32 },
            }],
            parent: None,
        },
    );
    cfg
}

fn register_extents(suite: &mut BenchmarkSuite<'_>) {
    let guarded = guarded_cfg();

    benchmark_case!(
        suite,
        "api_recover_exclusive_extents",
        covers[recover_exclusive_extents, recover_exclusive_extents_with],
        || recover_exclusive_extents(&guarded),
        |extents: &Vec<ExclusiveExtent>| assert_eq!(extents.len(), 1)
    );
    benchmark_case!(
        suite,
        "api_promote_exclusive_extents",
        covers[promote_exclusive_extents],
        || {
            let mut candidate = guarded.clone();
            let bodies: Vec<(HandlerRef, BTreeSet<BlockId>)> =
                recover_exclusive_extents(&candidate)
                    .into_iter()
                    .map(|extent| (extent.handler, extent.blocks))
                    .collect();
            let decisions = promote_exclusive_extents(&mut candidate, |handler| {
                bodies
                    .iter()
                    .find(|(candidate, _)| *candidate == handler)
                    .map(|(_, blocks)| blocks.clone())
            });
            (candidate, decisions)
        },
        |(candidate, decisions): &(Cfg<ApiInst>, Vec<ExtentPromotionDecision>)| {
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
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    register_cleanup(suite);
    register_extents(suite);
    register_layout(suite);
    register_tail_duplication(suite);
    register_node_edits(suite);
    register_critical_edges(suite);
    register_dataflow_transforms(suite);
    register_loops(suite);
    register_frontend_utilities(suite);
    register_exception_regions(suite);
}
