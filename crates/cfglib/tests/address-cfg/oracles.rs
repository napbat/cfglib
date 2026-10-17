//! Independent oracles every built address CFG is held to.
//!
//! Each one restates a documented contract in a different shape than the
//! builder uses: block starts against the documented leader rules, edge
//! payloads against a hand-written role table, unwind edges against the
//! exception table, regions against the documented registration order,
//! dominators by node removal, and the control tree by instruction coverage.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use cfglib::{
    AddressInstruction, BasicBlock, BlockId, CallPolicy, Cfg, ControlTree, DominatorTree, EdgeKind,
    Flow, HandlerBody, HandlerRef, RegionId, verify,
};

use crate::harness::{Built, Role, Wire};
use crate::isa::Isa;

/// Runs every oracle over `built`.
///
/// # Panics
///
/// Panics, naming the violated contract, when one rejects the graph.
pub fn check<I: Isa>(built: &Built<I>) {
    let report = verify(built.cfg());
    assert!(report.is_ok(), "verify rejected the graph: {report:?}");
    check_partition(built);
    check_leaders(built);
    check_edge_payloads(built);
    check_unwind_edges(built);
    check_regions(built);
    check_dominators(built.cfg());
    check_control_tree(built);
}

/// Every instruction address maps to exactly one block that contains it, and
/// the blocks tile the stream in address order.
fn check_partition<I: Isa>(built: &Built<I>) {
    let count = built.layout.len();
    assert_eq!(built.graph.instruction_blocks.len(), count);
    let flattened: Vec<usize> = built.blocks().into_iter().flatten().collect();
    assert_eq!(
        flattened,
        (0..count).collect::<Vec<_>>(),
        "blocks tile the stream in address order"
    );
    assert!(
        built
            .cfg()
            .blocks()
            .flat_map(BasicBlock::instructions)
            .eq(built.layout.instructions()),
        "the blocks hold the decoded instructions unchanged"
    );
    for position in 0..count {
        let block = built.block_of(position);
        assert!(built.cfg().contains_block(block));
        let address = built.layout.address(position);
        let holders = built
            .cfg()
            .block(block)
            .instructions()
            .iter()
            .filter(|instruction| instruction.address() == address)
            .count();
        assert_eq!(holders, 1, "instruction {position} is held by its block");
    }
    if count > 0 {
        assert!(
            built.cfg().blocks().all(|block| !block.is_empty()),
            "no block is empty when the stream is not"
        );
    }
}

/// Block starts are exactly the leaders the builder documents.
fn check_leaders<I: Isa>(built: &Built<I>) {
    let starts: BTreeSet<usize> = built
        .blocks()
        .into_iter()
        .filter_map(|block| block.first().copied())
        .collect();
    assert_eq!(starts, expected_leaders(built));
}

fn expected_leaders<I: Isa>(built: &Built<I>) -> BTreeSet<usize> {
    let layout = &built.layout;
    let count = layout.len();
    let mut leaders = BTreeSet::new();
    if count == 0 {
        return leaders;
    }
    leaders.insert(0);
    leaders.extend(built.settings.entry);
    let flatten = built.settings.calls == CallPolicy::Flatten;
    for (position, instruction) in layout.instructions().iter().enumerate() {
        let end = instruction.end_address().expect("the toy code fits");
        let mark = |leaders: &mut BTreeSet<usize>, address| {
            leaders.extend(layout.position_of(address));
        };
        // A stream gap or an overlapping decode puts a leader at the next
        // instruction and at the instruction the end address names.
        let adjacent = layout
            .instructions()
            .get(position + 1)
            .is_some_and(|next| next.address() == end);
        if !adjacent {
            if position + 1 < count {
                leaders.insert(position + 1);
            }
            mark(&mut leaders, end);
        }
        let flow = instruction.flow();
        if matches!(flow, Flow::Next) || (flatten && flow.is_call()) {
            continue;
        }
        if position + 1 < count {
            leaders.insert(position + 1);
        }
        if flow.continues() {
            mark(&mut leaders, end);
        }
        match &flow {
            Flow::Jump { target }
            | Flow::Conditional { target }
            | Flow::Call { target }
            | Flow::Exceptional { target } => mark(&mut leaders, *target),
            Flow::Switch { default, cases } => {
                mark(&mut leaders, *default);
                for (_, target) in cases {
                    mark(&mut leaders, *target);
                }
            }
            Flow::Next | Flow::Return | Flow::Throw | Flow::Indirect | Flow::IndirectCall => {}
        }
        if matches!(flow, Flow::Exceptional { .. }) {
            leaders.insert(position);
        }
    }
    for entry in &built.table {
        leaders.insert(entry.protected.start);
        if entry.protected.end < count {
            leaders.insert(entry.protected.end);
        }
        leaders.insert(entry.entry);
        leaders.extend(entry.protected.clone());
    }
    leaders
}

/// The hand-written role-to-kind table of the edge vocabulary.
fn kind_of(role: Role) -> EdgeKind {
    match role {
        Role::Sequential => EdgeKind::Fallthrough,
        Role::Taken => EdgeKind::ConditionalTrue,
        Role::FallThrough => EdgeKind::ConditionalFalse,
        Role::Jump => EdgeKind::Jump,
        Role::Default | Role::Case { .. } => EdgeKind::SwitchCase,
        Role::Call => EdgeKind::Call,
        Role::Continuation { .. } => EdgeKind::CallReturn,
        Role::Exceptional | Role::Unwind { .. } => EdgeKind::ExceptionUnwind,
    }
}

/// The payload callback runs once per edge, and each payload describes its
/// own edge: structural kind, source instruction, and target block start.
fn check_edge_payloads<I: Isa>(built: &Built<I>) {
    let cfg = built.cfg();
    assert_eq!(built.payload_calls, cfg.edge_count());
    for edge in cfg.edges() {
        let wire = *edge.payload();
        assert_eq!(
            wire.kind,
            edge.kind(),
            "the payload carries the stored kind"
        );
        assert_eq!(wire.kind, kind_of(wire.role), "role {:?}", wire.role);
        let source: u64 = cfg
            .block(edge.source())
            .instructions()
            .last()
            .expect("an edge leaves a non-empty block")
            .address()
            .into();
        let target: u64 = cfg
            .block(edge.target())
            .instructions()
            .first()
            .expect("an edge enters a non-empty block")
            .address()
            .into();
        assert_eq!(
            wire.source, source,
            "{wire:?} names its transferring instruction"
        );
        assert_eq!(wire.target, target, "{wire:?} names its target block start");
        if let Role::Continuation { call_site } = wire.role {
            assert_eq!(call_site, wire.source);
        }
    }
}

/// Unwind edges are exactly one per protected instruction that retains its
/// exception edge and per table entry covering it, and each protected
/// instruction leads a block of its own.
fn check_unwind_edges<I: Isa>(built: &Built<I>) {
    let cfg = built.cfg();
    let mut expected = Vec::new();
    for (handler, entry) in built.table.iter().enumerate() {
        for position in entry.protected.clone() {
            let block = built.block_of(position);
            assert_eq!(
                cfg.block(block).instructions().len(),
                1,
                "protected instruction {position} leads its own block"
            );
            if built.layout.instructions()[position].retains_exception_edge() {
                expected.push((block.index(), built.block_of(entry.entry).index(), handler));
            }
        }
    }
    let mut actual: Vec<_> = cfg
        .edges()
        .filter_map(|edge| match edge.payload().role {
            Role::Unwind { handler } => {
                Some((edge.source().index(), edge.target().index(), handler))
            }
            _ => None,
        })
        .collect();
    expected.sort_unstable();
    actual.sort_unstable();
    assert_eq!(actual, expected);
}

fn encloses(outer: &Range<usize>, inner: &Range<usize>) -> bool {
    outer != inner && outer.start <= inner.start && inner.end <= outer.end
}

/// Regions follow the documented registration contract: handlers sharing a
/// range share a region, regions are ordered by descending span with table
/// order between equal spans, each names its innermost enclosing parent, and
/// every table entry has one handler reference in table order.
fn check_regions<I: Isa>(built: &Built<I>) {
    let cfg = built.cfg();
    let table = &built.table;
    assert_eq!(built.graph.handler_refs.len(), table.len());
    let mut ranges: Vec<(Range<usize>, Vec<usize>)> = Vec::new();
    for (index, entry) in table.iter().enumerate() {
        match ranges
            .iter_mut()
            .find(|(range, _)| *range == entry.protected)
        {
            Some((_, handlers)) => handlers.push(index),
            None => ranges.push((entry.protected.clone(), vec![index])),
        }
    }
    ranges.sort_by_key(|(range, _)| std::cmp::Reverse(built.layout.byte_span(range.clone())));
    assert_eq!(cfg.regions().len(), ranges.len());
    for (region_index, (range, handler_indices)) in ranges.iter().enumerate() {
        let region = &cfg.regions()[region_index];
        assert_eq!(region.id.index(), region_index);
        let protected: BTreeSet<BlockId> = range.clone().map(|p| built.block_of(p)).collect();
        assert_eq!(region.protected_blocks, protected);
        assert_eq!(region.handlers.len(), handler_indices.len());
        for (slot, &index) in handler_indices.iter().enumerate() {
            let handler = &region.handlers[slot];
            assert_eq!(handler.entry, built.block_of(table[index].entry));
            assert_eq!(handler.kind, table[index].kind);
            assert_eq!(handler.body, HandlerBody::Unknown);
            assert_eq!(
                built.graph.handler_refs[index],
                HandlerRef::new(region.id, slot)
            );
        }
        let parent = (0..ranges.len())
            .filter(|&other| encloses(&ranges[other].0, range))
            .min_by_key(|&other| built.layout.byte_span(ranges[other].0.clone()));
        assert_eq!(region.parent.map(RegionId::index), parent);
    }
}

/// The blocks reachable from `roots`, never entering `removed`.
pub fn reach<I, E>(
    cfg: &Cfg<I, E>,
    roots: &[BlockId],
    removed: Option<BlockId>,
) -> BTreeSet<BlockId> {
    let mut seen = BTreeSet::new();
    let mut pending: Vec<BlockId> = roots
        .iter()
        .copied()
        .filter(|&r| Some(r) != removed)
        .collect();
    while let Some(block) = pending.pop() {
        if seen.insert(block) {
            pending.extend(cfg.successors(block).filter(|&next| Some(next) != removed));
        }
    }
    seen
}

/// `DominatorTree` against dominance by node removal: `d` dominates `n` when
/// `n` is reachable from the entry and stops being reachable once `d` is
/// removed.
fn check_dominators<I, E>(cfg: &Cfg<I, E>) {
    let entry = cfg.entry();
    let blocks: Vec<BlockId> = cfg.block_ids().collect();
    let reachable = reach(cfg, &[entry], None);
    let without: BTreeMap<BlockId, BTreeSet<BlockId>> = blocks
        .iter()
        .map(|&block| (block, reach(cfg, &[entry], Some(block))))
        .collect();
    let dominates = |dominator: BlockId, node: BlockId| {
        reachable.contains(&node) && (dominator == node || !without[&dominator].contains(&node))
    };
    let tree = DominatorTree::compute(cfg);
    for &node in &blocks {
        assert_eq!(tree.is_reachable(node), reachable.contains(&node));
        for &dominator in &blocks {
            assert_eq!(
                tree.dominates(dominator, node),
                dominator == node || dominates(dominator, node),
                "{dominator} dominates {node}"
            );
        }
        if !reachable.contains(&node) {
            continue;
        }
        let strict: Vec<BlockId> = blocks
            .iter()
            .copied()
            .filter(|&d| d != node && dominates(d, node))
            .collect();
        assert_eq!(tree.level(node), Some(u32::try_from(strict.len()).unwrap()));
        let immediate = strict
            .iter()
            .copied()
            .find(|&s| strict.iter().all(|&t| dominates(t, s)));
        assert_eq!(tree.idom(node), immediate, "idom of {node}");
    }
    for &first in blocks.iter().filter(|b| reachable.contains(b)) {
        for &second in blocks.iter().filter(|b| reachable.contains(b)) {
            let common: Vec<BlockId> = blocks
                .iter()
                .copied()
                .filter(|&d| dominates(d, first) && dominates(d, second))
                .collect();
            let deepest = common
                .iter()
                .copied()
                .find(|&c| common.iter().all(|&t| dominates(t, c)));
            assert_eq!(tree.nearest_common_dominator(first, second), deepest);
        }
    }
}

/// The documented precondition of structuring a region: no handler entry is
/// among its own region's protected blocks, and ordinary flow enters a
/// handler (whose extent is only its entry here) from nowhere else.
fn regions_are_well_formed<I, E>(cfg: &Cfg<I, E>) -> bool {
    cfg.regions().iter().all(|region| {
        region.handlers.iter().all(|handler| {
            !region.protected_blocks.contains(&handler.entry)
                && cfg.incoming(handler.entry).all(|edge| {
                    let edge = cfg.edge(edge);
                    edge.kind().is_exceptional() || edge.source() == handler.entry
                })
        })
    })
}

/// The instruction addresses held by `blocks`.
fn addresses_in<I: Isa>(cfg: &Cfg<I::Insn, Wire>, blocks: &BTreeSet<BlockId>) -> BTreeSet<u64> {
    blocks
        .iter()
        .flat_map(|&block| cfg.block(block).instructions())
        .map(|instruction| instruction.address().into())
        .collect()
}

/// The control tree holds each owned instruction exactly once.
///
/// A block is owned when the entry reaches it or when it is a handler entry.
/// The tree may also lift what flows out of a dead handler pad, but it never
/// holds a block nothing reaches. Regions that break the structuring
/// precondition promise nothing, so they are not checked.
fn check_control_tree<I: Isa>(built: &Built<I>) {
    let cfg = built.cfg();
    if !regions_are_well_formed(cfg) {
        return;
    }
    let (tree, _) = ControlTree::compute_with_report(cfg);
    let mut visited: Vec<u64> = Vec::new();
    tree.for_each_instruction(cfg, |_, instruction| {
        visited.push(instruction.address().into());
    });
    let held = visited.len();
    let visited: BTreeSet<u64> = visited.into_iter().collect();
    assert_eq!(visited.len(), held, "no instruction appears twice");

    let entries: Vec<BlockId> = cfg
        .regions()
        .iter()
        .flat_map(|region| region.handlers.iter().map(|handler| handler.entry))
        .collect();
    let mut owned = reach(cfg, &[cfg.entry()], None);
    owned.extend(entries.iter().copied());
    let owned = addresses_in::<I>(cfg, &owned);
    let missing: Vec<_> = owned.difference(&visited).collect();
    assert!(
        missing.is_empty(),
        "owned instructions {missing:?} were dropped"
    );

    let mut roots = entries;
    roots.push(cfg.entry());
    let candidates = addresses_in::<I>(cfg, &reach(cfg, &roots, None));
    let invented: Vec<_> = visited.difference(&candidates).collect();
    assert!(
        invented.is_empty(),
        "unreachable instructions {invented:?} were kept"
    );
}
