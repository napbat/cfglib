//! Linearization of a CFG into a flat stream: block order, markers, and the
//! jumps a layout needs.

use std::collections::BTreeSet;

use cfglib::{BlockId, BlockOrder, Cfg, EdgeKind, LinearInst, linearize};

use crate::programs::{Rng, random_program};
use crate::toy::{Builder, INPUTS, Inst, Kind, OBSERVED, ToyEmitter, run, run_linear};

/// A linearizer's answer, whether or not linearizing can fail.
pub trait Stream {
    fn stream(self) -> Vec<LinearInst<Inst>>;
}

impl Stream for Vec<LinearInst<Inst>> {
    fn stream(self) -> Self {
        self
    }
}

impl<E: std::fmt::Debug> Stream for Result<Vec<LinearInst<Inst>>, E> {
    fn stream(self) -> Vec<LinearInst<Inst>> {
        self.expect("every toy block that branches has an instruction to branch on")
    }
}

/// The one place that calls the linearizer.
pub fn linearized(cfg: &Cfg<Inst>, order: BlockOrder) -> Vec<LinearInst<Inst>> {
    linearize(cfg, order, &ToyEmitter).stream()
}

fn layout_of(stream: &[LinearInst<Inst>]) -> Vec<BlockId> {
    stream
        .iter()
        .filter(|entry| matches!(entry.inst.kind, Kind::Label(_)))
        .map(|entry| entry.block)
        .collect()
}

fn is_fallthrough(kind: EdgeKind) -> bool {
    matches!(
        kind,
        EdgeKind::Fallthrough | EdgeKind::ConditionalFalse | EdgeKind::CallReturn
    )
}

/// Checks the stream against the document of `linearize`, block by block: a
/// marker, the block's own instructions, then exactly the branches its edges
/// need given which block comes next.
fn assert_stream_follows_the_cfg(
    cfg: &Cfg<Inst>,
    stream: &[LinearInst<Inst>],
    expected_blocks: &BTreeSet<BlockId>,
    context: &str,
) {
    let layout = layout_of(stream);
    assert_eq!(
        layout.iter().copied().collect::<BTreeSet<_>>(),
        *expected_blocks,
        "{context}: wrong block set"
    );
    assert_eq!(
        layout.len(),
        expected_blocks.len(),
        "{context}: a block is emitted twice"
    );
    assert_eq!(
        stream[0].block,
        cfg.entry(),
        "{context}: the entry comes first"
    );

    let mut position = 0;
    for (index, &block) in layout.iter().enumerate() {
        let run_length = stream[position..]
            .iter()
            .take_while(|entry| entry.block == block)
            .count();
        let run = &stream[position..position + run_length];
        position += run_length;

        assert_eq!(run[0].inst.kind, Kind::Label(block.index()), "{context}");
        assert_eq!(run[0].index, usize::MAX, "{context}");
        let own = cfg.block(block).instructions();
        for (offset, instruction) in own.iter().enumerate() {
            assert_eq!(run[1 + offset].index, offset, "{context}");
            assert_eq!(&run[1 + offset].inst, instruction, "{context}");
        }
        let synthetic: Vec<Kind> = run[1 + own.len()..]
            .iter()
            .map(|entry| {
                assert_eq!(entry.index, usize::MAX, "{context}: only synthetics follow");
                entry.inst.kind
            })
            .collect();

        let next = layout.get(index + 1).copied();
        let mut expected = Vec::new();
        let mut trailing = None;
        for edge in cfg.outgoing(block) {
            let (kind, target) = (cfg.edge(edge).kind(), cfg.edge(edge).target());
            if kind == EdgeKind::ConditionalTrue {
                expected.push(Kind::BranchIf(target.index()));
            } else if is_fallthrough(kind) {
                if next != Some(target) {
                    trailing = Some(Kind::Jump(target.index()));
                }
            } else {
                expected.push(Kind::Jump(target.index()));
            }
        }
        expected.extend(trailing);
        assert_eq!(synthetic, expected, "{context}: branches of {block}");
        if trailing.is_some() {
            assert_eq!(
                synthetic.last().copied(),
                trailing,
                "{context}: the jump closes the block"
            );
        }
    }
    assert_eq!(position, stream.len(), "{context}");
}

fn shuffled_order(cfg: &Cfg<Inst>, seed: u64) -> Vec<BlockId> {
    let mut rng = Rng::new(seed);
    let mut rest: Vec<BlockId> = cfg
        .block_ids()
        .filter(|&block| block != cfg.entry())
        .collect();
    for index in (1..rest.len()).rev() {
        rest.swap(index, rng.below(index + 1));
    }
    let mut order = vec![cfg.entry()];
    order.extend(rest);
    order
}

fn assert_stream_runs_like_the_cfg(cfg: &Cfg<Inst>, stream: &[LinearInst<Inst>], context: &str) {
    for inputs in INPUTS {
        let expected = run(cfg, &inputs, &OBSERVED);
        assert!(expected.is_ok(), "{context}: {expected:?}");
        assert_eq!(
            run_linear(stream, &inputs, &OBSERVED),
            expected,
            "{context}: {inputs:?}"
        );
    }
}

#[test]
fn every_reachable_block_is_emitted_once_in_reverse_postorder_and_runs_like_the_cfg() {
    for seed in 0..60 {
        let cfg = random_program::<()>(seed, true);
        let reachable: BTreeSet<BlockId> = cfg.reverse_postorder().into_iter().collect();
        let stream = linearized(&cfg, BlockOrder::ReversePostorder);
        let context = format!("seed {seed}");

        assert_stream_follows_the_cfg(&cfg, &stream, &reachable, &context);
        assert_stream_runs_like_the_cfg(&cfg, &stream, &context);
    }
}

#[test]
fn allocation_order_emits_every_live_block_including_unreachable_ones() {
    for seed in 0..60 {
        let mut cfg = random_program::<()>(seed, true);
        let live: BTreeSet<BlockId> = cfg.block_ids().collect();
        let stream = linearized(&cfg, BlockOrder::AllocationOrder);
        let context = format!("seed {seed}");
        assert_stream_follows_the_cfg(&cfg, &stream, &live, &context);
        assert_stream_runs_like_the_cfg(&cfg, &stream, &context);
        assert_eq!(
            layout_of(&stream),
            live.iter().copied().collect::<Vec<_>>(),
            "{context}: ascending identity order"
        );

        let doomed = cfg.block_ids().last().unwrap();
        cfg.remove_block(doomed);
        if cfg.entry() != doomed {
            let after = linearized(&cfg, BlockOrder::AllocationOrder);
            assert!(
                !layout_of(&after).contains(&doomed),
                "{context}: removed slots are skipped"
            );
        }
    }
}

#[test]
fn any_caller_chosen_order_still_runs_like_the_cfg_by_adding_jumps() {
    for seed in 0..60 {
        let cfg = random_program::<()>(seed, true);
        let order = shuffled_order(&cfg, seed);
        let live: BTreeSet<BlockId> = cfg.block_ids().collect();
        let stream = linearized(&cfg, BlockOrder::Custom(order.clone()));
        let context = format!("seed {seed}");

        assert_eq!(layout_of(&stream), order, "{context}");
        assert_stream_follows_the_cfg(&cfg, &stream, &live, &context);
        assert_stream_runs_like_the_cfg(&cfg, &stream, &context);
    }
}

#[test]
fn exceptional_edges_are_not_control_transfers_of_the_stream() {
    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    builder.push(entry, Inst::print(0));
    let next = builder.block([Inst::print(2), Inst::ret()]);
    let handler = builder.block([Inst::print(3), Inst::ret()]);
    builder.edge(entry, next, EdgeKind::Fallthrough);
    builder.edge(entry, handler, EdgeKind::ExceptionUnwind);
    let cfg = builder.finish();

    let stream = linearized(&cfg, BlockOrder::AllocationOrder);

    assert!(
        !stream
            .iter()
            .any(|entry| entry.inst.kind == Kind::Jump(handler.index())),
        "normal flow never jumps into a handler"
    );
    assert_stream_runs_like_the_cfg(&cfg, &stream, "unwind edge");
}

#[test]
fn a_loop_in_reverse_postorder_closes_with_one_back_jump() {
    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    builder.push(entry, Inst::constant(5, 0));
    let header = builder.block([Inst::less(9, 5, 1), Inst::branch(9)]);
    let body = builder.block([Inst::constant(4, 1), Inst::add(5, 5, 4), Inst::print(5)]);
    let exit = builder.block([Inst::ret()]);
    builder.edge(entry, header, EdgeKind::Fallthrough);
    builder.edge(header, body, EdgeKind::ConditionalTrue);
    builder.edge(header, exit, EdgeKind::ConditionalFalse);
    builder.edge(body, header, EdgeKind::Back);
    let cfg = builder.finish();

    let stream = linearized(&cfg, BlockOrder::ReversePostorder);

    let kinds: Vec<Kind> = stream.iter().map(|entry| entry.inst.kind).collect();
    let jumps_to_header = kinds
        .iter()
        .filter(|&&kind| kind == Kind::Jump(header.index()))
        .count();
    assert_eq!(jumps_to_header, 1, "only the latch jumps back");
    assert_eq!(
        stream
            .iter()
            .filter(|entry| matches!(entry.inst.kind, Kind::Jump(_)))
            .count(),
        1,
        "every other edge is a fallthrough in this order: {kinds:?}"
    );
    assert_stream_follows_the_cfg(&cfg, &stream, &cfg.block_ids().collect(), "loop");
    assert_stream_runs_like_the_cfg(&cfg, &stream, "loop");
}
