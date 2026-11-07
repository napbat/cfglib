//! The analyses read the operands of instructions that own none through
//! the context a [`WithOperands`] hands them, and answer exactly as they do
//! for the same program whose instructions own their operands.
//!
//! The toy instruction is a [`Row`] of runs into its function's [`Pool`], the
//! layout of a consumer that keeps one operand pool per function. A row
//! cannot answer [`InstrInfo`](crate::InstrInfo) at all, so every fact below
//! is read through the pool.

extern crate alloc;

use alloc::vec::Vec;
use core::ops::Range;

use super::fixture;
use crate::analysis::dead_code::DeadCode;
use crate::cfg::Cfg;
use crate::dataflow::cfg_view::WithOperands;
use crate::dataflow::liveness::Liveness;
use crate::dataflow::memory::{ExactMemoryAlias, MemorySsa};
use crate::dataflow::reaching::ReachingDefs;
use crate::dataflow::ssa::{PhiPlacements, SsaForm, SsaPruning};
use crate::edge::EdgeKind;
use crate::flow::FlowEffect;
use crate::graph::dominator::DominatorTree;
use crate::memory::{MemoryAccess, MemoryEvent, MemoryTrace};
use crate::test_util::{
    DfInst, MemInst, TestEffect, df_const, df_def, df_impure, df_op, df_throwing, df_use,
};
use crate::transform::dce::{dead_code_elimination, dead_code_elimination_with_operands};
use crate::{EffectOperands, InstrOperands, MemoryEventOperands, VariableId};

/// One instruction of a pooled function: four runs of its function's pool
/// and its unwind flag, and no operand of its own.
#[derive(Debug, Clone)]
struct Row {
    uses: Range<u16>,
    defs: Range<u16>,
    effects: Range<u16>,
    events: Range<u16>,
    throws: bool,
}

/// Every operand of one function, which its rows name runs of.
#[derive(Debug)]
struct Pool<V> {
    variables: Vec<V>,
    effects: Vec<TestEffect>,
    events: Vec<MemoryEvent<u32, V, ()>>,
}

/// Appends `items` to `pool` and returns the run they occupy.
fn run<T>(pool: &mut Vec<T>, items: impl IntoIterator<Item = T>) -> Range<u16> {
    let start = u16::try_from(pool.len()).expect("a toy pool fits u16");
    pool.extend(items);
    start..u16::try_from(pool.len()).expect("a toy pool fits u16")
}

/// The entries of `pool` that `run` names.
fn slice<'a, T>(pool: &'a [T], run: &Range<u16>) -> &'a [T] {
    &pool[usize::from(run.start)..usize::from(run.end)]
}

impl<V: Clone> Pool<V> {
    /// Moves the operands of one instruction into the pool.
    fn push(
        &mut self,
        uses: &[V],
        defs: &[V],
        effects: &[TestEffect],
        events: &[MemoryEvent<u32, V, ()>],
        throws: bool,
    ) -> Row {
        Row {
            uses: run(&mut self.variables, uses.iter().cloned()),
            defs: run(&mut self.variables, defs.iter().cloned()),
            effects: run(&mut self.effects, effects.iter().copied()),
            events: run(&mut self.events, events.iter().cloned()),
            throws,
        }
    }
}

impl<V: VariableId> InstrOperands<Row> for Pool<V> {
    type Variable = V;

    fn uses<'a>(&'a self, instruction: &'a Row) -> &'a [V] {
        slice(&self.variables, &instruction.uses)
    }

    fn defs<'a>(&'a self, instruction: &'a Row) -> &'a [V] {
        slice(&self.variables, &instruction.defs)
    }

    fn may_unwind(&self, instruction: &Row) -> bool {
        instruction.throws
    }
}

impl<V: VariableId> EffectOperands<Row> for Pool<V> {
    type Effect = TestEffect;

    fn effects<'a>(&'a self, instruction: &'a Row) -> &'a [TestEffect] {
        slice(&self.effects, &instruction.effects)
    }
}

impl<V: VariableId> MemoryEventOperands<Row> for Pool<V> {
    type Location = u32;
    type Fence = ();

    fn memory_events<'a>(
        &'a self,
        instruction: &'a Row,
    ) -> impl Iterator<Item = MemoryEvent<u32, V, ()>> + 'a {
        slice(&self.events, &instruction.events).iter().cloned()
    }
}

/// The graph of `cfg` with every instruction moved into one pool by `row`.
fn pooled<I, V>(cfg: &Cfg<I>, mut row: impl FnMut(&mut Pool<V>, &I) -> Row) -> (Cfg<Row>, Pool<V>) {
    let mut pool = Pool {
        variables: Vec::new(),
        effects: Vec::new(),
        events: Vec::new(),
    };
    let mut rows: Cfg<Row> = Cfg::new();
    for block in cfg.block_ids().skip(1) {
        assert_eq!(rows.new_block(), block, "the fixture removes no block");
    }
    for block in cfg.block_ids() {
        for instruction in cfg.block(block).instructions() {
            let row = row(&mut pool, instruction);
            rows.block_mut(block).push(row);
        }
    }
    for edge in cfg.edge_ids() {
        let edge = cfg.edge(edge);
        rows.add_edge(edge.source(), edge.target(), edge.kind());
    }
    (rows, pool)
}

/// The rows of a data-flow program.
fn pooled_data_flow(cfg: &Cfg<DfInst>) -> (Cfg<Row>, Pool<u16>) {
    pooled(cfg, |pool, instruction| {
        pool.push(
            &instruction.uses,
            &instruction.defs,
            &instruction.side_effects,
            &[],
            instruction.effect == FlowEffect::MayThrow,
        )
    })
}

#[test]
fn liveness_reads_operands_through_a_context() {
    let cfg = fixture();
    let (rows, pool) = pooled_data_flow(&cfg);
    let view = WithOperands::new(&rows, &pool);
    let expected = Liveness::compute(&cfg);
    let actual = Liveness::compute(&view);
    for block in cfg.block_ids() {
        assert_eq!(actual.live_in(block), expected.live_in(block));
        assert_eq!(actual.live_out(block), expected.live_out(block));
        assert_eq!(
            actual.live_on_unwind(&view, block),
            expected.live_on_unwind(&cfg, block)
        );
        assert_eq!(
            actual.live_before_instructions(&view, block),
            expected.live_before_instructions(&cfg, block)
        );
        assert_eq!(
            actual.live_after_instructions(&view, block),
            expected.live_after_instructions(&cfg, block)
        );
    }
    assert_eq!(
        actual.all_live_variables(&view),
        expected.all_live_variables(&cfg)
    );
    assert!(
        actual.live_on_unwind(&view, rows.entry()).contains(&1),
        "the context reports the throwing definition of r1"
    );
}

#[test]
fn reaching_definitions_read_operands_through_a_context() {
    let cfg = fixture();
    let (rows, pool) = pooled_data_flow(&cfg);
    let expected = ReachingDefs::compute(&cfg);
    let actual = ReachingDefs::compute(&WithOperands::new(&rows, &pool));
    for block in cfg.block_ids() {
        assert_eq!(actual.reaching_in(block), expected.reaching_in(block));
        assert_eq!(actual.reaching_out(block), expected.reaching_out(block));
    }
}

#[test]
fn ssa_reads_operands_through_a_context() {
    let cfg = fixture();
    let (rows, pool) = pooled_data_flow(&cfg);
    let view = WithOperands::new(&rows, &pool);
    let dominators = DominatorTree::compute(&rows);
    let expected = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
    let actual = SsaForm::compute(&view, &dominators);
    assert_eq!(actual, expected);
    assert!(actual.phis().count() > 0, "the join needs a phi for r2");

    let placed = |placements: &PhiPlacements<u16>| {
        placements
            .iter()
            .map(|(block, phi)| (block, phi.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        placed(&PhiPlacements::compute(&view, &dominators)),
        placed(&PhiPlacements::compute(&cfg, &DominatorTree::compute(&cfg)))
    );
}

#[test]
fn pruned_ssa_reads_operands_through_a_context() {
    let cfg = fixture();
    let (rows, pool) = pooled_data_flow(&cfg);
    let view = WithOperands::new(&rows, &pool);
    let (dominators, owned_dominators) =
        (DominatorTree::compute(&rows), DominatorTree::compute(&cfg));
    let placed = |placements: &PhiPlacements<u16>| {
        placements
            .iter()
            .map(|(block, phi)| (block, phi.clone()))
            .collect::<Vec<_>>()
    };
    let mut phi_counts = Vec::new();
    for pruning in [
        SsaPruning::Minimal,
        SsaPruning::SemiPruned,
        SsaPruning::Pruned,
    ] {
        let expected = SsaForm::compute_with_pruning(&cfg, &owned_dominators, pruning);
        let actual = SsaForm::compute_with_pruning(&view, &dominators, pruning);
        assert_eq!(actual, expected, "{pruning:?}");
        assert_eq!(
            placed(&PhiPlacements::compute_with_pruning(
                &view,
                &dominators,
                pruning
            )),
            placed(&PhiPlacements::compute_with_pruning(
                &cfg,
                &owned_dominators,
                pruning
            )),
            "{pruning:?}"
        );
        phi_counts.push(actual.phis().count());
    }
    // r2 is dead entering the loop body that writes it, so only the
    // liveness the context answers drops its phi there; the handler's phi
    // for r1 stays, since the handler reads the value before the throw.
    assert!(phi_counts[2] < phi_counts[0], "{phi_counts:?}");
    assert!(phi_counts[2] > 0, "{phi_counts:?}");
}

/// A dead chain, a definition an effect keeps, and a throwing definition a
/// handler reads.
fn dead_code_fixture() -> Cfg<DfInst> {
    let mut cfg: Cfg<DfInst> = Cfg::new();
    let entry = cfg.entry();
    let exit = cfg.new_block();
    let handler = cfg.new_block();
    cfg.block_mut(entry).push(df_def("r0 dead", 0));
    cfg.block_mut(entry).push(df_op("r1 dead", "add", 1, &[0]));
    cfg.block_mut(entry)
        .push(df_impure("store", TestEffect::MemoryWrite));
    cfg.block_mut(entry).push(df_const("r2", 2, 4));
    cfg.block_mut(entry).push(df_throwing(df_def("r3", 3)));
    cfg.block_mut(exit).push(df_use("use r2", 2));
    cfg.block_mut(handler).push(df_use("use r2", 2));
    cfg.add_edge(entry, exit, EdgeKind::Fallthrough);
    cfg.add_edge(entry, handler, EdgeKind::ExceptionUnwind);
    cfg
}

#[test]
fn dead_code_reads_effects_through_a_context() {
    let mut cfg = dead_code_fixture();
    let (mut rows, pool) = pooled_data_flow(&cfg);
    let expected = DeadCode::compute(&cfg);
    assert_eq!(
        DeadCode::compute(&WithOperands::new(&rows, &pool)),
        expected
    );
    assert_eq!(expected.instructions.len(), 2, "the dead chain");

    let removed = dead_code_elimination_with_operands(&mut rows, &pool, |_| Vec::new());
    assert_eq!(removed, dead_code_elimination(&mut cfg));
    for block in cfg.block_ids() {
        assert_eq!(
            rows.block(block)
                .instructions()
                .iter()
                .map(|row| pool.defs(row).to_vec())
                .collect::<Vec<_>>(),
            cfg.block(block)
                .instructions()
                .iter()
                .map(|instruction| instruction.defs.clone())
                .collect::<Vec<_>>(),
        );
    }
}

/// A store to location 7, a store to it that may unwind, a handler that
/// loads it, and a load of it after the join.
fn memory_fixture() -> Cfg<MemInst<u32, ()>> {
    let mut cfg: Cfg<MemInst<u32, ()>> = Cfg::new();
    let entry = cfg.entry();
    let body = cfg.new_block();
    let handler = cfg.new_block();
    let join = cfg.new_block();
    cfg.block_mut(entry).push(MemInst::with_data_flow(
        [1],
        [],
        [MemoryEvent::Access(MemoryAccess::write(7, [1]))],
    ));
    cfg.block_mut(body).push(
        MemInst::with_data_flow([2], [], [MemoryEvent::Access(MemoryAccess::write(7, [2]))])
            .unwinding(),
    );
    cfg.block_mut(handler).push(MemInst::with_data_flow(
        [],
        [3],
        [MemoryEvent::Access(MemoryAccess::read(7, [3]))],
    ));
    cfg.block_mut(join).push(MemInst::with_data_flow(
        [],
        [4],
        [MemoryEvent::Access(MemoryAccess::read(7, [4]))],
    ));
    cfg.add_edge(entry, body, EdgeKind::Fallthrough);
    cfg.add_edge(body, join, EdgeKind::Fallthrough);
    cfg.add_edge(body, handler, EdgeKind::ExceptionUnwind);
    cfg.add_edge(handler, join, EdgeKind::Fallthrough);
    cfg
}

#[test]
fn memory_ssa_reads_events_through_a_context() {
    let cfg = memory_fixture();
    let (rows, pool) = pooled(&cfg, |pool, instruction| {
        pool.push(
            &instruction.uses,
            &instruction.defs,
            &[],
            &instruction.events,
            instruction.may_unwind,
        )
    });
    let view = WithOperands::new(&rows, &pool);

    let expected = MemorySsa::compute(&cfg, &ExactMemoryAlias);
    let actual = MemorySsa::compute(&view, &ExactMemoryAlias);
    assert_eq!(actual, expected);
    assert!(!actual.phis().is_empty(), "the join merges two stores");
    assert_eq!(MemoryTrace::compute(&view), MemoryTrace::compute(&cfg));
}
