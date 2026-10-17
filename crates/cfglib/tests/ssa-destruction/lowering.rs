//! Lowering a program out of SSA the way a backend would: forward moves,
//! materialize the phi copies, and sequence them.

use std::collections::BTreeMap;

use cfglib::{
    BasicBlock, BlockId, Cfg, PhiCopy, SsaForm, SsaValue, copies_by_predecessor, eliminate_phis,
    split_critical_edges,
};

use crate::machine::{Behavior, Insn, Opcode, Reg, execute, input_vectors};
use crate::programs::{Program, analyze};

/// How a lowered program places and orders the copies of one departure.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Style {
    /// One parallel move per departure.
    Parallel,
    /// Plain moves in a cycle-free order, with a temporary to break cycles.
    Sequenced,
    /// Plain moves in the order the copies were listed: wrong for a cycle.
    Naive,
}

/// Order the parallel assignment `group` as plain moves, breaking cycles
/// with the fresh register `temporary`.
pub fn sequence(group: &[&PhiCopy<Reg>], temporary: &SsaValue<Reg>) -> Vec<Insn<SsaValue<Reg>>> {
    let mut pending: Vec<(SsaValue<Reg>, SsaValue<Reg>)> = group
        .iter()
        .map(|copy| (copy.destination.clone(), copy.source.clone()))
        .filter(|(destination, source)| destination != source)
        .collect();
    let mut moves = Vec::new();
    while !pending.is_empty() {
        let ready = pending
            .iter()
            .position(|(destination, _)| pending.iter().all(|(_, source)| source != destination));
        if let Some(index) = ready {
            let (destination, source) = pending.remove(index);
            moves.push(Insn::mov(destination, source));
        } else {
            // Every destination is still a pending source: a cycle. Save one
            // destination's value and let its readers take the saved copy.
            let saved = pending[0].0.clone();
            moves.push(Insn::mov(temporary.clone(), saved.clone()));
            for (_, source) in &mut pending {
                if *source == saved {
                    *source = temporary.clone();
                }
            }
        }
    }
    moves
}

/// Lower `cfg` to a program over SSA values.
///
/// With `propagate_moves`, every `Move` is deleted and its readers (and the
/// phi copies' sources) read the moved value directly, as an SSA optimizer
/// does before destruction. That is what makes phis feed each other.
pub fn lower(
    cfg: &Cfg<Insn<Reg>>,
    ssa: &SsaForm<Reg>,
    propagate_moves: bool,
    style: Style,
) -> Cfg<Insn<SsaValue<Reg>>> {
    let mut forwarded: BTreeMap<SsaValue<Reg>, SsaValue<Reg>> = BTreeMap::new();
    if propagate_moves {
        for block in cfg.block_ids() {
            for (insn, annotation) in cfg
                .block(block)
                .instructions()
                .iter()
                .zip(&ssa.block(block).instructions)
            {
                if insn.opcode == Opcode::Move {
                    forwarded.insert(annotation.defs()[0].clone(), annotation.uses()[0].clone());
                }
            }
        }
    }
    let resolve = |mut value: SsaValue<Reg>| {
        while let Some(next) = forwarded.get(&value) {
            value = next.clone();
        }
        value
    };

    let mut lowered: Cfg<Insn<SsaValue<Reg>>> = Cfg::new();
    let mut ids = vec![lowered.entry()];
    ids.extend((1..cfg.block_bound()).map(|_| lowered.new_block()));
    for edge in cfg.edges() {
        lowered.add_edge(
            ids[edge.source().index()],
            ids[edge.target().index()],
            edge.kind(),
        );
    }
    // Where each original instruction index lands once moves are deleted.
    let mut new_index: BTreeMap<(BlockId, usize), usize> = BTreeMap::new();
    for block in cfg.block_ids() {
        let mut kept = 0;
        for (index, insn) in cfg.block(block).instructions().iter().enumerate() {
            new_index.insert((block, index), kept);
            if !(propagate_moves && insn.opcode == Opcode::Move) {
                kept += 1;
            }
        }
        new_index.insert((block, cfg.block(block).instructions().len()), kept);
    }
    for block in cfg.block_ids() {
        for (insn, annotation) in cfg
            .block(block)
            .instructions()
            .iter()
            .zip(&ssa.block(block).instructions)
        {
            if propagate_moves && insn.opcode == Opcode::Move {
                continue;
            }
            let uses = annotation.uses().iter().cloned().map(&resolve).collect();
            lowered.block_mut(ids[block.index()]).push(Insn::new(
                insn.opcode,
                annotation.defs().to_vec(),
                uses,
            ));
        }
    }

    let copies: Vec<PhiCopy<Reg>> = eliminate_phis(ssa)
        .into_iter()
        .map(|copy| PhiCopy {
            source: resolve(copy.source.clone()),
            ..copy
        })
        .collect();
    let mut temporaries = 0;
    for (point, group) in copies_by_predecessor(&copies).iter().rev() {
        let at = new_index[&(point.block, point.inst_idx)];
        let block = lowered
            .block_mut(ids[point.block.index()])
            .instructions_mut();
        let at = if at == block.len()
            && block
                .last()
                .is_some_and(|last| last.opcode == Opcode::BranchNonZero)
        {
            // Copies placed on a branching block run on both arms.
            at - 1
        } else {
            at
        };
        let insns = match style {
            Style::Parallel => vec![Insn::parallel(
                group.iter().map(|copy| copy.destination.clone()).collect(),
                group.iter().map(|copy| copy.source.clone()).collect(),
            )],
            Style::Sequenced => {
                temporaries += 1;
                sequence(group, &SsaValue::new(Reg("tmp"), temporaries))
            }
            Style::Naive => group
                .iter()
                .map(|copy| Insn::mov(copy.destination.clone(), copy.source.clone()))
                .collect(),
        };
        block.splice(at..at, insns);
    }
    lowered
}

/// The registers a lowered program starts with: each input's version-zero
/// value, and zero for every other register it names (a phi may copy a
/// live-in nothing reads).
pub fn lowered_inputs(
    cfg: &Cfg<Insn<Reg>>,
    inputs: &BTreeMap<Reg, i64>,
) -> BTreeMap<SsaValue<Reg>, i64> {
    cfg.blocks()
        .flat_map(BasicBlock::instructions)
        .flat_map(|insn| insn.defs.iter().chain(&insn.uses))
        .map(|register| {
            (
                SsaValue::live_in(*register),
                inputs.get(register).copied().unwrap_or(0),
            )
        })
        .collect()
}

pub fn lowered_behaviors(
    program: &Program,
    style: Style,
    propagate_moves: bool,
    split: bool,
) -> Vec<(Behavior, Behavior)> {
    let mut cfg = program.cfg.clone();
    if split {
        split_critical_edges(&mut cfg);
    }
    let ssa = analyze(&cfg);
    let lowered = lower(&cfg, &ssa, propagate_moves, style);
    input_vectors(program.inputs)
        .into_iter()
        .map(|inputs| {
            (
                execute(&lowered, &lowered_inputs(&cfg, &inputs)),
                execute(&program.cfg, &inputs),
            )
        })
        .collect()
}

pub fn assert_lowering_preserves_behavior(
    name: &str,
    program: &Program,
    style: Style,
    propagate_moves: bool,
) {
    for (lowered, original) in lowered_behaviors(program, style, propagate_moves, true) {
        assert_eq!(lowered, original, "{name}");
    }
}
