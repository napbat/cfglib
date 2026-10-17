//! Lowering a program out of SSA the way a backend would: forward moves,
//! materialize the phi copies, and sequence them.

use std::collections::BTreeMap;

use cfglib::{
    BasicBlock, BlockId, Cfg, EdgeKind, PhiCopy, PhiCopyOrigin, SsaForm, SsaValue, SsaVersion,
    copies_by_predecessor, eliminate_phis, split_critical_edges,
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

/// Where the copies that run once, when the function starts, are placed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum StartCopies {
    /// In a new block in front of the entry, which becomes the entry: the
    /// copies run once, and no back edge reaches them.
    InFrontOfEntry,
    /// At the top of the entry block: wrong when that block is a loop
    /// header, since every back edge then runs them again.
    AtTopOfEntry,
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

/// Every `Move`'s destination mapped to its source: what a reader of the
/// destination reads once the moves are deleted.
fn forwarded_moves(
    cfg: &Cfg<Insn<Reg>>,
    ssa: &SsaForm<Reg>,
) -> BTreeMap<SsaValue<Reg>, SsaValue<Reg>> {
    let mut forwarded = BTreeMap::new();
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
    forwarded
}

/// Follow `forwarded` from `value` to the value its readers read.
fn resolve_forwarded(
    forwarded: &BTreeMap<SsaValue<Reg>, SsaValue<Reg>>,
    mut value: SsaValue<Reg>,
) -> SsaValue<Reg> {
    while let Some(next) = forwarded.get(&value) {
        value = next.clone();
    }
    value
}

/// A program with the blocks and edges of `cfg` and no instructions, and the
/// lowered block each original block became, by block index.
fn lowered_skeleton(cfg: &Cfg<Insn<Reg>>) -> (Cfg<Insn<SsaValue<Reg>>>, Vec<BlockId>) {
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
    (lowered, ids)
}

/// Where each original instruction index lands once moves are deleted: every
/// instruction, and the end of every block.
fn surviving_positions(
    cfg: &Cfg<Insn<Reg>>,
    propagate_moves: bool,
) -> BTreeMap<(BlockId, usize), usize> {
    let mut new_index = BTreeMap::new();
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
    new_index
}

/// Push the instructions of `cfg` onto their blocks in `lowered` (`ids` maps
/// each original block to its lowered one), reading operands from the `ssa`
/// annotations and resolving uses through `forwarded`. With `propagate_moves`
/// the moves themselves are dropped.
fn push_lowered_instructions(
    cfg: &Cfg<Insn<Reg>>,
    ssa: &SsaForm<Reg>,
    propagate_moves: bool,
    forwarded: &BTreeMap<SsaValue<Reg>, SsaValue<Reg>>,
    lowered: &mut Cfg<Insn<SsaValue<Reg>>>,
    ids: &[BlockId],
) {
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
            let uses = annotation
                .uses()
                .iter()
                .cloned()
                .map(|value| resolve_forwarded(forwarded, value))
                .collect();
            lowered.block_mut(ids[block.index()]).push(Insn::new(
                insn.opcode,
                annotation.defs().to_vec(),
                uses,
            ));
        }
    }
}

/// The instructions that carry out one departure's `group` of phi copies in
/// `style`. A sequenced group takes the next of the fresh temporaries that
/// `temporaries` counts.
fn phi_copy_insns(
    group: &[&PhiCopy<Reg>],
    style: Style,
    temporaries: &mut SsaVersion,
) -> Vec<Insn<SsaValue<Reg>>> {
    match style {
        Style::Parallel => vec![Insn::parallel(
            group.iter().map(|copy| copy.destination.clone()).collect(),
            group.iter().map(|copy| copy.source.clone()).collect(),
        )],
        Style::Sequenced => {
            *temporaries += 1;
            sequence(group, &SsaValue::new(Reg("tmp"), *temporaries))
        }
        Style::Naive => group
            .iter()
            .map(|copy| Insn::mov(copy.destination.clone(), copy.source.clone()))
            .collect(),
    }
}

/// Insert the phi copies of `ssa` into `lowered` in `style`, their sources
/// resolved through `forwarded`. Each departure's copies go where
/// `new_index` puts its departure point in the block `ids` maps it to, and
/// the function-start copies go where `start` says.
fn place_phi_copies(
    ssa: &SsaForm<Reg>,
    forwarded: &BTreeMap<SsaValue<Reg>, SsaValue<Reg>>,
    new_index: &BTreeMap<(BlockId, usize), usize>,
    ids: &[BlockId],
    style: Style,
    start: StartCopies,
    lowered: &mut Cfg<Insn<SsaValue<Reg>>>,
) {
    let copies: Vec<PhiCopy<Reg>> = eliminate_phis(ssa)
        .into_iter()
        .map(|copy| PhiCopy {
            source: resolve_forwarded(forwarded, copy.source.clone()),
            ..copy
        })
        .collect();
    let mut temporaries = 0;
    for group in copies_by_predecessor(&copies).iter().rev() {
        let insns = phi_copy_insns(&group.copies, style, &mut temporaries);
        match group.from {
            PhiCopyOrigin::FunctionStart => {
                place_start_copies(lowered, ids[group.to_block.index()], insns, start);
            }
            PhiCopyOrigin::Departure(point) => {
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
                block.splice(at..at, insns);
            }
        }
    }
}

/// Run `insns` once when the function starts, before `entry` runs.
fn place_start_copies(
    lowered: &mut Cfg<Insn<SsaValue<Reg>>>,
    entry: BlockId,
    insns: Vec<Insn<SsaValue<Reg>>>,
    start: StartCopies,
) {
    match start {
        StartCopies::InFrontOfEntry => {
            let prologue = lowered.new_block();
            lowered.add_edge(prologue, entry, EdgeKind::Fallthrough);
            lowered.set_entry(prologue);
            lowered.block_mut(prologue).instructions_mut().extend(insns);
        }
        StartCopies::AtTopOfEntry => {
            lowered
                .block_mut(entry)
                .instructions_mut()
                .splice(0..0, insns);
        }
    }
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
    lower_with(
        cfg,
        ssa,
        propagate_moves,
        style,
        StartCopies::InFrontOfEntry,
    )
}

/// [`lower`] with the function-start copies placed as `start` says.
pub fn lower_with(
    cfg: &Cfg<Insn<Reg>>,
    ssa: &SsaForm<Reg>,
    propagate_moves: bool,
    style: Style,
    start: StartCopies,
) -> Cfg<Insn<SsaValue<Reg>>> {
    let forwarded = if propagate_moves {
        forwarded_moves(cfg, ssa)
    } else {
        BTreeMap::new()
    };
    let (mut lowered, ids) = lowered_skeleton(cfg);
    let new_index = surviving_positions(cfg, propagate_moves);
    push_lowered_instructions(cfg, ssa, propagate_moves, &forwarded, &mut lowered, &ids);
    place_phi_copies(
        ssa,
        &forwarded,
        &new_index,
        &ids,
        style,
        start,
        &mut lowered,
    );
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
