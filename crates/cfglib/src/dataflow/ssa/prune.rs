//! Which variables and blocks receive phis: the minimal, semi-pruned and
//! pruned forms of SSA.
//!
//! Minimal SSA places a phi at every iterated dominance frontier of every
//! variable's definitions. The two pruned forms leave out phis that no
//! instruction can read. They change no value an instruction reads, only the
//! version numbers, because fewer phis consume versions.

extern crate alloc;

use alloc::vec::Vec;

use crate::block::BlockId;
use crate::dataflow::cfg_view::CfgView;
use crate::dataflow::unwind::is_unwind_edge;
use crate::graph::epoch::EpochMarks;

/// Which phis SSA construction places.
///
/// Every form gives each instruction operand the same definition: the same
/// instruction definition, the same phi, or the version-zero live-in. A
/// pruned form only leaves out phis whose value no instruction reads, so it
/// numbers the remaining definitions with fewer versions.
///
/// The phis left out are the ones that dominate the cost of a function with
/// wide joins. A phi has an operand per edge into its block, so minimal SSA
/// over a join with thousands of predecessors pays operands for every
/// variable defined on its way, read after the join or not.
///
/// [`SsaForm::value_at`](super::SsaForm::value_at) answers the same
/// definition in every form wherever the variable is live. Where a pruned
/// form left out a phi, no instruction reads the variable before redefining
/// it, and `value_at` names the nearest dominating definition instead.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum SsaPruning {
    /// Place a phi at every iterated dominance frontier of each variable's
    /// definitions (minimal SSA).
    #[default]
    Minimal,
    /// Place phis only for a variable that some block reads before it
    /// writes it (semi-pruned SSA, Briggs et al.).
    ///
    /// A variable that every block writes before it reads is read only in
    /// the block that wrote it, so no phi of it can be read. This costs one
    /// pass over the instructions.
    SemiPruned,
    /// Place a phi only at a block the variable is live on entry to (pruned
    /// SSA).
    ///
    /// Liveness is the one [`Liveness`](crate::Liveness) computes, including
    /// its unwind rule, solved per variable over the blocks it is live in.
    /// It places the fewest phis of the three.
    Pruned,
}

/// Mark every block that `variable` is live on entry to, given the blocks
/// that read it before writing it.
///
/// This walks backward from those reads and stops at a block that writes
/// the variable. An edge that [departs before a throwing
/// instruction](crate::departs_before_throws) carries the value before that
/// instruction, so the block it leaves only stops the walk when it writes
/// the variable before its first throwing instruction. It is the
/// per-variable form of the [`Liveness`](crate::Liveness) equation, and it
/// visits only the blocks the variable is live in.
///
/// `writes` marks the blocks that write the variable anywhere, `throwing` the
/// blocks with an unwind edge and a throwing instruction, and
/// `writes_before_throw` lists, in ascending block order, the blocks that
/// write the variable before their first throwing instruction. The answer is
/// left in `live`, and `worklist` is left empty.
pub(super) fn mark_live_blocks<G, V>(
    cfg: &G,
    reads: impl IntoIterator<Item = BlockId>,
    writes: &EpochMarks,
    throwing: &EpochMarks,
    writes_before_throw: &[(V, BlockId)],
    live: &mut EpochMarks,
    worklist: &mut Vec<BlockId>,
) where
    G: CfgView,
{
    live.reset();
    worklist.clear();
    for block in reads {
        if !live.is_marked(block.index()) {
            live.mark(block.index());
            worklist.push(block);
        }
    }
    while let Some(block) = worklist.pop() {
        for edge in cfg.incoming(block) {
            let edge = cfg.edge(edge);
            let source = edge.source();
            if live.is_marked(source.index()) {
                continue;
            }
            let killed = if is_unwind_edge(edge) && throwing.is_marked(source.index()) {
                writes_before_throw
                    .binary_search_by_key(&source, |&(_, block)| block)
                    .is_ok()
            } else {
                writes.is_marked(source.index())
            };
            if !killed {
                live.mark(source.index());
                worklist.push(source);
            }
        }
    }
}

/// The run of `sorted` that belongs to `variable`, advancing `cursor` past
/// it.
///
/// `sorted` ascends by variable, and the callers ask for variables in
/// ascending order, so one cursor walks each list once alongside the
/// definition map.
pub(super) fn run_of<'a, V: Ord, T>(
    sorted: &'a [(V, T)],
    cursor: &mut usize,
    variable: &V,
) -> &'a [(V, T)] {
    while sorted
        .get(*cursor)
        .is_some_and(|(other, _)| other < variable)
    {
        *cursor += 1;
    }
    let start = *cursor;
    while sorted
        .get(*cursor)
        .is_some_and(|(other, _)| other == variable)
    {
        *cursor += 1;
    }
    &sorted[start..*cursor]
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::collections::{BTreeMap, BTreeSet};
    use alloc::vec::Vec;

    use super::SsaPruning;
    use crate::block::BlockId;
    use crate::cfg::Cfg;
    use crate::dataflow::liveness::Liveness;
    use crate::dataflow::program_point::ProgramPoint;
    use crate::dataflow::ssa::{PhiPlacements, SsaForm, SsaScratch, SsaValue};
    use crate::edge::EdgeKind;
    use crate::graph::dominator::DominatorTree;
    use crate::test_util::rng::Rng;
    use crate::test_util::{DfInst, df_def, df_ff, df_throwing, df_use, protected_region};

    const PRUNED_FORMS: [SsaPruning; 2] = [SsaPruning::SemiPruned, SsaPruning::Pruned];

    fn instruction(uses: &[u16], defs: &[u16]) -> DfInst {
        DfInst {
            uses: uses.to_vec(),
            defs: defs.to_vec(),
            ..df_ff("instruction")
        }
    }

    fn form(cfg: &Cfg<DfInst>, pruning: SsaPruning) -> SsaForm<u16> {
        SsaForm::compute_with_pruning(cfg, &DominatorTree::compute(cfg), pruning)
    }

    /// The phi count and the phi operand count of each form, minimal first.
    fn counts(cfg: &Cfg<DfInst>) -> [[usize; 2]; 3] {
        [
            SsaPruning::Minimal,
            SsaPruning::SemiPruned,
            SsaPruning::Pruned,
        ]
        .map(|pruning| {
            let form = form(cfg, pruning);
            let operands = form.phis().map(|(_, phi)| phi.operands().len()).sum();
            [form.phis().count(), operands]
        })
    }

    /// Where a value is defined, which does not depend on version numbers.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Origin {
        Instruction(ProgramPoint),
        Phi(BlockId),
        Start,
    }

    fn origins(form: &SsaForm<u16>) -> BTreeMap<SsaValue<u16>, Origin> {
        let mut origins = BTreeMap::new();
        for block in form.blocks() {
            for phi in &block.phis {
                origins.insert(phi.result.clone(), Origin::Phi(block.block));
            }
            for instruction in &block.instructions {
                for def in instruction.defs() {
                    origins.insert(def.clone(), Origin::Instruction(instruction.point));
                }
            }
        }
        origins
    }

    fn origin(origins: &BTreeMap<SsaValue<u16>, Origin>, value: &SsaValue<u16>) -> Origin {
        if value.is_live_in() {
            Origin::Start
        } else {
            origins[value]
        }
    }

    /// One phi: its block, its variable, and each operand's departure point
    /// and origin.
    type PhiRow = (BlockId, u16, Vec<(ProgramPoint, Origin)>);

    /// Every phi of a form, in the form's order.
    fn phis(form: &SsaForm<u16>) -> Vec<PhiRow> {
        let origins = origins(form);
        form.phis()
            .map(|(block, phi)| {
                let operands = phi
                    .operands()
                    .iter()
                    .map(|(point, value)| (*point, origin(&origins, value)))
                    .collect();
                (block, phi.result.variable, operands)
            })
            .collect()
    }

    /// The variables some block reads before it writes them.
    fn upward_exposed(cfg: &Cfg<DfInst>) -> BTreeSet<u16> {
        let mut exposed = BTreeSet::new();
        for block in cfg.block_ids() {
            let mut written = BTreeSet::new();
            for instruction in cfg.block(block).instructions() {
                exposed.extend(
                    instruction
                        .uses
                        .iter()
                        .filter(|&use_| !written.contains(use_)),
                );
                written.extend(instruction.defs.iter().copied());
            }
        }
        exposed
    }

    /// A CFG of up to ten blocks over four variables, with random edges,
    /// unwind edges, throwing instructions and unreachable blocks.
    fn random_cfg(rng: &mut Rng) -> Cfg<DfInst> {
        fn variable(rng: &mut Rng) -> u16 {
            u16::try_from(rng.below(4)).expect("a variable below four fits in u16")
        }
        let mut cfg = Cfg::new();
        let count = 2 + rng.below(9);
        let mut blocks = alloc::vec![cfg.entry()];
        blocks.extend((1..count).map(|_| cfg.new_block()));
        for &block in &blocks {
            for _ in 0..rng.below(4) {
                let uses: Vec<u16> = (0..rng.below(3)).map(|_| variable(rng)).collect();
                let defs: Vec<u16> = (0..rng.below(2)).map(|_| variable(rng)).collect();
                let mut random = instruction(&uses, &defs);
                if rng.one_in(4) {
                    random = df_throwing(random);
                }
                cfg.block_mut(block).push(random);
            }
            for _ in 0..rng.below(3) {
                let kind = if rng.one_in(4) {
                    EdgeKind::ExceptionUnwind
                } else {
                    EdgeKind::Fallthrough
                };
                cfg.add_edge(block, blocks[rng.below(count)], kind);
            }
        }
        cfg
    }

    #[test]
    fn a_wide_join_keeps_only_the_phis_its_reads_need() {
        // Every arm writes the merged result 0, a temporary 1 it reads at
        // once, and a value 2 that its own tail reads. Only 0 is read after
        // the join.
        let mut cfg = Cfg::<DfInst>::new();
        let join = cfg.new_block();
        for _ in 0..8 {
            let arm = cfg.new_block();
            let tail = cfg.new_block();
            cfg.add_edge(cfg.entry(), arm, EdgeKind::Fallthrough);
            cfg.add_edge(arm, tail, EdgeKind::Fallthrough);
            cfg.add_edge(tail, join, EdgeKind::Fallthrough);
            cfg.block_mut(arm).instructions_mut().extend([
                df_def("temporary", 1),
                df_use("read temporary", 1),
                df_def("result", 0),
                df_def("arm value", 2),
            ]);
            cfg.block_mut(tail).push(df_use("read arm value", 2));
        }
        cfg.block_mut(join).push(df_use("read result", 0));

        assert_eq!(counts(&cfg), [[3, 24], [2, 16], [1, 8]]);
        let pruned = form(&cfg, SsaPruning::Pruned);
        let [(_, phi)] = pruned.phis().collect::<Vec<_>>()[..] else {
            panic!("the join merges the result alone");
        };
        assert_eq!(phi.result.variable, 0);
        assert_eq!(pruned.block(join).instructions[0].uses()[0], phi.result);
    }

    #[test]
    fn a_variable_written_before_every_read_needs_no_phi() {
        let mut cfg = Cfg::<DfInst>::new();
        let body = cfg.new_block();
        let exit = cfg.new_block();
        cfg.add_edge(cfg.entry(), body, EdgeKind::Fallthrough);
        cfg.add_edge(body, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(body, exit, EdgeKind::ConditionalFalse);
        cfg.block_mut(body)
            .instructions_mut()
            .extend([df_def("scratch", 0), df_use("read scratch", 0)]);

        assert_eq!(counts(&cfg), [[1, 2], [0, 0], [0, 0]]);
        assert_eq!(form(&cfg, SsaPruning::SemiPruned).max_version(&0), 1);
    }

    #[test]
    fn a_handler_read_keeps_the_phi_of_the_value_before_the_throw() {
        for (handler_reads, expected) in [
            (true, [[1, 1], [1, 1], [1, 1]]),
            (false, [[1, 1], [0, 0], [0, 0]]),
        ] {
            let (mut cfg, [_, _, handler]) = protected_region([
                df_def("before call", 0),
                df_throwing(df_ff("call")),
                df_def("after call", 0),
            ]);
            if handler_reads {
                cfg.block_mut(handler).push(df_use("catch", 0));
            }
            assert_eq!(counts(&cfg), expected, "handler reads: {handler_reads}");
        }
    }

    #[test]
    fn an_unwind_keeps_a_variable_live_that_its_block_writes_after_the_throw() {
        // The protected block is a loop header that writes 0 only after its
        // throwing call, so the handler reads the value entering the header.
        // Writing 0 ahead of the call instead makes it dead there.
        for (write_first, phi_at_header) in [(false, true), (true, false)] {
            let mut cfg = Cfg::<DfInst>::new();
            let header = cfg.new_block();
            let latch = cfg.new_block();
            let handler = cfg.new_block();
            cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
            cfg.add_edge(header, latch, EdgeKind::Fallthrough);
            cfg.add_edge(header, handler, EdgeKind::ExceptionUnwind);
            cfg.add_edge(latch, header, EdgeKind::Fallthrough);
            cfg.block_mut(cfg.entry()).push(df_def("initial", 0));
            let call = df_throwing(df_ff("call"));
            let write = df_def("write", 0);
            cfg.block_mut(header)
                .instructions_mut()
                .extend(if write_first {
                    [write, call]
                } else {
                    [call, write]
                });
            cfg.block_mut(handler).push(df_use("catch", 0));

            let pruned = form(&cfg, SsaPruning::Pruned);
            assert_eq!(
                pruned.block(header).phis.len(),
                usize::from(phi_at_header),
                "write first: {write_first}"
            );
            let minimal = form(&cfg, SsaPruning::Minimal);
            let catch = ProgramPoint {
                block: handler,
                inst_idx: 0,
            };
            let read = |form: &SsaForm<u16>| {
                let catch = form
                    .instruction(catch)
                    .expect("the handler's read is annotated");
                origin(&origins(form), &catch.uses()[0])
            };
            assert_eq!(read(&pruned), read(&minimal), "write first: {write_first}");
        }
    }

    #[test]
    fn pruned_forms_keep_every_read_and_the_phis_liveness_asks_for() {
        for seed in 0..400 {
            let cfg = random_cfg(&mut Rng::new(seed));
            let dominators = DominatorTree::compute(&cfg);
            let minimal = SsaForm::compute(&cfg, &dominators);
            let minimal_origins = origins(&minimal);
            let liveness = Liveness::compute(&cfg);
            let exposed = upward_exposed(&cfg);
            for pruning in PRUNED_FORMS {
                let pruned = SsaForm::compute_with_pruning(&cfg, &dominators, pruning);
                let expected: Vec<_> = phis(&minimal)
                    .into_iter()
                    .filter(|(block, variable, _)| match pruning {
                        SsaPruning::SemiPruned => exposed.contains(variable),
                        _ => liveness.is_live_in(variable, *block),
                    })
                    .collect();
                assert_eq!(phis(&pruned), expected, "seed {seed}, {pruning:?}");

                let pruned_origins = origins(&pruned);
                for block in cfg.block_ids() {
                    let live = liveness.live_before_instructions(&cfg, block);
                    let instructions = pruned.block(block).instructions.iter();
                    for (index, instruction) in instructions.enumerate() {
                        let reads = |form: &SsaForm<u16>, origins| {
                            form.instruction(instruction.point)
                                .expect("both forms annotate every instruction")
                                .uses()
                                .iter()
                                .map(|value| origin(origins, value))
                                .collect::<Vec<_>>()
                        };
                        assert_eq!(
                            reads(&pruned, &pruned_origins),
                            reads(&minimal, &minimal_origins),
                            "seed {seed}, {pruning:?}, {:?}",
                            instruction.point
                        );
                        for variable in &live[index] {
                            let point = instruction.point;
                            assert_eq!(
                                origin(&pruned_origins, &pruned.value_at(point, variable)),
                                origin(&minimal_origins, &minimal.value_at(point, variable)),
                                "seed {seed}, {pruning:?}, {point:?}, variable {variable}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn one_scratch_builds_every_form_of_every_procedure() {
        let mut scratch = SsaScratch::new();
        for seed in 0..100 {
            let cfg = random_cfg(&mut Rng::new(seed));
            let dominators = DominatorTree::compute(&cfg);
            for pruning in [
                SsaPruning::Pruned,
                SsaPruning::Minimal,
                SsaPruning::SemiPruned,
            ] {
                let reused =
                    SsaForm::compute_with_pruning_in(&mut scratch, &cfg, &dominators, pruning);
                assert_eq!(
                    reused,
                    SsaForm::compute_with_pruning(&cfg, &dominators, pruning),
                    "seed {seed}, {pruning:?}"
                );
            }
        }
    }

    #[test]
    fn placement_alone_places_what_the_form_renames() {
        for seed in 0..100 {
            let cfg = random_cfg(&mut Rng::new(seed));
            // Placement alone does not complete a dominator forest, so only
            // a CFG whose every block is reachable is compared.
            let dominators = DominatorTree::compute(&cfg);
            if cfg.block_ids().any(|block| !dominators.is_reachable(block)) {
                continue;
            }
            for pruning in PRUNED_FORMS {
                let placements = PhiPlacements::compute_with_pruning(&cfg, &dominators, pruning);
                let placed: Vec<_> = placements
                    .iter()
                    .map(|(block, phi)| (block, phi.variable, phi.predecessors().to_vec()))
                    .collect();
                let renamed: Vec<_> = SsaForm::compute_with_pruning(&cfg, &dominators, pruning)
                    .phis()
                    .map(|(block, phi)| {
                        let points = phi.operands().iter().map(|(point, _)| *point).collect();
                        (block, phi.result.variable, points)
                    })
                    .collect();
                assert_eq!(placed, renamed, "seed {seed}, {pruning:?}");
            }
        }
    }
}
