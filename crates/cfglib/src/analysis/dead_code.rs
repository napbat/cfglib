//! Dead-code analysis — what is dead, reported without removing it.
//!
//! [`DeadCode::compute`] is the single source of truth for deadness:
//! [`dead_code_elimination`](crate::dead_code_elimination) applies its
//! instruction list, and [`remove_dead_code`](crate::remove_dead_code)
//! additionally consumes the structure the removals leave behind. Computing
//! the analysis alone serves the reporting shapes — dead-code diagnostics,
//! IDE greying, metrics — where knowing what is dead matters and deleting it
//! is someone else's decision.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::dataflow::cfg_view::CfgView;
use crate::dataflow::liveness::step_backward;
use crate::dataflow::{EffectInfo, InstrInfo, ProgramPoint, VariableId, departs_before_throws};
use crate::graph::traverse::{TraversalDirection, reachable};

/// Everything dead in a CFG, found without mutating it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeadCode {
    /// Instructions whose definitions no needed instruction reads, in block
    /// order and ascending instruction order within a block.
    ///
    /// An instruction is needed when it has declared side effects, when it
    /// [`may_unwind`](crate::InstrInfo::may_unwind), when the caller observes
    /// it, when it defines nothing, or when it defines something a needed
    /// instruction reads; such an instruction is never listed. The reads of a
    /// dead instruction keep nothing alive, across blocks and around loops
    /// as well as within a block, so a chain feeding only a dead instruction
    /// is listed with it by one analysis. A throwing instruction keeps what
    /// its unwind successors read live before it. An unreachable block is
    /// judged like any other: its definition is listed only if no needed
    /// instruction reads it, and a read by a reachable block keeps it.
    ///
    /// [`ProgramPoint`] indices are positions, not identities: any
    /// instruction edit invalidates the analysis.
    pub instructions: Vec<ProgramPoint>,
    /// Blocks with no path from the entry, in allocation order.
    ///
    /// Reported separately from [`instructions`](Self::instructions) because
    /// they are dead for a different reason: no path executes them, side
    /// effects notwithstanding.
    pub unreachable_blocks: Vec<BlockId>,
}

impl DeadCode {
    /// Analyze `cfg` for dead instructions and unreachable blocks.
    ///
    /// Nothing leaves the graph: a definition survives only because
    /// something inside reads it. [`compute_with_exits`](Self::compute_with_exits)
    /// is the same analysis told what leaves.
    ///
    /// Requiring [`EffectInfo`] is deliberate, exactly as in
    /// [`dead_code_elimination`](crate::dead_code_elimination): a consumer
    /// must state an effect vocabulary before anything is called dead, which
    /// prevents silently classifying side-effecting statements as removable.
    #[must_use]
    pub fn compute<G>(cfg: &G) -> Self
    where
        G: CfgView,
        G::Inst: EffectInfo,
    {
        Self::compute_with_exits(cfg, |_| Vec::new())
    }

    /// Analyze `cfg` with `live_out` stating what leaves each block.
    ///
    /// A function's own graph never says that a value outlives it — what a
    /// return hands back, the storage a caller reads again, the state
    /// something reached through a non-returning exit observes — so the
    /// definitions that produce those values read as dead. `live_out`
    /// states them: its variables are live at the end of the named block,
    /// and everything their definitions transitively read stays live with
    /// them. A caller normally answers for the blocks with no successors
    /// and hands back an empty vector everywhere else.
    #[must_use]
    pub fn compute_with_exits<G>(
        cfg: &G,
        live_out: impl Fn(BlockId) -> Vec<<G::Inst as InstrInfo>::Variable>,
    ) -> Self
    where
        G: CfgView,
        G::Inst: EffectInfo,
    {
        Self::compute_with_roots(cfg, live_out, |_, _| false)
    }

    /// Analyzes `cfg` with exit values and additional observed instructions.
    ///
    /// `observed` retains an instruction and the definitions that supply its
    /// inputs. It does not declare a side effect or alter the graph. This is
    /// useful when a consumer observes a definition through metadata rather
    /// than an explicit variable read.
    #[must_use]
    pub fn compute_with_roots<G>(
        cfg: &G,
        live_out: impl Fn(BlockId) -> Vec<<G::Inst as InstrInfo>::Variable>,
        observed: impl Fn(ProgramPoint, &G::Inst) -> bool,
    ) -> Self
    where
        G: CfgView,
        G::Inst: EffectInfo,
    {
        let bound = cfg.node_bound();
        let mut seeds = vec![Vec::new(); bound];
        let mut forced = vec![Vec::new(); bound];
        for block in cfg.node_ids() {
            seeds[block.index()] = live_out(block);
            forced[block.index()] = cfg
                .instructions(block)
                .iter()
                .enumerate()
                .map(|(inst_idx, inst)| {
                    !inst.effects().is_empty()
                        || inst.may_unwind()
                        || observed(ProgramPoint { block, inst_idx }, inst)
                })
                .collect();
        }
        let live_in = needed_live_in(cfg, &seeds, &forced);

        let mut instructions = Vec::new();
        for block in cfg.node_ids() {
            let index = block.index();
            let (mut live, unwind) = exits_of(cfg, block, &live_in, &seeds[index]);
            let mut dead = Vec::new();
            replay(cfg, block, &forced[index], &mut live, &unwind, |inst_idx| {
                dead.push(inst_idx);
            });
            instructions.extend(
                dead.into_iter()
                    .rev()
                    .map(|inst_idx| ProgramPoint { block, inst_idx }),
            );
        }

        let reached = reachable(cfg, [cfg.root()], TraversalDirection::Outgoing);
        let unreachable_blocks = cfg
            .node_ids()
            .filter(|block| !reached[block.index()])
            .collect();

        Self {
            instructions,
            unreachable_blocks,
        }
    }

    /// Whether nothing dead was found.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty() && self.unreachable_blocks.is_empty()
    }
}

/// What is live at the end of `block` once only needed instructions read
/// anything: the seed joined with the live-in sets of the normal successors,
/// and, separately, what an unwind out of the block reads.
fn exits_of<G, V>(
    cfg: &G,
    block: BlockId,
    live_in: &[BTreeSet<V>],
    seed: &[V],
) -> (BTreeSet<V>, BTreeSet<V>)
where
    G: CfgView,
    G::Inst: InstrInfo<Variable = V>,
    V: VariableId,
{
    let mut out: BTreeSet<V> = seed.iter().cloned().collect();
    let mut unwind = BTreeSet::new();
    for edge in cfg.outgoing(block) {
        let edge = cfg.edge(edge);
        let reads = live_in[edge.target().index()].iter().cloned();
        if departs_before_throws(cfg, edge) {
            unwind.extend(reads);
        } else {
            out.extend(reads);
        }
    }
    (out, unwind)
}

/// Replay `block` backward from `live`, the variables live at its end.
///
/// An instruction is needed when `forced` says so (an effect, an unwind, an
/// observer), when it defines nothing, or when it defines something live; it
/// then takes the ordinary liveness step. Any other instruction is dead and
/// is reported to `dead`, last instruction first. It takes no step, so its
/// reads keep nothing alive: the chain that feeds only it dies with it.
fn replay<G, V>(
    cfg: &G,
    block: BlockId,
    forced: &[bool],
    live: &mut BTreeSet<V>,
    unwind: &BTreeSet<V>,
    mut dead: impl FnMut(usize),
) where
    G: CfgView,
    G::Inst: InstrInfo<Variable = V>,
    V: VariableId,
{
    for (index, inst) in cfg.instructions(block).iter().enumerate().rev() {
        let needed = forced[index]
            || inst.defs().is_empty()
            || inst.defs().iter().any(|def| live.contains(def));
        if needed {
            step_backward(live, inst, unwind);
        } else {
            dead(index);
        }
    }
}

/// The least fixpoint of liveness in which only needed instructions read.
///
/// Ordinary liveness counts the reads of a dead instruction, so a chain whose
/// only reader is dead looks live across a block boundary, and a loop-carried
/// update nothing else reads keeps itself live around the back edge. Here the
/// live-in sets start empty and grow only through needed instructions, so
/// neither does. `forced` states, per block and instruction, which
/// instructions are needed whatever is live.
fn needed_live_in<G, V>(cfg: &G, seeds: &[Vec<V>], forced: &[Vec<bool>]) -> Vec<BTreeSet<V>>
where
    G: CfgView,
    G::Inst: InstrInfo<Variable = V>,
    V: VariableId,
{
    let bound = cfg.node_bound();
    let mut live_in: Vec<BTreeSet<V>> = vec![BTreeSet::new(); bound];
    let mut queued = vec![false; bound];
    let mut worklist: Vec<BlockId> = cfg.node_ids().collect();
    for block in &worklist {
        queued[block.index()] = true;
    }
    while let Some(block) = worklist.pop() {
        let index = block.index();
        queued[index] = false;
        let (mut live, unwind) = exits_of(cfg, block, &live_in, &seeds[index]);
        replay(cfg, block, &forced[index], &mut live, &unwind, |_| {});
        // The transfer is monotone, so this union only ever adds what the
        // replay found; it also guarantees the loop ends.
        let before = live_in[index].len();
        live_in[index].extend(live);
        if live_in[index].len() != before {
            for predecessor in cfg.predecessors(block) {
                if !queued[predecessor.index()] {
                    queued[predecessor.index()] = true;
                    worklist.push(predecessor);
                }
            }
        }
    }
    live_in
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::flow::FlowEffect;
    use crate::test_util::{DfInst, TestEffect, df_def, df_impure, df_op, df_use, df_with_effect};

    #[test]
    fn a_dead_definition_is_reported_and_a_live_one_is_not() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_def("dead", 0));
        cfg.block_mut(cfg.entry()).push(df_def("live", 1));
        cfg.block_mut(exit).push(df_use("use", 1));
        cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);

        let dead = DeadCode::compute(&cfg);
        assert_eq!(
            dead.instructions,
            alloc::vec![ProgramPoint {
                block: cfg.entry(),
                inst_idx: 0,
            }]
        );
        assert_eq!(dead.unreachable_blocks, [] as [crate::BlockId; 0]);
    }

    #[test]
    fn a_chain_feeding_only_a_dead_instruction_dies_with_it() {
        // a = ...; b = a; nothing uses b — both are dead in one analysis.
        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).push(df_def("a", 0));
        cfg.block_mut(cfg.entry()).push({
            let mut inst = df_def("b", 1);
            inst.uses.push(0);
            inst
        });

        let dead = DeadCode::compute(&cfg);
        let indices: Vec<usize> = dead.instructions.iter().map(|p| p.inst_idx).collect();
        assert_eq!(indices, alloc::vec![0, 1]);
    }

    #[test]
    fn a_side_effecting_instruction_is_never_reported() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let mut store = df_impure("volatile", TestEffect::MemoryWrite);
        store.defs.push(0);
        cfg.block_mut(cfg.entry()).push(store);

        let dead = DeadCode::compute(&cfg);
        assert!(dead.is_empty());
    }

    #[test]
    fn an_unreachable_block_is_reported_separately() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let orphan = cfg.new_block();
        cfg.block_mut(orphan).push(df_def("stranded", 0));

        let dead = DeadCode::compute(&cfg);
        assert_eq!(dead.unreachable_blocks, alloc::vec![orphan]);
        // Its non-effectful definition is liveness-dead as well.
        assert_eq!(
            dead.instructions,
            alloc::vec![ProgramPoint {
                block: orphan,
                inst_idx: 0,
            }]
        );
    }

    #[test]
    fn a_definition_in_an_unreachable_block_stays_live_while_a_reachable_block_reads_it() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let orphan = cfg.new_block();
        let reader = cfg.new_block();
        cfg.block_mut(orphan).push(df_def("stranded", 0));
        cfg.block_mut(reader).push(df_use("read", 0));
        cfg.add_edge(cfg.entry(), reader, EdgeKind::Fallthrough);
        cfg.add_edge(orphan, reader, EdgeKind::Fallthrough);

        let dead = DeadCode::compute(&cfg);

        assert_eq!(dead.unreachable_blocks, alloc::vec![orphan]);
        assert_eq!(dead.instructions, []);
    }

    /// A definition nothing inside the graph reads is dead, with the chain
    /// feeding it, and the same definitions are live once an exit says the
    /// value leaves.
    #[test]
    fn a_definition_the_exit_seed_observes_is_not_dead() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_def("source", 0));
        cfg.block_mut(exit).push({
            let mut inst = df_def("leaves", 1);
            inst.uses.push(0);
            inst
        });
        cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);

        let unseeded = DeadCode::compute(&cfg);
        assert_eq!(
            unseeded.instructions,
            alloc::vec![
                ProgramPoint {
                    block: cfg.entry(),
                    inst_idx: 0,
                },
                ProgramPoint {
                    block: exit,
                    inst_idx: 0,
                },
            ],
            "nothing inside the graph reads what the exit produces, so its source dies with it"
        );

        let seeded = DeadCode::compute_with_exits(&cfg, |block| {
            if block == exit {
                alloc::vec![1]
            } else {
                Vec::new()
            }
        });
        assert!(
            seeded.instructions.is_empty(),
            "the seeded definition and its source both survive: {:?}",
            seeded.instructions
        );
    }

    /// A seed names one variable, not the whole exit: an unrelated
    /// definition in the same block stays dead.
    #[test]
    fn an_exit_seed_keeps_only_what_it_names() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).push(df_def("kept", 0));
        cfg.block_mut(cfg.entry()).push(df_def("dropped", 1));

        let dead = DeadCode::compute_with_exits(&cfg, |_| alloc::vec![0]);
        assert_eq!(
            dead.instructions,
            alloc::vec![ProgramPoint {
                block: cfg.entry(),
                inst_idx: 1,
            }]
        );
    }

    #[test]
    fn an_observed_root_keeps_its_dependencies_not_later_overwrites() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let block = cfg.new_block();
        cfg.add_edge(cfg.entry(), block, EdgeKind::Fallthrough);
        cfg.block_mut(cfg.entry()).push(df_def("input", 0));
        let mut derived = df_def("derived", 1);
        derived.uses.push(0);
        cfg.block_mut(block).push(derived);
        let mut observed = df_def("observed", 2);
        observed.uses.push(1);
        cfg.block_mut(block).push(observed);
        cfg.block_mut(block).push(df_def("later", 2));
        cfg.block_mut(block).push(df_def("unrelated", 3));

        let dead = DeadCode::compute_with_roots(
            &cfg,
            |_| Vec::new(),
            |point, _| point == (ProgramPoint { block, inst_idx: 1 }),
        );
        assert_eq!(
            dead.instructions,
            alloc::vec![
                ProgramPoint { block, inst_idx: 2 },
                ProgramPoint { block, inst_idx: 3 },
            ]
        );
    }

    #[test]
    fn a_chain_that_spans_blocks_is_dead_in_one_analysis() {
        // bb0: r10 = r0 + r2; bb1: r11 = r10 + r10; bb2: r12 = r11 + r11;
        // print r0. Nothing reads r12, so all three additions are dead.
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let second = cfg.new_block();
        let third = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .push(df_op("a", "add", 10, &[0, 2]));
        cfg.block_mut(second).push(df_op("b", "add", 11, &[10, 10]));
        cfg.block_mut(third).push(df_op("c", "add", 12, &[11, 11]));
        cfg.block_mut(third).push(df_use("print", 0));
        cfg.add_edge(cfg.entry(), second, EdgeKind::Fallthrough);
        cfg.add_edge(second, third, EdgeKind::Fallthrough);

        let dead = DeadCode::compute(&cfg);

        assert_eq!(
            dead.instructions,
            alloc::vec![
                ProgramPoint {
                    block: cfg.entry(),
                    inst_idx: 0
                },
                ProgramPoint {
                    block: second,
                    inst_idx: 0
                },
                ProgramPoint {
                    block: third,
                    inst_idx: 0
                },
            ]
        );
    }

    #[test]
    fn an_update_only_itself_reads_around_a_loop_is_dead() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let header = cfg.new_block();
        let body = cfg.new_block();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_def("init", 0));
        cfg.block_mut(header).push(df_use("test", 1));
        cfg.block_mut(body).push(df_op("step", "add", 0, &[0]));
        cfg.block_mut(exit).push(df_use("done", 1));
        cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
        cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(body, header, EdgeKind::Back);

        let dead = DeadCode::compute(&cfg);

        assert_eq!(
            dead.instructions,
            alloc::vec![
                ProgramPoint {
                    block: cfg.entry(),
                    inst_idx: 0
                },
                ProgramPoint {
                    block: body,
                    inst_idx: 0
                },
            ],
            "the counter and its seed feed only the update"
        );
    }

    #[test]
    fn a_loop_value_a_needed_instruction_reads_stays_live_around_the_loop() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let header = cfg.new_block();
        let body = cfg.new_block();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_def("init", 0));
        cfg.block_mut(header).push(df_use("test", 0));
        cfg.block_mut(body).push(df_op("step", "add", 0, &[0]));
        cfg.block_mut(exit).push(df_use("done", 0));
        cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
        cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(body, header, EdgeKind::Back);

        assert_eq!(DeadCode::compute(&cfg).instructions, []);
    }

    #[test]
    fn a_clean_cfg_reports_nothing() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).push(df_def("keep", 0));
        cfg.block_mut(cfg.entry()).push(df_use("use", 0));
        assert!(DeadCode::compute(&cfg).is_empty());
    }

    /// `protected: r0 = first; r7 = load (throws); r0 = second`, falling
    /// through to an empty `normal` and unwinding to a handler that reads
    /// r0. Only the write after the throw is dead; the faulting load stays
    /// although nothing reads what it loads.
    #[test]
    fn a_handler_keeps_the_write_before_a_throw_alive() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let protected = cfg.new_block();
        let normal = cfg.new_block();
        let handler = cfg.new_block();
        cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
        cfg.add_edge(protected, normal, EdgeKind::Fallthrough);
        cfg.add_edge(protected, handler, EdgeKind::ExceptionUnwind);
        cfg.block_mut(protected).instructions_mut().extend([
            df_def("first", 0),
            df_with_effect(df_def("load", 7), FlowEffect::MayThrow),
            df_def("second", 0),
        ]);
        cfg.block_mut(handler).push(df_use("catch", 0));

        let dead = DeadCode::compute(&cfg);
        assert_eq!(
            dead.instructions,
            alloc::vec![ProgramPoint {
                block: protected,
                inst_idx: 2,
            }]
        );
    }
}
