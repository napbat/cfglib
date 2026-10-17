//! Two interpreters for the toy instruction set: one over the plain CFG and
//! one over its `SsaForm`. The pipeline oracle is that every program, before
//! and after every transformation, produces the same observable behavior on
//! every input vector.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;

use cfglib::{BlockId, Cfg, EdgeKind, ProgramPoint, SsaForm, SsaValue};

use crate::toy::{Inst, Op, Var, compute};

/// Instructions an execution may run before the test calls it divergent.
const FUEL: usize = 50_000;

/// How an execution ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Termination {
    /// A `Return` instruction ran.
    Returned(i64),
    /// Control left a block that has no successor and no `Return`.
    FellOffEnd,
    /// A throwing instruction ran with no unwind edge to take.
    Trapped,
}

/// Everything an outside observer sees of one execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    /// The values `Print` emitted, in order.
    pub output: Vec<i64>,
    /// How the execution ended.
    pub termination: Termination,
}

/// The variables an execution starts with.
pub type Inputs = BTreeMap<Var, i64>;

/// What one execution visited, for the analyses' soundness checks.
#[derive(Debug, Clone)]
pub struct Trace<V> {
    /// Every block entered.
    pub blocks: BTreeSet<BlockId>,
    /// Every edge taken, by endpoints.
    pub edges: BTreeSet<(BlockId, BlockId)>,
    /// The environment at each block entry, in execution order.
    pub entry_states: Vec<(BlockId, BTreeMap<V, i64>)>,
    /// The environment at each block exit, in execution order.
    pub exit_states: Vec<(BlockId, BTreeMap<V, i64>)>,
}

impl<V> Default for Trace<V> {
    fn default() -> Self {
        Self {
            blocks: BTreeSet::new(),
            edges: BTreeSet::new(),
            entry_states: Vec::new(),
            exit_states: Vec::new(),
        }
    }
}

/// What one instruction does to the machine, given its argument values.
enum Step {
    Define(Vec<i64>),
    Output(i64),
    Branch(bool),
    Return(i64),
    Trap,
    Continue,
}

fn step<V>(inst: &Inst<V>, arguments: &[i64]) -> Step {
    match inst.op {
        Op::ParallelCopy => Step::Define(arguments.to_vec()),
        Op::Print => Step::Output(arguments[0]),
        Op::Branch => Step::Branch(arguments[0] != 0),
        Op::Return => Step::Return(arguments[0]),
        Op::Marker => Step::Continue,
        op => compute(op, arguments).map_or(Step::Trap, |value| Step::Define(vec![value])),
    }
}

/// The successor a block's terminator selects, or the departure that ends
/// the execution.
enum Departure {
    Edge(BlockId, BlockId),
    End(Termination),
}

fn normal_successor<V>(
    cfg: &Cfg<Inst<V>>,
    block: BlockId,
    decision: Option<bool>,
) -> Option<BlockId> {
    let normal: Vec<_> = cfg
        .outgoing(block)
        .map(|edge| cfg.edge(edge))
        .filter(|edge| edge.kind() != EdgeKind::ExceptionUnwind)
        .collect();
    if let Some(taken) = decision {
        let wanted = if taken {
            EdgeKind::ConditionalTrue
        } else {
            EdgeKind::ConditionalFalse
        };
        let edge = normal
            .iter()
            .find(|edge| edge.kind() == wanted)
            .unwrap_or_else(|| panic!("{block:?} branches {taken} but has no {wanted:?} edge"));
        return Some(edge.target());
    }
    match normal.as_slice() {
        [] => None,
        [only] => Some(only.target()),
        _ => panic!("{block:?} has several successors and no branch"),
    }
}

fn unwind_target<V>(cfg: &Cfg<Inst<V>>, block: BlockId) -> Option<BlockId> {
    cfg.outgoing(block)
        .map(|edge| cfg.edge(edge))
        .find(|edge| edge.kind() == EdgeKind::ExceptionUnwind)
        .map(|edge| edge.target())
}

fn read<V: Ord + Debug>(environment: &BTreeMap<V, i64>, variable: &V, inst: &Inst<V>) -> i64 {
    *environment
        .get(variable)
        .unwrap_or_else(|| panic!("{inst:?} reads {variable:?} before any definition"))
}

/// Run `cfg` from its entry over `inputs`.
///
/// # Panics
///
/// Panics when the program reads an undefined variable, branches without the
/// matching edge, or exhausts the fuel: all of them are bugs in the program
/// or in a transformation, never an observable outcome.
pub fn run<V: Clone + Ord + Debug>(
    cfg: &Cfg<Inst<V>>,
    inputs: &BTreeMap<V, i64>,
) -> (Outcome, Trace<V>) {
    let mut environment = inputs.clone();
    let mut output = Vec::new();
    let mut trace = Trace::default();
    let mut block = cfg.entry();
    let mut fuel = FUEL;
    loop {
        trace.blocks.insert(block);
        trace.entry_states.push((block, environment.clone()));
        let mut decision = None;
        let mut departure = None;
        for inst in cfg.block(block).instructions() {
            fuel = fuel.checked_sub(1).expect("the program terminates");
            let arguments: Vec<i64> = inst
                .uses
                .iter()
                .map(|used| read(&environment, used, inst))
                .collect();
            match step(inst, &arguments) {
                Step::Define(values) => environment.extend(inst.defs.iter().cloned().zip(values)),
                Step::Output(value) => output.push(value),
                Step::Branch(taken) => decision = Some(taken),
                Step::Return(value) => {
                    departure = Some(Departure::End(Termination::Returned(value)));
                    break;
                }
                Step::Trap => {
                    departure = Some(
                        unwind_target(cfg, block)
                            .map_or(Departure::End(Termination::Trapped), |handler| {
                                Departure::Edge(block, handler)
                            }),
                    );
                    break;
                }
                Step::Continue => {}
            }
        }
        let departure = departure.unwrap_or_else(|| {
            normal_successor(cfg, block, decision)
                .map_or(Departure::End(Termination::FellOffEnd), |next| {
                    Departure::Edge(block, next)
                })
        });
        trace.exit_states.push((block, environment.clone()));
        match departure {
            Departure::End(termination) => {
                return (
                    Outcome {
                        output,
                        termination,
                    },
                    trace,
                );
            }
            Departure::Edge(source, target) => {
                trace.edges.insert((source, target));
                block = target;
            }
        }
    }
}

/// What an SSA execution observed: every value each SSA name ever held.
pub type ObservedValues = BTreeMap<SsaValue<Var>, BTreeSet<i64>>;

/// Record that `name` now holds `value`, and that it held `value` at some
/// point of the execution.
fn define(
    environment: &mut BTreeMap<SsaValue<Var>, i64>,
    observed: &mut ObservedValues,
    name: SsaValue<Var>,
    value: i64,
) {
    observed.entry(name.clone()).or_default().insert(value);
    environment.insert(name, value);
}

/// The value each phi of `block` takes when control arrives from `departure`,
/// or, when `departure` is `None`, when the function starts at `block`. Each
/// phi reads its input from `environment` before any phi result is written,
/// and one whose input holds no value yet takes none.
///
/// Panics when a phi has no operand for `departure`, or no start value when
/// the function starts.
fn phi_arrivals(
    ssa: &SsaForm<Var>,
    block: BlockId,
    departure: Option<ProgramPoint>,
    environment: &BTreeMap<SsaValue<Var>, i64>,
) -> Vec<(SsaValue<Var>, i64)> {
    ssa.block(block)
        .phis
        .iter()
        .filter_map(|phi| {
            let input = match departure {
                Some(departure) => phi
                    .operands()
                    .iter()
                    .find(|(point, _)| *point == departure)
                    .unwrap_or_else(|| panic!("a phi of {block:?} has no operand for {departure}"))
                    .1
                    .clone(),
                None => phi
                    .start_value()
                    .unwrap_or_else(|| panic!("a phi of {block:?} has no start value")),
            };
            environment
                .get(&input)
                .map(|value| (phi.result.clone(), *value))
        })
        .collect()
}

/// Run `cfg` through its `SsaForm`: instructions read and write renamed
/// values, and a phi picks the operand of the departure point control came
/// from. Returns what the program did and every value each SSA name held.
///
/// # Panics
///
/// Panics like [`run`], and when an annotation does not belong to the
/// instruction it sits on.
pub fn run_ssa(
    cfg: &Cfg<Inst<Var>>,
    ssa: &SsaForm<Var>,
    inputs: &Inputs,
) -> (Outcome, ObservedValues) {
    let mut observed = ObservedValues::new();
    let mut environment: BTreeMap<SsaValue<Var>, i64> = BTreeMap::new();
    for (variable, value) in inputs {
        define(
            &mut environment,
            &mut observed,
            SsaValue::live_in(*variable),
            *value,
        );
    }
    let mut output = Vec::new();
    let mut block = cfg.entry();
    let mut arrived_from: Option<ProgramPoint> = None;
    let mut fuel = FUEL;
    loop {
        // Every phi reads its input before any phi result is written. The
        // first block runs with no departure: the function has just started.
        for (name, value) in phi_arrivals(ssa, block, arrived_from, &environment) {
            define(&mut environment, &mut observed, name, value);
        }
        let instructions = cfg.block(block).instructions();
        let annotations = &ssa.block(block).instructions;
        assert_eq!(instructions.len(), annotations.len());
        let mut decision = None;
        let mut next: Option<(BlockId, ProgramPoint)> = None;
        let mut ended = None;
        for (index, (inst, annotation)) in instructions.iter().zip(annotations).enumerate() {
            fuel = fuel.checked_sub(1).expect("the program terminates");
            let point = ProgramPoint {
                block,
                inst_idx: index,
            };
            assert_eq!(annotation.point, point);
            let arguments: Vec<i64> = annotation
                .uses()
                .iter()
                .map(|used| {
                    *environment
                        .get(used)
                        .unwrap_or_else(|| panic!("{inst:?} reads undefined {used:?}"))
                })
                .collect();
            match step(inst, &arguments) {
                Step::Define(values) => {
                    for (name, value) in annotation.defs().iter().zip(values) {
                        define(&mut environment, &mut observed, name.clone(), value);
                    }
                }
                Step::Output(value) => output.push(value),
                Step::Branch(taken) => decision = Some(taken),
                Step::Return(value) => {
                    ended = Some(Termination::Returned(value));
                    break;
                }
                Step::Trap => {
                    match unwind_target(cfg, block) {
                        Some(handler) => next = Some((handler, point)),
                        None => ended = Some(Termination::Trapped),
                    }
                    break;
                }
                Step::Continue => {}
            }
        }
        if next.is_none() && ended.is_none() {
            let end = ProgramPoint {
                block,
                inst_idx: instructions.len(),
            };
            match normal_successor(cfg, block, decision) {
                Some(target) => next = Some((target, end)),
                None => ended = Some(Termination::FellOffEnd),
            }
        }
        if let Some(termination) = ended {
            return (
                Outcome {
                    output,
                    termination,
                },
                observed,
            );
        }
        let (target, departure) = next.expect("an execution that continues has a successor");
        block = target;
        arrived_from = Some(departure);
    }
}

/// Every input vector over `names`: each variable takes each of a handful of
/// small values, negative, zero, and positive.
pub fn input_vectors(names: &[&'static str]) -> Vec<Inputs> {
    const VALUES: [i64; 5] = [-2, 0, 1, 3, 5];
    let mut vectors = vec![Inputs::new()];
    for name in names {
        vectors = vectors
            .into_iter()
            .flat_map(|partial| {
                VALUES.iter().map(move |value| {
                    let mut extended = partial.clone();
                    extended.insert(Var(name), *value);
                    extended
                })
            })
            .collect();
    }
    vectors
}
