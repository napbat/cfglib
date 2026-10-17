//! The toy register machine and its interpreter.

use std::collections::BTreeMap;
use std::fmt::Debug;

use cfglib::{Cfg, EdgeKind, InstrInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Reg(pub &'static str);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    LoadImmediate(i64),
    Add,
    Move,
    SetLess,
    Divide,
    BranchNonZero,
    Output,
    Return,
    ParallelMove,
}

#[derive(Debug, Clone)]
pub struct Insn<R> {
    pub opcode: Opcode,
    pub defs: Vec<R>,
    pub uses: Vec<R>,
}

impl<R> Insn<R> {
    pub fn new(opcode: Opcode, defs: Vec<R>, uses: Vec<R>) -> Self {
        Self { opcode, defs, uses }
    }

    pub fn mov(dst: R, src: R) -> Self {
        Self::new(Opcode::Move, vec![dst], vec![src])
    }

    pub fn parallel(defs: Vec<R>, uses: Vec<R>) -> Self {
        Self::new(Opcode::ParallelMove, defs, uses)
    }
}

impl Insn<Reg> {
    pub fn li(dst: &'static str, value: i64) -> Self {
        Self::new(Opcode::LoadImmediate(value), vec![Reg(dst)], Vec::new())
    }

    pub fn binary(opcode: Opcode, dst: &'static str, a: &'static str, b: &'static str) -> Self {
        Self::new(opcode, vec![Reg(dst)], vec![Reg(a), Reg(b)])
    }

    pub fn add(dst: &'static str, a: &'static str, b: &'static str) -> Self {
        Self::binary(Opcode::Add, dst, a, b)
    }

    pub fn slt(dst: &'static str, a: &'static str, b: &'static str) -> Self {
        Self::binary(Opcode::SetLess, dst, a, b)
    }

    pub fn div(dst: &'static str, a: &'static str, b: &'static str) -> Self {
        Self::binary(Opcode::Divide, dst, a, b)
    }

    pub fn bnez(condition: &'static str) -> Self {
        Self::new(Opcode::BranchNonZero, Vec::new(), vec![Reg(condition)])
    }

    pub fn out(src: &'static str) -> Self {
        Self::new(Opcode::Output, Vec::new(), vec![Reg(src)])
    }

    pub fn ret(src: &'static str) -> Self {
        Self::new(Opcode::Return, Vec::new(), vec![Reg(src)])
    }
}

impl<R: Clone + Ord> InstrInfo for Insn<R> {
    type Variable = R;

    fn uses(&self) -> &[R] {
        &self.uses
    }

    fn defs(&self) -> &[R] {
        &self.defs
    }

    fn may_unwind(&self) -> bool {
        self.opcode == Opcode::Divide
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum End {
    Returned(i64),
    Trapped,
    FellOff,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Behavior {
    pub output: Vec<i64>,
    pub end: End,
}

/// Run `cfg` from its entry; `inputs` are the registers it starts with.
///
/// # Panics
///
/// Panics when the program does not terminate within the step budget.
pub fn execute<R: Clone + Ord + Debug>(cfg: &Cfg<Insn<R>>, inputs: &BTreeMap<R, i64>) -> Behavior {
    try_execute(cfg, inputs).expect("the program does not terminate")
}

/// Run `cfg` from its entry, or `None` when it has not ended within the step
/// budget.
pub fn try_execute<R: Clone + Ord + Debug>(
    cfg: &Cfg<Insn<R>>,
    inputs: &BTreeMap<R, i64>,
) -> Option<Behavior> {
    let mut registers = inputs.clone();
    let mut output = Vec::new();
    let mut block = cfg.entry();
    for _ in 0..20_000 {
        let mut branch = None;
        let mut trapped = false;
        for insn in cfg.block(block).instructions() {
            let args: Vec<i64> = insn
                .uses
                .iter()
                .map(|used| {
                    *registers
                        .get(used)
                        .unwrap_or_else(|| panic!("{insn:?} reads undefined {used:?}"))
                })
                .collect();
            match (insn.opcode, args.as_slice()) {
                (Opcode::LoadImmediate(value), []) => {
                    registers.insert(insn.defs[0].clone(), value);
                }
                (Opcode::Add, [a, b]) => {
                    registers.insert(insn.defs[0].clone(), a.wrapping_add(*b));
                }
                (Opcode::SetLess, [a, b]) => {
                    registers.insert(insn.defs[0].clone(), i64::from(a < b));
                }
                (Opcode::Divide, [_, 0]) => {
                    trapped = true;
                    break;
                }
                (Opcode::Divide, [a, b]) => {
                    registers.insert(insn.defs[0].clone(), a.wrapping_div(*b));
                }
                (Opcode::Move, [value]) => {
                    registers.insert(insn.defs[0].clone(), *value);
                }
                (Opcode::ParallelMove, values) => {
                    registers.extend(insn.defs.iter().cloned().zip(values.iter().copied()));
                }
                (Opcode::BranchNonZero, [condition]) => branch = Some(*condition != 0),
                (Opcode::Output, [value]) => output.push(*value),
                (Opcode::Return, [value]) => {
                    return Some(Behavior {
                        output,
                        end: End::Returned(*value),
                    });
                }
                other => panic!("malformed instruction {other:?}"),
            }
        }
        let edges = cfg.outgoing(block).map(|edge| cfg.edge(edge));
        let next = if trapped {
            let mut unwinds = edges.filter(|edge| edge.kind() == EdgeKind::ExceptionUnwind);
            match unwinds.next() {
                Some(edge) => edge.target(),
                None => {
                    return Some(Behavior {
                        output,
                        end: End::Trapped,
                    });
                }
            }
        } else {
            let mut normal = edges.filter(|edge| edge.kind() != EdgeKind::ExceptionUnwind);
            if let Some(taken) = branch {
                let wanted = if taken {
                    EdgeKind::ConditionalTrue
                } else {
                    EdgeKind::ConditionalFalse
                };
                normal
                    .find(|edge| edge.kind() == wanted)
                    .expect("a branch has both arms")
                    .target()
            } else {
                match normal.next() {
                    Some(edge) => edge.target(),
                    None => {
                        return Some(Behavior {
                            output,
                            end: End::FellOff,
                        });
                    }
                }
            }
        };
        block = next;
    }
    None
}

/// Every assignment of the listed registers from a few small values.
pub fn input_vectors(names: &[&'static str]) -> Vec<BTreeMap<Reg, i64>> {
    let mut vectors = vec![BTreeMap::new()];
    for name in names {
        vectors = vectors
            .into_iter()
            .flat_map(|partial| {
                [-2, 0, 1, 3, 5].into_iter().map(move |value| {
                    let mut extended = partial.clone();
                    extended.insert(Reg(name), value);
                    extended
                })
            })
            .collect();
    }
    vectors
}
