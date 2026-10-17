//! Deterministic generator of structured toy programs with if, if-else, and
//! counted loops, plus the dead code, empty blocks, redefinitions, and
//! unreachable blocks the cleanup passes exist for. Every generated program
//! terminates and reads only registers that are definitely assigned.

use cfglib::{BlockId, Cfg, EdgeKind};

use crate::toy::{BinaryOp, Builder, Inst, Payload, Reg};

/// xorshift64*: enough randomness for fixture shapes, reproducible per seed.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(bound).unwrap()).unwrap()
    }

    fn percent(&mut self, chance: usize) -> bool {
        self.below(100) < chance
    }
}

const DATA: [Reg; 2] = [2, 3];
const BOUND: Reg = 1;
const ONE: Reg = 4;
const ZERO: Reg = 5;
const FIRST_TEMPORARY: Reg = 10;

struct Generator<E: Payload> {
    builder: Builder<E>,
    rng: Rng,
    /// Registers that hold a value on every path to the current point.
    readable: Vec<Reg>,
    /// The readable registers a statement may overwrite: never a loop
    /// counter, the loop bound, or a constant the generator relies on.
    writable: Vec<Reg>,
    next_temporary: Reg,
    /// Generate no loops, and never assign a register twice.
    single_assignment: bool,
}

impl<E: Payload> Generator<E> {
    fn fresh(&mut self) -> Reg {
        let register = self.next_temporary;
        self.next_temporary += 1;
        register
    }

    fn pick(&mut self, from: &[Reg]) -> Reg {
        from[self.rng.below(from.len())]
    }

    fn operand(&mut self) -> Reg {
        let readable = self.readable.clone();
        self.pick(&readable)
    }

    fn destination(&mut self) -> Reg {
        if self.single_assignment || self.rng.percent(65) {
            let register = self.fresh();
            self.readable.push(register);
            self.writable.push(register);
            register
        } else {
            let writable = self.writable.clone();
            self.pick(&writable)
        }
    }

    fn assignment(&mut self, block: BlockId) {
        let instruction = match self.rng.below(10) {
            0 => {
                let value = i64::try_from(self.rng.below(7)).unwrap() - 2;
                Inst::constant(self.destination(), value)
            }
            1 => {
                let source = self.operand();
                Inst::mov(self.destination(), source)
            }
            _ => {
                let op = [BinaryOp::Add, BinaryOp::Subtract, BinaryOp::Multiply][self.rng.below(3)];
                let (left, right) = (self.operand(), self.operand());
                Inst::binary(op, self.destination(), left, right)
            }
        };
        self.builder.push(block, instruction);
    }

    /// Emits statements into `block`, returns the block control leaves from.
    fn statements(&mut self, mut block: BlockId, depth: usize, count: usize) -> BlockId {
        for _ in 0..count {
            block = match self.rng.below(100) {
                0..=54 => {
                    self.assignment(block);
                    block
                }
                55..=69 => {
                    let source = self.operand();
                    self.builder.push(block, Inst::print(source));
                    block
                }
                70..=84 if depth < 3 => self.conditional(block, depth),
                85..=94 if depth < 2 && !self.single_assignment => self.counted_loop(block, depth),
                _ if self.rng.percent(50) => self.trampoline(block),
                _ => block,
            };
        }
        block
    }

    /// Unreachable blocks that nevertheless feed live ones.
    fn attach_orphans(&mut self) {
        let live: Vec<BlockId> = self.builder.cfg.block_ids().collect();
        for _ in 0..=self.rng.below(2) {
            let orphan = self.builder.block([Inst::print(DATA[0])]);
            let target = live[self.rng.below(live.len())];
            self.builder.edge(orphan, target, EdgeKind::Fallthrough);
        }
    }

    fn trampoline(&mut self, block: BlockId) -> BlockId {
        let empty = self.builder.block([]);
        self.builder.edge(block, empty, EdgeKind::Fallthrough);
        empty
    }

    fn scoped_statements(&mut self, block: BlockId, depth: usize) -> BlockId {
        let (readable, writable) = (self.readable.len(), self.writable.len());
        let count = self.rng.below(4);
        let end = self.statements(block, depth + 1, count);
        self.readable.truncate(readable);
        self.writable.truncate(writable);
        end
    }

    fn conditional(&mut self, block: BlockId, depth: usize) -> BlockId {
        let (left, right) = (self.operand(), self.operand());
        let condition = self.fresh();
        self.builder.push(block, Inst::less(condition, left, right));
        self.builder.push(block, Inst::branch(condition));

        let then_block = self.builder.block([]);
        let join = self.builder.block([]);
        self.builder
            .edge(block, then_block, EdgeKind::ConditionalTrue);
        let then_end = self.scoped_statements(then_block, depth);
        self.builder.edge(then_end, join, EdgeKind::Fallthrough);

        if self.rng.percent(50) {
            let else_block = self.builder.block([]);
            self.builder
                .edge(block, else_block, EdgeKind::ConditionalFalse);
            let else_end = self.scoped_statements(else_block, depth);
            self.builder.edge(else_end, join, EdgeKind::Fallthrough);
        } else {
            self.builder.edge(block, join, EdgeKind::ConditionalFalse);
        }
        join
    }

    fn counted_loop(&mut self, block: BlockId, depth: usize) -> BlockId {
        let counter = self.fresh();
        self.readable.push(counter);
        self.builder.push(block, Inst::mov(counter, ZERO));

        let header = self.builder.block([]);
        let body = self.builder.block([]);
        let exit = self.builder.block([]);
        let condition = self.fresh();
        self.builder.edge(block, header, EdgeKind::Fallthrough);
        self.builder
            .push(header, Inst::less(condition, counter, BOUND));
        self.builder.push(header, Inst::branch(condition));
        self.builder.edge(header, body, EdgeKind::ConditionalTrue);
        self.builder.edge(header, exit, EdgeKind::ConditionalFalse);

        let latch = self.scoped_statements(body, depth);
        self.builder.push(latch, Inst::add(counter, counter, ONE));
        self.builder.edge(latch, header, EdgeKind::Back);
        exit
    }
}

/// A random terminating program. Registers 0 through 3 are inputs; with
/// `orphans`, unreachable blocks that feed live blocks are added as well.
pub fn random_program<E: Payload>(seed: u64, orphans: bool) -> Cfg<Inst, E> {
    generate(seed, orphans, false)
}

/// A random loop-free program in which every register is assigned at most
/// once, so a register's value has one well-defined expression tree.
pub fn random_single_assignment_program<E: Payload>(seed: u64) -> Cfg<Inst, E> {
    generate(seed, false, true)
}

fn generate<E: Payload>(seed: u64, orphans: bool, single_assignment: bool) -> Cfg<Inst, E> {
    let mut builder = Builder::<E>::new();
    let entry = builder.entry();
    builder.push(entry, Inst::constant(ONE, 1));
    builder.push(entry, Inst::constant(ZERO, 0));
    let mut generator = Generator {
        builder,
        rng: Rng::new(seed),
        readable: vec![0, BOUND, ONE, ZERO, DATA[0], DATA[1]],
        writable: DATA.to_vec(),
        next_temporary: FIRST_TEMPORARY,
        single_assignment,
    };
    let statements = 4 + generator.rng.below(8);
    let last = generator.statements(entry, 0, statements);
    for _ in 0..2 {
        let source = generator.operand();
        generator.builder.push(last, Inst::print(source));
    }
    generator.builder.push(last, Inst::ret());
    if orphans {
        generator.attach_orphans();
    }
    generator.builder.finish()
}
