//! Tessera: a wide-slot machine with a vocabulary that is nothing like Kestrel.
//!
//! The roots are numbered slots whose order runs against the neutral numbering,
//! the operators and effects are strings, and the outcomes and domains are
//! plain integers. The storage owns every statement on its own, keeps only
//! forward adjacency, and uses `u32` blocks and instructions with `usize`
//! values, so nothing about it matches the pooled Kestrel storage.

use cfglib::ir::rtl::{
    BitSlice, Bits, ConstBits, Control, Dialect, Fold, FunctionView, InstructionBuilder, Place,
    Raise, RecordView, Selection, SpanAction, Statement, ValueRange, WriteTarget,
};
use cfglib::{GraphView, RootedView, scan_predecessors};

use crate::bits;
use crate::contract::{self, SealError};
use crate::model::{Store, Target};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Tessera;

/// A storage slot. The slot of neutral id `n` is `100 - 7 * n`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Slot(pub u16);

impl Slot {
    pub fn of(id: u8) -> Self {
        Self(100 - 7 * u16::from(id))
    }

    pub fn id(self) -> u8 {
        u8::try_from((100 - self.0) / 7).expect("a slot of the test vocabulary")
    }
}

impl Dialect for Tessera {
    type Root = Slot;
    type Operator = String;
    type EffectOp = String;
    type WritePolicy = bool;
    type Outcome = u16;
    type NondetDomain = u8;

    fn root_width(root: Slot) -> Bits {
        match root.id() {
            0..=3 | 9 => bits(64),
            4..=7 => Bits::BOOL,
            8 => bits(256),
            other => panic!("slot {other} is not part of the test vocabulary"),
        }
    }
}

impl Target for Tessera {
    fn root_of(id: u8) -> Slot {
        Slot::of(id)
    }

    fn id_of(root: Slot) -> u8 {
        root.id()
    }

    fn operator() -> String {
        "xor".to_owned()
    }

    fn effect() -> String {
        "fence".to_owned()
    }

    fn policy() -> bool {
        true
    }

    fn outcome() -> u16 {
        0x0e
    }

    fn domain() -> u8 {
        3
    }

    fn add() -> String {
        "add".to_owned()
    }

    fn sub() -> String {
        "sub".to_owned()
    }

    fn and() -> String {
        "and".to_owned()
    }

    fn xor() -> String {
        "xor".to_owned()
    }

    fn equal() -> String {
        "eq".to_owned()
    }

    fn opaque() -> String {
        "opaque".to_owned()
    }
}

impl Fold for Tessera {
    fn fold(operator: &String, args: &[ConstBits<'_>], results: &[Bits], out: &mut [u64]) -> bool {
        match operator.as_str() {
            "xor" | "and" => {
                for (position, slot) in out[..results[0].words()].iter_mut().enumerate() {
                    let (left, right) = (args[0].words[position], args[1].words[position]);
                    *slot = if operator == "and" {
                        left & right
                    } else {
                        left ^ right
                    };
                }
                true
            }
            "eq" => {
                out[0] = u64::from(args[0].words == args[1].words);
                true
            }
            "add" | "sub" => {
                let (Some(left), Some(right)) = (value(&args[0]), value(&args[1])) else {
                    return false;
                };
                let sum = if operator == "add" {
                    left.wrapping_add(right)
                } else {
                    left.wrapping_sub(right)
                };
                out[0] = u64::try_from(sum & u128::from(u64::MAX)).expect("masked");
                if results[0].words() > 1 {
                    out[1] = u64::try_from(sum >> 64).expect("two words");
                }
                true
            }
            // Quotient and remainder of two one-word values. A zero divisor
            // has no value, so the operator does not fold.
            "divmod" => {
                let (dividend, divisor) = (args[0].words[0], args[1].words[0]);
                if divisor == 0 {
                    return false;
                }
                out[0] = dividend / divisor;
                out[1] = dividend % divisor;
                true
            }
            // Exchanges two values of different widths; the results come in
            // the order of the swapped operands.
            "swap" => {
                let split = args[1].words.len();
                out[..split].copy_from_slice(args[1].words);
                out[split..split + args[0].words.len()].copy_from_slice(args[0].words);
                true
            }
            _ => false,
        }
    }
}

/// The value of an operand of at most 128 bits.
fn value(arg: &ConstBits<'_>) -> Option<u128> {
    if arg.width.width() > 128 {
        return None;
    }
    let high = arg.words.get(1).copied().map_or(0, u128::from);
    Some(u128::from(arg.words[0]) | (high << 64))
}

enum OwnedSource {
    Direct(Place<Tessera>),
    Finite {
        index: usize,
        candidates: Box<[Place<Tessera>]>,
    },
    Sparse {
        bank: Slot,
        key: usize,
        slice: BitSlice,
    },
}

enum OwnedTarget {
    Root {
        root: Slot,
        plan: Box<[SpanAction<Tessera, usize>]>,
    },
    Finite {
        index: usize,
        candidates: Box<[Place<Tessera>]>,
        value: usize,
    },
    Sparse {
        bank: Slot,
        key: usize,
        slice: BitSlice,
        value: usize,
    },
}

enum OwnedStatement {
    Read {
        result: usize,
        source: OwnedSource,
    },
    Constant {
        result: usize,
        words: Box<[u64]>,
        width: Bits,
    },
    Apply {
        results: ValueRange<usize>,
        operator: String,
        args: Box<[usize]>,
    },
    Nondet {
        result: usize,
        width: Bits,
        domain: u8,
    },
    Effect {
        results: ValueRange<usize>,
        operation: String,
        args: Box<[usize]>,
        outcomes: Box<[u16]>,
    },
    Raise {
        outcome: u16,
        guard: Option<usize>,
        payload: Box<[usize]>,
        binds: Box<[(Slot, usize)]>,
    },
    Write {
        target: OwnedTarget,
        policy: bool,
    },
}

impl OwnedStatement {
    fn lend(&self) -> Statement<'_, Tessera, usize> {
        match self {
            Self::Read { result, source } => Statement::Read {
                result: *result,
                source: match source {
                    OwnedSource::Direct(place) => Selection::Direct(*place),
                    OwnedSource::Finite { index, candidates } => Selection::FiniteIndexed {
                        index: *index,
                        candidates,
                    },
                    OwnedSource::Sparse { bank, key, slice } => Selection::SparseIndexed {
                        bank: *bank,
                        key: *key,
                        slice: *slice,
                    },
                },
            },
            Self::Constant {
                result,
                words,
                width,
            } => Statement::Constant {
                result: *result,
                words,
                width: *width,
            },
            Self::Apply {
                results,
                operator,
                args,
            } => Statement::Apply {
                results: *results,
                operator,
                args,
            },
            Self::Nondet {
                result,
                width,
                domain,
            } => Statement::Nondet {
                result: *result,
                width: *width,
                domain: *domain,
            },
            Self::Effect {
                results,
                operation,
                args,
                outcomes,
            } => Statement::Effect {
                results: *results,
                operation,
                args,
                outcomes,
            },
            Self::Raise {
                outcome,
                guard,
                payload,
                binds,
            } => Statement::Raise(Raise {
                outcome: *outcome,
                guard: *guard,
                payload,
                binds,
            }),
            Self::Write { target, policy } => Statement::Write {
                target: match target {
                    OwnedTarget::Root { root, plan } => WriteTarget::Root { root: *root, plan },
                    OwnedTarget::Finite {
                        index,
                        candidates,
                        value,
                    } => WriteTarget::FiniteIndexed {
                        index: *index,
                        candidates,
                        value: *value,
                    },
                    OwnedTarget::Sparse {
                        bank,
                        key,
                        slice,
                        value,
                    } => WriteTarget::SparseIndexed {
                        bank: *bank,
                        key: *key,
                        slice: *slice,
                        value: *value,
                    },
                },
                policy: *policy,
            },
        }
    }
}

struct Block {
    insts: Vec<u32>,
    successors: Vec<u32>,
}

struct Record {
    statements: Vec<OwnedStatement>,
    control: Control<usize>,
}

/// A Tessera function. Only forward adjacency is stored.
pub struct TesseraFunction {
    blocks: Vec<Option<Block>>,
    records: Vec<Record>,
    widths: Vec<Bits>,
}

impl TesseraFunction {
    /// Creates a function whose only block is the entry block `0`.
    pub fn new() -> Self {
        let mut function = Self {
            blocks: Vec::new(),
            records: Vec::new(),
            widths: Vec::new(),
        };
        function.add_block();
        function
    }

    pub fn add_block(&mut self) -> u32 {
        self.blocks.push(Some(Block {
            insts: Vec::new(),
            successors: Vec::new(),
        }));
        u32::try_from(self.blocks.len() - 1).expect("few blocks")
    }

    pub fn add_edge(&mut self, from: u32, to: u32) {
        self.live_mut(from).successors.push(to);
    }

    pub fn remove_block(&mut self, block: u32) {
        assert_ne!(block, 0, "the entry block stays");
        for other in self.blocks.iter_mut().flatten() {
            other.successors.retain(|&target| target != block);
        }
        self.blocks[usize::try_from(block).expect("small")] = None;
    }

    /// Seals one record at the end of `block`.
    ///
    /// # Panics
    ///
    /// Panics when the producer returns without sealing a record.
    pub fn append(
        &mut self,
        block: u32,
        produce: impl FnOnce(TesseraBuilder<'_>) -> Result<(), SealError>,
    ) -> Result<u32, SealError> {
        let inst = u32::try_from(self.records.len()).expect("few records");
        produce(TesseraBuilder {
            function: self,
            statements: Vec::new(),
        })?;
        assert_eq!(self.records.len(), usize::try_from(inst).unwrap() + 1);
        self.live_mut(block).insts.push(inst);
        Ok(inst)
    }

    fn live(&self, block: u32) -> &Block {
        self.blocks[usize::try_from(block).expect("small")]
            .as_ref()
            .expect("the block is live")
    }

    fn live_mut(&mut self, block: u32) -> &mut Block {
        self.blocks[usize::try_from(block).expect("small")]
            .as_mut()
            .expect("the block is live")
    }
}

impl GraphView for TesseraFunction {
    type NodeId = u32;

    fn node_bound(&self) -> usize {
        self.blocks.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.blocks.len())
            .filter(|&index| self.blocks[index].is_some())
            .map(|index| u32::try_from(index).expect("few blocks"))
    }

    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.live(node).successors.iter().copied()
    }

    fn predecessors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        scan_predecessors(self, node)
    }
}

impl RootedView for TesseraFunction {
    fn root(&self) -> u32 {
        0
    }
}

pub struct TesseraRecord<'f> {
    function: &'f TesseraFunction,
    record: &'f Record,
}

impl RecordView<Tessera> for TesseraRecord<'_> {
    type Value = usize;

    fn statements(&self) -> impl ExactSizeIterator<Item = Statement<'_, Tessera, usize>> + '_ {
        self.record.statements.iter().map(OwnedStatement::lend)
    }

    fn control(&self) -> Control<usize> {
        self.record.control
    }

    fn width(&self, value: usize) -> Bits {
        self.function.widths[value]
    }
}

impl FunctionView<Tessera> for TesseraFunction {
    type Inst = u32;
    type Value = usize;
    type Record<'a> = TesseraRecord<'a>;

    fn inst_bound(&self) -> usize {
        self.records.len()
    }

    fn value_bound(&self) -> usize {
        self.widths.len()
    }

    fn instructions(&self, block: u32) -> &[u32] {
        &self.live(block).insts
    }

    fn record(&self, inst: u32) -> TesseraRecord<'_> {
        TesseraRecord {
            function: self,
            record: &self.records[usize::try_from(inst).expect("small")],
        }
    }
}

pub struct TesseraBuilder<'f> {
    function: &'f mut TesseraFunction,
    statements: Vec<OwnedStatement>,
}

impl TesseraBuilder<'_> {
    fn define(&mut self, width: Bits) -> usize {
        self.function.widths.push(width);
        self.function.widths.len() - 1
    }

    fn define_run(&mut self, widths: &[Bits]) -> ValueRange<usize> {
        let first = self.function.widths.len();
        self.function.widths.extend_from_slice(widths);
        ValueRange::new(first, u32::try_from(widths.len()).expect("few results"))
    }
}

impl InstructionBuilder<Tessera> for TesseraBuilder<'_> {
    type Value = usize;
    type Error = SealError;

    fn read(&mut self, source: Selection<'_, Tessera, usize>) -> usize {
        let (width, source) = match source {
            Selection::Direct(place) => (place.slice.width, OwnedSource::Direct(place)),
            Selection::FiniteIndexed { index, candidates } => (
                candidates[0].slice.width,
                OwnedSource::Finite {
                    index,
                    candidates: candidates.into(),
                },
            ),
            Selection::SparseIndexed { bank, key, slice } => {
                (slice.width, OwnedSource::Sparse { bank, key, slice })
            }
        };
        let result = self.define(width);
        self.statements
            .push(OwnedStatement::Read { result, source });
        result
    }

    fn constant(&mut self, words: &[u64], width: Bits) -> usize {
        let result = self.define(width);
        self.statements.push(OwnedStatement::Constant {
            result,
            words: words.into(),
            width,
        });
        result
    }

    fn apply(&mut self, operator: &String, args: &[usize], results: &[Bits]) -> ValueRange<usize> {
        let results = self.define_run(results);
        self.statements.push(OwnedStatement::Apply {
            results,
            operator: operator.clone(),
            args: args.into(),
        });
        results
    }

    fn nondet(&mut self, width: Bits, domain: u8) -> usize {
        let result = self.define(width);
        self.statements.push(OwnedStatement::Nondet {
            result,
            width,
            domain,
        });
        result
    }

    fn effect(
        &mut self,
        operation: &String,
        args: &[usize],
        results: &[Bits],
        outcomes: &[u16],
    ) -> ValueRange<usize> {
        let results = self.define_run(results);
        self.statements.push(OwnedStatement::Effect {
            results,
            operation: operation.clone(),
            args: args.into(),
            outcomes: outcomes.into(),
        });
        results
    }

    fn raise(&mut self, raise: Raise<'_, Tessera, usize>) {
        self.statements.push(OwnedStatement::Raise {
            outcome: raise.outcome,
            guard: raise.guard,
            payload: raise.payload.into(),
            binds: raise.binds.into(),
        });
    }

    fn write(&mut self, target: WriteTarget<'_, Tessera, usize>, policy: bool) {
        let target = match target {
            WriteTarget::Root { root, plan } => OwnedTarget::Root {
                root,
                plan: plan.into(),
            },
            WriteTarget::FiniteIndexed {
                index,
                candidates,
                value,
            } => OwnedTarget::Finite {
                index,
                candidates: candidates.into(),
                value,
            },
            WriteTarget::SparseIndexed {
                bank,
                key,
                slice,
                value,
            } => OwnedTarget::Sparse {
                bank,
                key,
                slice,
                value,
            },
        };
        self.statements
            .push(OwnedStatement::Write { target, policy });
    }

    fn finish(self, control: Control<usize>) -> Result<(), SealError> {
        let function = self.function;
        let lent = self.statements.iter().map(OwnedStatement::lend);
        contract::check::<Tessera, usize>(lent, control, &|value| function.widths[value])?;
        function.records.push(Record {
            statements: self.statements,
            control,
        });
        Ok(())
    }
}

impl Store<Tessera> for TesseraFunction {
    type Builder<'f>
        = TesseraBuilder<'f>
    where
        Self: 'f;

    fn start() -> Self {
        Self::new()
    }

    fn new_block(&mut self) -> u32 {
        self.add_block()
    }

    fn connect(&mut self, from: u32, to: u32) {
        self.add_edge(from, to);
    }

    fn remove(&mut self, block: u32) {
        self.remove_block(block);
    }

    fn seal(
        &mut self,
        block: u32,
        produce: impl FnOnce(TesseraBuilder<'_>) -> Result<(), SealError>,
    ) -> Result<u32, SealError> {
        self.append(block, produce)
    }
}
