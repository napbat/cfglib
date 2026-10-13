//! A toy register-transfer dialect over `Vec`-backed storage.
//!
//! The dialect has a few 64-bit roots, a one-bit flag, a three-bit stack
//! pointer, four stack slots that an indexed access can select, and one
//! sparse bank. The storage holds each record in owned vectors and lends
//! them back through the view traits. Tests author records through the
//! [`InstructionBuilder`] protocol.

extern crate alloc;

use alloc::vec::Vec;
use core::convert::Infallible;

use crate::DenseBits;
use crate::graph::view::{DenseId, GraphView, RootedView};
use crate::ir::rtl::{
    Action, BitSlice, Bits, ConstBits, Control, Dialect, Fold, FunctionView, InstructionBuilder,
    Place, Raise, RecordView, RootTable, Selection, SpanAction, Statement, ValueRange, WriteTarget,
};

macro_rules! dense_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub(super) struct $name(pub(super) u32);

        impl DenseId for $name {
            fn from_index(index: usize) -> Self {
                Self(u32::try_from(index).unwrap())
            }

            fn index(self) -> usize {
                usize::try_from(self.0).unwrap()
            }
        }
    };
}

dense_id!(
    /// A block of a toy function.
    Block
);
dense_id!(
    /// An instruction of a toy function.
    Inst
);
dense_id!(
    /// A value of a toy function.
    Value
);

/// The marker of the toy dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Toy;

/// A storage root of the toy dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Root {
    /// A 64-bit accumulator.
    Acc,
    /// A 64-bit scratch register.
    Tmp,
    /// A one-bit flag.
    Flag,
    /// The three-bit stack pointer.
    Top,
    /// The first of four stack slots.
    Slot0,
    /// The second stack slot.
    Slot1,
    /// The third stack slot.
    Slot2,
    /// The fourth stack slot.
    Slot3,
    /// A sparse bank of 64-bit elements.
    Bank,
}

/// The stack slots in index order.
pub(super) const SLOTS: [Root; 4] = [Root::Slot0, Root::Slot1, Root::Slot2, Root::Slot3];

/// A pure operator of the toy dialect.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum Operator {
    /// The wrapping sum of two operands.
    Add,
    /// The bitwise and of two operands.
    And,
}

/// An ordered effect of the toy dialect.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum Effect {
    /// A load from memory.
    Load,
}

impl Dialect for Toy {
    type Root = Root;
    type Operator = Operator;
    type EffectOp = Effect;
    type WritePolicy = ();
    type Outcome = ();
    type NondetDomain = ();

    fn root_width(root: Root) -> Bits {
        let width = match root {
            Root::Flag => 1,
            Root::Top => 3,
            Root::Acc
            | Root::Tmp
            | Root::Slot0
            | Root::Slot1
            | Root::Slot2
            | Root::Slot3
            | Root::Bank => 64,
        };
        Bits::new(width).unwrap()
    }
}

impl Fold for Toy {
    fn fold(
        operator: &Operator,
        args: &[ConstBits<'_>],
        results: &[Bits],
        out: &mut [u64],
    ) -> bool {
        let ([left, right], [width]) = (args, results) else {
            return false;
        };
        if width.width() > 64 || left.width.width() > 64 {
            return false;
        }
        let mask = u64::MAX >> (64 - u32::from(width.width()));
        out[0] = match operator {
            Operator::Add => left.words[0].wrapping_add(right.words[0]),
            Operator::And => left.words[0] & right.words[0],
        } & mask;
        true
    }
}

/// A selection in owned storage.
#[derive(Debug)]
enum StoredSelection {
    Direct(Place<Toy>),
    Finite {
        index: Value,
        candidates: Vec<Place<Toy>>,
    },
    Sparse {
        bank: Root,
        key: Value,
        slice: BitSlice,
    },
}

impl StoredSelection {
    fn view(&self) -> Selection<'_, Toy, Value> {
        match self {
            Self::Direct(place) => Selection::Direct(*place),
            Self::Finite { index, candidates } => Selection::FiniteIndexed {
                index: *index,
                candidates,
            },
            Self::Sparse { bank, key, slice } => Selection::SparseIndexed {
                bank: *bank,
                key: *key,
                slice: *slice,
            },
        }
    }
}

/// A write target in owned storage.
#[derive(Debug)]
enum StoredTarget {
    Root {
        root: Root,
        plan: Vec<SpanAction<Toy, Value>>,
    },
    Finite {
        index: Value,
        candidates: Vec<Place<Toy>>,
        value: Value,
    },
    Sparse {
        bank: Root,
        key: Value,
        slice: BitSlice,
        value: Value,
    },
}

impl StoredTarget {
    fn view(&self) -> WriteTarget<'_, Toy, Value> {
        match self {
            Self::Root { root, plan } => WriteTarget::Root { root: *root, plan },
            Self::Finite {
                index,
                candidates,
                value,
            } => WriteTarget::FiniteIndexed {
                index: *index,
                candidates,
                value: *value,
            },
            Self::Sparse {
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
        }
    }
}

/// A statement in owned storage.
#[derive(Debug)]
enum Stored {
    Read {
        result: Value,
        source: StoredSelection,
    },
    Constant {
        result: Value,
        words: Vec<u64>,
        width: Bits,
    },
    Apply {
        results: ValueRange<Value>,
        operator: Operator,
        args: Vec<Value>,
    },
    Nondet {
        result: Value,
        width: Bits,
    },
    Effect {
        results: ValueRange<Value>,
        operation: Effect,
        args: Vec<Value>,
    },
    Raise {
        guard: Option<Value>,
        payload: Vec<Value>,
        binds: Vec<(Root, Value)>,
    },
    Write {
        target: StoredTarget,
    },
}

impl Stored {
    fn view(&self) -> Statement<'_, Toy, Value> {
        match self {
            Self::Read { result, source } => Statement::Read {
                result: *result,
                source: source.view(),
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
            Self::Nondet { result, width } => Statement::Nondet {
                result: *result,
                width: *width,
                domain: (),
            },
            Self::Effect {
                results,
                operation,
                args,
            } => Statement::Effect {
                results: *results,
                operation,
                args,
                outcomes: &[],
            },
            Self::Raise {
                guard,
                payload,
                binds,
            } => Statement::Raise(Raise {
                outcome: (),
                guard: *guard,
                payload,
                binds,
            }),
            Self::Write { target } => Statement::Write {
                target: target.view(),
                policy: (),
            },
        }
    }
}

/// One sealed record in owned storage.
#[derive(Debug)]
struct Record {
    statements: Vec<Stored>,
    control: Control<Value>,
}

/// A toy function. The first block is the entry block.
#[derive(Debug, Default)]
pub(super) struct Function {
    blocks: Vec<Vec<Inst>>,
    successors: Vec<Vec<Block>>,
    records: Vec<Record>,
    widths: Vec<Bits>,
}

impl Function {
    /// Creates a function without blocks.
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Adds a block.
    pub(super) fn block(&mut self) -> Block {
        self.blocks.push(Vec::new());
        self.successors.push(Vec::new());
        Block::from_index(self.blocks.len() - 1)
    }

    /// Adds the flow edges that leave `block`.
    pub(super) fn flow(&mut self, block: Block, targets: &[Block]) {
        self.successors[block.index()].extend_from_slice(targets);
    }

    /// Starts a record at the end of `block`.
    pub(super) fn record_in(&mut self, block: Block) -> Builder<'_> {
        Builder {
            function: self,
            block,
            statements: Vec::new(),
        }
    }

    /// Returns the instruction at `position` of `block`.
    pub(super) fn inst(&self, block: Block, position: usize) -> Inst {
        self.blocks[block.index()][position]
    }
}

impl GraphView for Function {
    type NodeId = Block;

    fn node_bound(&self) -> usize {
        self.blocks.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = Block> + '_ {
        (0..self.blocks.len()).map(Block::from_index)
    }

    fn successors(&self, node: Block) -> impl Iterator<Item = Block> + '_ {
        self.successors[node.index()].iter().copied()
    }

    fn predecessors(&self, node: Block) -> impl Iterator<Item = Block> + '_ {
        self.node_ids().flat_map(move |source| {
            self.successors[source.index()]
                .iter()
                .filter(move |&&target| target == node)
                .map(move |_| source)
        })
    }
}

impl RootedView for Function {
    fn root(&self) -> Block {
        Block(0)
    }
}

/// A lent view of one record.
pub(super) struct RecordRef<'a> {
    function: &'a Function,
    record: &'a Record,
}

impl RecordView<Toy> for RecordRef<'_> {
    type Value = Value;

    fn statements(&self) -> impl ExactSizeIterator<Item = Statement<'_, Toy, Value>> + '_ {
        self.record.statements.iter().map(Stored::view)
    }

    fn control(&self) -> Control<Value> {
        self.record.control
    }

    fn width(&self, value: Value) -> Bits {
        self.function.widths[value.index()]
    }
}

impl FunctionView<Toy> for Function {
    type Inst = Inst;
    type Value = Value;
    type Record<'a> = RecordRef<'a>;

    fn inst_bound(&self) -> usize {
        self.records.len()
    }

    fn value_bound(&self) -> usize {
        self.widths.len()
    }

    fn instructions(&self, block: Block) -> &[Inst] {
        &self.blocks[block.index()]
    }

    fn record(&self, inst: Inst) -> RecordRef<'_> {
        RecordRef {
            function: self,
            record: &self.records[inst.index()],
        }
    }
}

/// The builder of one record.
pub(super) struct Builder<'f> {
    function: &'f mut Function,
    block: Block,
    statements: Vec<Stored>,
}

impl Builder<'_> {
    fn value(&mut self, width: Bits) -> Value {
        self.function.widths.push(width);
        Value::from_index(self.function.widths.len() - 1)
    }

    /// Reads the whole of `root`.
    pub(super) fn read_whole(&mut self, root: Root) -> Value {
        self.read(Selection::Direct(Place::whole(root)))
    }

    /// Creates a constant of `width` bits from one word.
    pub(super) fn constant_value(&mut self, word: u64, width: u16) -> Value {
        self.constant(&[word], Bits::new(width).unwrap())
    }

    /// Writes `value` over the whole of `root`.
    pub(super) fn write_whole(&mut self, root: Root, value: Value) {
        let plan = [SpanAction {
            span: BitSlice::whole(Toy::root_width(root)),
            action: Action::TakeInput { value, offset: 0 },
        }];
        self.write(WriteTarget::Root { root, plan: &plan }, ());
    }

    /// Writes `value` into `span` of `root` and preserves the other bits.
    pub(super) fn write_slice(&mut self, root: Root, span: BitSlice, value: Value) {
        let width = Toy::root_width(root);
        let mut plan = Vec::new();
        if span.offset > 0 {
            plan.push(SpanAction {
                span: BitSlice::new(0, Bits::new(span.offset).unwrap()),
                action: Action::PreservePreState,
            });
        }
        plan.push(SpanAction {
            span,
            action: Action::TakeInput { value, offset: 0 },
        });
        if span.end() < u32::from(width.width()) {
            let rest = u16::try_from(u32::from(width.width()) - span.end()).unwrap();
            plan.push(SpanAction {
                span: BitSlice::new(u16::try_from(span.end()).unwrap(), Bits::new(rest).unwrap()),
                action: Action::PreservePreState,
            });
        }
        self.write(WriteTarget::Root { root, plan: &plan }, ());
    }

    /// Seals the record with `control`.
    pub(super) fn end(self, control: Control<Value>) -> Inst {
        let inst = Inst::from_index(self.function.records.len());
        self.finish(control).unwrap();
        inst
    }
}

impl InstructionBuilder<Toy> for Builder<'_> {
    type Value = Value;
    type Error = Infallible;

    fn read(&mut self, source: Selection<'_, Toy, Value>) -> Value {
        let (width, source) = match source {
            Selection::Direct(place) => (place.slice.width, StoredSelection::Direct(place)),
            Selection::FiniteIndexed { index, candidates } => (
                candidates[0].slice.width,
                StoredSelection::Finite {
                    index,
                    candidates: candidates.to_vec(),
                },
            ),
            Selection::SparseIndexed { bank, key, slice } => {
                (slice.width, StoredSelection::Sparse { bank, key, slice })
            }
        };
        let result = self.value(width);
        self.statements.push(Stored::Read { result, source });
        result
    }

    fn constant(&mut self, words: &[u64], width: Bits) -> Value {
        let result = self.value(width);
        self.statements.push(Stored::Constant {
            result,
            words: words.to_vec(),
            width,
        });
        result
    }

    fn apply(
        &mut self,
        operator: &Operator,
        args: &[Value],
        results: &[Bits],
    ) -> ValueRange<Value> {
        let values: Vec<Value> = results.iter().map(|&width| self.value(width)).collect();
        let range = ValueRange::new(values[0], u32::try_from(values.len()).unwrap());
        self.statements.push(Stored::Apply {
            results: range,
            operator: operator.clone(),
            args: args.to_vec(),
        });
        range
    }

    fn nondet(&mut self, width: Bits, _domain: ()) -> Value {
        let result = self.value(width);
        self.statements.push(Stored::Nondet { result, width });
        result
    }

    fn effect(
        &mut self,
        operation: &Effect,
        args: &[Value],
        results: &[Bits],
        _outcomes: &[()],
    ) -> ValueRange<Value> {
        let values: Vec<Value> = results.iter().map(|&width| self.value(width)).collect();
        let range = ValueRange::new(
            values.first().copied().unwrap_or(Value(0)),
            u32::try_from(values.len()).unwrap(),
        );
        self.statements.push(Stored::Effect {
            results: range,
            operation: operation.clone(),
            args: args.to_vec(),
        });
        range
    }

    fn raise(&mut self, raise: Raise<'_, Toy, Value>) {
        self.statements.push(Stored::Raise {
            guard: raise.guard,
            payload: raise.payload.to_vec(),
            binds: raise.binds.to_vec(),
        });
    }

    fn write(&mut self, target: WriteTarget<'_, Toy, Value>, _policy: ()) {
        let target = match target {
            WriteTarget::Root { root, plan } => StoredTarget::Root {
                root,
                plan: plan.to_vec(),
            },
            WriteTarget::FiniteIndexed {
                index,
                candidates,
                value,
            } => StoredTarget::Finite {
                index,
                candidates: candidates.to_vec(),
                value,
            },
            WriteTarget::SparseIndexed {
                bank,
                key,
                slice,
                value,
            } => StoredTarget::Sparse {
                bank,
                key,
                slice,
                value,
            },
        };
        self.statements.push(Stored::Write { target });
    }

    fn finish(self, control: Control<Value>) -> Result<(), Infallible> {
        let inst = Inst::from_index(self.function.records.len());
        self.function.records.push(Record {
            statements: self.statements,
            control,
        });
        self.function.blocks[self.block.index()].push(inst);
        Ok(())
    }
}

/// Returns the places that name the whole of each of `roots`.
pub(super) fn whole_places(roots: &[Root]) -> Vec<Place<Toy>> {
    roots.iter().map(|&root| Place::whole(root)).collect()
}

/// Returns the roots that the dense indices in `set` stand for, in order.
pub(super) fn roots_in(table: &RootTable<Toy>, set: &DenseBits) -> Vec<Root> {
    set.ones().map(|index| table.root(index)).collect()
}
