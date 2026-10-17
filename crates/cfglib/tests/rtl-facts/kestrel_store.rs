//! Kestrel storage: one flat pool per kind of operand.
//!
//! A statement stores index ranges into the pools, and the record view lends
//! the pool slices back as the plain-data types of `cfglib::ir::rtl`, the way
//! the module docs say a consumer can. Blocks keep both adjacency lists, and a
//! removed block leaves a tombstone slot.

use std::ops::Range;

use cfglib::ir::rtl::{
    BitSlice, Bits, Control, FunctionView, InstructionBuilder, Place, Raise, RecordView, Selection,
    SpanAction, Statement, ValueRange, WriteTarget,
};
use cfglib::{DenseId, GraphView, RootedView};

use crate::contract::{self, SealError};
use crate::kestrel::{Fault, KEffect, KOp, Kestrel, Policy, Reg, Undef};
use crate::model::Store;

macro_rules! dense_handle {
    ($($name:ident),*) => {$(
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        pub struct $name(u32);

        impl DenseId for $name {
            fn from_index(index: usize) -> Self {
                Self(u32::try_from(index).expect("handles fit in u32"))
            }

            fn index(self) -> usize {
                usize::try_from(self.0).expect("u32 fits in usize")
            }
        }
    )*};
}

dense_handle!(BlockRef, InstRef, ValueRef);

/// A run of one pool.
#[derive(Clone, Copy)]
struct Span {
    start: usize,
    end: usize,
}

impl Span {
    fn range(self) -> Range<usize> {
        self.start..self.end
    }
}

fn extend<T: Clone>(pool: &mut Vec<T>, items: &[T]) -> Span {
    let start = pool.len();
    pool.extend_from_slice(items);
    Span {
        start,
        end: pool.len(),
    }
}

#[derive(Clone, Copy)]
enum StoredSource {
    Direct(Place<Kestrel>),
    Finite {
        index: ValueRef,
        candidates: Span,
    },
    Sparse {
        bank: Reg,
        key: ValueRef,
        slice: BitSlice,
    },
}

#[derive(Clone, Copy)]
enum StoredTarget {
    Root {
        root: Reg,
        plan: Span,
    },
    Finite {
        index: ValueRef,
        candidates: Span,
        value: ValueRef,
    },
    Sparse {
        bank: Reg,
        key: ValueRef,
        slice: BitSlice,
        value: ValueRef,
    },
}

#[derive(Clone, Copy)]
enum StoredStatement {
    Read {
        result: ValueRef,
        source: StoredSource,
    },
    Constant {
        result: ValueRef,
        words: Span,
        width: Bits,
    },
    Apply {
        results: ValueRange<ValueRef>,
        operator: usize,
        args: Span,
    },
    Nondet {
        result: ValueRef,
        width: Bits,
        domain: Undef,
    },
    Effect {
        results: ValueRange<ValueRef>,
        operation: usize,
        args: Span,
        outcomes: Span,
    },
    Raise {
        outcome: Fault,
        guard: Option<ValueRef>,
        payload: Span,
        binds: Span,
    },
    Write {
        target: StoredTarget,
        policy: Policy,
    },
}

#[derive(Default)]
struct Pools {
    places: Vec<Place<Kestrel>>,
    plans: Vec<SpanAction<Kestrel, ValueRef>>,
    values: Vec<ValueRef>,
    binds: Vec<(Reg, ValueRef)>,
    words: Vec<u64>,
    operators: Vec<KOp>,
    effects: Vec<KEffect>,
    outcomes: Vec<Fault>,
}

impl Pools {
    /// Lends one stored statement as the plain-data statement of the facade.
    fn lend(&self, statement: &StoredStatement) -> Statement<'_, Kestrel, ValueRef> {
        match *statement {
            StoredStatement::Read { result, source } => Statement::Read {
                result,
                source: match source {
                    StoredSource::Direct(place) => Selection::Direct(place),
                    StoredSource::Finite { index, candidates } => Selection::FiniteIndexed {
                        index,
                        candidates: &self.places[candidates.range()],
                    },
                    StoredSource::Sparse { bank, key, slice } => {
                        Selection::SparseIndexed { bank, key, slice }
                    }
                },
            },
            StoredStatement::Constant {
                result,
                words,
                width,
            } => Statement::Constant {
                result,
                words: &self.words[words.range()],
                width,
            },
            StoredStatement::Apply {
                results,
                operator,
                args,
            } => Statement::Apply {
                results,
                operator: &self.operators[operator],
                args: &self.values[args.range()],
            },
            StoredStatement::Nondet {
                result,
                width,
                domain,
            } => Statement::Nondet {
                result,
                width,
                domain,
            },
            StoredStatement::Effect {
                results,
                operation,
                args,
                outcomes,
            } => Statement::Effect {
                results,
                operation: &self.effects[operation],
                args: &self.values[args.range()],
                outcomes: &self.outcomes[outcomes.range()],
            },
            StoredStatement::Raise {
                outcome,
                guard,
                payload,
                binds,
            } => Statement::Raise(Raise {
                outcome,
                guard,
                payload: &self.values[payload.range()],
                binds: &self.binds[binds.range()],
            }),
            StoredStatement::Write { target, policy } => Statement::Write {
                target: match target {
                    StoredTarget::Root { root, plan } => WriteTarget::Root {
                        root,
                        plan: &self.plans[plan.range()],
                    },
                    StoredTarget::Finite {
                        index,
                        candidates,
                        value,
                    } => WriteTarget::FiniteIndexed {
                        index,
                        candidates: &self.places[candidates.range()],
                        value,
                    },
                    StoredTarget::Sparse {
                        bank,
                        key,
                        slice,
                        value,
                    } => WriteTarget::SparseIndexed {
                        bank,
                        key,
                        slice,
                        value,
                    },
                },
                policy,
            },
        }
    }
}

struct BlockData {
    insts: Vec<InstRef>,
    successors: Vec<BlockRef>,
    predecessors: Vec<BlockRef>,
}

struct Record {
    statements: Vec<StoredStatement>,
    control: Control<ValueRef>,
}

/// A Kestrel function: blocks with tombstones, sealed records, and the pools.
pub struct KestrelFunction {
    blocks: Vec<Option<BlockData>>,
    records: Vec<Record>,
    widths: Vec<Bits>,
    pools: Pools,
}

impl KestrelFunction {
    /// Creates a function whose only block is the entry block.
    pub fn new() -> Self {
        let mut function = Self {
            blocks: Vec::new(),
            records: Vec::new(),
            widths: Vec::new(),
            pools: Pools::default(),
        };
        function.add_block();
        function
    }

    pub fn entry(&self) -> BlockRef {
        debug_assert!(self.blocks[0].is_some(), "the entry block is never removed");
        BlockRef::from_index(0)
    }

    pub fn add_block(&mut self) -> BlockRef {
        self.blocks.push(Some(BlockData {
            insts: Vec::new(),
            successors: Vec::new(),
            predecessors: Vec::new(),
        }));
        BlockRef::from_index(self.blocks.len() - 1)
    }

    pub fn add_edge(&mut self, from: BlockRef, to: BlockRef) {
        self.live_mut(from).successors.push(to);
        self.live_mut(to).predecessors.push(from);
    }

    /// Removes a block and every edge that touches it, leaving its slot.
    pub fn remove_block(&mut self, block: BlockRef) {
        assert_ne!(block, self.entry(), "the entry block stays");
        for other in self.blocks.iter_mut().flatten() {
            other.successors.retain(|&target| target != block);
            other.predecessors.retain(|&source| source != block);
        }
        self.blocks[block.index()] = None;
    }

    /// Seals one record at the end of `block`.
    ///
    /// # Panics
    ///
    /// Panics when the producer returns without sealing a record.
    pub fn append(
        &mut self,
        block: BlockRef,
        produce: impl FnOnce(KestrelBuilder<'_>) -> Result<(), SealError>,
    ) -> Result<InstRef, SealError> {
        let inst = InstRef::from_index(self.records.len());
        produce(KestrelBuilder {
            function: self,
            statements: Vec::new(),
        })?;
        assert_eq!(self.records.len(), inst.index() + 1, "the producer sealed");
        self.live_mut(block).insts.push(inst);
        Ok(inst)
    }

    fn live(&self, block: BlockRef) -> &BlockData {
        self.blocks[block.index()]
            .as_ref()
            .expect("the block is live")
    }

    fn live_mut(&mut self, block: BlockRef) -> &mut BlockData {
        self.blocks[block.index()]
            .as_mut()
            .expect("the block is live")
    }
}

impl GraphView for KestrelFunction {
    type NodeId = BlockRef;

    fn node_bound(&self) -> usize {
        self.blocks.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = BlockRef> + '_ {
        (0..self.blocks.len())
            .filter(|&index| self.blocks[index].is_some())
            .map(BlockRef::from_index)
    }

    fn successors(&self, node: BlockRef) -> impl Iterator<Item = BlockRef> + '_ {
        self.live(node).successors.iter().copied()
    }

    fn predecessors(&self, node: BlockRef) -> impl Iterator<Item = BlockRef> + '_ {
        self.live(node).predecessors.iter().copied()
    }
}

impl RootedView for KestrelFunction {
    fn root(&self) -> BlockRef {
        self.entry()
    }
}

/// One sealed record, lent out of the pools.
pub struct KestrelRecord<'f> {
    function: &'f KestrelFunction,
    record: &'f Record,
}

impl RecordView<Kestrel> for KestrelRecord<'_> {
    type Value = ValueRef;

    fn statements(&self) -> impl ExactSizeIterator<Item = Statement<'_, Kestrel, ValueRef>> + '_ {
        let pools = &self.function.pools;
        self.record
            .statements
            .iter()
            .map(move |statement| pools.lend(statement))
    }

    fn control(&self) -> Control<ValueRef> {
        self.record.control
    }

    fn width(&self, value: ValueRef) -> Bits {
        self.function.widths[value.index()]
    }
}

impl FunctionView<Kestrel> for KestrelFunction {
    type Inst = InstRef;
    type Value = ValueRef;
    type Record<'a> = KestrelRecord<'a>;

    fn inst_bound(&self) -> usize {
        self.records.len()
    }

    fn value_bound(&self) -> usize {
        self.widths.len()
    }

    fn instructions(&self, block: BlockRef) -> &[InstRef] {
        &self.live(block).insts
    }

    fn record(&self, inst: InstRef) -> KestrelRecord<'_> {
        KestrelRecord {
            function: self,
            record: &self.records[inst.index()],
        }
    }
}

/// The builder of one record. It is dropped without effect unless `finish`
/// seals the record.
pub struct KestrelBuilder<'f> {
    function: &'f mut KestrelFunction,
    statements: Vec<StoredStatement>,
}

impl KestrelBuilder<'_> {
    fn define(&mut self, width: Bits) -> ValueRef {
        self.function.widths.push(width);
        ValueRef::from_index(self.function.widths.len() - 1)
    }

    fn define_run(&mut self, widths: &[Bits]) -> ValueRange<ValueRef> {
        let first = ValueRef::from_index(self.function.widths.len());
        self.function.widths.extend_from_slice(widths);
        ValueRange::new(first, u32::try_from(widths.len()).expect("few results"))
    }
}

impl InstructionBuilder<Kestrel> for KestrelBuilder<'_> {
    type Value = ValueRef;
    type Error = SealError;

    fn read(&mut self, source: Selection<'_, Kestrel, ValueRef>) -> ValueRef {
        let pools = &mut self.function.pools;
        let (width, source) = match source {
            Selection::Direct(place) => (place.slice.width, StoredSource::Direct(place)),
            Selection::FiniteIndexed { index, candidates } => (
                candidates[0].slice.width,
                StoredSource::Finite {
                    index,
                    candidates: extend(&mut pools.places, candidates),
                },
            ),
            Selection::SparseIndexed { bank, key, slice } => {
                (slice.width, StoredSource::Sparse { bank, key, slice })
            }
        };
        let result = self.define(width);
        self.statements
            .push(StoredStatement::Read { result, source });
        result
    }

    fn constant(&mut self, words: &[u64], width: Bits) -> ValueRef {
        let words = extend(&mut self.function.pools.words, words);
        let result = self.define(width);
        self.statements.push(StoredStatement::Constant {
            result,
            words,
            width,
        });
        result
    }

    fn apply(
        &mut self,
        operator: &KOp,
        args: &[ValueRef],
        results: &[Bits],
    ) -> ValueRange<ValueRef> {
        let pools = &mut self.function.pools;
        pools.operators.push(operator.clone());
        let operator = pools.operators.len() - 1;
        let args = extend(&mut pools.values, args);
        let results = self.define_run(results);
        self.statements.push(StoredStatement::Apply {
            results,
            operator,
            args,
        });
        results
    }

    fn nondet(&mut self, width: Bits, domain: Undef) -> ValueRef {
        let result = self.define(width);
        self.statements.push(StoredStatement::Nondet {
            result,
            width,
            domain,
        });
        result
    }

    fn effect(
        &mut self,
        operation: &KEffect,
        args: &[ValueRef],
        results: &[Bits],
        outcomes: &[Fault],
    ) -> ValueRange<ValueRef> {
        let pools = &mut self.function.pools;
        pools.effects.push(operation.clone());
        let operation = pools.effects.len() - 1;
        let args = extend(&mut pools.values, args);
        let outcomes = extend(&mut pools.outcomes, outcomes);
        let results = self.define_run(results);
        self.statements.push(StoredStatement::Effect {
            results,
            operation,
            args,
            outcomes,
        });
        results
    }

    fn raise(&mut self, raise: Raise<'_, Kestrel, ValueRef>) {
        let pools = &mut self.function.pools;
        let payload = extend(&mut pools.values, raise.payload);
        let binds = extend(&mut pools.binds, raise.binds);
        self.statements.push(StoredStatement::Raise {
            outcome: raise.outcome,
            guard: raise.guard,
            payload,
            binds,
        });
    }

    fn write(&mut self, target: WriteTarget<'_, Kestrel, ValueRef>, policy: Policy) {
        let pools = &mut self.function.pools;
        let target = match target {
            WriteTarget::Root { root, plan } => StoredTarget::Root {
                root,
                plan: extend(&mut pools.plans, plan),
            },
            WriteTarget::FiniteIndexed {
                index,
                candidates,
                value,
            } => StoredTarget::Finite {
                index,
                candidates: extend(&mut pools.places, candidates),
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
        self.statements
            .push(StoredStatement::Write { target, policy });
    }

    fn finish(self, control: Control<ValueRef>) -> Result<(), SealError> {
        let function = self.function;
        let lent = self
            .statements
            .iter()
            .map(|statement| function.pools.lend(statement));
        contract::check::<Kestrel, ValueRef>(lent, control, &|value| {
            function.widths[value.index()]
        })?;
        function.records.push(Record {
            statements: self.statements,
            control,
        });
        Ok(())
    }
}

impl Store<Kestrel> for KestrelFunction {
    type Builder<'f>
        = KestrelBuilder<'f>
    where
        Self: 'f;

    fn start() -> Self {
        Self::new()
    }

    fn new_block(&mut self) -> BlockRef {
        self.add_block()
    }

    fn connect(&mut self, from: BlockRef, to: BlockRef) {
        self.add_edge(from, to);
    }

    fn remove(&mut self, block: BlockRef) {
        self.remove_block(block);
    }

    fn seal(
        &mut self,
        block: BlockRef,
        produce: impl FnOnce(KestrelBuilder<'_>) -> Result<(), SealError>,
    ) -> Result<InstRef, SealError> {
        self.append(block, produce)
    }
}
