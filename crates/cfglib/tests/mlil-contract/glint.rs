//! Glint: a shader-style SSA bytecode, as a consumer would store it.
//!
//! One table per kind of entity. A block lists its parameters and its
//! instructions, an instruction owns its operands, results, and (when it
//! terminates a block) its successor records, and a value table says where
//! each value is defined. Predecessors are never stored: the graph view derives
//! them from the successor records, one entry per record.
//!
//! The builder checks nothing. Valid functions come from the fixtures, and the
//! negative tests break them on purpose through the same calls.

use cfglib::ir::mlil::{self, Successor, ValueDef};
use cfglib::{DenseId, GraphView, RootedView};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Glint;

/// One operation with its immediate attributes.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Op {
    Const(i32),
    Add,
    Mul,
    Sub,
    /// Signed less-than, with a Boolean result.
    Lt,
    /// `operands = [condition, if_true, if_false]`.
    Select,
    Jump,
    /// `operands = [condition]`; the records are `Taken` then `NotTaken`.
    Branch,
    /// `operands = [scrutinee]`; one `Case` record per arm, then `Default`.
    Switch,
    /// `operands = [value]`.
    Return,
    Discard,
}

impl Op {
    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            Self::Jump | Self::Branch | Self::Switch | Self::Return | Self::Discard
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Ty {
    Bool,
    I32,
}

/// The metadata of one successor record.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Arm {
    Always,
    Taken,
    NotTaken,
    Case(i32),
    Default,
}

impl mlil::Dialect for Glint {
    type Op = Op;
    type Type = Ty;
    type Edge = Arm;
}

macro_rules! dense_handle {
    ($($name:ident),*) => {$(
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        pub struct $name(u16);

        impl DenseId for $name {
            fn from_index(index: usize) -> Self {
                Self(u16::try_from(index).expect("test functions are small"))
            }

            fn index(self) -> usize {
                usize::from(self.0)
            }
        }
    )*};
}

dense_handle!(Block, Inst, Value);

struct SuccessorData {
    target: Block,
    args: Vec<Value>,
    edge: Arm,
}

struct InstData {
    block: Block,
    op: Op,
    operands: Vec<Value>,
    results: Vec<Value>,
    successors: Vec<SuccessorData>,
}

struct ValueData {
    ty: Ty,
    def: ValueDef<Inst, Block>,
}

struct BlockData {
    params: Vec<Value>,
    insts: Vec<Inst>,
}

/// A Glint function. Block `0` is the entry block.
pub struct Function {
    blocks: Vec<Option<BlockData>>,
    insts: Vec<InstData>,
    values: Vec<ValueData>,
}

impl Function {
    pub fn new() -> Self {
        let mut function = Self {
            blocks: Vec::new(),
            insts: Vec::new(),
            values: Vec::new(),
        };
        function.add_block();
        function
    }

    pub fn entry(&self) -> Block {
        debug_assert!(self.blocks[0].is_some(), "the entry block is never removed");
        Block::from_index(0)
    }

    pub fn add_block(&mut self) -> Block {
        self.blocks.push(Some(BlockData {
            params: Vec::new(),
            insts: Vec::new(),
        }));
        Block::from_index(self.blocks.len() - 1)
    }

    pub fn add_param(&mut self, block: Block, ty: Ty) -> Value {
        let index = u32::try_from(self.block(block).params.len()).expect("few params");
        let value = self.new_value(ty, ValueDef::Param { block, index });
        self.block_mut(block).params.push(value);
        value
    }

    /// Appends an instruction without successors and returns its results.
    pub fn add_inst(
        &mut self,
        block: Block,
        op: Op,
        operands: &[Value],
        results: &[Ty],
    ) -> Vec<Value> {
        self.append(block, op, operands, results, Vec::new()).1
    }

    /// Appends a terminator with its successor records `(target, args, edge)`.
    pub fn terminate(
        &mut self,
        block: Block,
        op: Op,
        operands: &[Value],
        successors: Vec<(Block, Vec<Value>, Arm)>,
    ) -> Inst {
        let records = successors
            .into_iter()
            .map(|(target, args, edge)| SuccessorData { target, args, edge })
            .collect();
        self.append(block, op, operands, &[], records).0
    }

    /// Removes a block, leaving its slot, its instructions, and its values
    /// below the bounds.
    ///
    /// # Panics
    ///
    /// Panics when a live block still has a successor record that names it.
    pub fn remove_block(&mut self, block: Block) {
        assert_ne!(block, self.entry(), "the entry block stays");
        for (index, slot) in self.blocks.iter().enumerate() {
            let Some(live) = slot else { continue };
            for &inst in &live.insts {
                assert!(
                    self.insts[inst.index()]
                        .successors
                        .iter()
                        .all(|record| record.target != block),
                    "block {index} still branches to the removed block"
                );
            }
        }
        self.blocks[block.index()] = None;
    }

    /// Breaks the definition that `def` reports for `value`.
    pub fn override_def(&mut self, value: Value, def: ValueDef<Inst, Block>) {
        self.values[value.index()].def = def;
    }

    /// Breaks the block that `block_of` reports for `inst`.
    pub fn override_block_of(&mut self, inst: Inst, block: Block) {
        self.insts[inst.index()].block = block;
    }

    /// Lists `value` among the parameters of `block` too, so that two places
    /// define it.
    pub fn alias_param(&mut self, block: Block, value: Value) {
        self.block_mut(block).params.push(value);
    }

    fn append(
        &mut self,
        block: Block,
        op: Op,
        operands: &[Value],
        results: &[Ty],
        successors: Vec<SuccessorData>,
    ) -> (Inst, Vec<Value>) {
        let inst = Inst::from_index(self.insts.len());
        let values: Vec<Value> = results
            .iter()
            .enumerate()
            .map(|(index, &ty)| {
                let index = u32::try_from(index).expect("few results");
                self.new_value(ty, ValueDef::Result { inst, index })
            })
            .collect();
        self.insts.push(InstData {
            block,
            op,
            operands: operands.to_vec(),
            results: values.clone(),
            successors,
        });
        self.block_mut(block).insts.push(inst);
        (inst, values)
    }

    fn new_value(&mut self, ty: Ty, def: ValueDef<Inst, Block>) -> Value {
        self.values.push(ValueData { ty, def });
        Value::from_index(self.values.len() - 1)
    }

    fn block(&self, block: Block) -> &BlockData {
        self.blocks[block.index()]
            .as_ref()
            .expect("the block is live")
    }

    fn block_mut(&mut self, block: Block) -> &mut BlockData {
        self.blocks[block.index()]
            .as_mut()
            .expect("the block is live")
    }

    /// The records of the last instruction of `block`, if there is one.
    fn records(&self, block: Block) -> &[SuccessorData] {
        match self.block(block).insts.last() {
            Some(inst) => &self.insts[inst.index()].successors,
            None => &[],
        }
    }
}

impl GraphView for Function {
    type NodeId = Block;

    fn node_bound(&self) -> usize {
        self.blocks.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = Block> + '_ {
        (0..self.blocks.len())
            .filter(|&index| self.blocks[index].is_some())
            .map(Block::from_index)
    }

    fn successors(&self, node: Block) -> impl Iterator<Item = Block> + '_ {
        self.records(node).iter().map(|record| record.target)
    }

    /// One entry per successor record that names `node`, so parallel records
    /// stay parallel.
    fn predecessors(&self, node: Block) -> impl Iterator<Item = Block> + '_ {
        self.node_ids().flat_map(move |source| {
            self.records(source)
                .iter()
                .filter(move |record| record.target == node)
                .map(move |_| source)
        })
    }
}

impl RootedView for Function {
    fn root(&self) -> Block {
        self.entry()
    }
}

impl mlil::FunctionView<Glint> for Function {
    type Block = Block;
    type Inst = Inst;
    type Value = Value;

    fn inst_bound(&self) -> usize {
        self.insts.len()
    }

    fn value_bound(&self) -> usize {
        self.values.len()
    }

    fn params(&self, block: Block) -> &[Value] {
        &self.block(block).params
    }

    fn insts(&self, block: Block) -> &[Inst] {
        &self.block(block).insts
    }

    fn op(&self, inst: Inst) -> &Op {
        &self.insts[inst.index()].op
    }

    fn operands(&self, inst: Inst) -> &[Value] {
        &self.insts[inst.index()].operands
    }

    fn results(&self, inst: Inst) -> &[Value] {
        &self.insts[inst.index()].results
    }

    fn successors(
        &self,
        block: Block,
    ) -> impl ExactSizeIterator<Item = Successor<'_, Block, Value, Arm>> + '_ {
        self.records(block).iter().map(|record| Successor {
            target: record.target,
            args: &record.args,
            edge: &record.edge,
        })
    }

    fn block_of(&self, inst: Inst) -> Block {
        self.insts[inst.index()].block
    }

    fn def(&self, value: Value) -> ValueDef<Inst, Block> {
        self.values[value.index()].def
    }

    fn ty(&self, value: Value) -> &Ty {
        &self.values[value.index()].ty
    }
}
