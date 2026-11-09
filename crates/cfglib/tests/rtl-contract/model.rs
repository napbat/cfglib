//! A neutral description of a program, and the code that emits it through the
//! builder of any dialect.
//!
//! The model states how each instruction ends. The storages receive the same
//! model through `InstructionBuilder`, so a test that reads it back from both
//! storages cannot depend on one vocabulary.

use cfglib::DenseId;
use cfglib::ir::rtl::{Bits, Control, Dialect, FunctionView, InstructionBuilder, Raise};

use crate::bits;
use crate::contract::SealError;

/// What a dialect must supply so the neutral model can be emitted into it.
pub trait Target: Dialect {
    fn root_of(id: u8) -> Self::Root;
    fn operator() -> Self::Operator;
    fn effect() -> Self::EffectOp;
    fn policy() -> Self::WritePolicy;
    fn outcome() -> Self::Outcome;
    fn domain() -> Self::NondetDomain;
}

/// How an instruction ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ending {
    Advance,
    Call,
    Jump,
    Branch,
    Dispatch,
    Return,
    /// An unconditional raise: the instruction never continues.
    Trap,
}

#[derive(Clone, Debug)]
pub struct Inst {
    pub ending: Ending,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub insts: Vec<Inst>,
    pub successors: Vec<usize>,
}

/// A function: block `0` is the entry block.
#[derive(Clone, Debug)]
pub struct Model {
    pub blocks: Vec<Block>,
}

/// A block. The last instruction ends the way its out-degree says, unless
/// the caller already chose a call or a trap.
pub fn block(mut insts: Vec<Inst>, successors: &[usize]) -> Block {
    if let Some(last) = insts.last_mut()
        && last.ending == Ending::Advance
    {
        last.ending = match successors.len() {
            0 => Ending::Return,
            1 => Ending::Advance,
            2 => Ending::Branch,
            _ => Ending::Dispatch,
        };
    }
    Block {
        insts,
        successors: successors.to_vec(),
    }
}

impl Model {
    /// Checks that the description is a legal input to every builder.
    pub fn new(blocks: Vec<Block>) -> Self {
        assert!(!blocks.is_empty(), "the entry block exists");
        for block in &blocks {
            for &target in &block.successors {
                assert!(target < blocks.len(), "an edge reaches a block");
            }
        }
        Self { blocks }
    }
}

fn emit_ending<D: Target, B: InstructionBuilder<D>>(
    builder: &mut B,
    ending: Ending,
) -> Control<B::Value> {
    let address = |builder: &mut B| builder.constant(&[0x40], bits(32));
    match ending {
        Ending::Advance => Control::Advance,
        Ending::Call => Control::Call {
            target: address(builder),
        },
        Ending::Jump => Control::Jump {
            target: address(builder),
        },
        Ending::Return => Control::Return {
            target: address(builder),
        },
        Ending::Branch => {
            let condition = builder.constant(&[1], Bits::BOOL);
            Control::Branch {
                condition,
                taken: address(builder),
                not_taken: None,
            }
        }
        Ending::Dispatch => Control::Dispatch {
            scrutinee: address(builder),
        },
        Ending::Trap => {
            builder.raise(Raise {
                outcome: D::outcome(),
                guard: None,
                payload: &[],
                binds: &[],
            });
            Control::Terminating
        }
    }
}

/// Emits one instruction through `builder` and seals it.
///
/// # Errors
///
/// Returns the error of the builder when the record breaks the contract.
pub fn emit_instruction<D: Target, B: InstructionBuilder<D>>(
    mut builder: B,
    inst: &Inst,
) -> Result<(), B::Error> {
    let control = emit_ending::<D, B>(&mut builder, inst.ending);
    builder.finish(control)
}

/// The storage side of a test dialect: a function that grows one block, edge,
/// and sealed record at a time.
///
/// Both test storages implement it, so a program is written once, against the
/// builder trait, and runs in every storage.
pub trait Store<D: Target>: FunctionView<D> + Sized {
    type Builder<'f>: InstructionBuilder<D, Value = Self::Value, Error = SealError>
    where
        Self: 'f;

    /// A function that holds only its entry block.
    fn start() -> Self;

    fn new_block(&mut self) -> Self::NodeId;

    fn connect(&mut self, from: Self::NodeId, to: Self::NodeId);

    /// Seals one record at the end of `block`.
    ///
    /// # Errors
    ///
    /// Returns the error that the builder reports.
    fn seal(
        &mut self,
        block: Self::NodeId,
        produce: impl FnOnce(Self::Builder<'_>) -> Result<(), SealError>,
    ) -> Result<Self::Inst, SealError>;
}

/// Builds the model in storage `S`: records first, then edges.
pub fn build<D: Target, S: Store<D>>(model: &Model) -> S {
    let mut function = S::start();
    let blocks: Vec<S::NodeId> = (0..model.blocks.len())
        .map(|index| {
            if index == 0 {
                S::NodeId::from_index(0)
            } else {
                function.new_block()
            }
        })
        .collect();
    for (index, block) in model.blocks.iter().enumerate() {
        for inst in &block.insts {
            function
                .seal(blocks[index], |builder| {
                    emit_instruction::<D, _>(builder, inst)
                })
                .expect("the model emits legal records");
        }
    }
    for (index, block) in model.blocks.iter().enumerate() {
        for &target in &block.successors {
            function.connect(blocks[index], blocks[target]);
        }
    }
    function
}
