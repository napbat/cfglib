//! A neutral description of a program, and the code that emits it through the
//! builder of any dialect.
//!
//! The model names roots by a small number, not by a dialect type, and states
//! what each instruction does to them. The oracles read the model; the
//! storages receive the same model through `InstructionBuilder`. A fact that
//! agrees with the oracle on both storages cannot depend on one vocabulary.

use std::collections::BTreeSet;

use cfglib::DenseId;
use cfglib::ir::rtl::{
    Action, BitSlice, Bits, Control, Dialect, FunctionView, InstructionBuilder, Place, Raise,
    Selection, SpanAction, WriteTarget,
};

use crate::bits;
use crate::contract::SealError;
use crate::kestrel::Kestrel;
use crate::kestrel_store::KestrelFunction;
use crate::tessera::{Tessera, TesseraFunction};

/// Neutral root ids: `0..=3` data roots, `4..=7` one-bit flags, `8` a wide
/// root, `9` the bank of a sparse space.
pub const ROOT_COUNT: u8 = 10;

/// The class of a root, which fixes its width inside every dialect.
pub fn class(id: u8) -> u8 {
    match id {
        0..=3 => 0,
        4..=7 => 1,
        8 => 2,
        _ => 3,
    }
}

/// What a dialect must supply so the neutral model can be emitted into it.
pub trait Target: Dialect {
    fn root_of(id: u8) -> Self::Root;
    fn id_of(root: Self::Root) -> u8;
    fn operator() -> Self::Operator;
    fn effect() -> Self::EffectOp;
    fn policy() -> Self::WritePolicy;
    fn outcome() -> Self::Outcome;
    fn domain() -> Self::NondetDomain;
    /// Wrapping sum of two values of one width.
    fn add() -> Self::Operator;
    /// Wrapping difference of two values of one width.
    fn sub() -> Self::Operator;
    fn and() -> Self::Operator;
    fn xor() -> Self::Operator;
    /// Equality of two values of one width, with a one-bit result.
    fn equal() -> Self::Operator;
    /// An operator that has no constant evaluation.
    fn opaque() -> Self::Operator;
}

/// One thing that an instruction does to the roots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Access {
    /// A direct read of a whole root.
    Read(u8),
    /// An indexed read: any candidate can be selected.
    ReadIndexed(Vec<u8>),
    /// A read of an element of a sparse bank.
    ReadSparse(u8),
    /// A write that replaces the whole root.
    WriteFull(u8),
    /// A write of the low byte that preserves the rest of the root.
    WritePartial(u8),
    /// An indexed write: one candidate is replaced, chosen at run time.
    WriteIndexed(Vec<u8>),
    /// A write of one element of a sparse bank.
    WriteSparse(u8),
    /// A guarded raise that publishes the root as a binding.
    RaiseBinding(u8),
    /// Statements that touch no root: a nondeterministic choice, an operator,
    /// and an effect.
    Noise,
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
    pub accesses: Vec<Access>,
    pub ending: Ending,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub insts: Vec<Inst>,
    pub successors: Vec<usize>,
}

#[derive(Clone, Debug)]
pub enum BlockSlot {
    Live(Block),
    /// A block that was removed after its instructions were emitted: the slot
    /// stays, and its records stay in the function.
    Removed {
        stale: Vec<Inst>,
    },
}

/// A function: slot `0` is the entry block.
#[derive(Clone, Debug)]
pub struct Model {
    pub slots: Vec<BlockSlot>,
}

/// The roots whose whole content a record's write may replace. A sparse bank
/// is opaque, so it does not count.
pub fn written_roots(access: &Access) -> Vec<u8> {
    match access {
        Access::WriteFull(id) | Access::WritePartial(id) => vec![*id],
        Access::WriteIndexed(ids) => ids.clone(),
        _ => Vec::new(),
    }
}

pub fn inst(accesses: Vec<Access>) -> Inst {
    Inst {
        accesses,
        ending: Ending::Advance,
    }
}

pub fn call(accesses: Vec<Access>) -> Inst {
    Inst {
        accesses,
        ending: Ending::Call,
    }
}

/// A live block. The last instruction ends the way its out-degree says,
/// unless the caller already chose a call or a trap.
pub fn block(mut insts: Vec<Inst>, successors: &[usize]) -> BlockSlot {
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
    BlockSlot::Live(Block {
        insts,
        successors: successors.to_vec(),
    })
}

pub fn removed(stale: Vec<Inst>) -> BlockSlot {
    BlockSlot::Removed { stale }
}

impl Model {
    /// Checks that the description is a legal input to every builder.
    pub fn new(slots: Vec<BlockSlot>) -> Self {
        let model = Self { slots };
        assert!(
            matches!(model.slots[0], BlockSlot::Live(_)),
            "entry is live"
        );
        for (_, block) in model.live() {
            for &target in &block.successors {
                assert!(model.is_live(target), "an edge reaches a removed block");
            }
        }
        for inst in model.all_insts() {
            let written: Vec<u8> = inst.accesses.iter().flat_map(written_roots).collect();
            let distinct: BTreeSet<u8> = written.iter().copied().collect();
            assert_eq!(
                written.len(),
                distinct.len(),
                "two writes of one record cover the same root"
            );
            let writes = inst.accesses.iter().any(|access| {
                matches!(
                    access,
                    Access::WriteFull(_)
                        | Access::WritePartial(_)
                        | Access::WriteIndexed(_)
                        | Access::WriteSparse(_)
                )
            });
            assert!(
                !(inst.ending == Ending::Trap && writes),
                "a trapping record holds no write"
            );
            for access in &inst.accesses {
                match access {
                    Access::ReadIndexed(ids) | Access::WriteIndexed(ids) => {
                        assert_ne!(ids.len(), 0, "an indexed access has a candidate");
                        assert!(ids.iter().all(|&id| class(id) == class(ids[0])));
                    }
                    Access::WritePartial(id) => assert!(matches!(class(*id), 0 | 2)),
                    _ => {}
                }
            }
        }
        model
    }

    pub fn is_live(&self, block: usize) -> bool {
        matches!(self.slots[block], BlockSlot::Live(_))
    }

    pub fn live(&self) -> impl Iterator<Item = (usize, &Block)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| match slot {
                BlockSlot::Live(block) => Some((index, block)),
                BlockSlot::Removed { .. } => None,
            })
    }

    fn all_insts(&self) -> impl Iterator<Item = &Inst> {
        self.slots.iter().flat_map(|slot| match slot {
            BlockSlot::Live(block) => block.insts.iter(),
            BlockSlot::Removed { stale } => stale.iter(),
        })
    }

    /// The live predecessors of `block`, once per edge.
    pub fn predecessors(&self, block: usize) -> Vec<usize> {
        self.live()
            .flat_map(|(source, live)| {
                live.successors
                    .iter()
                    .filter(move |&&target| target == block)
                    .map(move |_| source)
            })
            .collect()
    }

    /// The roots that live blocks name: the table the facts must build.
    pub fn named_roots(&self) -> BTreeSet<u8> {
        let mut named = BTreeSet::new();
        for (_, block) in self.live() {
            for inst in &block.insts {
                for access in &inst.accesses {
                    match access {
                        Access::Read(id)
                        | Access::ReadSparse(id)
                        | Access::WriteFull(id)
                        | Access::WritePartial(id)
                        | Access::WriteSparse(id)
                        | Access::RaiseBinding(id) => {
                            named.insert(*id);
                        }
                        Access::ReadIndexed(ids) | Access::WriteIndexed(ids) => {
                            named.extend(ids);
                        }
                        Access::Noise => {}
                    }
                }
            }
        }
        named
    }
}

/// An index value for an indexed access: one byte, zero.
fn index_value<D: Target, B: InstructionBuilder<D>>(builder: &mut B) -> B::Value {
    builder.constant(&[0], bits(8))
}

fn zero_value<D: Target, B: InstructionBuilder<D>>(builder: &mut B, width: Bits) -> B::Value {
    builder.constant(&vec![0; width.words()], width)
}

fn places<D: Target>(ids: &[u8]) -> Vec<Place<D>> {
    ids.iter().map(|&id| Place::whole(D::root_of(id))).collect()
}

fn emit_access<D: Target, B: InstructionBuilder<D>>(builder: &mut B, access: &Access) {
    match access {
        Access::Read(id) => {
            builder.read(Selection::Direct(Place::whole(D::root_of(*id))));
        }
        Access::ReadIndexed(ids) => {
            let index = index_value::<D, B>(builder);
            builder.read(Selection::FiniteIndexed {
                index,
                candidates: &places::<D>(ids),
            });
        }
        Access::ReadSparse(id) => {
            let bank = D::root_of(*id);
            let key = builder.constant(&[7], bits(16));
            let slice = BitSlice::whole(D::root_width(bank));
            builder.read(Selection::SparseIndexed { bank, key, slice });
        }
        Access::WriteFull(id) => {
            let root = D::root_of(*id);
            let width = D::root_width(root);
            let value = zero_value::<D, B>(builder, width);
            let plan = [SpanAction {
                span: BitSlice::whole(width),
                action: Action::TakeInput { value, offset: 0 },
            }];
            builder.write(WriteTarget::Root { root, plan: &plan }, D::policy());
        }
        Access::WritePartial(id) => {
            let root = D::root_of(*id);
            let width = D::root_width(root).width();
            let low = bits(8);
            let value = zero_value::<D, B>(builder, low);
            let plan = [
                SpanAction {
                    span: BitSlice::new(0, low),
                    action: Action::TakeInput { value, offset: 0 },
                },
                SpanAction {
                    span: BitSlice::new(8, bits(width - 8)),
                    action: Action::PreservePreState,
                },
            ];
            builder.write(WriteTarget::Root { root, plan: &plan }, D::policy());
        }
        Access::WriteIndexed(ids) => {
            let index = index_value::<D, B>(builder);
            let candidates = places::<D>(ids);
            let width = candidates[0].slice.width;
            let value = zero_value::<D, B>(builder, width);
            builder.write(
                WriteTarget::FiniteIndexed {
                    index,
                    candidates: &candidates,
                    value,
                },
                D::policy(),
            );
        }
        Access::WriteSparse(id) => {
            let bank = D::root_of(*id);
            let key = builder.constant(&[7], bits(16));
            let slice = BitSlice::whole(D::root_width(bank));
            let value = zero_value::<D, B>(builder, slice.width);
            builder.write(
                WriteTarget::SparseIndexed {
                    bank,
                    key,
                    slice,
                    value,
                },
                D::policy(),
            );
        }
        Access::RaiseBinding(id) => {
            let root = D::root_of(*id);
            let guard = builder.constant(&[1], Bits::BOOL);
            let bound = zero_value::<D, B>(builder, D::root_width(root));
            builder.raise(Raise {
                outcome: D::outcome(),
                guard: Some(guard),
                payload: &[],
                binds: &[(root, bound)],
            });
        }
        Access::Noise => {
            let seed = builder.constant(&[3], bits(8));
            let chosen = builder.nondet(bits(8), D::domain());
            let mixed = builder.apply(&D::operator(), &[seed, chosen], &[bits(8)]);
            builder.effect(&D::effect(), &[mixed.first], &[bits(8)], &[D::outcome()]);
        }
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
    for access in &inst.accesses {
        emit_access::<D, B>(&mut builder, access);
    }
    let control = emit_ending::<D, B>(&mut builder, inst.ending);
    builder.finish(control)
}

fn slot_insts(slot: &BlockSlot) -> &[Inst] {
    match slot {
        BlockSlot::Live(block) => &block.insts,
        BlockSlot::Removed { stale } => stale,
    }
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

    /// Removes a block and the edges that touch it, leaving its slot.
    fn remove(&mut self, block: Self::NodeId);

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

/// Builds the model in storage `S`: records first, then edges, then the
/// removal of the dead slots.
pub fn build<D: Target, S: Store<D>>(model: &Model) -> S {
    let mut function = S::start();
    let blocks: Vec<S::NodeId> = (0..model.slots.len())
        .map(|index| {
            if index == 0 {
                S::NodeId::from_index(0)
            } else {
                function.new_block()
            }
        })
        .collect();
    for (index, slot) in model.slots.iter().enumerate() {
        for inst in slot_insts(slot) {
            function
                .seal(blocks[index], |builder| {
                    emit_instruction::<D, _>(builder, inst)
                })
                .expect("the model emits legal records");
        }
    }
    for (index, block) in model.live() {
        for &target in &block.successors {
            function.connect(blocks[index], blocks[target]);
        }
    }
    for (index, slot) in model.slots.iter().enumerate() {
        if matches!(slot, BlockSlot::Removed { .. }) {
            function.remove(blocks[index]);
        }
    }
    function
}

/// Builds the model in the pooled Kestrel storage.
pub fn build_kestrel(model: &Model) -> KestrelFunction {
    build::<Kestrel, KestrelFunction>(model)
}

/// Builds the model in the owned Tessera storage.
pub fn build_tessera(model: &Model) -> TesseraFunction {
    build::<Tessera, TesseraFunction>(model)
}
