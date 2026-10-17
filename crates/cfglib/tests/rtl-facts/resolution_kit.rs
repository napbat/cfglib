//! Emitters shared by the selection-resolution tests.
//!
//! Every helper is written against the builder trait and the neutral root
//! numbering, so one scenario runs in every dialect. The data roots `0..=3`
//! and the bank root `9` all have the data width of their dialect.

use cfglib::ir::rtl::{
    Action, BitSlice, Bits, Fold, FunctionView, InstructionBuilder, Place, RecordView,
    RootConstant, RootTable, Selection, SelectionResolution, SpanAction, Statement, WriteTarget,
};

use crate::model::{Store, Target};

/// The width of the data roots of the dialect.
pub fn data<D: Target>() -> Bits {
    D::root_width(D::root_of(0))
}

/// A constant of `width` bits whose low word is `value`.
pub fn word<D: Target, B: InstructionBuilder<D>>(
    builder: &mut B,
    value: u64,
    width: Bits,
) -> B::Value {
    let mut words = vec![0; width.words()];
    words[0] = value;
    builder.constant(&words, width)
}

pub fn read_root<D: Target, B: InstructionBuilder<D>>(builder: &mut B, id: u8) -> B::Value {
    builder.read(Selection::Direct(Place::whole(D::root_of(id))))
}

/// `value & mask` at the data width.
pub fn masked<D: Target, B: InstructionBuilder<D>>(
    builder: &mut B,
    value: B::Value,
    mask: u64,
) -> B::Value {
    let width = data::<D>();
    let mask = word::<D, B>(builder, mask, width);
    builder.apply(&D::and(), &[value, mask], &[width]).first
}

/// `(root + add) & mask`, the way a rotating register window computes a slot.
pub fn computed_index<D: Target, B: InstructionBuilder<D>>(
    builder: &mut B,
    id: u8,
    add: u64,
    mask: u64,
) -> B::Value {
    let width = data::<D>();
    let top = read_root::<D, B>(builder, id);
    let offset = word::<D, B>(builder, add, width);
    let sum = builder.apply(&D::add(), &[top, offset], &[width]).first;
    masked::<D, B>(builder, sum, mask)
}

/// `count` places of equal width, for the candidates of a read.
pub fn places<D: Target>(count: usize) -> Vec<Place<D>> {
    (0..count)
        .map(|position| {
            let id = u8::try_from(position % 4).expect("below four");
            Place::whole(D::root_of(id))
        })
        .collect()
}

/// Reads the candidate that `index` selects out of `count` places.
pub fn select<D: Target, B: InstructionBuilder<D>>(builder: &mut B, index: B::Value, count: usize) {
    builder.read(Selection::FiniteIndexed {
        index,
        candidates: &places::<D>(count),
    });
}

/// Replaces the whole of root `id` with `value`.
pub fn write_value<D: Target, B: InstructionBuilder<D>>(builder: &mut B, id: u8, value: B::Value) {
    let root = D::root_of(id);
    let plan = [SpanAction {
        span: BitSlice::whole(D::root_width(root)),
        action: Action::TakeInput { value, offset: 0 },
    }];
    builder.write(WriteTarget::Root { root, plan: &plan }, D::policy());
}

/// Replaces the whole of root `id` with the constant `value`.
pub fn set_root<D: Target, B: InstructionBuilder<D>>(builder: &mut B, id: u8, value: u64) {
    let value = word::<D, B>(builder, value, D::root_width(D::root_of(id)));
    write_value::<D, B>(builder, id, value);
}

/// The positions of the indexed reads and writes of one record.
pub fn selection_positions<D: Target, F: FunctionView<D>>(
    function: &F,
    inst: F::Inst,
) -> Vec<usize> {
    function
        .record(inst)
        .statements()
        .enumerate()
        .filter(|(_, statement)| {
            matches!(
                statement,
                Statement::Read {
                    source: Selection::FiniteIndexed { .. },
                    ..
                } | Statement::Write {
                    target: WriteTarget::FiniteIndexed { .. },
                    ..
                }
            )
        })
        .map(|(position, _)| position)
        .collect()
}

/// Runs the resolution with the given roots known on entry.
pub fn resolve<D: Target + Fold, S: Store<D>>(
    function: &S,
    entry: &[(u8, &[u64])],
) -> SelectionResolution<S::Inst> {
    let roots = RootTable::collect(function);
    let constants: Vec<RootConstant<'_, D>> = entry
        .iter()
        .map(|&(id, words)| RootConstant {
            root: D::root_of(id),
            words,
        })
        .collect();
    SelectionResolution::compute(function, &roots, &constants)
}

/// What `candidate` should say for an index below `count`, or none.
pub fn expect_candidate(index: u64, count: usize) -> Option<usize> {
    let index = usize::try_from(index).ok()?;
    (index < count).then_some(index)
}
