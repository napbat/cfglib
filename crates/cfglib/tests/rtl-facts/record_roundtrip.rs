//! `InstructionBuilder` to sealed records read back through `RecordView`.
//!
//! One producer, written against the builder trait only, drives both
//! storages. A recording wrapper keeps an owned copy of everything the
//! producer passed in, so the statements that the view yields can be compared
//! with the input, value handle for value handle.

use cfglib::DenseId;
use cfglib::ir::rtl::{
    Action, BitSlice, Bits, Control, Dialect, FunctionView, InstructionBuilder, Place, Raise,
    RecordView, Selection, SpanAction, Statement, ValueRange, WriteTarget,
};

use crate::bits;
use crate::contract::SealError;
use crate::kestrel::Kestrel;
use crate::kestrel_store::KestrelFunction;
use crate::model::Target;
use crate::tessera::{Tessera, TesseraFunction};

#[derive(Debug, Clone, PartialEq)]
pub enum OwnedSelection<D: Dialect, V> {
    Direct(Place<D>),
    Finite {
        index: V,
        candidates: Vec<Place<D>>,
    },
    Sparse {
        bank: D::Root,
        key: V,
        slice: BitSlice,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum OwnedTarget<D: Dialect, V> {
    Root {
        root: D::Root,
        plan: Vec<SpanAction<D, V>>,
    },
    Finite {
        index: V,
        candidates: Vec<Place<D>>,
        value: V,
    },
    Sparse {
        bank: D::Root,
        key: V,
        slice: BitSlice,
        value: V,
    },
}

/// A statement with owned parts, comparable with what a view lends.
#[derive(Debug, Clone, PartialEq)]
pub enum Owned<D: Dialect, V> {
    Read {
        result: V,
        source: OwnedSelection<D, V>,
    },
    Constant {
        result: V,
        words: Vec<u64>,
        width: Bits,
    },
    Apply {
        results: ValueRange<V>,
        operator: D::Operator,
        args: Vec<V>,
    },
    Nondet {
        result: V,
        width: Bits,
        domain: D::NondetDomain,
    },
    Effect {
        results: ValueRange<V>,
        operation: D::EffectOp,
        args: Vec<V>,
        outcomes: Vec<D::Outcome>,
    },
    Raise {
        outcome: D::Outcome,
        guard: Option<V>,
        payload: Vec<V>,
        binds: Vec<(D::Root, V)>,
    },
    Write {
        target: OwnedTarget<D, V>,
        policy: D::WritePolicy,
    },
}

fn own_source<D: Dialect, V: Copy>(source: Selection<'_, D, V>) -> OwnedSelection<D, V> {
    match source {
        Selection::Direct(place) => OwnedSelection::Direct(place),
        Selection::FiniteIndexed { index, candidates } => OwnedSelection::Finite {
            index,
            candidates: candidates.to_vec(),
        },
        Selection::SparseIndexed { bank, key, slice } => {
            OwnedSelection::Sparse { bank, key, slice }
        }
    }
}

fn own_target<D: Dialect, V: Copy>(target: WriteTarget<'_, D, V>) -> OwnedTarget<D, V> {
    match target {
        WriteTarget::Root { root, plan } => OwnedTarget::Root {
            root,
            plan: plan.to_vec(),
        },
        WriteTarget::FiniteIndexed {
            index,
            candidates,
            value,
        } => OwnedTarget::Finite {
            index,
            candidates: candidates.to_vec(),
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
    }
}

/// Copies a lent statement.
pub fn own<D: Dialect, V: Copy>(statement: &Statement<'_, D, V>) -> Owned<D, V> {
    match *statement {
        Statement::Read { result, source } => Owned::Read {
            result,
            source: own_source(source),
        },
        Statement::Constant {
            result,
            words,
            width,
        } => Owned::Constant {
            result,
            words: words.to_vec(),
            width,
        },
        Statement::Apply {
            results,
            operator,
            args,
        } => Owned::Apply {
            results,
            operator: operator.clone(),
            args: args.to_vec(),
        },
        Statement::Nondet {
            result,
            width,
            domain,
        } => Owned::Nondet {
            result,
            width,
            domain,
        },
        Statement::Effect {
            results,
            operation,
            args,
            outcomes,
        } => Owned::Effect {
            results,
            operation: operation.clone(),
            args: args.to_vec(),
            outcomes: outcomes.to_vec(),
        },
        Statement::Raise(raise) => Owned::Raise {
            outcome: raise.outcome,
            guard: raise.guard,
            payload: raise.payload.to_vec(),
            binds: raise.binds.to_vec(),
        },
        Statement::Write { target, policy } => Owned::Write {
            target: own_target(target),
            policy,
        },
    }
}

/// Forwards every call to a builder and keeps an owned copy of the input.
struct Recording<D: Dialect, B: InstructionBuilder<D>> {
    inner: B,
    statements: Vec<Owned<D, B::Value>>,
    widths: Vec<(B::Value, Bits)>,
}

impl<D: Dialect, B: InstructionBuilder<D>> Recording<D, B> {
    fn new(inner: B) -> Self {
        Self {
            inner,
            statements: Vec::new(),
            widths: Vec::new(),
        }
    }

    fn read(&mut self, source: Selection<'_, D, B::Value>, width: Bits) -> B::Value {
        let owned = own_source(source);
        let result = self.inner.read(source);
        self.widths.push((result, width));
        self.statements.push(Owned::Read {
            result,
            source: owned,
        });
        result
    }

    fn constant(&mut self, words: &[u64], width: Bits) -> B::Value {
        let result = self.inner.constant(words, width);
        self.widths.push((result, width));
        self.statements.push(Owned::Constant {
            result,
            words: words.to_vec(),
            width,
        });
        result
    }

    fn apply(
        &mut self,
        operator: &D::Operator,
        args: &[B::Value],
        results: &[Bits],
    ) -> ValueRange<B::Value> {
        let range = self.inner.apply(operator, args, results);
        self.widths
            .extend(range.iter().zip(results.iter().copied()));
        self.statements.push(Owned::Apply {
            results: range,
            operator: operator.clone(),
            args: args.to_vec(),
        });
        range
    }

    fn nondet(&mut self, width: Bits, domain: D::NondetDomain) -> B::Value {
        let result = self.inner.nondet(width, domain);
        self.widths.push((result, width));
        self.statements.push(Owned::Nondet {
            result,
            width,
            domain,
        });
        result
    }

    fn effect(
        &mut self,
        operation: &D::EffectOp,
        args: &[B::Value],
        results: &[Bits],
        outcomes: &[D::Outcome],
    ) -> ValueRange<B::Value> {
        let range = self.inner.effect(operation, args, results, outcomes);
        self.widths
            .extend(range.iter().zip(results.iter().copied()));
        self.statements.push(Owned::Effect {
            results: range,
            operation: operation.clone(),
            args: args.to_vec(),
            outcomes: outcomes.to_vec(),
        });
        range
    }

    fn raise(&mut self, raise: Raise<'_, D, B::Value>) {
        self.statements.push(Owned::Raise {
            outcome: raise.outcome,
            guard: raise.guard,
            payload: raise.payload.to_vec(),
            binds: raise.binds.to_vec(),
        });
        self.inner.raise(raise);
    }

    fn write(&mut self, target: WriteTarget<'_, D, B::Value>, policy: D::WritePolicy) {
        self.statements.push(Owned::Write {
            target: own_target(target),
            policy,
        });
        self.inner.write(target, policy);
    }
}

/// What the producer emitted and what a faithful view must return.
struct Expected<D: Dialect, V> {
    statements: Vec<Owned<D, V>>,
    widths: Vec<(V, Bits)>,
    control: Control<V>,
}

/// Emits one record with every statement shape, every selection, every write
/// target, and a multi-way branch, through the builder trait only.
fn kitchen_sink<D: Target, B: InstructionBuilder<D>>(
    builder: B,
) -> Result<Expected<D, B::Value>, B::Error> {
    let root = D::root_of;
    let width_of = |id: u8| D::root_width(root(id));
    let mut rec = Recording::new(builder);

    let whole = rec.read(Selection::Direct(Place::whole(root(0))), width_of(0));
    let field = Place {
        root: root(1),
        slice: BitSlice::new(8, bits(16)),
    };
    let part = rec.read(Selection::Direct(field), bits(16));
    let index = rec.constant(&[2], bits(8));
    let candidates = [
        Place::whole(root(0)),
        Place::whole(root(1)),
        Place::whole(root(2)),
    ];
    let picked = rec.read(
        Selection::FiniteIndexed {
            index,
            candidates: &candidates,
        },
        width_of(0),
    );
    let key = rec.constant(&[0x1b], bits(16));
    let element = BitSlice::new(0, bits(8));
    let from_bank = rec.read(
        Selection::SparseIndexed {
            bank: root(9),
            key,
            slice: element,
        },
        bits(8),
    );
    let wide = rec.constant(&[0xdead_beef, 0, 0, 0xff], bits(200));
    let sum = rec.apply(&D::operator(), &[part, from_bank], &[bits(16), Bits::BOOL]);
    let loaded = rec.effect(
        &D::effect(),
        &[sum.first],
        &[bits(32), bits(8)],
        &[D::outcome()],
    );
    let choice = rec.nondet(bits(4), D::domain());
    let guard = rec.constant(&[1], Bits::BOOL);
    let flag = rec.constant(&[0], Bits::BOOL);
    rec.raise(Raise {
        outcome: D::outcome(),
        guard: Some(guard),
        payload: &[choice, loaded.first, whole, wide],
        binds: &[(root(4), flag)],
    });

    // Keep the low byte, replace the middle, take a fresh choice for the rest.
    let rest = width_of(1).width() - 24;
    let plan = [
        SpanAction {
            span: BitSlice::new(0, bits(8)),
            action: Action::PreservePreState,
        },
        SpanAction {
            span: BitSlice::new(8, bits(16)),
            action: Action::TakeInput {
                value: part,
                offset: 0,
            },
        },
        SpanAction {
            span: BitSlice::new(24, bits(rest)),
            action: Action::Nondeterministic(D::domain()),
        },
    ];
    rec.write(
        WriteTarget::Root {
            root: root(1),
            plan: &plan,
        },
        D::policy(),
    );
    let targets = [Place::whole(root(2)), Place::whole(root(3))];
    rec.write(
        WriteTarget::FiniteIndexed {
            index,
            candidates: &targets,
            value: picked,
        },
        D::policy(),
    );
    let byte = rec.constant(&[0x7f], bits(8));
    rec.write(
        WriteTarget::SparseIndexed {
            bank: root(9),
            key,
            slice: element,
            value: byte,
        },
        D::policy(),
    );

    let taken = rec.constant(&[0x1000], bits(32));
    let fallthrough = rec.constant(&[0x1004], bits(32));
    let control = Control::Branch {
        condition: guard,
        taken,
        not_taken: Some(fallthrough),
    };
    let Recording {
        inner,
        statements,
        widths,
    } = rec;
    inner.finish(control)?;
    Ok(Expected {
        statements,
        widths,
        control,
    })
}

fn assert_reads_back<D: Target, F: FunctionView<D>>(
    function: &F,
    inst: F::Inst,
    expected: &Expected<D, F::Value>,
) {
    let record = function.record(inst);
    let statements = record.statements();
    assert_eq!(statements.len(), expected.statements.len());
    let actual: Vec<_> = statements.map(|statement| own(&statement)).collect();
    assert_eq!(actual, expected.statements);
    assert_eq!(record.control(), expected.control);
    for &(value, width) in &expected.widths {
        assert_eq!(
            record.width(value),
            width,
            "width of value {}",
            value.index()
        );
        assert!(value.index() < function.value_bound());
    }
}

#[test]
fn kestrel_storage_reads_back_every_statement_shape_unchanged() {
    let mut function = KestrelFunction::new();
    let entry = function.entry();
    let mut expected = None;
    let inst = function
        .append(entry, |builder| {
            expected = Some(kitchen_sink::<Kestrel, _>(builder)?);
            Ok(())
        })
        .expect("the record seals");
    assert_reads_back(&function, inst, &expected.expect("sealed"));
    assert_eq!(function.instructions(entry), [inst]);
}

#[test]
fn tessera_storage_reads_back_every_statement_shape_unchanged() {
    let mut function = TesseraFunction::new();
    let mut expected = None;
    let inst = function
        .append(0, |builder| {
            expected = Some(kitchen_sink::<Tessera, _>(builder)?);
            Ok(())
        })
        .expect("the record seals");
    assert_reads_back(&function, inst, &expected.expect("sealed"));
    assert_eq!(function.instructions(0), [inst]);
}

#[test]
fn instructions_keep_their_order_and_handles_stay_dense_across_records() {
    let mut function = KestrelFunction::new();
    let entry = function.entry();
    let mut insts = Vec::new();
    let mut handles = Vec::new();
    for round in 0..3 {
        let inst = function
            .append(entry, |mut builder| {
                let first = builder.constant(&[round], bits(8));
                let second = builder.constant(&[round + 1], bits(8));
                handles.push((first, second));
                builder.finish(Control::Advance)
            })
            .expect("seals");
        insts.push(inst);
    }
    assert_eq!(function.instructions(entry), insts);
    assert_eq!(function.inst_bound(), 3);
    assert_eq!(function.value_bound(), 6);
    let mut indices: Vec<usize> = handles
        .iter()
        .flat_map(|&(a, b)| [a.index(), b.index()])
        .collect();
    indices.sort_unstable();
    assert_eq!(indices, [0, 1, 2, 3, 4, 5]);
}

#[test]
fn value_ranges_hold_consecutive_handles() {
    let range = ValueRange::new(5_u32, 3);
    assert_eq!(range.iter().collect::<Vec<_>>(), [5, 6, 7]);
    assert_eq!(range.iter().len(), 3);
    assert_eq!(
        (range.get(0), range.get(2), range.get(3)),
        (Some(5), Some(7), None)
    );
    let empty = ValueRange::new(9_u32, 0);
    assert_eq!(empty.iter().next(), None);
    assert_eq!(empty.get(0), None);
}

#[test]
fn a_multi_result_operator_returns_one_handle_per_result_in_order() {
    let mut function = KestrelFunction::new();
    let entry = function.entry();
    function
        .append(entry, |mut builder| {
            let a = builder.constant(&[1], bits(32));
            let b = builder.constant(&[2], bits(32));
            let both = builder.apply(
                &crate::kestrel::KOp::AddCarry,
                &[a, b],
                &[bits(32), Bits::BOOL],
            );
            assert_eq!(both.len, 2);
            assert_eq!(
                both.get(1).map(DenseId::index),
                Some(both.first.index() + 1)
            );
            builder.finish(Control::Advance)
        })
        .expect("seals");
}

/// The contract violations and the one legal record next to each.
#[derive(Clone, Copy)]
enum Case {
    UnknownOperand,
    PlanWithAGap,
    PlanThatStopsShort,
    PlanThatSkipsTheLowBits,
    TwoRootWritesOfOneRoot,
    RootWriteBesideAnIndexedWriteOfIt,
    IndexedWritesOfDisjointSlices,
    NonBooleanRaiseGuard,
    NonBooleanBranchCondition,
    UnconditionalRaiseThatAdvances,
    StatementAfterAnUnconditionalRaise,
    WriteBesideAnUnconditionalRaise,
    UnconditionalRaiseThatTerminates,
    DuplicateBinding,
    BindingOfTheWrongWidth,
    GuardedRaiseBesideAWrite,
}

const CASES: [(Case, Result<(), SealError>); 16] = [
    (Case::UnknownOperand, Err(SealError::UnknownOperand(999))),
    (Case::PlanWithAGap, Err(SealError::IncompletePlan)),
    (Case::PlanThatStopsShort, Err(SealError::IncompletePlan)),
    (
        Case::PlanThatSkipsTheLowBits,
        Err(SealError::IncompletePlan),
    ),
    (
        Case::TwoRootWritesOfOneRoot,
        Err(SealError::OverlappingWrites),
    ),
    (
        Case::RootWriteBesideAnIndexedWriteOfIt,
        Err(SealError::OverlappingWrites),
    ),
    (Case::IndexedWritesOfDisjointSlices, Ok(())),
    (Case::NonBooleanRaiseGuard, Err(SealError::GuardNotBoolean)),
    (
        Case::NonBooleanBranchCondition,
        Err(SealError::GuardNotBoolean),
    ),
    (
        Case::UnconditionalRaiseThatAdvances,
        Err(SealError::RaiseNotTerminal),
    ),
    (
        Case::StatementAfterAnUnconditionalRaise,
        Err(SealError::StatementAfterRaise),
    ),
    (
        Case::WriteBesideAnUnconditionalRaise,
        Err(SealError::WriteBesideRaise),
    ),
    (Case::UnconditionalRaiseThatTerminates, Ok(())),
    (Case::DuplicateBinding, Err(SealError::DuplicateBinding)),
    (Case::BindingOfTheWrongWidth, Err(SealError::BindingWidth)),
    (Case::GuardedRaiseBesideAWrite, Ok(())),
];

/// A span of `width` bits at `offset` that takes a fresh constant.
fn take<D: Target, B: InstructionBuilder<D>>(
    builder: &mut B,
    offset: u16,
    width: u16,
) -> SpanAction<D, B::Value> {
    let value = builder.constant(&vec![0; bits(width).words()], bits(width));
    SpanAction {
        span: BitSlice::new(offset, bits(width)),
        action: Action::TakeInput { value, offset: 0 },
    }
}

fn write_plan<D: Target, B: InstructionBuilder<D>>(
    builder: &mut B,
    id: u8,
    plan: &[SpanAction<D, B::Value>],
) {
    builder.write(
        WriteTarget::Root {
            root: D::root_of(id),
            plan,
        },
        D::policy(),
    );
}

fn full_plan<D: Target, B: InstructionBuilder<D>>(builder: &mut B, id: u8) {
    let width = D::root_width(D::root_of(id)).width();
    let span = take::<D, B>(builder, 0, width);
    write_plan::<D, B>(builder, id, &[span]);
}

fn raise_with<D: Target, B: InstructionBuilder<D>>(
    builder: &mut B,
    guard: Option<B::Value>,
    binds: &[(D::Root, B::Value)],
) {
    builder.raise(Raise {
        outcome: D::outcome(),
        guard,
        payload: &[],
        binds,
    });
}

/// Emits the record of `case` and seals it.
fn produce<D: Target, B: InstructionBuilder<D>>(mut b: B, case: Case) -> Result<(), B::Error> {
    let root = D::root_of;
    let width = |id: u8| D::root_width(root(id)).width();
    let control = match case {
        Case::UnknownOperand => {
            let stray = B::Value::from_index(999);
            let candidates = [Place::whole(root(0)), Place::whole(root(1))];
            b.read(Selection::FiniteIndexed {
                index: stray,
                candidates: &candidates,
            });
            Control::Advance
        }
        Case::PlanWithAGap => {
            let low = take::<D, B>(&mut b, 0, 8);
            let high = take::<D, B>(&mut b, 16, width(0) - 16);
            write_plan::<D, B>(&mut b, 0, &[low, high]);
            Control::Advance
        }
        Case::PlanThatStopsShort => {
            let low = take::<D, B>(&mut b, 0, 8);
            write_plan::<D, B>(&mut b, 0, &[low]);
            Control::Advance
        }
        Case::PlanThatSkipsTheLowBits => {
            let high = take::<D, B>(&mut b, 8, width(0) - 8);
            write_plan::<D, B>(&mut b, 0, &[high]);
            Control::Advance
        }
        Case::TwoRootWritesOfOneRoot => {
            full_plan::<D, B>(&mut b, 1);
            full_plan::<D, B>(&mut b, 1);
            Control::Advance
        }
        Case::RootWriteBesideAnIndexedWriteOfIt => {
            full_plan::<D, B>(&mut b, 1);
            let index = b.constant(&[0], bits(8));
            let candidates = [Place::whole(root(0)), Place::whole(root(1))];
            let value = b.constant(&vec![0; bits(width(0)).words()], bits(width(0)));
            b.write(
                WriteTarget::FiniteIndexed {
                    index,
                    candidates: &candidates,
                    value,
                },
                D::policy(),
            );
            Control::Advance
        }
        Case::IndexedWritesOfDisjointSlices => {
            let index = b.constant(&[0], bits(8));
            let low = [Place {
                root: root(0),
                slice: BitSlice::new(0, bits(16)),
            }];
            let high = [Place {
                root: root(0),
                slice: BitSlice::new(16, bits(16)),
            }];
            let value = b.constant(&[0], bits(16));
            for candidates in [&low, &high] {
                b.write(
                    WriteTarget::FiniteIndexed {
                        index,
                        candidates,
                        value,
                    },
                    D::policy(),
                );
            }
            Control::Advance
        }
        Case::NonBooleanRaiseGuard => {
            let guard = b.constant(&[1], bits(8));
            raise_with::<D, B>(&mut b, Some(guard), &[]);
            Control::Advance
        }
        Case::NonBooleanBranchCondition => {
            let condition = b.constant(&[1], bits(8));
            let taken = b.constant(&[8], bits(32));
            Control::Branch {
                condition,
                taken,
                not_taken: None,
            }
        }
        Case::UnconditionalRaiseThatAdvances => {
            raise_with::<D, B>(&mut b, None, &[]);
            Control::Advance
        }
        Case::StatementAfterAnUnconditionalRaise => {
            raise_with::<D, B>(&mut b, None, &[]);
            b.constant(&[1], bits(8));
            Control::Terminating
        }
        Case::WriteBesideAnUnconditionalRaise => {
            full_plan::<D, B>(&mut b, 0);
            raise_with::<D, B>(&mut b, None, &[]);
            Control::Terminating
        }
        Case::UnconditionalRaiseThatTerminates => {
            b.read(Selection::Direct(Place::whole(root(0))));
            raise_with::<D, B>(&mut b, None, &[]);
            Control::Terminating
        }
        Case::DuplicateBinding => {
            let value = b.constant(&[0], Bits::BOOL);
            raise_with::<D, B>(&mut b, None, &[(root(4), value), (root(4), value)]);
            Control::Terminating
        }
        Case::BindingOfTheWrongWidth => {
            let value = b.constant(&[0], bits(8));
            raise_with::<D, B>(&mut b, None, &[(root(4), value)]);
            Control::Terminating
        }
        Case::GuardedRaiseBesideAWrite => {
            full_plan::<D, B>(&mut b, 0);
            let guard = b.constant(&[1], Bits::BOOL);
            raise_with::<D, B>(&mut b, Some(guard), &[]);
            Control::Advance
        }
    };
    b.finish(control)
}

#[test]
fn the_consumer_reports_each_contract_violation_when_it_seals_a_record_in_kestrel() {
    for (case, expected) in CASES {
        let mut function = KestrelFunction::new();
        let entry = function.entry();
        let result = function.append(entry, |builder| produce::<Kestrel, _>(builder, case));
        assert_eq!(result.map(|_| ()), expected);
        let sealed = usize::from(expected.is_ok());
        assert_eq!(function.instructions(entry).len(), sealed);
        assert_eq!(function.inst_bound(), sealed);
    }
}

#[test]
fn the_consumer_reports_each_contract_violation_when_it_seals_a_record_in_tessera() {
    for (case, expected) in CASES {
        let mut function = TesseraFunction::new();
        let result = function.append(0, |builder| produce::<Tessera, _>(builder, case));
        assert_eq!(result.map(|_| ()), expected);
        let sealed = usize::from(expected.is_ok());
        assert_eq!(function.instructions(0).len(), sealed);
        assert_eq!(function.inst_bound(), sealed);
    }
}

/// The name of the variant that a sealed record ends with.
fn boundary<V>(control: &Control<V>) -> &'static str {
    match control {
        Control::Advance => "advance",
        Control::Jump { .. } => "jump",
        Control::Branch { .. } => "branch",
        Control::Call { .. } => "call",
        Control::Return { .. } => "return",
        Control::Dispatch { .. } => "dispatch",
        Control::Terminating => "terminating",
    }
}

in_both_dialects! {
    fn every_control_boundary_is_read_back_as_it_was_sealed<D, S>() {
        use crate::model::{Ending, Inst, Model, block, build};

        let endings = [
            Ending::Advance,
            Ending::Call,
            Ending::Jump,
            Ending::Branch,
            Ending::Dispatch,
            Ending::Return,
            Ending::Trap,
        ];
        let insts: Vec<Inst> = endings
            .map(|ending| Inst { accesses: Vec::new(), ending })
            .to_vec();
        let function: S = build::<D, S>(&Model::new(vec![block(insts, &[])]));
        let entry = S::NodeId::from_index(0);
        let names: Vec<_> = function
            .instructions(entry)
            .iter()
            .map(|&inst| boundary(&function.record(inst).control()))
            .collect();
        assert_eq!(
            names,
            ["advance", "call", "jump", "branch", "dispatch", "return", "terminating"]
        );

        let record_of = |position: usize| function.record(function.instructions(entry)[position]);
        // A branch names its taken target and no other target, so a not-taken
        // arm continues at the next instruction.
        let Control::Branch { condition, taken, not_taken } = record_of(3).control() else {
            unreachable!("position three holds a branch");
        };
        assert_eq!(not_taken, None);
        assert_eq!(record_of(3).width(condition), Bits::BOOL);
        assert_eq!(record_of(3).width(taken), bits(32));
        // A call and a jump carry an address value.
        let Control::Call { target } = record_of(1).control() else {
            unreachable!("position one holds a call");
        };
        assert_eq!(record_of(1).width(target), bits(32));
        // The record that terminates ends in an unconditional raise.
        let terminating = record_of(6);
        let statements: Vec<_> = terminating.statements().collect();
        assert!(matches!(
            statements.last(),
            Some(Statement::Raise(raise)) if raise.guard.is_none()
        ));
    }
}
