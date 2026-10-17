//! The consumer's check of the `InstructionBuilder` contract.
//!
//! The trait leaves the check to the consumer: `finish` seals the record and
//! reports a violation through its error type. Both test storages run this
//! check over the statements they are about to seal, using only the public
//! statement types.

use std::collections::BTreeSet;

use cfglib::DenseId;
use cfglib::ir::rtl::{
    Action, BitSlice, Bits, Control, Dialect, Raise, Selection, SpanAction, Statement, WriteTarget,
};

/// Why a record cannot be sealed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SealError {
    /// An operand is a handle that this record has not defined yet.
    UnknownOperand(usize),
    /// A root write whose plan leaves a gap, overlaps, or misses the root width.
    IncompletePlan,
    /// Two writes of one record cover the same bit.
    OverlappingWrites,
    /// A guard or condition that is not one bit wide.
    GuardNotBoolean,
    /// A statement follows an unconditional raise.
    StatementAfterRaise,
    /// An unconditional raise whose record does not end in `Terminating`.
    RaiseNotTerminal,
    /// A record with an unconditional raise holds a write.
    WriteBesideRaise,
    /// Two bindings of one raise name the same root.
    DuplicateBinding,
    /// A bound value whose width is not the width of its root.
    BindingWidth,
}

/// Checks one record, given its statements in order and its control.
pub fn check<'a, D: Dialect, V: DenseId + 'a>(
    statements: impl Iterator<Item = Statement<'a, D, V>>,
    control: Control<V>,
    width_of: &dyn Fn(V) -> Bits,
) -> Result<(), SealError> {
    let mut defined = BTreeSet::new();
    let mut writes: Vec<(D::Root, BitSlice)> = Vec::new();
    let (mut raised, mut wrote) = (false, false);
    for statement in statements {
        if raised {
            return Err(SealError::StatementAfterRaise);
        }
        for operand in operands(&statement) {
            if !defined.contains(&operand.index()) {
                return Err(SealError::UnknownOperand(operand.index()));
            }
        }
        match statement {
            Statement::Read { result, .. }
            | Statement::Constant { result, .. }
            | Statement::Nondet { result, .. } => {
                defined.insert(result.index());
            }
            Statement::Apply { results, .. } | Statement::Effect { results, .. } => {
                defined.extend(results.iter().map(DenseId::index));
            }
            Statement::Raise(raise) => {
                check_raise(&raise, width_of)?;
                raised |= raise.guard.is_none();
            }
            Statement::Write { target, .. } => {
                wrote = true;
                if let WriteTarget::Root { root, plan } = target
                    && !plan_covers::<D, V>(root, plan)
                {
                    return Err(SealError::IncompletePlan);
                }
                record_write(&target, &mut writes)?;
            }
        }
    }
    if raised && !matches!(control, Control::Terminating) {
        return Err(SealError::RaiseNotTerminal);
    }
    if raised && wrote {
        return Err(SealError::WriteBesideRaise);
    }
    for operand in control_operands(control) {
        if !defined.contains(&operand.index()) {
            return Err(SealError::UnknownOperand(operand.index()));
        }
    }
    if let Control::Branch { condition, .. } = control
        && width_of(condition) != Bits::BOOL
    {
        return Err(SealError::GuardNotBoolean);
    }
    Ok(())
}

fn check_raise<D: Dialect, V: DenseId>(
    raise: &Raise<'_, D, V>,
    width_of: &dyn Fn(V) -> Bits,
) -> Result<(), SealError> {
    if let Some(guard) = raise.guard
        && width_of(guard) != Bits::BOOL
    {
        return Err(SealError::GuardNotBoolean);
    }
    let mut seen = BTreeSet::new();
    for &(root, value) in raise.binds {
        if !seen.insert(root) {
            return Err(SealError::DuplicateBinding);
        }
        if width_of(value) != D::root_width(root) {
            return Err(SealError::BindingWidth);
        }
    }
    Ok(())
}

/// Whether the spans of `plan` are ascending, start at bit zero, leave no gap,
/// and end at the root width.
fn plan_covers<D: Dialect, V>(root: D::Root, plan: &[SpanAction<D, V>]) -> bool {
    let mut next = 0_u32;
    for span in plan {
        if u32::from(span.span.offset) != next {
            return false;
        }
        next = span.span.end();
    }
    next == u32::from(D::root_width(root).width())
}

/// Adds the bits that `target` can write and rejects an overlap with an
/// earlier write. A sparse bank element is opaque, so it never conflicts.
fn record_write<D: Dialect, V>(
    target: &WriteTarget<'_, D, V>,
    writes: &mut Vec<(D::Root, BitSlice)>,
) -> Result<(), SealError> {
    let covered: Vec<(D::Root, BitSlice)> = match *target {
        WriteTarget::Root { root, .. } => vec![(root, BitSlice::whole(D::root_width(root)))],
        WriteTarget::FiniteIndexed { candidates, .. } => candidates
            .iter()
            .map(|place| (place.root, place.slice))
            .collect(),
        WriteTarget::SparseIndexed { .. } => Vec::new(),
    };
    for &(root, slice) in &covered {
        let clash = writes.iter().any(|&(other_root, other)| {
            other_root == root
                && u32::from(slice.offset) < other.end()
                && u32::from(other.offset) < slice.end()
        });
        if clash {
            return Err(SealError::OverlappingWrites);
        }
    }
    writes.extend(covered);
    Ok(())
}

fn operands<D: Dialect, V: DenseId>(statement: &Statement<'_, D, V>) -> Vec<V> {
    match *statement {
        Statement::Read { source, .. } => match source {
            Selection::Direct(_) => Vec::new(),
            Selection::FiniteIndexed { index, .. } => vec![index],
            Selection::SparseIndexed { key, .. } => vec![key],
        },
        Statement::Constant { .. } | Statement::Nondet { .. } => Vec::new(),
        Statement::Apply { args, .. } | Statement::Effect { args, .. } => args.to_vec(),
        Statement::Raise(raise) => raise
            .guard
            .into_iter()
            .chain(raise.payload.iter().copied())
            .chain(raise.binds.iter().map(|&(_, value)| value))
            .collect(),
        Statement::Write { target, .. } => match target {
            WriteTarget::Root { plan, .. } => plan
                .iter()
                .filter_map(|span| match span.action {
                    Action::TakeInput { value, .. } => Some(value),
                    _ => None,
                })
                .collect(),
            WriteTarget::FiniteIndexed { index, value, .. } => vec![index, value],
            WriteTarget::SparseIndexed { key, value, .. } => vec![key, value],
        },
    }
}

fn control_operands<V: Copy>(control: Control<V>) -> Vec<V> {
    match control {
        Control::Advance | Control::Terminating => Vec::new(),
        Control::Jump { target } | Control::Call { target } | Control::Return { target } => {
            vec![target]
        }
        Control::Branch {
            condition,
            taken,
            not_taken,
        } => [Some(condition), Some(taken), not_taken]
            .into_iter()
            .flatten()
            .collect(),
        Control::Dispatch { scrutinee } => vec![scrutinee],
    }
}
