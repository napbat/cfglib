//! The effect of one block on the constants of its roots.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use super::ResolvedSelection;
use super::state::{ConstState, ValueConsts};
use crate::graph::view::DenseId;
use crate::ir::rtl::facts::roots::RootTable;
use crate::ir::rtl::facts::words::{copy_bits, to_usize};
use crate::ir::rtl::{
    Action, BitSlice, Bits, Control, Fold, FunctionView, Place, RecordView, Selection, SpanAction,
    Statement, ValueRange, WriteTarget,
};

/// The working storage that runs blocks over constant states.
pub(super) struct Transfer<'a, D: Fold, F: FunctionView<D>> {
    /// The function whose blocks run.
    function: &'a F,
    /// The roots of the function.
    roots: &'a RootTable<D>,
    /// The constants of the values of the record in hand.
    values: ValueConsts,
    /// The result types of the operator in hand.
    widths: Vec<Bits>,
    /// The result words of the operator in hand.
    out: Vec<u64>,
    /// The writes of the record in hand: the root and its new value, or
    /// `None` when the new value is unknown.
    pending: Vec<(usize, Option<Box<[u64]>>)>,
}

impl<'a, D: Fold, F: FunctionView<D>> Transfer<'a, D, F> {
    /// Creates the working storage for `function`.
    pub(super) fn new(function: &'a F, roots: &'a RootTable<D>) -> Self {
        Self {
            function,
            roots,
            values: ValueConsts::new(function.value_bound()),
            widths: Vec::new(),
            out: Vec::new(),
            pending: Vec::new(),
        }
    }

    /// Runs the instructions of `block` over `state`.
    ///
    /// The call appends every indexed selection with a constant index to
    /// `report` when it has a report.
    pub(super) fn run_block(
        &mut self,
        block: F::NodeId,
        state: &mut ConstState,
        mut report: Option<&mut Vec<ResolvedSelection<F::Inst>>>,
    ) {
        let function = self.function;
        for &inst in function.instructions(block) {
            let record = function.record(inst);
            self.run_record(inst, &record, state, report.as_deref_mut());
        }
    }

    /// Runs one record over `state`.
    ///
    /// Every read sees `state` as it was before the record, and the writes of
    /// the record take effect together after the last statement.
    fn run_record(
        &mut self,
        inst: F::Inst,
        record: &F::Record<'_>,
        state: &mut ConstState,
        mut report: Option<&mut Vec<ResolvedSelection<F::Inst>>>,
    ) {
        self.values.clear();
        self.pending.clear();
        for (position, statement) in record.statements().enumerate() {
            match statement {
                Statement::Read { result, source } => {
                    let place = match source {
                        Selection::Direct(place) => Some(place),
                        Selection::FiniteIndexed { index, candidates } => {
                            let chosen = self.constant_index(record, index, candidates.len());
                            note(report.as_deref_mut(), inst, position, chosen);
                            chosen.map(|candidate| candidates[candidate])
                        }
                        Selection::SparseIndexed { .. } => None,
                    };
                    if let Some(place) = place {
                        self.read_place(state, result, place);
                    }
                }
                Statement::Constant {
                    result,
                    words,
                    width,
                } => {
                    let value = self.values.insert(result.index(), width);
                    copy_bits(value, 0, words, 0, usize::from(width.width()));
                }
                Statement::Apply {
                    results,
                    operator,
                    args,
                } => self.apply(record, results, operator, args),
                Statement::Write { target, .. } => {
                    self.stage_write(
                        record,
                        state,
                        target,
                        (inst, position),
                        report.as_deref_mut(),
                    );
                }
                Statement::Nondet { .. } | Statement::Effect { .. } | Statement::Raise(_) => {}
            }
        }
        for (root, value) in self.pending.drain(..) {
            match value {
                Some(value) => state.set(root, value),
                None => state.remove(root),
            }
        }
        // The callee can write any root, and the dialect does not say which.
        if matches!(record.control(), Control::Call { .. }) {
            state.clear();
        }
    }

    /// Returns the candidate that a constant `index` selects, when `index` is
    /// a constant below `candidates`.
    fn constant_index(
        &self,
        record: &F::Record<'_>,
        index: F::Value,
        candidates: usize,
    ) -> Option<usize> {
        let value = self.values.get(index.index(), record.width(index))?;
        to_usize(value.words).filter(|&chosen| chosen < candidates)
    }

    /// Makes `result` the constant that `place` holds in `state`, when the
    /// root of `place` is a known constant.
    fn read_place(&mut self, state: &ConstState, result: F::Value, place: Place<D>) {
        let Some(root) = state.get(self.roots.index_of_named(place.root)) else {
            return;
        };
        let width = place.slice.width;
        let value = self.values.insert(result.index(), width);
        copy_bits(
            value,
            0,
            root,
            usize::from(place.slice.offset),
            usize::from(width.width()),
        );
    }

    /// Evaluates a pure operator when every operand is a constant.
    fn apply(
        &mut self,
        record: &F::Record<'_>,
        results: ValueRange<F::Value>,
        operator: &D::Operator,
        args: &[F::Value],
    ) {
        if !args.iter().all(|arg| self.values.contains(arg.index())) {
            return;
        }
        let operands: Vec<_> = args
            .iter()
            .filter_map(|&arg| self.values.get(arg.index(), record.width(arg)))
            .collect();
        self.widths.clear();
        self.widths
            .extend(results.iter().map(|result| record.width(result)));
        self.out.clear();
        self.out
            .resize(self.widths.iter().map(|width| width.words()).sum(), 0);
        if !D::fold(operator, &operands, &self.widths, &mut self.out) {
            return;
        }
        let mut start = 0;
        for (result, &width) in results.iter().zip(&self.widths) {
            let end = start + width.words();
            self.values
                .insert(result.index(), width)
                .copy_from_slice(&self.out[start..end]);
            start = end;
        }
    }

    /// Stages the effect of one write on the roots of the function.
    fn stage_write(
        &mut self,
        record: &F::Record<'_>,
        state: &ConstState,
        target: WriteTarget<'_, D, F::Value>,
        site: (F::Inst, usize),
        report: Option<&mut Vec<ResolvedSelection<F::Inst>>>,
    ) {
        match target {
            WriteTarget::Root { root, plan } => {
                let value = self.assemble(record, state, root, plan);
                self.pending.push((self.roots.index_of_named(root), value));
            }
            WriteTarget::FiniteIndexed {
                index,
                candidates,
                value,
            } => {
                let chosen = self.constant_index(record, index, candidates.len());
                note(report, site.0, site.1, chosen);
                if let Some(candidate) = chosen {
                    let place = candidates[candidate];
                    let spliced = self.splice(record, state, place, value);
                    self.pending
                        .push((self.roots.index_of_named(place.root), spliced));
                } else {
                    for place in candidates {
                        self.pending
                            .push((self.roots.index_of_named(place.root), None));
                    }
                }
            }
            WriteTarget::SparseIndexed { bank, .. } => {
                self.pending.push((self.roots.index_of_named(bank), None));
            }
        }
    }

    /// Returns the new value of `root` after a write with `plan`, when every
    /// span of the plan has a known value.
    fn assemble(
        &self,
        record: &F::Record<'_>,
        state: &ConstState,
        root: D::Root,
        plan: &[SpanAction<D, F::Value>],
    ) -> Option<Box<[u64]>> {
        let before = state.get(self.roots.index_of_named(root));
        let mut words = vec![0; D::root_width(root).words()].into_boxed_slice();
        for span in plan {
            let offset = usize::from(span.span.offset);
            let width = usize::from(span.span.width.width());
            match span.action {
                Action::PreservePreState => copy_bits(&mut words, offset, before?, offset, width),
                Action::TakeInput {
                    value,
                    offset: first,
                } => {
                    let input = self.values.get(value.index(), record.width(value))?;
                    copy_bits(&mut words, offset, input.words, usize::from(first), width);
                }
                Action::Nondeterministic(_) => return None,
            }
        }
        Some(words)
    }

    /// Returns the value of the root `root` once the writes staged so far have
    /// taken effect, when it is known.
    ///
    /// Two writes of one record can land in disjoint slices of one root. Each
    /// builds on what the earlier ones staged, so the later write does not
    /// restore the bits that the earlier one changed, and a root that an
    /// earlier write left unknown stays unknown.
    fn staged<'s>(&'s self, state: &'s ConstState, root: usize) -> Option<&'s [u64]> {
        match self
            .pending
            .iter()
            .rev()
            .find(|&&(staged, _)| staged == root)
        {
            Some((_, value)) => value.as_deref(),
            None => state.get(root),
        }
    }

    /// Returns the new value of the root of `place` after a write of `value`
    /// into `place`, when `value` is a constant and the write either covers
    /// the whole root or the root is a known constant.
    fn splice(
        &self,
        record: &F::Record<'_>,
        state: &ConstState,
        place: Place<D>,
        value: F::Value,
    ) -> Option<Box<[u64]>> {
        let input = self.values.get(value.index(), record.width(value))?;
        let root_width = D::root_width(place.root);
        let mut words = match self.staged(state, self.roots.index_of_named(place.root)) {
            Some(before) => Box::<[u64]>::from(before),
            None if place.slice == BitSlice::whole(root_width) => {
                vec![0; root_width.words()].into_boxed_slice()
            }
            None => return None,
        };
        copy_bits(
            &mut words,
            usize::from(place.slice.offset),
            input.words,
            0,
            usize::from(place.slice.width.width()),
        );
        Some(words)
    }
}

/// Reports that the selection at statement `statement` of `inst` resolved to
/// `chosen`, when there is a report and a constant index.
fn note<I>(
    report: Option<&mut Vec<ResolvedSelection<I>>>,
    inst: I,
    statement: usize,
    chosen: Option<usize>,
) {
    if let (Some(report), Some(candidate)) = (report, chosen) {
        report.push(ResolvedSelection {
            inst,
            statement,
            candidate,
        });
    }
}
