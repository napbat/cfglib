//! The ordered driver: units evaluated one at a time in sweep order, each
//! reporting the slots it read.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::mem;

use super::SlotFixpointReport;
use crate::dataflow::worklist::Worklist;

/// The slots one evaluation read and the slots it changed.
///
/// [`ordered_fixpoint`] hands each evaluation an empty access. A read slot
/// is a dependency: the unit runs again after a later change of it. A
/// changed slot queues every unit whose latest evaluation read it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotAccess<K> {
    reads: Vec<K>,
    changes: Vec<K>,
}

impl<K> SlotAccess<K> {
    const fn new() -> Self {
        Self {
            reads: Vec::new(),
            changes: Vec::new(),
        }
    }

    /// Record that the evaluation read slot `key`. Repeated reads count once.
    pub fn read(&mut self, key: K) {
        self.reads.push(key);
    }

    /// Record that the evaluation changed slot `key`.
    pub fn changed(&mut self, key: K) {
        self.changes.push(key);
    }
}

/// Run the units `0..units` to a fixpoint in ordered sweeps.
///
/// A sweep evaluates its dirty units in ascending order, one at a time, so a
/// slot changed earlier in a sweep is visible to every later evaluation of
/// that sweep. Every unit is dirty in sweep 0. Each evaluation records the
/// slots it read and changed through its [`SlotAccess`]; its reads replace
/// the dependencies of its previous evaluation. When it changes a slot, each
/// unit whose latest evaluation read that slot becomes dirty: in the current
/// sweep when it sorts after the evaluated unit, and in the next sweep
/// otherwise, the evaluated unit included.
///
/// The units evaluated are exactly those that a full sweep in unit order
/// would evaluate if it skipped each unit whose latest evaluation read no
/// slot that changed since, so a deterministic evaluation that depends only
/// on the slots it records yields the same slots as that full sweep while the
/// driver never visits a clean unit. A unit whose latest evaluation recorded
/// no read does not run again.
///
/// A sweep after the first runs only when the sweep before it changed a slot,
/// and then even when no unit is dirty, so the sweep count is that of the
/// full sweep, which stops after its first sweep without a change. Before
/// each sweep the driver calls `proceed` with the sweep index; `false` stops
/// the fixpoint unconverged, which is where a sweep cap or a work budget
/// goes.
///
/// # Examples
///
/// ```
/// use cfglib::ordered_fixpoint;
///
/// // Unit `u` copies slot `u + 1` into slot `u`, and slot 3 starts set, so
/// // the chain settles one unit per sweep against the sweep order.
/// let mut slots = [false, false, false, true];
/// let report = ordered_fixpoint(3, |_| true, |unit, access| {
///     access.read(unit + 1);
///     if !slots[unit] && slots[unit + 1] {
///         slots[unit] = true;
///         access.changed(unit);
///     }
/// });
/// assert_eq!(slots, [true; 4]);
/// assert_eq!(report.round_count(), 4);
/// assert_eq!(report.evaluation_count(), 5);
/// assert!(report.converged());
/// ```
pub fn ordered_fixpoint<K: Ord + Clone>(
    units: usize,
    mut proceed: impl FnMut(usize) -> bool,
    mut evaluate: impl FnMut(usize, &mut SlotAccess<K>),
) -> SlotFixpointReport {
    let mut report = SlotFixpointReport::default();
    // The sorted, distinct slots each unit's latest evaluation read, and the
    // same relation inverted.
    let mut reads: Vec<Vec<K>> = (0..units).map(|_| Vec::new()).collect();
    let mut readers: BTreeSet<(K, usize)> = BTreeSet::new();
    let mut current = Worklist::of(units, 0..units);
    let mut next = Worklist::new(units);
    let mut access = SlotAccess::new();
    for sweep in 0.. {
        if !proceed(sweep) {
            return report;
        }
        report.rounds += 1;
        let mut changed = false;
        while let Some(unit) = current.pop_first() {
            access.reads.clear();
            evaluate(unit, &mut access);
            report.evaluations += 1;
            access.reads.sort_unstable();
            access.reads.dedup();
            // The new reads go in before the changes queue their readers: a
            // unit that changes a slot it read runs again next sweep.
            if access.reads != reads[unit] {
                for key in reads[unit].drain(..) {
                    readers.remove(&(key, unit));
                }
                readers.extend(access.reads.iter().map(|key| (key.clone(), unit)));
                mem::swap(&mut reads[unit], &mut access.reads);
            }
            for key in access.changes.drain(..) {
                changed = true;
                for &(_, reader) in readers.range((key.clone(), 0)..=(key, usize::MAX)) {
                    if reader > unit {
                        current.insert(reader);
                    } else {
                        next.insert(reader);
                    }
                }
            }
        }
        if !changed {
            break;
        }
        mem::swap(&mut current, &mut next);
    }
    report.converged = true;
    report
}
