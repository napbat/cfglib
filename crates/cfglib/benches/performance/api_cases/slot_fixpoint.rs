use std::collections::VecDeque;

use cfglib::{Fanout, SequentialExecutor, SolveError, SolveStats, ordered_fixpoint, wave_fixpoint};

use super::BenchmarkSuite;
use crate::harness::benchmark_case;

/// Slots in the ring, one unit or wave group each.
const SLOT_COUNT: usize = 4_096;

/// The value of a slot that no read has reached yet.
const UNREACHED: u32 = u32::MAX;

/// What a run hands its oracle: the outcome and the final slots.
type Run = (Result<SolveStats, SolveError>, Vec<u32>);

/// A keyed-slot program in the shape of re-export depth over files. Unit `u`
/// owns slot `u` and reads three slots: its ring predecessor and the images
/// of `u` under two fixed affine permutations. A unit lowers its slot to one
/// more than the smallest slot it reads, and slot 0 starts at zero. So the
/// fixpoint is the breadth-first distance of each slot from slot 0 along the
/// reads.
struct Ring {
    reads: Vec<[usize; 3]>,
}

impl Ring {
    fn new() -> Self {
        let reads = (0..SLOT_COUNT)
            .map(|unit| {
                [
                    (unit + SLOT_COUNT - 1) % SLOT_COUNT,
                    (unit * 1_597 + 211) % SLOT_COUNT,
                    (unit * 2_917 + 1_009) % SLOT_COUNT,
                ]
            })
            .collect();
        Self { reads }
    }

    /// Every slot unreached except slot 0.
    fn start() -> Vec<u32> {
        let mut slots = vec![UNREACHED; SLOT_COUNT];
        slots[0] = 0;
        slots
    }

    /// The new value of slot `unit` when its reads lower it.
    fn lowered(&self, unit: usize, slots: &[u32]) -> Option<u32> {
        let reached = self.reads[unit]
            .iter()
            .map(|&slot| slots[slot].saturating_add(1))
            .min()?;
        (reached < slots[unit]).then_some(reached)
    }

    /// The groups that read each slot, which the wave driver requeues.
    fn readers(&self) -> Fanout<usize, usize> {
        Fanout::from_pairs(
            self.reads
                .iter()
                .enumerate()
                .flat_map(|(unit, reads)| reads.iter().map(move |&slot| (slot, unit)))
                .collect(),
        )
    }

    /// The breadth-first distance of each slot from slot 0, found without
    /// either driver.
    fn distances(&self) -> Vec<u32> {
        let mut readers = vec![Vec::new(); SLOT_COUNT];
        for (unit, reads) in self.reads.iter().enumerate() {
            for &slot in reads {
                readers[slot].push(unit);
            }
        }
        let mut distances = vec![UNREACHED; SLOT_COUNT];
        distances[0] = 0;
        let mut queue = VecDeque::from([0]);
        while let Some(slot) = queue.pop_front() {
            let next = distances[slot] + 1;
            for &reader in &readers[slot] {
                if distances[reader] == UNREACHED {
                    distances[reader] = next;
                    queue.push_back(reader);
                }
            }
        }
        distances
    }
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let ring = Ring::new();
    let readers = ring.readers();
    let expected = ring.distances();
    // Both drivers reach the breadth-first distances. The steps and rounds
    // go next to the time, because the sweep order is what changes them.
    let check = |name: &str, run: &Run| {
        let (outcome, slots) = run;
        let stats = outcome.as_ref().expect("an unbounded fixpoint converges");
        assert_eq!(slots, &expected);
        println!(
            "{name:<36} {:>12} steps  {} rounds",
            stats.step_count(),
            stats.round_count()
        );
    };

    benchmark_case!(
        suite,
        "api_ordered_fixpoint_ring",
        covers[ordered_fixpoint],
        || {
            let mut slots = Ring::start();
            let outcome = ordered_fixpoint(SLOT_COUNT, None, |unit, access| {
                for &slot in &ring.reads[unit] {
                    access.read(slot);
                }
                if let Some(value) = ring.lowered(unit, &slots) {
                    slots[unit] = value;
                    access.changed(unit);
                }
            });
            (outcome, slots)
        },
        |run: &Run| check("api_ordered_fixpoint_ring", run)
    );
    benchmark_case!(
        suite,
        "api_wave_fixpoint_ring",
        covers[wave_fixpoint],
        || {
            let mut slots = Ring::start();
            let outcome = wave_fixpoint(
                SLOT_COUNT,
                &readers,
                &mut slots,
                &mut SequentialExecutor,
                None,
                |group, slots: &Vec<u32>| ring.lowered(group, slots),
                |slots, group, lowered, requeue| {
                    if let Some(value) = lowered {
                        slots[group] = value;
                        requeue.changed(&group);
                    }
                },
            );
            (outcome, slots)
        },
        |run: &Run| check("api_wave_fixpoint_ring", run)
    );
}
