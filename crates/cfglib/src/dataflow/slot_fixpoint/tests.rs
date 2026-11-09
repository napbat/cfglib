extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;

use super::{
    SequentialExecutor, SlotAccess, SlotFixpointStats, WaveExecutor, ordered_fixpoint,
    wave_fixpoint,
};
use crate::graph::fanout::Fanout;
use crate::test_util::rng::Rng;

/// Runs each wave on up to two scoped threads, so the evaluations of a wave
/// finish in an order the driver does not choose.
struct Threads;

impl WaveExecutor for Threads {
    fn run<T, F>(&mut self, wave: &[usize], evaluate: F) -> Option<Vec<T>>
    where
        T: Send,
        F: Fn(usize) -> T + Sync,
    {
        let evaluate = &evaluate;
        let half = wave.len().div_ceil(2).max(1);
        Some(std::thread::scope(|scope| {
            let parts: Vec<_> = wave
                .chunks(half)
                .map(|part| {
                    scope.spawn(move || {
                        part.iter()
                            .map(|&group| evaluate(group))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            parts
                .into_iter()
                .flat_map(|part| part.join().expect("an evaluation does not panic"))
                .collect()
        }))
    }
}

/// Declines every wave after the first `waves`.
struct Budget {
    waves: usize,
}

impl WaveExecutor for Budget {
    fn run<T, F>(&mut self, wave: &[usize], evaluate: F) -> Option<Vec<T>>
    where
        T: Send,
        F: Fn(usize) -> T + Sync,
    {
        self.waves = self.waves.checked_sub(1)?;
        SequentialExecutor.run(wave, evaluate)
    }
}

/// A monotone wave program: group `g` owns the slots `owned[g]` and raises
/// each to one more than the largest slot it reads, capped at `height`. The
/// seed slots start at one.
struct WaveProgram {
    height: u8,
    owned: Vec<Vec<usize>>,
    reads: Vec<Vec<usize>>,
    seeds: Vec<usize>,
    slot_count: usize,
}

impl WaveProgram {
    fn random(rng: &mut Rng, height: u8) -> Self {
        let groups = 1 + rng.below(12);
        let slot_count = 1 + rng.below(24);
        let mut owned = vec![Vec::new(); groups];
        for slot in 0..slot_count {
            owned[rng.below(groups)].push(slot);
        }
        let reads = (0..groups)
            .map(|_| (0..rng.below(4)).map(|_| rng.below(slot_count)).collect())
            .collect();
        let seeds = (0..slot_count).filter(|_| rng.one_in(4)).collect();
        Self {
            height,
            owned,
            reads,
            seeds,
            slot_count,
        }
    }

    fn start(&self) -> Vec<u8> {
        let mut slots = vec![0; self.slot_count];
        for &seed in &self.seeds {
            slots[seed] = 1;
        }
        slots
    }

    /// The new value of each slot `group` owns, given `slots`.
    fn evaluate(&self, group: usize, slots: &[u8]) -> Vec<(usize, u8)> {
        let input = self.reads[group].iter().map(|&slot| slots[slot]).max();
        let raised = input
            .filter(|&value| value > 0)
            .map_or(0, |value| value.saturating_add(1).min(self.height));
        self.owned[group]
            .iter()
            .map(|&slot| (slot, slots[slot].max(raised)))
            .collect()
    }

    fn readers(&self) -> Fanout<usize, usize> {
        Fanout::from_pairs(
            self.reads
                .iter()
                .enumerate()
                .flat_map(|(group, reads)| reads.iter().map(move |&slot| (slot, group)))
                .collect(),
        )
    }

    /// Round-robin over every group until a whole round changes nothing.
    fn naive(&self) -> Vec<u8> {
        let mut slots = self.start();
        loop {
            let mut changed = false;
            for group in 0..self.owned.len() {
                for (slot, value) in self.evaluate(group, &slots) {
                    changed |= slots[slot] != value;
                    slots[slot] = value;
                }
            }
            if !changed {
                return slots;
            }
        }
    }
}

/// Runs `program` through the wave driver with `executor`, returning the
/// slots, the stats, the merge order and the readers of every change.
fn run_waves(
    program: &WaveProgram,
    executor: &mut impl WaveExecutor,
) -> (Vec<u8>, SlotFixpointStats, Vec<usize>, usize) {
    let readers = program.readers();
    let mut slots = program.start();
    let mut merged = Vec::new();
    let mut requeued = 0;
    let stats = wave_fixpoint(
        program.owned.len(),
        &readers,
        &mut slots,
        executor,
        |group, slots: &Vec<u8>| program.evaluate(group, slots),
        |slots, group, values, requeue| {
            merged.push(group);
            for (slot, value) in values {
                if slots[slot] != value {
                    slots[slot] = value;
                    requeued += requeue.changed(&slot).len();
                }
            }
        },
    );
    (slots, stats, merged, requeued)
}

#[test]
fn waves_reach_the_round_robin_fixpoint_on_random_graphs() {
    for seed in 0..400 {
        let mut rng = Rng::new(seed);
        let height = [1, 2, 5][rng.below(3)];
        let program = WaveProgram::random(&mut rng, height);
        let expected = program.naive();
        let (sequential, stats, _, requeued) = run_waves(&program, &mut SequentialExecutor);
        assert_eq!(sequential, expected, "seed {seed}");
        assert!(stats.converged(), "seed {seed}");
        assert!(
            stats.evaluation_count() <= program.owned.len() + requeued,
            "seed {seed}: {} evaluations for {} groups and {requeued} readers of changes",
            stats.evaluation_count(),
            program.owned.len(),
        );
        let (threaded, threaded_stats, _, _) = run_waves(&program, &mut Threads);
        assert_eq!(threaded, expected, "seed {seed}");
        assert_eq!(threaded_stats, stats, "seed {seed}");
    }
}

#[test]
fn each_wave_merges_in_ascending_group_order() {
    // Slot 0 is seeded. Groups 0 and 3 raise slots 1 and 3 from it, and
    // groups 1 and 2 read slot 3, so they run again in wave 1.
    let program = WaveProgram {
        height: 3,
        owned: vec![vec![1], vec![2], vec![4], vec![3]],
        reads: vec![vec![0], vec![3], vec![3], vec![0]],
        seeds: vec![0],
        slot_count: 5,
    };
    let (slots, stats, merged, _) = run_waves(&program, &mut Threads);
    assert_eq!(slots, [1, 2, 3, 2, 3]);
    assert_eq!(merged, [0, 1, 2, 3, 1, 2]);
    assert_eq!(stats.round_count(), 2);
    assert!(stats.converged());
}

#[test]
fn a_declined_wave_stops_with_the_merged_facts() {
    // A chain: group `g` raises slot `g + 1` from slot `g`, one wave a link.
    let program = WaveProgram {
        height: 9,
        owned: (1..6).map(|slot| vec![slot]).collect(),
        reads: (0..5).map(|slot| vec![slot]).collect(),
        seeds: vec![0],
        slot_count: 6,
    };
    let (slots, stats, _, _) = run_waves(&program, &mut Budget { waves: 2 });
    assert_eq!(slots, [1, 2, 3, 0, 0, 0]);
    assert_eq!(stats.round_count(), 2);
    assert_eq!(stats.evaluation_count(), 6);
    assert!(!stats.converged());
    let (slots, stats, _, _) = run_waves(&program, &mut SequentialExecutor);
    assert_eq!(slots, [1, 2, 3, 4, 5, 6]);
    assert_eq!(stats.round_count(), 5);
    assert!(stats.converged());
}

#[test]
fn no_groups_run_no_wave() {
    let readers = Fanout::<usize, usize>::default();
    let stats = wave_fixpoint(
        0,
        &readers,
        &mut (),
        &mut SequentialExecutor,
        |_, ()| (),
        |(), _, (), _| {},
    );
    assert_eq!(stats.round_count(), 0);
    assert_eq!(stats.evaluation_count(), 0);
    assert!(stats.converged());
}

/// An order-dependent, first-value-wins program over one slot per unit, in
/// the shape of a type inference: a set slot is final, a seeded unit sets
/// its slot from nothing, and any other unit reads its candidates in order
/// and takes the first set one with an odd value. A unit may also read its
/// own slot. What a unit reads depends on the values it finds, so its
/// dependencies change between evaluations.
struct OrderedProgram {
    seeds: Vec<Option<u32>>,
    candidates: Vec<Vec<usize>>,
    self_read: Vec<bool>,
}

impl OrderedProgram {
    fn random(rng: &mut Rng) -> Self {
        let units = 1 + rng.below(30);
        Self {
            seeds: (0..units)
                .map(|_| {
                    rng.one_in(5)
                        .then(|| u32::try_from(rng.below(9)).expect("small"))
                })
                .collect(),
            candidates: (0..units)
                .map(|_| (0..rng.below(4)).map(|_| rng.below(units)).collect())
                .collect(),
            self_read: (0..units).map(|_| rng.one_in(3)).collect(),
        }
    }

    fn units(&self) -> usize {
        self.seeds.len()
    }

    /// Evaluates `unit` unless its slot is set, reporting each slot read.
    /// Returns whether it ran and whether it set the slot.
    fn evaluate(
        &self,
        unit: usize,
        slots: &mut [Option<u32>],
        mut read: impl FnMut(usize),
    ) -> (bool, bool) {
        if slots[unit].is_some() {
            return (false, false);
        }
        if self.self_read[unit] {
            read(unit);
        }
        let value = self.seeds[unit].or_else(|| {
            self.candidates[unit].iter().find_map(|&candidate| {
                read(candidate);
                slots[candidate]
                    .filter(|value| value % 2 == 1)
                    .map(|value| value.wrapping_mul(2).wrapping_add(1))
            })
        });
        slots[unit] = value;
        (true, value.is_some())
    }

    /// Full sweeps over every unit until a sweep changes nothing: the naive
    /// round robin. Returns the slots and the sweep count.
    fn naive(&self) -> (Vec<Option<u32>>, usize) {
        let mut slots = vec![None; self.units()];
        let mut sweeps = 0;
        loop {
            sweeps += 1;
            let mut changed = false;
            for unit in 0..self.units() {
                changed |= self.evaluate(unit, &mut slots, |_| {}).1;
            }
            if !changed {
                return (slots, sweeps);
            }
        }
    }

    /// Full sweeps that skip a unit whose latest evaluation read no slot
    /// changed since, on revision stamps: the scan the driver replaces.
    /// Returns the slots and each evaluation as `(sweep, unit)`.
    fn stamped(&self) -> (Vec<Option<u32>>, Vec<(usize, usize)>) {
        let mut slots = vec![None; self.units()];
        let mut now = 0_usize;
        let mut changed_at = vec![0_usize; self.units()];
        let mut runs: Vec<Option<(usize, Vec<usize>)>> = vec![None; self.units()];
        let mut log = Vec::new();
        for sweep in 0.. {
            let mut changed = false;
            for unit in 0..self.units() {
                let valid = runs[unit]
                    .as_ref()
                    .is_some_and(|(at, reads)| reads.iter().all(|&slot| changed_at[slot] <= *at));
                if slots[unit].is_some() || valid {
                    continue;
                }
                let mut reads = Vec::new();
                let at = now;
                let (_, set) = self.evaluate(unit, &mut slots, |slot| reads.push(slot));
                log.push((sweep, unit));
                runs[unit] = Some((at, reads));
                if set {
                    now += 1;
                    changed_at[unit] = now;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        (slots, log)
    }
}

/// What one run of the ordered driver did.
struct OrderedRun {
    slots: Vec<Option<u32>>,
    stats: SlotFixpointStats,
    /// Each evaluation that ran, as `(sweep, unit)`.
    log: Vec<(usize, usize)>,
    /// For each change, the units whose latest evaluation read the slot.
    readers_of_changes: usize,
}

fn run_ordered(program: &OrderedProgram, sweeps: usize) -> OrderedRun {
    let mut slots = vec![None; program.units()];
    let mut latest: Vec<Vec<usize>> = vec![Vec::new(); program.units()];
    let mut log = Vec::new();
    let mut readers_of_changes = 0;
    let sweep = RefCell::new(0);
    let stats = ordered_fixpoint(
        program.units(),
        |index| {
            *sweep.borrow_mut() = index;
            index < sweeps
        },
        |unit, access: &mut SlotAccess<usize>| {
            let mut reads = Vec::new();
            let (ran, set) = program.evaluate(unit, &mut slots, |slot| reads.push(slot));
            if ran {
                log.push((*sweep.borrow(), unit));
            }
            for &slot in &reads {
                access.read(slot);
            }
            latest[unit] = reads;
            if set {
                access.changed(unit);
                readers_of_changes += latest.iter().filter(|reads| reads.contains(&unit)).count();
            }
        },
    );
    OrderedRun {
        slots,
        stats,
        log,
        readers_of_changes,
    }
}

#[test]
fn ordered_sweeps_match_the_round_robin_and_the_stamped_scan() {
    for seed in 0..600 {
        let mut rng = Rng::new(seed);
        let program = OrderedProgram::random(&mut rng);
        let (naive, naive_sweeps) = program.naive();
        let (stamped, stamped_log) = program.stamped();
        let run = run_ordered(&program, usize::MAX);
        assert_eq!(run.slots, naive, "seed {seed}");
        assert_eq!(stamped, naive, "seed {seed}");
        assert_eq!(run.log, stamped_log, "seed {seed}");
        assert_eq!(run.stats.round_count(), naive_sweeps, "seed {seed}");
        assert!(run.stats.converged(), "seed {seed}");
        assert!(
            run.stats.evaluation_count() <= program.units() + run.readers_of_changes,
            "seed {seed}: {} evaluations for {} units and {} readers of changes",
            run.stats.evaluation_count(),
            program.units(),
            run.readers_of_changes,
        );
    }
}

#[test]
fn a_reverse_chain_settles_one_unit_a_sweep_without_rescanning() {
    // Unit `u` copies unit `u + 1`; only the last unit is seeded.
    const UNITS: usize = 200;
    let mut seeds = vec![None; UNITS];
    seeds[UNITS - 1] = Some(1);
    let program = OrderedProgram {
        seeds,
        candidates: (0..UNITS).map(|unit| vec![unit + 1]).collect(),
        self_read: vec![false; UNITS],
    };
    let run = run_ordered(&program, usize::MAX);
    assert!(run.slots.iter().all(Option::is_some));
    assert_eq!(run.stats.round_count(), UNITS + 1);
    // Every unit once in sweep 0, then one unit a sweep: linear, where the
    // full scan visits every unit in every sweep.
    assert_eq!(run.stats.evaluation_count(), 2 * UNITS - 1);
}

#[test]
fn a_forward_chain_settles_in_one_sweep() {
    // Unit `u` copies unit `u - 1`, so each change is seen later in the
    // same sweep and the next sweep has nothing to run.
    const UNITS: usize = 50;
    let mut seeds = vec![None; UNITS];
    seeds[0] = Some(1);
    let program = OrderedProgram {
        seeds,
        candidates: (0..UNITS)
            .map(|unit| vec![unit.saturating_sub(1)])
            .collect(),
        self_read: vec![false; UNITS],
    };
    let run = run_ordered(&program, usize::MAX);
    assert!(run.slots.iter().all(Option::is_some));
    assert_eq!(run.stats.round_count(), 2);
    assert_eq!(run.stats.evaluation_count(), UNITS);
}

#[test]
fn a_refused_sweep_stops_unconverged() {
    const UNITS: usize = 10;
    let mut seeds = vec![None; UNITS];
    seeds[UNITS - 1] = Some(1);
    let program = OrderedProgram {
        seeds,
        candidates: (0..UNITS).map(|unit| vec![unit + 1]).collect(),
        self_read: vec![false; UNITS],
    };
    let run = run_ordered(&program, 3);
    assert_eq!(run.stats.round_count(), 3);
    assert!(!run.stats.converged());
    let set = run.slots.iter().filter(|slot| slot.is_some()).count();
    assert_eq!(set, 3, "one link a sweep");
}

#[test]
fn a_dependency_dropped_by_a_later_evaluation_does_not_requeue() {
    // Unit 1 reads slot 2 only while slot 0 is unset. Slot 2 changes again
    // after unit 1 dropped it, which must not run unit 1 again.
    let mut evaluated = Vec::new();
    let mut slots = [0_u8; 3];
    let stats = ordered_fixpoint(
        3,
        |_| true,
        |unit, access: &mut SlotAccess<usize>| {
            evaluated.push(unit);
            match unit {
                0 => {
                    access.read(2);
                    if slots[2] > 0 && slots[0] == 0 {
                        slots[0] = 1;
                        access.changed(0);
                    }
                }
                1 => {
                    access.read(0);
                    if slots[0] == 0 {
                        access.read(2);
                    }
                }
                _ => {
                    access.read(0);
                    let target = slots[0] + 1;
                    if slots[2] < target {
                        slots[2] = target;
                        access.changed(2);
                    }
                }
            }
        },
    );
    // Sweep 0: unit 2 sets slot 2 and queues units 0 and 1 for sweep 1.
    // Sweep 1: unit 0 sets slot 0 and queues units 1 and 2 later in the
    // sweep; unit 1 drops slot 2; unit 2 raises slot 2 again, which queues
    // unit 0 alone. Sweep 2 changes nothing.
    assert_eq!(evaluated, [0, 1, 2, 0, 1, 2, 0]);
    assert_eq!(slots, [1, 0, 2]);
    assert_eq!(stats.round_count(), 3);
    assert!(stats.converged());
}

#[test]
fn no_units_run_one_empty_sweep() {
    let stats = ordered_fixpoint(0, |_| true, |_, _: &mut SlotAccess<usize>| {});
    assert_eq!(stats.round_count(), 1);
    assert_eq!(stats.evaluation_count(), 0);
    assert!(stats.converged());
}
