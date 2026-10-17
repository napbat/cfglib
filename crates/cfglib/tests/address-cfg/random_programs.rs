//! A deterministic sweep of random programs through the structural oracles.
//!
//! The harness already holds every built graph to the oracles in
//! `oracles.rs`; here the programs are generated instead of hand-written, so
//! leader splitting, unresolved targets, flattened calls, nested and
//! disjoint protected ranges, and explicit entries meet in combinations no
//! hand-derived scenario lists. The oracles are independent of the builder,
//! so a disagreement names a real defect or a wrong oracle, never a shared
//! mistake.

use cfglib::{CallPolicy, HandlerKind};

use crate::harness::{Settings, TryRange, build_with};
use crate::isa::Dest::{At, Beyond, Inside};
use crate::isa::{Dest, Isa, Op};

/// A small deterministic generator (`splitmix64`), so failures reproduce.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value in `0..bound`.
    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(bound).unwrap()).unwrap()
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
}

/// An operation shape before its destinations are chosen.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Work,
    Pure,
    Jump,
    BranchIf,
    Call,
    CallIndirect,
    JumpIndirect,
    Dispatch,
    Return,
    Raise,
    Probe,
}

const SHAPES: [(Shape, usize); 11] = [
    (Shape::Work, 28),
    (Shape::Pure, 8),
    (Shape::Jump, 8),
    (Shape::BranchIf, 12),
    (Shape::Call, 8),
    (Shape::CallIndirect, 4),
    (Shape::JumpIndirect, 3),
    (Shape::Dispatch, 6),
    (Shape::Return, 8),
    (Shape::Raise, 3),
    (Shape::Probe, 5),
];

fn shape(rng: &mut Rng) -> Shape {
    let mut roll = rng.below(SHAPES.iter().map(|&(_, weight)| weight).sum());
    for (shape, weight) in SHAPES {
        if roll < weight {
            return shape;
        }
        roll -= weight;
    }
    Shape::Work
}

/// Mostly an instruction, sometimes an address nobody can resolve.
fn destination(rng: &mut Rng, shapes: &[Shape]) -> Dest {
    let position = rng.below(shapes.len());
    if rng.chance(12) {
        Beyond(1 + u64::try_from(rng.below(40)).unwrap())
    } else if rng.chance(8) && shapes[position] == Shape::Work {
        Inside(position)
    } else {
        At(position)
    }
}

fn program(rng: &mut Rng) -> Vec<Op> {
    let length = 3 + rng.below(20);
    let shapes: Vec<Shape> = (0..length).map(|_| shape(rng)).collect();
    shapes
        .iter()
        .map(|&shape| match shape {
            Shape::Work => Op::Work,
            Shape::Pure => Op::Pure,
            Shape::Jump => Op::Jump(destination(rng, &shapes)),
            Shape::BranchIf => Op::BranchIf(destination(rng, &shapes)),
            Shape::Call => Op::Call(destination(rng, &shapes)),
            Shape::CallIndirect => Op::CallIndirect,
            Shape::JumpIndirect => Op::JumpIndirect,
            Shape::Dispatch => Op::Dispatch {
                default: destination(rng, &shapes),
                cases: (0..rng.below(4))
                    .map(|case| (i16::try_from(case * 7).unwrap(), destination(rng, &shapes)))
                    .collect(),
            },
            Shape::Return => Op::Return,
            Shape::Raise => Op::Raise,
            Shape::Probe => Op::Probe(destination(rng, &shapes)),
        })
        .collect()
}

/// Whether two ranges are nested, equal, or disjoint: a laminar family has
/// no two enclosing ranges of one range that tie on span.
fn laminar(first: &core::ops::Range<usize>, second: &core::ops::Range<usize>) -> bool {
    let disjoint = first.end <= second.start || second.end <= first.start;
    let nested = (first.start <= second.start && second.end <= first.end)
        || (second.start <= first.start && first.end <= second.end);
    disjoint || nested
}

fn table(rng: &mut Rng, length: usize) -> Vec<TryRange> {
    const KINDS: [HandlerKind; 4] = [
        HandlerKind::Catch,
        HandlerKind::CatchAll,
        HandlerKind::Finally,
        HandlerKind::Fault,
    ];
    let mut table: Vec<TryRange> = Vec::new();
    for _ in 0..rng.below(4) {
        let start = rng.below(length);
        let end = start + 1 + rng.below(length - start);
        let candidate = start..end;
        if table
            .iter()
            .all(|entry| laminar(&entry.protected, &candidate))
        {
            table.push(TryRange {
                protected: candidate,
                entry: rng.below(length),
                kind: KINDS[rng.below(KINDS.len())],
            });
        }
    }
    table
}

fn random_programs_satisfy_every_structural_oracle<I: Isa>() {
    for seed in 0..300 {
        let mut rng = Rng(seed);
        let ops = program(&mut rng);
        let table = table(&mut rng, ops.len());
        let entry = rng.chance(30).then(|| rng.below(ops.len()));
        for calls in [CallPolicy::Edges, CallPolicy::Flatten] {
            // The oracles run inside the harness; a failure names its seed.
            let outcome = std::panic::catch_unwind(|| {
                build_with::<I>(&ops, table.clone(), Settings { entry, calls })
            });
            assert!(
                outcome.is_ok(),
                "seed {seed} under {calls:?} broke an oracle"
            );
        }
    }
}

over_both_isas!(random_programs_satisfy_every_structural_oracle);
