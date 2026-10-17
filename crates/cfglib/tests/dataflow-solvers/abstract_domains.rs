//! `AbstractFacts::compute` with a parity domain, sound against concrete
//! executions and exact on programs whose parity is known by inspection.

use cfglib::{AbstractDomain, AbstractFacts, BlockId, Cfg, EdgeKind, InstrInfo, Lattice};

use crate::graphs::{Rng, Shape};

/// A one-register machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Set(i64),
    Add(i64),
    Double,
    Havoc,
    Call,
    Nop,
}

impl InstrInfo for Op {
    type Variable = u8;

    fn uses(&self) -> &[u8] {
        &[]
    }

    fn defs(&self) -> &[u8] {
        &[]
    }

    fn may_unwind(&self) -> bool {
        *self == Op::Call
    }
}

/// The parity of the register. `Unknown` is the least precise element (the
/// lattice bottom) and `Unconstrained` the most precise (the identity of
/// meet), following the crate's orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parity {
    Unknown,
    Even,
    Odd,
    Unconstrained,
}

fn parity_of(value: i64) -> Parity {
    if value.rem_euclid(2) == 0 {
        Parity::Even
    } else {
        Parity::Odd
    }
}

impl Parity {
    /// The identity of the meet. The solver never asks for it; the lattice
    /// laws below do.
    fn top() -> Self {
        Parity::Unconstrained
    }

    /// The lattice order, which the laws below check against the meet.
    fn leq(self, other: Self) -> bool {
        self == other || self == Parity::Unknown || other == Parity::Unconstrained
    }
}

impl Lattice for Parity {
    fn bottom() -> Self {
        Parity::Unknown
    }

    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (Parity::Unconstrained, x) | (x, Parity::Unconstrained) => *x,
            (a, b) if a == b => *a,
            _ => Parity::Unknown,
        }
    }
}

impl AbstractDomain<Op> for Parity {
    fn transfer(state: &Self, op: &Op) -> Self {
        match (op, state) {
            (Op::Set(value) | Op::Add(value), Parity::Even) | (Op::Set(value), _) => {
                parity_of(*value)
            }
            (Op::Add(value), Parity::Odd) => parity_of(*value + 1),
            (Op::Add(_) | Op::Nop, other) => *other,
            (Op::Double, _) => Parity::Even,
            (Op::Havoc | Op::Call, _) => Parity::Unknown,
        }
    }

    fn entry_value() -> Self {
        Parity::Even
    }
}

const ELEMENTS: [Parity; 4] = [
    Parity::Unknown,
    Parity::Even,
    Parity::Odd,
    Parity::Unconstrained,
];

#[test]
fn the_parity_lattice_satisfies_the_meet_semilattice_laws_and_orders_by_meet() {
    for a in ELEMENTS {
        assert_eq!(a.meet(&a), a, "idempotent");
        assert_eq!(a.meet(&Parity::top()), a, "top is the identity");
        assert_eq!(
            a.meet(&Parity::bottom()),
            Parity::bottom(),
            "bottom absorbs"
        );
        assert!(Parity::bottom().leq(a) && a.leq(Parity::top()));
        for b in ELEMENTS {
            assert_eq!(a.meet(&b), b.meet(&a), "commutative");
            assert_eq!(
                a.leq(b),
                a.meet(&b) == a,
                "{a:?} below {b:?} exactly when their meet is {a:?}"
            );
            for c in ELEMENTS {
                assert_eq!(a.meet(&b).meet(&c), a.meet(&b.meet(&c)), "associative");
            }
        }
    }
}

fn program(shape: &Shape, code: Vec<Vec<Op>>) -> Cfg<Op> {
    shape.build(code)
}

fn id(index: usize) -> BlockId {
    BlockId::from_index(index)
}

#[test]
fn straight_line_code_is_exact_and_each_block_boundary_is_reported() {
    let shape = Shape::new(2, &[(0, 1)]);
    let cfg = program(
        &shape,
        vec![vec![Op::Set(1), Op::Add(1)], vec![Op::Add(3), Op::Double]],
    );
    let facts = AbstractFacts::<Parity>::compute(&cfg);
    assert_eq!(
        facts.fact_in(id(0)),
        Some(&Parity::Even),
        "the entry starts at the domain's entry value"
    );
    assert_eq!(
        facts.fact_out(id(0)),
        Some(&Parity::Even),
        "set 1 then add 1"
    );
    assert_eq!(facts.fact_in(id(1)), Some(&Parity::Even));
    assert_eq!(facts.fact_out(id(1)), Some(&Parity::Even));
}

#[test]
fn a_join_of_different_parities_is_unknown_and_of_equal_parities_is_kept() {
    // entry -> left (odd) / right (even) -> join; and entry -> a / b -> merge, both odd.
    let shape = Shape::new(
        7,
        &[
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (0, 4),
            (0, 5),
            (4, 6),
            (5, 6),
        ],
    );
    let code = vec![
        vec![],
        vec![Op::Set(1)],
        vec![Op::Set(2)],
        vec![],
        vec![Op::Set(3)],
        vec![Op::Set(5)],
        vec![],
    ];
    let facts = AbstractFacts::<Parity>::compute(&program(&shape, code));
    assert_eq!(facts.fact_in(id(3)), Some(&Parity::Unknown));
    assert_eq!(facts.fact_in(id(6)), Some(&Parity::Odd));
}

#[test]
fn a_loop_whose_body_preserves_parity_keeps_it_through_the_header() {
    // entry -> head -> body -> head, head -> exit; the body adds 2.
    let shape = Shape::new(4, &[(0, 1), (1, 2), (2, 1), (1, 3)]);
    let code = vec![vec![Op::Set(0)], vec![], vec![Op::Add(2)], vec![]];
    let facts = AbstractFacts::<Parity>::compute(&program(&shape, code));
    assert_eq!(facts.fact_in(id(1)), Some(&Parity::Even));
    assert_eq!(facts.fact_in(id(3)), Some(&Parity::Even));

    let flipping = vec![vec![Op::Set(0)], vec![], vec![Op::Add(1)], vec![]];
    let facts = AbstractFacts::<Parity>::compute(&program(&shape, flipping));
    assert_eq!(
        facts.fact_in(id(1)),
        Some(&Parity::Unknown),
        "even on entry, odd around the back edge"
    );
}

#[test]
fn an_entry_that_heads_a_loop_still_starts_from_the_entry_value() {
    // The entry is also the loop header: entry -> body -> entry. The body
    // preserves parity, so the header's input is the entry value met with
    // the back edge's agreeing parity.
    let shape = Shape::new(3, &[(0, 1), (1, 0), (0, 2)]);
    let code = vec![vec![], vec![Op::Add(2)], vec![]];
    let facts = AbstractFacts::<Parity>::compute(&program(&shape, code));
    assert_eq!(facts.fact_in(id(0)), Some(&Parity::Even));
    assert_eq!(facts.fact_in(id(1)), Some(&Parity::Even));
    assert_eq!(facts.fact_in(id(2)), Some(&Parity::Even));
}

#[test]
fn unreached_blocks_retain_the_bottom_state() {
    let shape = Shape::new(3, &[(0, 1), (2, 1)]);
    let code = vec![vec![Op::Set(1)], vec![], vec![Op::Set(2)]];
    let facts = AbstractFacts::<Parity>::compute(&program(&shape, code));
    assert_eq!(facts.fact_in(id(2)), Some(&Parity::Unknown));
    assert_eq!(facts.fact_out(id(2)), Some(&Parity::Unknown));
    assert_eq!(
        facts.fact_in(id(1)),
        Some(&Parity::Odd),
        "the unreached block contributes nothing to the join"
    );
}

#[test]
fn an_unwind_edge_carries_the_meet_of_the_states_before_each_throwing_instruction() {
    let shape = Shape {
        blocks: 3,
        edges: vec![
            (0, 1, EdgeKind::Fallthrough),
            (0, 2, EdgeKind::ExceptionUnwind),
        ],
        hole: None,
    };
    // Even before the first call, odd before the second: the handler sees
    // their meet; the normal successor sees the state after the last call.
    let code = vec![
        vec![Op::Set(0), Op::Call, Op::Set(1), Op::Call, Op::Set(2)],
        vec![],
        vec![],
    ];
    let facts = AbstractFacts::<Parity>::compute(&program(&shape, code));
    assert_eq!(facts.fact_in(id(2)), Some(&Parity::Unknown));
    assert_eq!(facts.fact_in(id(1)), Some(&Parity::Even));

    // Both throwing points agree, so the handler keeps the parity.
    let agreeing = vec![
        vec![Op::Set(0), Op::Call, Op::Set(2), Op::Call],
        vec![],
        vec![],
    ];
    let facts = AbstractFacts::<Parity>::compute(&program(&shape, agreeing));
    assert_eq!(facts.fact_in(id(2)), Some(&Parity::Even));
    assert_eq!(
        facts.fact_in(id(1)),
        Some(&Parity::Unknown),
        "the call clobbers the register"
    );
}

fn random_ops(rng: &mut Rng) -> Vec<Op> {
    (0..rng.below(4))
        .map(|_| match rng.below(6) {
            0 => Op::Set(i64::try_from(rng.below(9)).expect("small") - 4),
            1 => Op::Add(i64::try_from(rng.below(5)).expect("small") - 2),
            2 => Op::Double,
            3 => Op::Havoc,
            4 => Op::Call,
            _ => Op::Nop,
        })
        .collect()
}

/// One concrete execution: the register at each block entry, where control
/// goes at random and a call may throw to the block's handler.
fn execute(cfg: &Cfg<Op>, rng: &mut Rng) -> Vec<(BlockId, i64)> {
    let mut entered = Vec::new();
    let mut register = 0_i64;
    let mut block = cfg.entry();
    for _ in 0..40 {
        entered.push((block, register));
        let mut thrown_to = None;
        for op in cfg.block(block).instructions() {
            if *op == Op::Call {
                let handler = cfg
                    .outgoing(block)
                    .map(|edge| cfg.edge(edge))
                    .find(|edge| edge.kind() == EdgeKind::ExceptionUnwind)
                    .map(|edge| edge.target());
                if let Some(handler) = handler.filter(|_| rng.below(2) == 0) {
                    thrown_to = Some(handler);
                    break;
                }
            }
            register = match op {
                Op::Set(value) => *value,
                Op::Add(value) => register.wrapping_add(*value),
                Op::Double => register.wrapping_mul(2),
                Op::Havoc | Op::Call => i64::try_from(rng.below(21)).expect("small") - 10,
                Op::Nop => register,
            };
        }
        let normal: Vec<BlockId> = cfg
            .outgoing(block)
            .map(|edge| cfg.edge(edge))
            .filter(|edge| edge.kind() != EdgeKind::ExceptionUnwind)
            .map(|edge| edge.target())
            .collect();
        block = match thrown_to {
            Some(handler) => handler,
            None if normal.is_empty() => return entered,
            None => normal[rng.below(normal.len())],
        };
    }
    entered
}

#[test]
fn abstract_states_cover_the_register_in_every_concrete_execution() {
    let mut rng = Rng::new(51);
    for _ in 0..200 {
        let shape = Shape::random(&mut rng, false, true);
        let code = (0..shape.blocks).map(|_| random_ops(&mut rng)).collect();
        let cfg = shape.build(code);
        let facts = AbstractFacts::<Parity>::compute(&cfg);
        for _ in 0..20 {
            for (block, value) in execute(&cfg, &mut rng) {
                match facts.fact_in(block) {
                    Some(Parity::Even) => {
                        assert_eq!(value.rem_euclid(2), 0, "even at {block:?} in {shape:?}");
                    }
                    Some(Parity::Odd) => {
                        assert_eq!(value.rem_euclid(2), 1, "odd at {block:?} in {shape:?}");
                    }
                    _ => {}
                }
            }
        }
    }
}
