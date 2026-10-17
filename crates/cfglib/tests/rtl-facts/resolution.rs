//! `SelectionResolution`: flow of constants into indexed selections.
//!
//! Every scenario is generic over the dialect and the storage and runs in both
//! of them. The expected candidates come from plain integer arithmetic and
//! from concrete execution of the little programs, written in the test.

use std::collections::BTreeSet;

use cfglib::DenseId;
use cfglib::ir::rtl::{
    Action, BitSlice, Control, InstructionBuilder, Place, RootTable, SpanAction, WriteTarget,
};

use crate::bits;
use crate::resolution_kit::{
    computed_index, data, expect_candidate, masked, read_root, resolve, select,
    selection_positions, set_root, word, write_value,
};

in_both_dialects! {
    fn a_constant_index_resolves_to_its_candidate<D, S>() {
        for top in [6_u64, 4, 1, 2, 0, 5, 13] {
            let mut function = S::start();
            let entry = S::NodeId::from_index(0);
            let inst = function
                .seal(entry, |mut b| {
                    let index = computed_index::<D, _>(&mut b, 0, 3, 7);
                    select::<D, _>(&mut b, index, 5);
                    b.finish(Control::Advance)
                })
                .expect("seals");
            let resolution = resolve::<D, S>(&function, &[(0, &[top])]);
            let position = selection_positions::<D, S>(&function, inst)[0];
            // The window computes (top + 3) & 7 and has five slots.
            assert_eq!(
                resolution.candidate(inst, position),
                expect_candidate((top + 3) & 7, 5),
                "top = {top}"
            );
        }
    }

    fn an_index_that_depends_on_an_unknown_root_is_not_resolved<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        let inst = function
            .seal(entry, |mut b| {
                let index = computed_index::<D, _>(&mut b, 0, 3, 7);
                select::<D, _>(&mut b, index, 5);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<D, S>(&function, &[(1, &[6])]);
        assert_eq!(resolution.resolved(), []);
        assert_eq!(resolution.candidate(inst, 5), None);
    }

    fn an_entry_constant_for_a_root_the_function_does_not_name_is_ignored<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        let inst = function
            .seal(entry, |mut b| {
                let index = computed_index::<D, _>(&mut b, 0, 0, 7);
                select::<D, _>(&mut b, index, 8);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let table = RootTable::collect(&function);
        assert_eq!(table.index_of(D::root_of(7)), None);
        let resolution = resolve::<D, S>(&function, &[(7, &[1]), (0, &[3])]);
        let position = selection_positions::<D, S>(&function, inst)[0];
        assert_eq!(resolution.candidate(inst, position), Some(3));
    }

    fn an_empty_function_resolves_nothing<D, S>() {
        let function = S::start();
        assert_eq!(resolve::<D, S>(&function, &[]).resolved(), []);
    }

    fn joins_keep_a_constant_only_when_every_path_agrees<D, S>() {
        let cases: [(Option<u64>, Option<u64>, Option<u64>); 9] = [
            (None, Some(2), Some(2)),
            (None, Some(2), Some(3)),
            (Some(2), None, Some(2)),
            (Some(2), None, None),
            (Some(2), None, Some(1)),
            (None, Some(2), None),
            (Some(1), Some(2), Some(2)),
            (None, Some(9), Some(9)),
            (None, Some(5), Some(5)),
        ];
        for (entry, left, right) in cases {
            let mut function = S::start();
            let top = S::NodeId::from_index(0);
            let (l, r, join) = (function.new_block(), function.new_block(), function.new_block());
            for (from, to) in [(top, l), (top, r), (l, join), (r, join)] {
                function.connect(from, to);
            }
            for (arm, value) in [(l, left), (r, right)] {
                if let Some(value) = value {
                    function
                        .seal(arm, |mut b| {
                            set_root::<D, _>(&mut b, 0, value);
                            b.finish(Control::Advance)
                        })
                        .expect("seals");
                }
            }
            let inst = function
                .seal(join, |mut b| {
                    let value = read_root::<D, _>(&mut b, 0);
                    let index = masked::<D, _>(&mut b, value, 7);
                    select::<D, _>(&mut b, index, 5);
                    b.finish(Control::Advance)
                })
                .expect("seals");

            let words = entry.map(|value| [value]);
            let known: Vec<(u8, &[u64])> = words.iter().map(|value| (0, &value[..])).collect();
            let resolution = resolve::<D, S>(&function, &known);
            let position = selection_positions::<D, S>(&function, inst)[0];

            // Follow both paths to the join: the value that reaches it is
            // known when the arm wrote it or the entry gave it.
            let expected = match [left.or(entry), right.or(entry)] {
                [Some(a), Some(b)] if a == b => expect_candidate(a & 7, 5),
                _ => None,
            };
            assert_eq!(
                resolution.candidate(inst, position),
                expected,
                "entry {entry:?}, left {left:?}, right {right:?}"
            );
        }
    }

    fn a_loop_keeps_a_constant_only_when_every_turn_gives_the_root_the_same_value<D, S>() {
        #[derive(Clone, Copy)]
        enum Body {
            Idle,
            Set(u64),
            Increment,
        }
        // What the body does to `R0`, and the same step run on plain numbers.
        let bodies = [
            ("leaves it alone", Body::Idle),
            ("rewrites the entry value", Body::Set(2)),
            ("writes another constant", Body::Set(5)),
            ("adds one", Body::Increment),
        ];
        for (what, step) in bodies {
            let mut function = S::start();
            let top = S::NodeId::from_index(0);
            let (header, body, exit) =
                (function.new_block(), function.new_block(), function.new_block());
            for (from, to) in [(top, header), (header, body), (body, header), (header, exit)] {
                function.connect(from, to);
            }
            if !matches!(step, Body::Idle) {
                function
                    .seal(body, |mut b| {
                        match step {
                            Body::Set(value) => set_root::<D, _>(&mut b, 0, value),
                            Body::Idle => unreachable!("an idle body has no record"),
                            Body::Increment => {
                                let width = data::<D>();
                                let current = read_root::<D, _>(&mut b, 0);
                                let one = word::<D, _>(&mut b, 1, width);
                                let next = b.apply(&D::add(), &[current, one], &[width]).first;
                                write_value::<D, _>(&mut b, 0, next);
                            }
                        }
                        b.finish(Control::Advance)
                    })
                    .expect("seals");
            }
            let probes: Vec<_> = [header, exit]
                .into_iter()
                .map(|block| {
                    function
                        .seal(block, |mut b| {
                            let value = read_root::<D, _>(&mut b, 0);
                            let index = masked::<D, _>(&mut b, value, 7);
                            select::<D, _>(&mut b, index, 8);
                            b.finish(Control::Advance)
                        })
                        .expect("seals")
                })
                .collect();
            let resolution = resolve::<D, S>(&function, &[(0, &[2])]);

            // Run the loop for a few turns: the header sees one value, or
            // several.
            let mut seen = BTreeSet::new();
            let mut value = 2_u64;
            for _ in 0..4 {
                seen.insert(value);
                value = match step {
                    Body::Idle => value,
                    Body::Set(constant) => constant,
                    Body::Increment => value + 1,
                };
            }
            let expected = if seen.len() == 1 { Some(2) } else { None };
            for inst in probes {
                let position = selection_positions::<D, S>(&function, inst)[0];
                assert_eq!(
                    resolution.candidate(inst, position),
                    expected,
                    "a body that {what}"
                );
            }
        }
    }

    fn an_unreachable_predecessor_does_not_poison_a_join<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        let (join, orphan) = (function.new_block(), function.new_block());
        function.connect(entry, join);
        function.connect(orphan, join);
        function
            .seal(orphan, |mut b| {
                set_root::<D, _>(&mut b, 0, 9);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let inst = function
            .seal(join, |mut b| {
                let value = read_root::<D, _>(&mut b, 0);
                let index = masked::<D, _>(&mut b, value, 7);
                select::<D, _>(&mut b, index, 8);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<D, S>(&function, &[(0, &[2])]);
        let position = selection_positions::<D, S>(&function, inst)[0];
        // Only the entry reaches the join, so the orphan's write never lands.
        assert_eq!(resolution.candidate(inst, position), Some(2));
    }

    fn a_call_makes_every_root_unknown_after_its_record_but_not_inside_it<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        let mut insts = Vec::new();
        for calls in [false, true, false] {
            insts.push(
                function
                    .seal(entry, |mut b| {
                        let value = read_root::<D, _>(&mut b, 0);
                        let index = masked::<D, _>(&mut b, value, 7);
                        select::<D, _>(&mut b, index, 8);
                        if calls {
                            let target = word::<D, _>(&mut b, 0x40, bits(32));
                            b.finish(Control::Call { target })
                        } else {
                            b.finish(Control::Advance)
                        }
                    })
                    .expect("seals"),
            );
        }
        let resolution = resolve::<D, S>(&function, &[(0, &[2])]);
        let answers: Vec<_> = insts
            .iter()
            .map(|&inst| {
                let position = selection_positions::<D, S>(&function, inst)[0];
                resolution.candidate(inst, position)
            })
            .collect();
        // The record before the call, the call record itself (its reads see
        // the state before it), and the record after the call.
        assert_eq!(answers, [Some(2), Some(2), None]);
    }

    fn a_complete_write_of_a_constant_gives_later_reads_that_constant<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        function
            .seal(entry, |mut b| {
                set_root::<D, _>(&mut b, 1, 3);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let inst = function
            .seal(entry, |mut b| {
                let value = read_root::<D, _>(&mut b, 1);
                let index = masked::<D, _>(&mut b, value, 7);
                select::<D, _>(&mut b, index, 8);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<D, S>(&function, &[]);
        let position = selection_positions::<D, S>(&function, inst)[0];
        assert_eq!(resolution.candidate(inst, position), Some(3));
    }

    fn a_partial_write_keeps_the_known_bits_around_it<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        function
            .seal(entry, |mut b| {
                let low = word::<D, _>(&mut b, 2, bits(8));
                let rest = D::root_width(D::root_of(0)).width() - 8;
                let plan = [
                    SpanAction {
                        span: BitSlice::new(0, bits(8)),
                        action: Action::TakeInput { value: low, offset: 0 },
                    },
                    SpanAction {
                        span: BitSlice::new(8, bits(rest)),
                        action: Action::PreservePreState,
                    },
                ];
                b.write(WriteTarget::Root { root: D::root_of(0), plan: &plan }, D::policy());
                b.finish(Control::Advance)
            })
            .expect("seals");
        let inst = function
            .seal(entry, |mut b| {
                let width = data::<D>();
                let value = read_root::<D, _>(&mut b, 0);
                let low = masked::<D, _>(&mut b, value, 7);
                select::<D, _>(&mut b, low, 8);
                let whole = word::<D, _>(&mut b, 0x102, width);
                let same = b.apply(&D::equal(), &[value, whole], &[bits(1)]).first;
                select::<D, _>(&mut b, same, 2);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<D, S>(&function, &[(0, &[0x100])]);
        let positions = selection_positions::<D, S>(&function, inst);

        let after = (0x100_u64 & !0xFF) | 2;
        assert_eq!(
            resolution.candidate(inst, positions[0]),
            expect_candidate(after & 7, 8)
        );
        assert_eq!(
            resolution.candidate(inst, positions[1]),
            expect_candidate(u64::from(after == 0x102), 2)
        );
    }

    fn a_partial_write_to_an_unknown_root_leaves_it_unknown<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        function
            .seal(entry, |mut b| {
                let low = word::<D, _>(&mut b, 1, bits(8));
                let rest = D::root_width(D::root_of(0)).width() - 8;
                let plan = [
                    SpanAction {
                        span: BitSlice::new(0, bits(8)),
                        action: Action::TakeInput { value: low, offset: 0 },
                    },
                    SpanAction {
                        span: BitSlice::new(8, bits(rest)),
                        action: Action::PreservePreState,
                    },
                ];
                b.write(WriteTarget::Root { root: D::root_of(0), plan: &plan }, D::policy());
                b.finish(Control::Advance)
            })
            .expect("seals");
        let inst = function
            .seal(entry, |mut b| {
                let value = read_root::<D, _>(&mut b, 0);
                let index = masked::<D, _>(&mut b, value, 7);
                select::<D, _>(&mut b, index, 8);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<D, S>(&function, &[]);
        assert_eq!(resolution.resolved(), []);
        assert_eq!(resolution.candidate(inst, 3), None);
    }

    fn a_nondeterministic_span_makes_the_root_unknown<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        function
            .seal(entry, |mut b| {
                let width = D::root_width(D::root_of(0)).width();
                let low = word::<D, _>(&mut b, 1, bits(16));
                let plan = [
                    SpanAction {
                        span: BitSlice::new(0, bits(16)),
                        action: Action::TakeInput { value: low, offset: 0 },
                    },
                    SpanAction {
                        span: BitSlice::new(16, bits(width - 16)),
                        action: Action::Nondeterministic(D::domain()),
                    },
                ];
                b.write(WriteTarget::Root { root: D::root_of(0), plan: &plan }, D::policy());
                b.finish(Control::Advance)
            })
            .expect("seals");
        let inst = function
            .seal(entry, |mut b| {
                let value = read_root::<D, _>(&mut b, 0);
                let index = masked::<D, _>(&mut b, value, 7);
                select::<D, _>(&mut b, index, 8);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<D, S>(&function, &[(0, &[1])]);
        let position = selection_positions::<D, S>(&function, inst)[0];
        assert_eq!(resolution.candidate(inst, position), None);
    }

    fn a_write_of_a_value_that_is_not_constant_makes_the_root_unknown<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        function
            .seal(entry, |mut b| {
                let chosen = b.nondet(data::<D>(), D::domain());
                write_value::<D, _>(&mut b, 0, chosen);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let inst = function
            .seal(entry, |mut b| {
                let value = read_root::<D, _>(&mut b, 0);
                let index = masked::<D, _>(&mut b, value, 7);
                select::<D, _>(&mut b, index, 8);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<D, S>(&function, &[(0, &[1])]);
        let position = selection_positions::<D, S>(&function, inst)[0];
        assert_eq!(resolution.candidate(inst, position), None);
    }

    fn the_writes_of_one_record_land_together_after_its_reads<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        let first = function
            .seal(entry, |mut b| {
                // Selects by the old `R0`, then overwrites `R0`.
                let value = read_root::<D, _>(&mut b, 0);
                let index = masked::<D, _>(&mut b, value, 7);
                select::<D, _>(&mut b, index, 8);
                set_root::<D, _>(&mut b, 0, 5);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let second = function
            .seal(entry, |mut b| {
                let value = read_root::<D, _>(&mut b, 0);
                let index = masked::<D, _>(&mut b, value, 7);
                select::<D, _>(&mut b, index, 8);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<D, S>(&function, &[(0, &[2])]);
        let at = |inst| {
            let position = selection_positions::<D, S>(&function, inst)[0];
            resolution.candidate(inst, position)
        };
        assert_eq!((at(first), at(second)), (Some(2), Some(5)));
    }

    fn a_read_of_a_slice_takes_the_bits_of_the_known_root<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        let inst = function
            .seal(entry, |mut b| {
                let field = Place::<D> {
                    root: D::root_of(0),
                    slice: BitSlice::new(8, bits(8)),
                };
                let byte = b.read(cfglib::ir::rtl::Selection::Direct(field));
                select::<D, _>(&mut b, byte, 220);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<D, S>(&function, &[(0, &[0xAB_CDEF])]);
        let position = selection_positions::<D, S>(&function, inst)[0];
        assert_eq!(
            resolution.candidate(inst, position),
            expect_candidate((0xAB_CDEF >> 8) & 0xFF, 220)
        );
    }
}
