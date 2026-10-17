//! `SelectionResolution`: indexed and sparse writes, values wider than a word,
//! and the shape of what the analysis reports.

use cfglib::DenseId;
use cfglib::ir::rtl::{
    Action, BitSlice, Control, InstructionBuilder, Place, ResolvedSelection, Selection, SpanAction,
    WriteTarget,
};

use crate::bits;
use crate::kestrel::Kestrel;
use crate::kestrel_store::KestrelFunction;
use crate::model::{Store, Target};
use crate::resolution_kit::{
    computed_index, data, expect_candidate, masked, read_root, resolve, select,
    selection_positions, word,
};
use crate::tessera::{Tessera, TesseraFunction};

/// Emits `select(masked(read(id), 7), 8)`: the probe of one root.
fn probe<D: Target, B: InstructionBuilder<D>>(builder: &mut B, id: u8) {
    let value = read_root::<D, B>(builder, id);
    let index = masked::<D, B>(builder, value, 7);
    select::<D, B>(builder, index, 8);
}

in_both_dialects! {
    fn an_indexed_write_with_a_constant_index_changes_only_that_candidate<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        let write = function
            .seal(entry, |mut b| {
                let index = word::<D, _>(&mut b, 1, bits(8));
                let value = word::<D, _>(&mut b, 5, data::<D>());
                let candidates = [0, 1, 2].map(|id| Place::whole(D::root_of(id)));
                b.write(
                    WriteTarget::FiniteIndexed { index, candidates: &candidates, value },
                    D::policy(),
                );
                b.finish(Control::Advance)
            })
            .expect("seals");
        let probes = function
            .seal(entry, |mut b| {
                for id in 0..3 {
                    probe::<D, _>(&mut b, id);
                }
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution =
            resolve::<D, S>(&function, &[(0, &[4]), (1, &[0]), (2, &[6])]);

        // The write itself is reported, with the candidate its index picks.
        let at_write = selection_positions::<D, S>(&function, write)[0];
        assert_eq!(resolution.candidate(write, at_write), Some(1));
        // Only the chosen candidate changed.
        let mut registers = [4_u64, 0, 6];
        registers[1] = 5;
        for (position, register) in selection_positions::<D, S>(&function, probes)
            .into_iter()
            .zip(registers)
        {
            assert_eq!(
                resolution.candidate(probes, position),
                expect_candidate(register & 7, 8)
            );
        }
    }

    fn an_indexed_write_with_an_unknown_index_makes_every_candidate_unknown<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        function
            .seal(entry, |mut b| {
                let index = b.nondet(bits(8), D::domain());
                let value = word::<D, _>(&mut b, 5, data::<D>());
                let candidates = [0, 1, 2].map(|id| Place::whole(D::root_of(id)));
                b.write(
                    WriteTarget::FiniteIndexed { index, candidates: &candidates, value },
                    D::policy(),
                );
                b.finish(Control::Advance)
            })
            .expect("seals");
        let probes = function
            .seal(entry, |mut b| {
                for id in 0..4 {
                    probe::<D, _>(&mut b, id);
                }
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution =
            resolve::<D, S>(&function, &[(0, &[4]), (1, &[0]), (2, &[6]), (3, &[7])]);
        let answers: Vec<_> = selection_positions::<D, S>(&function, probes)
            .into_iter()
            .map(|position| resolution.candidate(probes, position))
            .collect();
        // The three candidates are unknown; the root outside the list is not.
        assert_eq!(answers, [None, None, None, Some(7)]);
    }

    fn an_indexed_write_of_a_slice_needs_a_known_root_or_it_leaves_the_root_unknown<D, S>() {
        for known in [true, false] {
            let mut function = S::start();
            let entry = S::NodeId::from_index(0);
            function
                .seal(entry, |mut b| {
                    let index = word::<D, _>(&mut b, 0, bits(8));
                    let value = word::<D, _>(&mut b, 0x7F, bits(8));
                    let candidates = [Place::<D> {
                        root: D::root_of(0),
                        slice: BitSlice::new(0, bits(8)),
                    }];
                    b.write(
                        WriteTarget::FiniteIndexed { index, candidates: &candidates, value },
                        D::policy(),
                    );
                    b.finish(Control::Advance)
                })
                .expect("seals");
            let inst = function
                .seal(entry, |mut b| {
                    let value = read_root::<D, _>(&mut b, 0);
                    let expected = word::<D, _>(&mut b, 0x17F, data::<D>());
                    let same = b.apply(&D::equal(), &[value, expected], &[bits(1)]).first;
                    select::<D, _>(&mut b, same, 2);
                    b.finish(Control::Advance)
                })
                .expect("seals");
            let entry_constants: &[(u8, &[u64])] = if known { &[(0, &[0x100])] } else { &[] };
            let resolution = resolve::<D, S>(&function, entry_constants);
            let position = selection_positions::<D, S>(&function, inst)[0];
            // 0x100 with its low byte replaced by 0x7F is 0x17F.
            let expected = known.then_some(1);
            assert_eq!(resolution.candidate(inst, position), expected, "known = {known}");
        }
    }

    fn a_sparse_write_makes_its_bank_unknown<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        let before = function
            .seal(entry, |mut b| {
                probe::<D, _>(&mut b, 9);
                b.finish(Control::Advance)
            })
            .expect("seals");
        function
            .seal(entry, |mut b| {
                let key = word::<D, _>(&mut b, 0x1b, bits(16));
                let value = word::<D, _>(&mut b, 1, data::<D>());
                b.write(
                    WriteTarget::SparseIndexed {
                        bank: D::root_of(9),
                        key,
                        slice: BitSlice::whole(data::<D>()),
                        value,
                    },
                    D::policy(),
                );
                b.finish(Control::Advance)
            })
            .expect("seals");
        let after = function
            .seal(entry, |mut b| {
                probe::<D, _>(&mut b, 9);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<D, S>(&function, &[(9, &[7])]);
        let at = |inst| {
            let position = selection_positions::<D, S>(&function, inst)[0];
            resolution.candidate(inst, position)
        };
        assert_eq!((at(before), at(after)), (Some(7), None));
    }

    fn an_index_wider_than_a_machine_word_resolves_only_when_its_high_words_are_zero<D, S>() {
        // (width, words, expected candidate among four)
        let cases: [(u16, Vec<u64>, Option<usize>); 5] = [
            (8192, vec![2; 1].into_iter().chain([0; 127]).collect(), Some(2)),
            (8192, [3].into_iter().chain([0; 126]).chain([1]).collect(), None),
            (128, vec![3, 1], None),
            (65, vec![3, 0], Some(3)),
            (64, vec![u64::MAX], None),
        ];
        for (width, words, expected) in cases {
            let mut function = S::start();
            let entry = S::NodeId::from_index(0);
            let inst = function
                .seal(entry, |mut b| {
                    let index = b.constant(&words, bits(width));
                    select::<D, _>(&mut b, index, 4);
                    b.finish(Control::Advance)
                })
                .expect("seals");
            let resolution = resolve::<D, S>(&function, &[]);
            let position = selection_positions::<D, S>(&function, inst)[0];
            assert_eq!(resolution.candidate(inst, position), expected, "width {width}");
        }
    }

    fn a_slice_that_crosses_a_word_boundary_is_read_from_both_words<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        let inst = function
            .seal(entry, |mut b| {
                let field = Place::<D> {
                    root: D::root_of(8),
                    slice: BitSlice::new(60, bits(8)),
                };
                let byte = b.read(Selection::Direct(field));
                select::<D, _>(&mut b, byte, 100);
                b.finish(Control::Advance)
            })
            .expect("seals");
        // Bits 60..64 are 0xA and bits 64..68 are 0x5, so the field is 0x5A.
        let words = [0xA_u64 << 60, 0x5, 0, 0];
        let resolution = resolve::<D, S>(&function, &[(8, &words)]);
        let position = selection_positions::<D, S>(&function, inst)[0];
        assert_eq!(resolution.candidate(inst, position), expect_candidate(0x5A, 100));
    }

    fn a_partial_write_to_a_wide_root_preserves_the_words_it_does_not_cover<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        function
            .seal(entry, |mut b| {
                let root = D::root_of(8);
                let rest = D::root_width(root).width() - 64;
                let low = word::<D, _>(&mut b, 9, bits(64));
                let plan = [
                    SpanAction {
                        span: BitSlice::new(0, bits(64)),
                        action: Action::TakeInput { value: low, offset: 0 },
                    },
                    SpanAction {
                        span: BitSlice::new(64, bits(rest)),
                        action: Action::PreservePreState,
                    },
                ];
                b.write(WriteTarget::Root { root, plan: &plan }, D::policy());
                b.finish(Control::Advance)
            })
            .expect("seals");
        let inst = function
            .seal(entry, |mut b| {
                for (offset, width) in [(0, 8), (64, 8), (128, 2)] {
                    let field = Place::<D> {
                        root: D::root_of(8),
                        slice: BitSlice::new(offset, bits(width)),
                    };
                    let value = b.read(Selection::Direct(field));
                    select::<D, _>(&mut b, value, 16);
                }
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<D, S>(&function, &[(8, &[1, 2, 3, 4])]);
        let answers: Vec<_> = selection_positions::<D, S>(&function, inst)
            .into_iter()
            .map(|position| resolution.candidate(inst, position))
            .collect();
        // The words of the entry value are [1, 2, 3, 4]; the write replaces
        // the first with 9 and keeps the others.
        assert_eq!(answers, [Some(9), Some(2), Some(3)]);
    }

    fn resolutions_are_sorted_by_instruction_and_statement_and_looked_up_by_both<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        let next = function.new_block();
        function.connect(entry, next);
        // The record of the second block is sealed first, so instruction
        // order is not block order.
        let in_next = function
            .seal(next, |mut b| {
                for choice in [1, 3] {
                    let index = word::<D, _>(&mut b, choice, bits(8));
                    select::<D, _>(&mut b, index, 4);
                    b.nondet(bits(8), D::domain());
                }
                b.finish(Control::Advance)
            })
            .expect("seals");
        let in_entry = function
            .seal(entry, |mut b| {
                let index = word::<D, _>(&mut b, 2, bits(8));
                select::<D, _>(&mut b, index, 4);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<D, S>(&function, &[]);
        let (first, second) = {
            let positions = selection_positions::<D, S>(&function, in_next);
            (positions[0], positions[1])
        };
        let third = selection_positions::<D, S>(&function, in_entry)[0];
        let mut expected = vec![
            ResolvedSelection { inst: in_next, statement: first, candidate: 1 },
            ResolvedSelection { inst: in_next, statement: second, candidate: 3 },
            ResolvedSelection { inst: in_entry, statement: third, candidate: 2 },
        ];
        expected.sort();
        assert_eq!(resolution.resolved(), expected);
        assert_eq!(resolution.candidate(in_next, second), Some(3));
        assert_eq!(resolution.candidate(in_next, second + 1), None);
        assert_eq!(resolution.clone(), resolution);
    }

    fn a_selection_in_an_unreachable_block_is_absent<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        let island = function.new_block();
        let reachable = function
            .seal(entry, |mut b| {
                probe::<D, _>(&mut b, 0);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let unreachable = function
            .seal(island, |mut b| {
                probe::<D, _>(&mut b, 0);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<D, S>(&function, &[(0, &[2])]);
        let in_reachable = selection_positions::<D, S>(&function, reachable)[0];
        let in_island = selection_positions::<D, S>(&function, unreachable)[0];
        assert_eq!(resolution.candidate(reachable, in_reachable), Some(2));
        assert_eq!(resolution.candidate(unreachable, in_island), None);
        assert_eq!(resolution.resolved().len(), 1);
    }

    fn a_removed_block_does_not_disturb_the_resolution<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        let dead = function.new_block();
        function
            .seal(dead, |mut b| {
                probe::<D, _>(&mut b, 1);
                b.finish(Control::Advance)
            })
            .expect("seals");
        function.remove(dead);
        let live = function
            .seal(entry, |mut b| {
                probe::<D, _>(&mut b, 0);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<D, S>(&function, &[(0, &[6]), (1, &[3])]);
        let position = selection_positions::<D, S>(&function, live)[0];
        assert_eq!(resolution.candidate(live, position), Some(6));
        assert_eq!(resolution.resolved().len(), 1);
    }

    fn an_operator_without_a_constant_evaluation_leaves_the_index_unknown<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        let inst = function
            .seal(entry, |mut b| {
                let width = data::<D>();
                let left = word::<D, _>(&mut b, 1, width);
                let right = word::<D, _>(&mut b, 2, width);
                let opaque = b.apply(&D::opaque(), &[left, right], &[width]).first;
                let index = masked::<D, _>(&mut b, opaque, 3);
                select::<D, _>(&mut b, index, 4);
                // The same operands through an operator that folds.
                let sum = b.apply(&D::add(), &[left, right], &[width]).first;
                select::<D, _>(&mut b, sum, 4);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<D, S>(&function, &[]);
        let positions = selection_positions::<D, S>(&function, inst);
        assert_eq!(resolution.candidate(inst, positions[0]), None);
        assert_eq!(resolution.candidate(inst, positions[1]), Some(3));
    }

    fn a_value_that_comes_from_an_effect_or_a_choice_is_never_constant<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        let inst = function
            .seal(entry, |mut b| {
                let width = data::<D>();
                let chosen = b.nondet(width, D::domain());
                select::<D, _>(&mut b, chosen, 4);
                let address = word::<D, _>(&mut b, 0x40, width);
                let loaded = b.effect(&D::effect(), &[address], &[width], &[D::outcome()]).first;
                select::<D, _>(&mut b, loaded, 4);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<D, S>(&function, &[]);
        assert_eq!(selection_positions::<D, S>(&function, inst).len(), 2);
        assert_eq!(resolution.resolved(), []);
    }

    fn the_window_arithmetic_wraps_inside_the_data_width<D, S>() {
        // The top of the window is the largest value of the data width, so
        // adding one wraps to zero before the mask.
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        let inst = function
            .seal(entry, |mut b| {
                let index = computed_index::<D, _>(&mut b, 0, 1, 7);
                select::<D, _>(&mut b, index, 8);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let top = if data::<D>().width() == 32 { u64::from(u32::MAX) } else { u64::MAX };
        let resolution = resolve::<D, S>(&function, &[(0, &[top])]);
        let position = selection_positions::<D, S>(&function, inst)[0];
        let wrapped = if data::<D>().width() == 32 {
            u64::from(u32::try_from(top).expect("u32 max").wrapping_add(1))
        } else {
            top.wrapping_add(1)
        };
        assert_eq!(wrapped, 0);
        assert_eq!(resolution.candidate(inst, position), expect_candidate(wrapped & 7, 8));
    }

    fn two_indexed_writes_of_disjoint_slices_of_one_root_both_take_effect<D, S>() {
        let mut function = S::start();
        let entry = S::NodeId::from_index(0);
        function
            .seal(entry, |mut b| {
                let index = word::<D, _>(&mut b, 0, bits(8));
                let low = [Place::<D> {
                    root: D::root_of(0),
                    slice: BitSlice::new(0, bits(16)),
                }];
                let high = [Place::<D> {
                    root: D::root_of(0),
                    slice: BitSlice::new(16, bits(16)),
                }];
                for (candidates, value) in [(&low, 1), (&high, 2)] {
                    let value = word::<D, _>(&mut b, value, bits(16));
                    b.write(
                        WriteTarget::FiniteIndexed { index, candidates, value },
                        D::policy(),
                    );
                }
                b.finish(Control::Advance)
            })
            .expect("seals");
        let inst = function
            .seal(entry, |mut b| {
                let width = data::<D>();
                let value = read_root::<D, _>(&mut b, 0);
                let both = word::<D, _>(&mut b, 0x2_0001, width);
                let same = b.apply(&D::equal(), &[value, both], &[bits(1)]).first;
                select::<D, _>(&mut b, same, 2);
                let low = masked::<D, _>(&mut b, value, 3);
                select::<D, _>(&mut b, low, 4);
                b.finish(Control::Advance)
            })
            .expect("seals");
        // Both writes read the root as it was, and each covers its own half,
        // so the root ends as 0x0002_0001.
        let resolution = resolve::<D, S>(&function, &[(0, &[0])]);
        let positions = selection_positions::<D, S>(&function, inst);
        assert_eq!(resolution.candidate(inst, positions[0]), Some(1));
        assert_eq!(resolution.candidate(inst, positions[1]), Some(1));
    }
}

#[test]
fn bits_past_the_root_width_of_an_entry_constant_are_ignored() {
    // `Wide` is 130 bits wide; the entry value sets every bit of its third
    // word, but only the low two bits of it belong to the root.
    let mut function = KestrelFunction::new();
    let entry = function.entry();
    let inst = function
        .seal(entry, |mut b| {
            let value = b.read(Selection::Direct(Place::whole(crate::kestrel::Reg::Wide)));
            let mut tail = [0_u64; 3];
            tail[2] = 3;
            let expected = b.constant(&tail, bits(130));
            let same = b
                .apply(&crate::kestrel::KOp::Eq, &[value, expected], &[bits(1)])
                .first;
            select::<Kestrel, _>(&mut b, same, 2);
            b.finish(Control::Advance)
        })
        .expect("seals");
    let resolution = resolve::<Kestrel, KestrelFunction>(&function, &[(8, &[0, 0, u64::MAX])]);
    let position = selection_positions::<Kestrel, KestrelFunction>(&function, inst)[0];
    assert_eq!(resolution.candidate(inst, position), Some(1));
}

#[test]
fn division_results_fill_both_handles_and_a_zero_divisor_does_not_fold() {
    let mut function = TesseraFunction::new();
    let entry = 0;
    let mut insts = Vec::new();
    for divisor in [5_u64, 0] {
        insts.push(
            function
                .seal(entry, |mut b| {
                    let width = bits(64);
                    let dividend = word::<Tessera, _>(&mut b, 17, width);
                    let divisor = word::<Tessera, _>(&mut b, divisor, width);
                    let both = b.apply(&"divmod".to_owned(), &[dividend, divisor], &[width, width]);
                    select::<Tessera, _>(&mut b, both.first, 8);
                    select::<Tessera, _>(&mut b, both.get(1).expect("two results"), 8);
                    b.finish(Control::Advance)
                })
                .expect("seals"),
        );
    }
    let resolution = resolve::<Tessera, TesseraFunction>(&function, &[]);
    let answers: Vec<_> = insts
        .iter()
        .map(|&inst| {
            selection_positions::<Tessera, TesseraFunction>(&function, inst)
                .into_iter()
                .map(|position| resolution.candidate(inst, position))
                .collect::<Vec<_>>()
        })
        .collect();
    // 17 / 5 = 3 and 17 % 5 = 2; with a zero divisor neither is known.
    assert_eq!(
        answers,
        [vec![Some(17 / 5), Some(17 % 5)], vec![None, None]]
    );
}

#[test]
fn results_of_different_word_counts_are_read_at_their_own_offsets() {
    let mut function = TesseraFunction::new();
    let inst = function
        .seal(0, |mut b| {
            let wide = b.constant(&[1, 2, 3, 4], bits(256));
            let small = word::<Tessera, _>(&mut b, 5, bits(64));
            // `swap` returns its operands exchanged: the 64-bit value first
            // (one output word), then the 256-bit value (four output words).
            let swapped = b.apply(&"swap".to_owned(), &[wide, small], &[bits(64), bits(256)]);
            select::<Tessera, _>(&mut b, swapped.first, 8);
            let back = swapped.get(1).expect("two results");
            let same = b.apply(&"eq".to_owned(), &[back, wide], &[bits(1)]).first;
            select::<Tessera, _>(&mut b, same, 2);
            b.finish(Control::Advance)
        })
        .expect("seals");
    let resolution = resolve::<Tessera, TesseraFunction>(&function, &[]);
    let positions = selection_positions::<Tessera, TesseraFunction>(&function, inst);
    assert_eq!(resolution.candidate(inst, positions[0]), Some(5));
    assert_eq!(resolution.candidate(inst, positions[1]), Some(1));
}
