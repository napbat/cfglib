//! `Cfg::split_block` and its relatives, on the graph itself.

use cfglib::{Cfg, EdgeKind, SplitPointError};

use crate::stmt::{Route, Stmt, block, fill};

fn five_statement_block() -> (Cfg<Stmt, Route>, cfglib::BlockId, cfglib::EdgeId) {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    fill(
        &mut cfg,
        entry,
        [
            Stmt::constant("a", 1),
            Stmt::constant("b", 2),
            Stmt::add("c", "a", "b"),
            Stmt::multiply("d", "c", "c"),
            Stmt::effect(&[crate::stmt::Effect::Output], &["d"]),
        ],
    );
    let exit = block(&mut cfg, [Stmt::ret()]);
    let leaving = cfg.add_edge_with_payload(entry, exit, EdgeKind::Jump, Route(7));
    (cfg, entry, leaving)
}

#[test]
fn splitting_moves_the_tail_and_every_exit_to_a_new_block_behind_a_fallthrough() {
    let (mut cfg, entry, leaving) = five_statement_block();
    let original: Vec<Stmt> = cfg.block(entry).instructions().to_vec();

    let tail = cfg.split_block(entry, 2);

    assert_eq!(cfg.block(entry).instructions(), &original[..2]);
    assert_eq!(cfg.block(tail).instructions(), &original[2..]);
    assert_eq!(
        cfg.edge(leaving).source(),
        tail,
        "exits leave from the tail"
    );
    assert_eq!(cfg.edge(leaving).payload(), &Route(7));
    let joining: Vec<_> = cfg.outgoing(entry).collect();
    assert_eq!(joining.len(), 1);
    assert_eq!(cfg.edge(joining[0]).target(), tail);
    assert_eq!(cfg.edge(joining[0]).kind(), EdgeKind::Fallthrough);
    assert_eq!(cfg.edge(joining[0]).payload(), &Route::default());
}

#[test]
fn splitting_at_either_end_leaves_one_part_empty() {
    for (at, head_len, tail_len) in [(0, 0, 5), (5, 5, 0)] {
        let (mut cfg, entry, _) = five_statement_block();
        let tail = cfg.split_block_with_payload(entry, at, Route(9));

        assert_eq!(cfg.block(entry).instructions().len(), head_len);
        assert_eq!(cfg.block(tail).instructions().len(), tail_len);
        let joining = cfg.outgoing(entry).next().unwrap();
        assert_eq!(
            cfg.edge(joining).payload(),
            &Route(9),
            "the explicit payload is used"
        );
    }
}

#[test]
#[should_panic(expected = "")]
fn splitting_beyond_the_last_instruction_panics() {
    let (mut cfg, entry, _) = five_statement_block();
    cfg.split_block(entry, 6);
}

#[test]
fn the_mapped_split_names_the_new_block_edge_and_the_moved_exits() {
    let (mut cfg, entry, leaving) = five_statement_block();

    let (tail, rewrite) = cfg.split_block_with_payload_mapped(entry, 3, Route(1));

    assert_eq!(rewrite.blocks(entry), Some([entry, tail].as_slice()));
    assert_eq!(rewrite.created_blocks(), [tail]);
    assert_eq!(
        rewrite.edges(leaving),
        Some([leaving].as_slice()),
        "moved, not replaced"
    );
    assert_eq!(rewrite.created_edges().len(), 1);
}

#[test]
fn splitting_at_several_points_validates_before_changing_anything() {
    let (mut cfg, entry, _) = five_statement_block();
    let before = cfg.clone();

    let out_of_bounds = cfg.split_block_at_points(entry, &[2, 9]).unwrap_err();
    let repeated = cfg.split_block_at_points(entry, &[2, 2]).unwrap_err();
    let decreasing = cfg.split_block_at_points(entry, &[3, 1]).unwrap_err();

    assert_eq!(
        out_of_bounds,
        SplitPointError::OutOfBounds {
            point: 9,
            instruction_count: 5
        }
    );
    assert_eq!(
        repeated,
        SplitPointError::NotStrictlyIncreasing {
            previous: 2,
            point: 2
        }
    );
    assert_eq!(
        decreasing,
        SplitPointError::NotStrictlyIncreasing {
            previous: 3,
            point: 1
        }
    );
    assert_eq!(
        cfg, before,
        "an invalid sequence leaves the graph unchanged"
    );

    let (parts, _) = cfg.split_block_at_points(entry, &[1, 4]).unwrap();
    let lengths: Vec<usize> = parts
        .iter()
        .map(|&part| cfg.block(part).instructions().len())
        .collect();
    assert_eq!(parts[0], entry);
    assert_eq!(
        lengths,
        [1, 3, 1],
        "points are offsets in the original block"
    );
}

#[test]
fn split_point_errors_describe_the_offending_numbers_and_work_as_std_errors() {
    let out_of_bounds = SplitPointError::OutOfBounds {
        point: 9,
        instruction_count: 5,
    };
    let unordered = SplitPointError::NotStrictlyIncreasing {
        previous: 3,
        point: 3,
    };

    let text = out_of_bounds.to_string();
    assert!(text.contains('9') && text.contains('5'), "{text}");
    let text = unordered.to_string();
    assert!(text.contains('3'), "{text}");
    let boxed: Box<dyn std::error::Error> = Box::new(unordered);
    assert_eq!(boxed.to_string(), unordered.to_string());
    assert_ne!(out_of_bounds, unordered);
    let copied = out_of_bounds;
    assert_eq!(copied, out_of_bounds);
}
