//! Integration tests: the text readers answer hostile indices with an error
//! instead of allocating for them.

use cfglib::{TextErrorKind, parse_cfg_text, parse_text};

/// The most removed slots a text document may ask a reader to recreate, as
/// the readers report it.
fn hole_limit() -> usize {
    let error = parse_text("n4294967294\n").expect_err("four billion slots are not a document");
    match error.kind {
        TextErrorKind::TooManyHoles { limit } => limit,
        other => panic!("expected the hole limit, got {other:?}"),
    }
}

fn blocks(text: &str) -> Result<usize, TextErrorKind> {
    parse_cfg_text(text, |line| Ok::<_, String>(line.to_owned()))
        .map(|cfg| cfg.block_count())
        .map_err(|error| error.kind)
}

#[test]
fn a_graph_index_far_beyond_the_declared_nodes_is_rejected() {
    for text in [
        "n4294967294\n",
        "n18446744073709551615\n",
        "n0 a\nn99999999999 b\n",
    ] {
        let error = parse_text(text).expect_err("the gap is not worth recreating");
        assert!(
            matches!(error.kind, TextErrorKind::TooManyHoles { .. }),
            "{text:?}: {error}"
        );
    }
}

#[test]
fn the_hole_limit_is_exact_and_counts_every_gap_in_the_document() {
    let limit = hole_limit();

    let graph = parse_text(&format!("n{limit} edge of the limit\n")).expect("exactly the limit");
    assert_eq!(graph.node_count(), 1);
    assert_eq!(graph.node_bound(), limit + 1);
    assert!(parse_text(&format!("n{} past it\n", limit + 1)).is_err());

    // Two gaps that are each within the limit but not together.
    let half = limit / 2 + 1;
    let error = parse_text(&format!("n{half}\nn{}\n", 2 * half + 1)).expect_err("gaps add up");
    assert!(matches!(error.kind, TextErrorKind::TooManyHoles { .. }));
}

#[test]
fn a_block_index_far_beyond_the_declared_blocks_is_rejected() {
    for text in [
        "bb4294967294:\n",
        "entry bb0\nbb0:\nbb18446744073709551615:\n",
    ] {
        assert!(
            matches!(blocks(text), Err(TextErrorKind::TooManyHoles { .. })),
            "{text:?}"
        );
    }
    let limit = hole_limit();
    assert_eq!(blocks(&format!("entry bb{limit}\nbb{limit}:\n")), Ok(1));
}
