//! The text form of a control-flow graph: a dump a human reads and a diff
//! compares, read back with an instruction syntax the consumer owns.

use std::borrow::Cow;
use std::fmt;

use cfglib::{
    BlockId, Cfg, CompletionReason, Continuation, DisplayInstr, DominatorTree, EdgeKind,
    ExceptionFlow, HandlerBody, HandlerKind, HandlerRef, Region, TextError, TextErrorKind,
    parse_cfg_text, verify,
};

use crate::fixture::{handler, region, rich_cfg, try_finally};
use crate::golden::assert_golden;
use crate::instruction::{Isa, Line, parse_isa};
use crate::labels::{TEXT_ALPHABET, strings_over};
use crate::sink::assert_failure_is_reported_at_every_length;

fn parse(text: &str) -> Result<Cfg<Line>, TextError> {
    parse_cfg_text(text, |line| Ok(Line::new(line)))
}

/// Everything the text form promises to carry, read through the accessors a
/// consumer has: the entry, the block slots, their labels and instructions,
/// the edges with their kinds in insertion order, and the regions.
#[derive(Debug, PartialEq, Eq)]
struct Shape {
    entry: BlockId,
    block_bound: usize,
    blocks: Vec<(BlockId, Option<String>, Vec<String>)>,
    edges: Vec<(BlockId, BlockId, EdgeKind)>,
    regions: Vec<Region>,
}

fn shape<E>(cfg: &Cfg<Line, E>) -> Shape {
    Shape {
        entry: cfg.entry(),
        block_bound: cfg.block_bound(),
        blocks: cfg
            .block_ids()
            .map(|id| {
                let block = cfg.block(id);
                (
                    id,
                    block.label().map(str::to_owned),
                    block
                        .instructions()
                        .iter()
                        .map(|line| line.0.clone())
                        .collect(),
                )
            })
            .collect(),
        edges: cfg
            .edges()
            .map(|edge| (edge.source(), edge.target(), edge.kind()))
            .collect(),
        regions: cfg.regions().to_vec(),
    }
}

#[test]
fn a_try_finally_is_written_as_blocks_edges_and_regions() {
    assert_golden("text-dot/cfg-try-finally.txt", &try_finally().to_text());
}

#[test]
fn every_edge_kind_is_written_under_its_canonical_name() {
    let mut cfg = Cfg::<Line>::new();
    for kind in EdgeKind::ALL {
        let target = cfg.new_block();
        cfg.add_edge(cfg.entry(), target, kind);
    }
    assert_golden("text-dot/cfg-every-edge-kind.txt", &cfg.to_text());

    let text = cfg.to_text();
    for (index, kind) in EdgeKind::ALL.into_iter().enumerate() {
        let expected = format!("bb0 -> bb{} {}", index + 1, kind.name());
        assert!(
            text.lines().any(|line| line == expected),
            "no line {expected:?}"
        );
    }
}

#[test]
fn parsing_a_written_cfg_reproduces_blocks_edges_regions_and_holes() {
    let original = rich_cfg();
    let parsed = parse(&original.to_text()).expect("the writer emits its own grammar");

    assert_eq!(shape(&parsed), shape(&original));
    assert_eq!(parsed.block_count(), original.block_count());
    assert_eq!(
        parsed.block_bound(),
        original.block_bound(),
        "removed blocks keep their slots, so every identity keeps its meaning"
    );
    assert_eq!(parsed.edge_count(), EdgeKind::ALL.len() + 4);
}

#[test]
fn writing_parsing_and_writing_again_is_a_fixed_point() {
    for original in [rich_cfg(), try_finally()] {
        let once = original.to_text();
        let twice = parse(&once).expect("its own grammar").to_text();
        assert_eq!(once, twice);
    }
}

#[test]
fn the_parsed_cfg_answers_analyses_like_the_original() {
    let original = rich_cfg();
    let parsed = parse(&original.to_text()).expect("its own grammar");

    assert_eq!(
        DominatorTree::compute(&parsed),
        DominatorTree::compute(&original)
    );
    assert_eq!(verify(&parsed).is_ok(), verify(&original).is_ok());
    for block in original.block_ids() {
        assert_eq!(
            parsed.protecting_region(block).map(|region| region.id),
            original.protecting_region(block).map(|region| region.id),
            "{block}"
        );
        assert_eq!(
            parsed.successors(block).collect::<Vec<_>>(),
            original.successors(block).collect::<Vec<_>>()
        );
    }
}

#[test]
fn parallel_edges_keep_their_count_kinds_and_order() {
    let mut cfg = Cfg::<Line>::new();
    let next = cfg.new_block();
    for kind in [
        EdgeKind::ConditionalTrue,
        EdgeKind::ConditionalTrue,
        EdgeKind::Fallthrough,
        EdgeKind::ConditionalTrue,
    ] {
        cfg.add_edge(cfg.entry(), next, kind);
    }
    cfg.add_edge(next, next, EdgeKind::Back);
    cfg.add_edge(next, next, EdgeKind::Back);

    let parsed = parse(&cfg.to_text()).expect("its own grammar");
    assert_eq!(shape(&parsed).edges, shape(&cfg).edges);
    assert_eq!(parsed.successors(cfg.entry()).count(), 4);
    assert_eq!(parsed.successors(next).collect::<Vec<_>>(), [next, next]);
}

#[test]
fn crlf_line_endings_comments_and_blank_lines_are_not_content() {
    let unix = "entry bb0\nbb0:\n    a = 1\nbb1: exit\n    return\nbb0 -> bb1 fallthrough\n";
    let noisy = "# dumped by a tool\r\n\r\nentry bb0\r\nbb0:\r\n    a = 1\r\n\r\n# between blocks\r\n\
                 bb1: exit\r\n    return\r\nbb0 -> bb1 fallthrough\r\n";
    let expected = parse(unix).expect("valid text");
    let parsed = parse(noisy).expect("valid text with CRLF line endings");
    assert_eq!(shape(&parsed), shape(&expected));
    assert_eq!(parsed.to_text(), unix);
}

#[test]
fn a_region_that_still_names_a_removed_block_is_written_without_it() {
    // A region's block sets may list blocks removed since the region was
    // added: compaction drops them, so the set means its live members.
    let mut cfg = Cfg::<Line>::new();
    let entry = cfg.entry();
    let body = cfg.new_block();
    let doomed = cfg.new_block();
    let landing = cfg.new_block();
    let stray = cfg.new_block();
    cfg.add_edge(entry, body, EdgeKind::Fallthrough);
    cfg.add_region(region(
        &[entry, body, doomed],
        vec![handler(
            landing,
            HandlerKind::Catch,
            HandlerBody::known([landing, stray]),
        )],
        None,
    ));
    cfg.remove_block(doomed);
    cfg.remove_block(stray);

    let parsed = parse(&cfg.to_text())
        .expect("the text of a CFG with a removed block in a region is still the grammar");
    let live = |blocks: &std::collections::BTreeSet<BlockId>| {
        blocks
            .iter()
            .copied()
            .filter(|&block| cfg.contains_block(block))
            .collect::<std::collections::BTreeSet<_>>()
    };
    let region = &cfg.regions()[0];
    let reread = &parsed.regions()[0];
    assert_eq!(reread.protected_blocks, live(&region.protected_blocks));
    assert_eq!(
        reread.handlers[0].body.blocks().cloned(),
        region.handlers[0].body.blocks().map(live)
    );
}

#[test]
fn an_entry_other_than_the_first_block_is_reproduced() {
    let mut cfg = Cfg::<Line>::new();
    let first = cfg.entry();
    let second = cfg.new_block();
    let third = cfg.new_block();
    cfg.add_edge(third, second, EdgeKind::Jump);
    cfg.set_entry(third);
    cfg.remove_block(first);

    let text = cfg.to_text();
    assert!(text.starts_with("entry bb2\n"), "{text}");
    let parsed = parse(&text).expect("its own grammar");
    assert_eq!(parsed.entry(), third);
    assert_eq!(shape(&parsed), shape(&cfg));
    assert!(!parsed.contains_block(first));
}

#[test]
fn empty_blocks_and_a_single_block_cfg_round_trip() {
    let cfg = Cfg::<Line>::new();
    assert_eq!(cfg.to_text(), "entry bb0\nbb0:\n");
    let parsed = parse(&cfg.to_text()).expect("its own grammar");
    assert_eq!(shape(&parsed), shape(&cfg));
    assert!(parsed.block(parsed.entry()).is_empty());
}

#[test]
fn an_instruction_that_renders_nothing_leaves_no_line_behind() {
    let mut cfg = Cfg::<Line>::new();
    for text in ["first", "", "second"] {
        cfg.block_mut(cfg.entry()).push(Line::new(text));
    }

    let text = cfg.to_text();
    assert!(
        text.lines().all(|line| !line.trim().is_empty()),
        "an empty instruction must not become a blank line: {text:?}"
    );
    let parsed = parse(&text).expect("its own grammar");
    let lines: Vec<_> = parsed
        .block(parsed.entry())
        .instructions()
        .iter()
        .map(|line| line.0.as_str())
        .collect();
    assert_eq!(lines, ["first", "second"]);
}

fn non_empty_strings() -> Vec<String> {
    strings_over(&TEXT_ALPHABET, 3)
        .into_iter()
        .filter(|text| !text.is_empty())
        .collect()
}

/// The instruction texts of a one-block CFG, in order.
fn instruction_texts(cfg: &Cfg<Line>) -> Vec<String> {
    cfg.block(cfg.entry())
        .instructions()
        .iter()
        .map(|line| line.0.clone())
        .collect()
}

#[test]
fn every_short_instruction_text_round_trips() {
    let strings = non_empty_strings();
    let mut cfg = Cfg::<Line>::new();
    for text in &strings {
        cfg.block_mut(cfg.entry()).push(Line::new(text));
    }

    let parsed = parse(&cfg.to_text()).expect("the writer escapes everything it must");
    let read_back = instruction_texts(&parsed);
    let kept: std::collections::BTreeSet<_> = read_back.iter().collect();
    let lost: Vec<_> = strings
        .iter()
        .filter(|text| !kept.contains(text))
        .take(5)
        .collect();
    assert!(
        lost.is_empty(),
        "instructions were lost or changed: {lost:?}"
    );
    assert_eq!(read_back, strings);
}

#[test]
fn an_instruction_that_starts_like_a_comment_is_still_an_instruction() {
    let mut cfg = Cfg::<Line>::new();
    cfg.block_mut(cfg.entry()).push(Line::new("#pragma once"));
    cfg.block_mut(cfg.entry()).push(Line::new("x = 1"));

    let parsed = parse(&cfg.to_text()).expect("its own grammar");
    assert_eq!(instruction_texts(&parsed), ["#pragma once", "x = 1"]);
}

#[test]
fn an_instruction_keeps_its_own_leading_whitespace() {
    let mut cfg = Cfg::<Line>::new();
    for text in ["    x = 1", "\ty = 2", "  "] {
        cfg.block_mut(cfg.entry()).push(Line::new(text));
    }

    let parsed = parse(&cfg.to_text()).expect("its own grammar");
    assert_eq!(instruction_texts(&parsed), ["    x = 1", "\ty = 2", "  "]);
}

#[test]
fn every_short_block_label_round_trips() {
    let strings = non_empty_strings();
    let mut cfg = Cfg::<Line>::new();
    for text in &strings {
        let block = cfg.new_block();
        cfg.block_mut(block).set_label(text.clone());
    }

    let parsed = parse(&cfg.to_text()).expect("the writer escapes everything it must");
    let relabeled: Vec<_> = cfg
        .block_ids()
        .filter(|&id| parsed.block(id).label() != cfg.block(id).label())
        .map(|id| cfg.block(id).label())
        .take(5)
        .collect();
    assert!(relabeled.is_empty(), "block labels changed: {relabeled:?}");
    assert_eq!(parsed.block_count(), cfg.block_count());
}

/// A statement kept as pieces and streamed into the sink one piece at a time,
/// the way an instruction that renders by formatting does.
struct Fragments(&'static [&'static str]);

impl DisplayInstr for Fragments {
    fn mnemonic(&self) -> Cow<'_, str> {
        Cow::Owned(self.0.concat())
    }

    fn write_mnemonic(&self, sink: &mut dyn fmt::Write) -> fmt::Result {
        self.0.iter().try_for_each(|piece| sink.write_str(piece))
    }
}

#[test]
fn an_instruction_streamed_in_pieces_is_escaped_as_one_label() {
    let mut cfg = Cfg::<Fragments>::new();
    let block = cfg.entry();
    cfg.block_mut(block).push(Fragments(&["", "-", "x"]));
    cfg.block_mut(block)
        .push(Fragments(&["a\\", "b", "\n", "c"]));
    cfg.block_mut(block).push(Fragments(&["", ""]));

    assert_eq!(
        cfg.to_text(),
        "entry bb0\nbb0:\n    \\-x\n    a\\\\b\\nc\n",
        "the leading hyphen is escaped although the first piece was empty, and an \
         instruction that streams nothing leaves no line"
    );
}

/// A guarded body with one cleanup handler, built over any edge payload that
/// has a default, with or without branch weights and a recorded cleanup.
fn guarded<E: Default>(weight: Option<f64>, record_cleanup: bool) -> Cfg<Line, E> {
    let mut cfg = Cfg::<Line, E>::with_edge_payload();
    let entry = cfg.entry();
    let body = cfg.new_block();
    let cleanup = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(body).set_label("guarded");
    cfg.block_mut(body).push(Line::new("use(x)"));
    cfg.block_mut(cleanup).push(Line::new("close(x)"));
    for (source, target, kind) in [
        (entry, body, EdgeKind::Fallthrough),
        (body, cleanup, EdgeKind::ExceptionUnwind),
        (body, exit, EdgeKind::ExceptionLeave),
        (cleanup, exit, EdgeKind::Fallthrough),
    ] {
        match weight {
            Some(weight) => cfg.add_weighted_edge(source, target, kind, weight),
            None => cfg.add_edge(source, target, kind),
        };
    }
    let region = cfg.add_region(region(
        &[body],
        vec![handler(
            cleanup,
            HandlerKind::Finally,
            HandlerBody::known([cleanup]),
        )],
        None,
    ));
    if record_cleanup {
        let finally = HandlerRef::new(region, 0);
        cfg.set_cleanup_resume(finally, cleanup);
        cfg.add_continuation(
            finally,
            Continuation {
                reason: CompletionReason::Normal,
                resume: exit,
            },
        );
    }
    cfg
}

#[test]
fn edge_payloads_do_not_change_the_text() {
    let unit: Cfg<Line> = guarded(None, true);
    let annotated: Cfg<Line, ExceptionFlow<u32>> = guarded(None, true);
    assert_eq!(annotated.to_text(), unit.to_text());
}

#[test]
fn weights_and_cleanup_records_are_outside_the_projection() {
    let full: Cfg<Line> = guarded(Some(0.25), true);
    let bare: Cfg<Line> = guarded(None, false);
    assert_eq!(full.cleanups().len(), 1);
    assert_eq!(bare.cleanups(), []);
    assert_eq!(full.to_text(), bare.to_text());

    let parsed = parse(&full.to_text()).expect("its own grammar");
    assert_eq!(parsed.cleanups(), []);
    assert!(parsed.edges().all(|edge| edge.weight().is_none()));
    assert_eq!(shape(&parsed), shape(&full));
}

#[test]
fn a_typed_instruction_syntax_round_trips_through_the_consumers_parser() {
    let mut cfg = Cfg::<Isa>::new();
    let join = cfg.new_block();
    let program = [
        Isa::Load {
            destination: 1,
            offset: -8,
        },
        Isa::Add {
            destination: 3,
            left: 1,
            right: 2,
        },
        Isa::Compare { left: 3, right: 1 },
        Isa::Return,
    ];
    for instruction in program {
        cfg.block_mut(cfg.entry()).push(instruction);
    }
    cfg.block_mut(join).push(Isa::Return);
    cfg.add_edge(cfg.entry(), join, EdgeKind::ConditionalFalse);

    let text = cfg.to_text();
    assert!(
        text.contains("    load r1, [-8]\n"),
        "a hyphen that is not leading is written as it is: {text}"
    );
    let parsed = parse_cfg_text(&text, parse_isa).expect("every line is the machine's syntax");
    assert_eq!(parsed.block(parsed.entry()).instructions(), program);
    assert_eq!(parsed.block(join).instructions(), [Isa::Return]);
    assert_eq!(parsed.to_text(), text);
}

#[test]
fn the_consumers_rejection_stops_the_parse_at_its_line() {
    let text = "entry bb0\nbb0:\n    ret\n    mul r1, r2\n    ret\n";
    let error = parse_cfg_text(text, parse_isa).expect_err("mul is not an instruction");
    assert_eq!(error.line, 4);
    assert_eq!(
        error.kind,
        TextErrorKind::Instruction("unknown instruction `mul r1, r2`".to_owned())
    );
}

#[test]
fn each_malformed_cfg_line_is_reported_with_its_number_and_kind() {
    let cases = [
        (
            "entry bb0\nbb0:\nbb1:\nbb0 -> bb1 sideways\n",
            4,
            TextErrorKind::EdgeKind,
        ),
        (
            "entry bb0\n    stray\nbb0:\n",
            2,
            TextErrorKind::Unrecognized,
        ),
        (
            "entry bb0\nbb1:\nbb0:\n",
            3,
            TextErrorKind::NodeOrder { expected: 2 },
        ),
        (
            "entry bb0\nbb0:\nbb0 -> bb4 jump\n",
            3,
            TextErrorKind::UnknownNode { index: 4 },
        ),
        (
            "entry bb0\nbb0:\nregion r1 protected=bb0\n",
            3,
            TextErrorKind::Region,
        ),
        (
            "entry bb0\nbb0:\nregion r0 protected=bb7\n",
            3,
            TextErrorKind::UnknownNode { index: 7 },
        ),
        (
            "entry bb0\nbb0:\nregion r0 colour=red\n",
            3,
            TextErrorKind::Region,
        ),
        (
            "entry bb0\nbb0:\nregion r0 protected=bb0\n    handler sneeze entry=bb0 body=?\n",
            4,
            TextErrorKind::Region,
        ),
        (
            "entry bb0\nbb0:\nregion r0 protected=bb0\n    handler catch entry=bb0\n",
            4,
            TextErrorKind::Region,
        ),
        (
            "entry bb0\nbb0:\nregion r0 protected=bb0\n    handler filter entry=bb0 body=?\n",
            4,
            TextErrorKind::Region,
        ),
        ("entry bb0\nbb0:\n    a\\qb\n", 3, TextErrorKind::Escape),
    ];
    for (text, line, kind) in cases {
        let error = parse(text).expect_err("the text is malformed");
        assert_eq!((error.line, error.kind), (line, kind), "{text:?}");
    }
}

#[test]
fn an_entry_naming_an_undeclared_block_is_rejected() {
    let error = parse("entry bb3\nbb0:\n").expect_err("bb3 was never declared");
    assert_eq!(error.kind, TextErrorKind::UnknownNode { index: 3 });
}

#[test]
fn write_text_streams_the_same_text_to_any_sink() {
    let cfg = rich_cfg();
    let mut streamed = String::new();
    cfg.write_text(&mut streamed).expect("a String never fails");
    assert_eq!(streamed, cfg.to_text());

    assert_failure_is_reported_at_every_length(&streamed, |sink| cfg.write_text(sink));
}
