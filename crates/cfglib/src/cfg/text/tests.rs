extern crate alloc;

use alloc::borrow::Cow;
use alloc::string::{String, ToString};

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::display::DisplayInstr;
use crate::edge::EdgeKind;
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};
use crate::test_util::golden::assert_golden;

use crate::TextError;

use super::parse_cfg_text;

/// The instruction the CFG text form round trips: its own line, verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Line(String);

impl DisplayInstr for Line {
    fn mnemonic(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.0)
    }
}

/// Read a CFG whose instructions are their own lines.
fn parse(text: &str) -> Result<Cfg<Line>, TextError> {
    parse_cfg_text(text, |line| Ok(Line(line.to_string())))
}

/// A CFG with a removed block, a labeled block, every edge shape the form
/// distinguishes, and a protected region with two handlers.
fn protected_cfg() -> Cfg<Line> {
    let mut cfg = Cfg::<Line>::new();
    let header = cfg.new_block();
    let removed = cfg.new_block();
    let handler = cfg.new_block();
    let filter = cfg.new_block();
    let exit = cfg.new_block();

    cfg.block_mut(cfg.entry())
        .push(Line("x = read()".to_string()));
    cfg.block_mut(cfg.entry())
        .push(Line("branch x".to_string()));
    cfg.block_mut(header).set_label("loop_header");
    cfg.block_mut(header).push(Line("y = x + 1".to_string()));
    cfg.block_mut(handler).push(Line("catch e".to_string()));
    cfg.block_mut(filter)
        .push(Line("e is Overflow".to_string()));
    cfg.block_mut(exit).push(Line("return y".to_string()));
    cfg.remove_block(removed);

    cfg.add_edge(cfg.entry(), header, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(header, header, EdgeKind::Back);
    cfg.add_edge(header, handler, EdgeKind::ExceptionUnwind);
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [cfg.entry(), header].into_iter().collect(),
        handlers: alloc::vec![
            Handler {
                entry: handler,
                body: HandlerBody::known([handler]),
                kind: HandlerKind::Catch,
            },
            Handler {
                entry: handler,
                body: HandlerBody::Unknown,
                kind: HandlerKind::Filter {
                    filter_block: filter,
                },
            },
        ],
        parent: None,
    });
    cfg
}

#[test]
fn a_cfg_writes_blocks_edges_and_regions() {
    assert_golden("text/cfg-protected-region.txt", &protected_cfg().to_text());
}

#[test]
fn writing_parsing_and_writing_again_is_a_fixed_point() {
    let once = protected_cfg().to_text();
    let twice = parse(&once)
        .expect("the writer emits its own grammar")
        .to_text();
    assert_eq!(once, twice);
}

#[test]
fn parsing_reproduces_identity_topology_labels_and_regions() {
    let original = protected_cfg();
    let parsed = parse(&original.to_text()).expect("its own grammar");

    assert_eq!(parsed.entry(), original.entry());
    assert_eq!(parsed.block_count(), original.block_count());
    assert_eq!(
        parsed.block_bound(),
        original.block_bound(),
        "the removed block keeps its slot"
    );
    for block in original.block_ids() {
        assert_eq!(parsed.block(block).label(), original.block(block).label());
        assert_eq!(
            parsed.block(block).instructions(),
            original.block(block).instructions()
        );
    }
    let edges = |cfg: &Cfg<Line>| {
        cfg.edge_ids()
            .map(|id| {
                let edge = cfg.edge(id);
                (edge.source(), edge.target(), edge.kind())
            })
            .collect::<alloc::vec::Vec<_>>()
    };
    assert_eq!(edges(&parsed), edges(&original));
    assert_eq!(parsed.regions(), original.regions());
}

#[test]
fn a_parsed_entry_may_be_any_declared_block() {
    let cfg = parse("entry bb1\nbb0:\nbb1:\nbb1 -> bb0 jump\n").expect("valid text");
    assert_eq!(cfg.entry(), BlockId::from_index(1));
}

#[test]
fn an_unknown_edge_kind_is_rejected() {
    let error = parse("entry bb0\nbb0:\nbb1:\nbb0 -> bb1 sideways\n")
        .expect_err("there is no sideways edge");
    assert_eq!(error.line, 4);
    assert_eq!(error.kind, crate::TextErrorKind::EdgeKind);
}

#[test]
fn an_instruction_outside_a_block_is_rejected() {
    let error = parse("entry bb0\n    stray\nbb0:\n").expect_err("nothing owns that line");
    assert_eq!(error.line, 2);
    assert_eq!(error.kind, crate::TextErrorKind::Unrecognized);
}

#[test]
fn a_rejected_instruction_reports_the_consumers_reason() {
    let error = parse_cfg_text("entry bb0\nbb0:\n    nope\n", |line| {
        Err::<String, _>(alloc::format!("{line} is not an instruction"))
    })
    .expect_err("the consumer rejected the line");
    assert_eq!(
        error.kind,
        crate::TextErrorKind::Instruction("nope is not an instruction".to_string())
    );
}

#[test]
fn an_instruction_is_read_back_whatever_its_first_character() {
    // Each of these would read as something else on its own line: a comment,
    // a blank line, an edge, or plain indentation.
    let texts = [
        "#pragma once",
        "    x = 1",
        "\ty = 2",
        "  ",
        "-> bb1 true",
        "\u{20}literal",
        "\u{a0}",
        "\\u{20} is a spelled-out escape",
    ];
    let mut cfg = Cfg::<Line>::new();
    let entry = cfg.entry();
    for text in texts {
        cfg.block_mut(entry).push(Line(text.to_string()));
    }

    let written = cfg.to_text();
    let parsed = parse(&written).expect("the writer emits its own grammar");
    assert_eq!(
        parsed.block(parsed.entry()).instructions(),
        cfg.block(entry).instructions()
    );
    assert_eq!(parsed.to_text(), written);
}
