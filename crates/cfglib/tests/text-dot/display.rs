//! The shared printing primitives a consumer's own printer is built from:
//! `IndentedWriter` for structure and `DisplayInstr` for instruction text.

use std::borrow::Cow;
use std::fmt::{self, Write as _};

use cfglib::{Cfg, DisplayInstr, IndentedWriter};

use crate::instruction::Line;

#[test]
fn braces_indent_and_close_inline_continues_the_construct() {
    let mut out = String::new();
    let mut printer = IndentedWriter::new(&mut out);
    printer.indent().unwrap();
    write!(printer, "if (x > 0)").unwrap();
    printer.open().unwrap();
    printer.line("y = 1;").unwrap();
    printer.close_inline().unwrap();
    printer.write_str(" else").unwrap();
    printer.open().unwrap();
    printer.indent().unwrap();
    write!(printer, "while (y < {})", 3).unwrap();
    printer.open().unwrap();
    printer.line("y += 1;").unwrap();
    printer.close().unwrap();
    printer.close().unwrap();

    assert_eq!(
        out,
        "if (x > 0) {\n    y = 1;\n} else {\n    while (y < 3) {\n        y += 1;\n    }\n}\n"
    );
}

#[test]
fn depth_follows_open_close_and_nested() {
    let mut out = String::new();
    let mut printer = IndentedWriter::new(&mut out);
    assert_eq!(printer.depth(), 0);
    printer.open().unwrap();
    assert_eq!(printer.depth(), 1);
    printer
        .nested(|inner| {
            assert_eq!(inner.depth(), 2);
            inner.open()?;
            assert_eq!(inner.depth(), 3);
            inner.close()
        })
        .unwrap();
    assert_eq!(
        printer.depth(),
        1,
        "nested restores the depth it entered at"
    );
    printer.close().unwrap();
    assert_eq!(printer.depth(), 0);
}

#[test]
fn nested_indents_one_level_without_a_brace() {
    let mut out = String::new();
    let mut printer = IndentedWriter::new(&mut out);
    printer.line("label:").unwrap();
    printer
        .nested(|body| {
            body.line("first")?;
            body.nested(|deeper| deeper.line("second"))?;
            body.line("third")
        })
        .unwrap();
    printer.line("after").unwrap();

    assert_eq!(out, "label:\n    first\n        second\n    third\nafter\n");
}

#[test]
fn a_failing_nested_body_still_restores_the_depth() {
    let mut out = String::new();
    let mut printer = IndentedWriter::new(&mut out);
    let result = printer.nested(|_| Err(fmt::Error));
    assert_eq!(result, Err(fmt::Error));
    assert_eq!(printer.depth(), 0);
}

#[test]
fn text_written_through_the_writer_is_not_indented_for_you() {
    let mut out = String::new();
    let mut printer = IndentedWriter::new(&mut out);
    printer
        .nested(|inner| {
            write!(inner, "raw ")?;
            inner.indent()?;
            write!(inner, "then indented")?;
            inner.end_line()
        })
        .unwrap();
    assert_eq!(out, "raw     then indented\n");
}

#[test]
fn an_optional_line_that_writes_nothing_leaves_no_line_and_no_indentation() {
    let mut out = String::new();
    let mut printer = IndentedWriter::new(&mut out);
    printer
        .nested(|inner| {
            inner.optional_line(|_| Ok(()))?;
            inner.optional_line(|line| line.write_str(""))?;
            inner.optional_line(|line| {
                line.write_str("")?;
                line.write_str("kept ")?;
                line.write_str("whole")
            })
        })
        .unwrap();
    assert_eq!(out, "    kept whole\n");
}

#[test]
fn an_unbalanced_close_inside_a_nested_body_does_not_corrupt_the_depth() {
    let mut out = String::new();
    let mut printer = IndentedWriter::new(&mut out);
    printer.nested(IndentedWriter::close).unwrap();
    assert_eq!(printer.depth(), 0, "the printer is back where it started");
    printer.line("still usable").unwrap();
    assert!(out.ends_with("still usable\n"), "{out:?}");
}

/// An instruction that renders differently through its two entry points, so a
/// caller's choice of entry point is observable.
struct Probe;

impl DisplayInstr for Probe {
    fn mnemonic(&self) -> Cow<'_, str> {
        Cow::Borrowed("owned form")
    }

    fn write_mnemonic(&self, sink: &mut dyn fmt::Write) -> fmt::Result {
        sink.write_str("streamed form")
    }
}

#[test]
fn the_streaming_form_defaults_to_the_owned_form() {
    let mut out = String::new();
    Line::new("add r1, r2").write_mnemonic(&mut out).unwrap();
    assert_eq!(out, "add r1, r2");
}

#[test]
fn a_reference_renders_through_the_same_two_entry_points_as_its_target() {
    let probe = Probe;
    let by_reference = &probe;
    assert_eq!(by_reference.mnemonic(), "owned form");
    let mut out = String::new();
    by_reference.write_mnemonic(&mut out).unwrap();
    assert_eq!(
        out, "streamed form",
        "the override survives the blanket impl"
    );
}

#[test]
fn a_cfg_of_borrowed_instructions_renders_like_one_of_owned_instructions() {
    let lines = [Line::new("a = 1"), Line::new("b = a")];
    let mut borrowed = Cfg::<&Line>::new();
    let mut owned = Cfg::<Line>::new();
    for line in &lines {
        borrowed.block_mut(borrowed.entry()).push(line);
        owned.block_mut(owned.entry()).push(line.clone());
    }
    assert_eq!(borrowed.to_text(), owned.to_text());
    assert_eq!(borrowed.to_dot(), owned.to_dot());
}
