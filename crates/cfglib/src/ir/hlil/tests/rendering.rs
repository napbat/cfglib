//! The pseudocode writer.

extern crate alloc;

use alloc::format;
use alloc::string::String;
use core::fmt;

use crate::ir::hlil::{FunctionBuilder, Handler, HandlerKind, pseudocode, write_pseudocode};
use crate::test_util::golden::assert_golden;

use super::fixtures::{bump, every_statement_shape};
use super::{EVERY_WRAP, Imp, Kit, Prim, ToyHlilBuilder, Ty, Wrap, nested, on_small_stack};

over_toys!(
    builder_constructs_verifies_and_renders,
    every_statement_shape_renders,
    pseudocode_writes_into_any_sink,
    a_deep_expression_prints_without_recursion,
    a_deep_nest_prints_without_recursion,
);

fn builder_constructs_verifies_and_renders<D: Kit>() {
    let function = bump::<D>();

    assert_golden(
        "hlil/builder-if-loop.pseudo",
        &pseudocode::<D, _>(&function),
    );
}

fn pseudocode_writes_into_any_sink<D: Kit>() {
    let function = bump::<D>();
    let mut sink = String::new();

    write_pseudocode::<D, _>(&function, &mut sink).unwrap();

    assert_eq!(sink, pseudocode::<D, _>(&function));
}

fn every_statement_shape_renders<D: Kit>() {
    let function = every_statement_shape::<D>();

    assert_golden(
        "hlil/every-statement-shape.pseudo",
        &pseudocode::<D, _>(&function),
    );
}

/// The writer keeps its own stack, so an operand chain as deep as generated
/// code can produce does not overflow the thread's.
fn a_deep_expression_prints_without_recursion<D: Kit>() {
    const DEPTH: usize = 50_000;

    let mut builder = ToyHlilBuilder::<D>::new();
    let counter = builder.int_variable(0);
    let mut sum = builder.read(counter);
    for _ in 0..DEPTH {
        let one = builder.number(1);
        sum = builder.call(Prim::Add, &[sum, one], Ty::Integer);
    }
    let statement = builder.expression_statement(sum);
    let function = builder
        .finish(&[statement])
        .expect("a deep chain is well formed");

    let text = pseudocode::<D, _>(&function);

    assert!(text.starts_with(&"add(".repeat(DEPTH)));
    assert!(text.ends_with(&format!("v0{};\n", ", 1)".repeat(DEPTH))));
}

/// What the printer wrote, without the text itself: indentation makes the
/// output of a nest quadratic in its depth.
#[derive(Default)]
struct Layout {
    lines: usize,
    /// The indentation of the most indented line, in spaces.
    widest_indent: usize,
    /// The indentation of the last line that had text, in spaces.
    last_indent: usize,
    indent: usize,
    at_line_start: bool,
}

impl fmt::Write for Layout {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.at_line_start {
            let spaces = text.bytes().take_while(|&byte| byte == b' ').count();
            self.indent += spaces;
            if spaces == text.len() {
                return Ok(());
            }
            self.widest_indent = self.widest_indent.max(self.indent);
            self.last_indent = self.indent;
            self.at_line_start = false;
        }
        if text.ends_with('\n') {
            self.lines += 1;
            self.indent = 0;
            self.at_line_start = true;
        }
        Ok(())
    }
}

/// The writer keeps its own stack, so statements nested as deep as untrusted
/// input can make them do not overflow the thread's.
///
/// The nest is shallower than the ones of the other walks: every line is
/// indented by its depth, so the text grows with the square of it.
fn a_deep_nest_prints_without_recursion<D: Kit>() {
    const DEPTH: usize = 2_000;
    const INDENT: usize = 4;
    for wraps in [&[Wrap::If][..], &EVERY_WRAP] {
        let layout = on_small_stack(|| {
            let function = nested::<D>(wraps, DEPTH, ToyHlilBuilder::noop);
            let mut layout = Layout {
                at_line_start: true,
                ..Layout::default()
            };
            write_pseudocode::<D, _>(&function, &mut layout).expect("a layout accepts text");
            layout
        });

        // The guard around the nest and each wrap indent what they hold one
        // level, and a `switch` holds its statements in an arm, which is one
        // more. The last line is the return, back at the outermost level.
        let levels = 1
            + (0..DEPTH)
                .map(|level| match wraps[level % wraps.len()] {
                    Wrap::Switch => 2,
                    _ => 1,
                })
                .sum::<usize>();
        assert_eq!(layout.widest_indent, INDENT * levels, "{wraps:?}");
        assert_eq!(layout.last_indent, 0, "{wraps:?}");
        assert!(layout.lines > 2 * DEPTH, "{wraps:?}");
    }
}

#[test]
fn caught_types_print_in_debug_form() {
    let mut builder = ToyHlilBuilder::<Imp>::new();
    let work = builder.effect(Prim::Call, &[]);
    let handler = Handler {
        kind: HandlerKind::Catch,
        binding: None,
        caught_types: &[Imp::ty(Ty::Integer), Imp::ty(Ty::Boolean)],
        body: &[],
    };
    let guarded = builder.try_(&[work], &[handler], &[]);
    let function = builder.finish(&[guarded]).unwrap();

    assert_eq!(
        pseudocode::<Imp, _>(&function),
        "try {\n    call();\n} catch (Integer | Boolean) {\n}\n"
    );
}
