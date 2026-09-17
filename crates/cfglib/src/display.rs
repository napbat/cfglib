//! Indented text output, and display-only instruction rendering.
//!
//! [`IndentedWriter`] is the one place indentation is decided. Every
//! pseudocode printer in the crate — [`ir::ast`](crate::ir::ast),
//! [`ir::rtl`](crate::ir::rtl), [`ir::hlil`](crate::ir::hlil) — writes
//! through it, so a nested construct is indented by the writer rather than
//! by each printer counting spaces.
//!
//! [`DisplayInstr`] is deliberately independent of
//! [`FlowControl`](crate::FlowControl): rendering a graph for humans (DOT
//! export, pseudocode) must not require implementing flow classification.
//! Consumers that only want a picture implement its one required method.

extern crate alloc;
use alloc::borrow::Cow;
use core::fmt;

/// One level of indentation.
const INDENT: &str = "    ";

/// A text sink that knows how deeply nested its writer currently is.
///
/// The writer owns two things and nothing else: the current depth, and the
/// `{`/`}` convention that changes it. Everything else is written straight
/// through — [`IndentedWriter`] implements [`fmt::Write`], so `write!` works
/// on it — which is what lets a printer emit a label without first
/// collecting it into a [`String`](alloc::string::String).
///
/// # Examples
///
/// ```
/// use core::fmt::Write as _;
/// use cfglib::IndentedWriter;
///
/// let mut out = String::new();
/// let mut printer = IndentedWriter::new(&mut out);
/// printer.indent().unwrap();
/// write!(printer, "if (x)").unwrap();
/// printer.open().unwrap();
/// printer.line("y = 1;").unwrap();
/// printer.close().unwrap();
///
/// assert_eq!(out, "if (x) {\n    y = 1;\n}\n");
/// ```
pub struct IndentedWriter<'sink> {
    sink: &'sink mut dyn fmt::Write,
    depth: usize,
}

impl<'sink> IndentedWriter<'sink> {
    /// Write into `sink` at the outermost level.
    pub fn new(sink: &'sink mut dyn fmt::Write) -> Self {
        Self { sink, depth: 0 }
    }

    /// The current nesting depth, in levels.
    #[must_use]
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Write the indentation for the current depth.
    ///
    /// # Errors
    ///
    /// Returns the sink's formatting error if a write fails.
    pub fn indent(&mut self) -> fmt::Result {
        for _ in 0..self.depth {
            self.sink.write_str(INDENT)?;
        }
        Ok(())
    }

    /// End the current line.
    ///
    /// # Errors
    ///
    /// Returns the sink's formatting error if a write fails.
    pub fn end_line(&mut self) -> fmt::Result {
        self.sink.write_str("\n")
    }

    /// Write one whole indented line.
    ///
    /// # Errors
    ///
    /// Returns the sink's formatting error if a write fails.
    pub fn line(&mut self, text: &str) -> fmt::Result {
        self.indent()?;
        self.sink.write_str(text)?;
        self.end_line()
    }

    /// Write an indented line whose content `body` produces.
    ///
    /// A `body` that writes nothing produces no line at all — not an empty
    /// one — so a printer can hand an instruction the sink without first
    /// collecting its label to find out whether there is one.
    ///
    /// # Errors
    ///
    /// Returns the sink's formatting error if a write fails.
    pub fn optional_line(
        &mut self,
        body: impl FnOnce(&mut dyn fmt::Write) -> fmt::Result,
    ) -> fmt::Result {
        let mut line = PendingLine {
            sink: &mut *self.sink,
            indent: self.depth,
            wrote: false,
        };
        body(&mut line)?;
        if line.wrote {
            self.end_line()?;
        }
        Ok(())
    }

    /// Write `body` one level deeper, with no brace to mark the level.
    ///
    /// This is what a label's statements are indented by: the nesting is
    /// real, but there is no block to open.
    ///
    /// # Errors
    ///
    /// Propagates whatever `body` returns.
    pub fn nested(&mut self, body: impl FnOnce(&mut Self) -> fmt::Result) -> fmt::Result {
        self.depth += 1;
        let result = body(self);
        self.depth -= 1;
        result
    }

    /// Open a block: ` {`, a line break, and one more level of indentation.
    ///
    /// The header is whatever was already written on the line.
    ///
    /// # Errors
    ///
    /// Returns the sink's formatting error if a write fails.
    pub fn open(&mut self) -> fmt::Result {
        self.sink.write_str(" {\n")?;
        self.depth += 1;
        Ok(())
    }

    /// Close a block: one level less indentation, then `}` on its own line.
    ///
    /// # Errors
    ///
    /// Returns the sink's formatting error if a write fails.
    pub fn close(&mut self) -> fmt::Result {
        self.close_inline()?;
        self.end_line()
    }

    /// Close a block without ending the line, so the caller can continue the
    /// construct — `} else {`, `} while (…)`, `} catch (…) {`.
    ///
    /// # Errors
    ///
    /// Returns the sink's formatting error if a write fails.
    pub fn close_inline(&mut self) -> fmt::Result {
        self.depth = self.depth.saturating_sub(1);
        self.indent()?;
        self.sink.write_str("}")
    }
}

/// A line that indents itself only once something is written to it.
struct PendingLine<'sink> {
    sink: &'sink mut dyn fmt::Write,
    indent: usize,
    wrote: bool,
}

impl fmt::Write for PendingLine<'_> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if text.is_empty() {
            return Ok(());
        }
        if !self.wrote {
            for _ in 0..self.indent {
                self.sink.write_str(INDENT)?;
            }
            self.wrote = true;
        }
        self.sink.write_str(text)
    }
}

impl fmt::Write for IndentedWriter<'_> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.sink.write_str(text)
    }
}

impl fmt::Debug for IndentedWriter<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IndentedWriter")
            .field("depth", &self.depth)
            .finish_non_exhaustive()
    }
}

/// Opt-in, display-only rendering of instructions.
///
/// Used by [`Cfg::write_dot`](crate::Cfg::write_dot),
/// [`Cfg::write_text`](crate::Cfg::write_text), and AST pseudocode output.
/// Return a short human-readable label: a machine mnemonic, a source
/// statement excerpt, or a node kind. Labels are escaped by the renderers,
/// so raw program text is safe to return.
///
/// [`mnemonic`](Self::mnemonic) is the owned form and the one method an
/// implementation must supply. An instruction that renders by formatting —
/// an MLIL instruction naming its operands, for instance — overrides
/// [`write_mnemonic`](Self::write_mnemonic) as well, so a renderer that only
/// needs the text in a sink never pays for the intermediate
/// [`String`](alloc::string::String).
pub trait DisplayInstr {
    /// Short label for this instruction (mnemonic, statement text, node kind).
    fn mnemonic(&self) -> Cow<'_, str>;

    /// Write the same label straight into `sink`.
    ///
    /// # Errors
    ///
    /// Returns the sink's formatting error if a write fails.
    fn write_mnemonic(&self, sink: &mut dyn fmt::Write) -> fmt::Result {
        sink.write_str(&self.mnemonic())
    }
}

impl<T: DisplayInstr + ?Sized> DisplayInstr for &T {
    fn mnemonic(&self) -> Cow<'_, str> {
        T::mnemonic(self)
    }

    fn write_mnemonic(&self, sink: &mut dyn fmt::Write) -> fmt::Result {
        T::write_mnemonic(self, sink)
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::string::String;
    use core::fmt::Write as _;

    use super::IndentedWriter;

    #[test]
    fn nested_blocks_indent_and_outdent() {
        let mut out = String::new();
        let mut printer = IndentedWriter::new(&mut out);
        printer.indent().unwrap();
        printer.write_str("loop").unwrap();
        printer.open().unwrap();
        printer.indent().unwrap();
        write!(printer, "if (x > {})", 0).unwrap();
        printer.open().unwrap();
        printer.line("break;").unwrap();
        printer.close_inline().unwrap();
        printer.write_str(" else").unwrap();
        printer.open().unwrap();
        printer.line("continue;").unwrap();
        printer.close().unwrap();
        printer.close().unwrap();

        assert_eq!(
            out,
            "loop {\n    if (x > 0) {\n        break;\n    } else {\n        continue;\n    }\n}\n"
        );
        assert_eq!(IndentedWriter::new(&mut String::new()).depth(), 0);
    }

    #[test]
    fn closing_an_unopened_block_stays_at_the_outermost_level() {
        let mut out = String::new();
        let mut printer = IndentedWriter::new(&mut out);
        printer.close().unwrap();
        assert_eq!(printer.depth(), 0);
        assert_eq!(out, "}\n");
    }
}
