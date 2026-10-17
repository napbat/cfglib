//! What the statements below a body contain, answered from one numbering of
//! the statements of the source.
//!
//! The recoveries that rewrite a body ask whether the body holds a `continue`
//! or assigns a variable. A scan of the body for each ask reads a nest of
//! such bodies once per level, so the statements are numbered once, in the
//! order that a walk visits them, and a question becomes a search for a
//! number in the range that the body spans.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::graph::view::DenseId;

use super::super::view::for_each_child_body;
use super::{
    ExprShape, Fallible, FunctionBuilder, FunctionView, Rebuilder, RecoverDialect, StmtShape,
};

/// The statements of a function in visit order, and where the statements
/// that matter to a recovery sit in it.
pub(super) struct Numbering {
    /// For each statement, the number of the first statement of its subtree
    /// and one past the last. The statement is the first of its own subtree.
    spans: Vec<(usize, usize)>,
    /// The numbers of the `continue` statements, ascending.
    continues: Vec<usize>,
    /// For each variable index, the numbers of the statements that assign it,
    /// ascending.
    assignments: BTreeMap<usize, Vec<usize>>,
}

impl Numbering {
    /// Whether any of `numbers` falls in the range that `ids` spans: the
    /// range from the first statement of `ids` to the end of the subtree of
    /// the last. The statements of one body are consecutive in visit order.
    fn spans_any(&self, ids: &[impl DenseId], numbers: &[usize]) -> bool {
        let (Some(first), Some(last)) = (ids.first(), ids.last()) else {
            return false;
        };
        let (start, end) = (self.spans[first.index()].0, self.spans[last.index()].1);
        let from = numbers.partition_point(|&number| number < start);
        numbers.get(from).is_some_and(|&number| number < end)
    }
}

/// One step of the numbering walk.
enum Visit<S> {
    Enter(S),
    Exit(S),
}

impl<'a, H, S, B> Rebuilder<'a, H, S, B>
where
    H: RecoverDialect,
    S: FunctionView<H>,
    B: FunctionBuilder<H>,
{
    /// Whether any statement of `ids`, or below them, is a `continue`. Nested
    /// loops are included, which keeps the checks that use this
    /// conservative.
    pub(super) fn contains_continue(&mut self, ids: &[S::Stmt]) -> Fallible<bool, B::Error> {
        let numbering = self.numbering()?;
        Ok(numbering.spans_any(ids, &numbering.continues))
    }

    /// Whether any statement of `ids`, or below them, assigns the variable.
    pub(super) fn assigns_variable(
        &mut self,
        ids: &[S::Stmt],
        variable: S::Var,
    ) -> Fallible<bool, B::Error> {
        let numbering = self.numbering()?;
        Ok(numbering
            .assignments
            .get(&variable.index())
            .is_some_and(|numbers| numbering.spans_any(ids, numbers)))
    }

    /// The numbering of the source, made when a recovery first asks.
    fn numbering(&mut self) -> Fallible<&Numbering, B::Error> {
        if self.numbering.is_none() {
            self.numbering = Some(self.number_statements()?);
        }
        Ok(self
            .numbering
            .as_ref()
            .expect("the numbering was made above"))
    }

    /// Numbers every statement below the body of the source, in the order of a
    /// walk that rebuilds the bodies of a statement in turn.
    fn number_statements(&self) -> Fallible<Numbering, B::Error> {
        let source: &'a S = self.source;
        let mut numbering = Numbering {
            spans: vec![(0, 0); source.stmt_count()],
            continues: Vec::new(),
            assignments: BTreeMap::new(),
        };
        let mut visits: Vec<Visit<S::Stmt>> = source
            .body()
            .iter()
            .rev()
            .map(|&id| Visit::Enter(id))
            .collect();
        let mut bodies: Vec<&'a [S::Stmt]> = Vec::new();
        let mut clock = 0;
        while let Some(visit) = visits.pop() {
            let id = match visit {
                Visit::Exit(id) => {
                    numbering.spans[id.index()].1 = clock;
                    continue;
                }
                Visit::Enter(id) => id,
            };
            let shape = self.stmt(id)?;
            numbering.spans[id.index()].0 = clock;
            match shape {
                StmtShape::Continue { .. } => numbering.continues.push(clock),
                StmtShape::Assign { target, .. } => {
                    if let ExprShape::Variable(variable) = self.expr(target)? {
                        numbering
                            .assignments
                            .entry(variable.index())
                            .or_default()
                            .push(clock);
                    }
                }
                _ => {}
            }
            clock += 1;
            visits.push(Visit::Exit(id));
            bodies.clear();
            for_each_child_body(source, id, |body, _| bodies.push(body));
            for body in bodies.iter().rev() {
                visits.extend(body.iter().rev().map(|&child| Visit::Enter(child)));
            }
        }
        Ok(numbering)
    }
}
