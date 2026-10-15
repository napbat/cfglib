//! Structural placement of presentation-level variable declarations.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use crate::graph::fanout::DenseFanout;
use crate::graph::view::DenseId;

use super::dialect::Dialect;
use super::shape::ExprShape;
use super::view::{
    ChildBodyKind, FunctionView, for_each_child_body, for_each_statement_expression,
    visit_expression_tree,
};

/// Lexically narrow declaration anchors for the variables of one function.
///
/// Each referenced variable is anchored before the first statement in the
/// narrowest lexical body that contains all of its occurrences. A `for`
/// initializer or update is part of the loop syntax rather than a declaration
/// body, so a variable that those clauses reference anchors before the `for`.
///
/// The analysis is deliberately structural. A source renderer remains
/// responsible for excluding the variables that a signature or a handler
/// declares, for hoisting live-in variables, and for satisfying the
/// declaration restrictions of its language.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariablePlacements<S: DenseId, V: DenseId> {
    /// The anchor of each variable, by variable index.
    placements: Vec<Option<S>>,
    /// The variables anchored before each statement, in variable order.
    before: DenseFanout<V, S>,
}

impl<S: DenseId, V: DenseId> VariablePlacements<S, V> {
    /// Computes the declaration anchors of every referenced variable of
    /// `function`.
    #[must_use]
    pub fn compute<H, F>(function: &F) -> Self
    where
        H: Dialect,
        F: FunctionView<H, Stmt = S, Var = V>,
    {
        let mut placements = vec![None; function.variable_count()];
        analyze_body(function, function.body(), &mut placements);
        let anchored = placements.iter().enumerate().filter_map(|(index, anchor)| {
            anchor.map(|statement| (statement, V::from_index(index)))
        });
        let before = DenseFanout::from_pairs(function.stmt_count(), anchored);
        Self { placements, before }
    }

    /// Returns the statement that `variable`'s declaration immediately
    /// precedes, or `None` when the variable has no occurrence.
    #[must_use]
    pub fn placement(&self, variable: V) -> Option<S> {
        self.placements.get(variable.index()).copied().flatten()
    }

    /// Returns the variables whose declarations anchor immediately before
    /// `statement`, in variable order.
    #[must_use]
    pub fn before(&self, statement: S) -> &[V] {
        if statement.index() < self.before.bound() {
            self.before.get(statement)
        } else {
            &[]
        }
    }
}

/// Anchors the variables inside `body` and returns every variable that
/// occurs in it.
fn analyze_body<H, F>(
    function: &F,
    body: &[F::Stmt],
    placements: &mut [Option<F::Stmt>],
) -> BTreeSet<F::Var>
where
    H: Dialect,
    F: FunctionView<H>,
{
    let mut contained = BTreeSet::new();
    let mut first_statement = BTreeMap::new();
    for &statement in body {
        for variable in analyze_statement(function, statement, placements) {
            if let Some(&first) = first_statement.get(&variable) {
                place(placements, variable, first);
            } else {
                first_statement.insert(variable, statement);
            }
            contained.insert(variable);
        }
    }
    contained
}

/// Anchors the variables of `statement` that no narrower body can hold and
/// returns every variable that occurs in it.
fn analyze_statement<H, F>(
    function: &F,
    statement: F::Stmt,
    placements: &mut [Option<F::Stmt>],
) -> BTreeSet<F::Var>
where
    H: Dialect,
    F: FunctionView<H>,
{
    let mut owned = BTreeSet::new();
    for_each_statement_expression(function, statement, |expression| {
        collect_expression(function, expression, &mut owned);
    });

    // A variable's components are the places it occurs in this statement:
    // its own operands, and each child body that mentions it.
    let mut components: BTreeMap<F::Var, usize> =
        owned.iter().map(|&variable| (variable, 1)).collect();
    for_each_child_body(function, statement, |body, kind| {
        let variables = match kind {
            ChildBodyKind::Lexical => analyze_body(function, body, placements),
            ChildBodyKind::Clause => {
                let variables = collect_body(function, body);
                owned.extend(variables.iter().copied());
                variables
            }
        };
        for variable in variables {
            *components.entry(variable).or_default() += 1;
        }
    });

    for (&variable, &component_count) in &components {
        if owned.contains(&variable) || component_count > 1 {
            place(placements, variable, statement);
        }
    }
    components.into_keys().collect()
}

fn collect_body<H, F>(function: &F, body: &[F::Stmt]) -> BTreeSet<F::Var>
where
    H: Dialect,
    F: FunctionView<H>,
{
    let mut variables = BTreeSet::new();
    for &statement in body {
        collect_statement(function, statement, &mut variables);
    }
    variables
}

fn collect_statement<H, F>(function: &F, statement: F::Stmt, variables: &mut BTreeSet<F::Var>)
where
    H: Dialect,
    F: FunctionView<H>,
{
    for_each_statement_expression(function, statement, |expression| {
        collect_expression(function, expression, variables);
    });
    for_each_child_body(function, statement, |body, _| {
        for &nested in body {
            collect_statement(function, nested, variables);
        }
    });
}

fn collect_expression<H, F>(function: &F, root: F::Expr, variables: &mut BTreeSet<F::Var>)
where
    H: Dialect,
    F: FunctionView<H>,
{
    visit_expression_tree(function, root, &mut |expression| {
        if let ExprShape::Variable(variable) = function.expr(expression) {
            variables.insert(variable);
        }
    });
}

fn place<S: DenseId, V: DenseId>(placements: &mut [Option<S>], variable: V, statement: S) {
    if let Some(slot) = placements.get_mut(variable.index()) {
        *slot = Some(statement);
    }
}

#[cfg(test)]
mod tests;
