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
    visit_expression_tree, visit_statements,
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

/// Every variable that `body` reads or writes, nested statements included.
fn collect_body<H, F>(function: &F, body: &[F::Stmt]) -> BTreeSet<F::Var>
where
    H: Dialect,
    F: FunctionView<H>,
{
    let mut variables = BTreeSet::new();
    visit_statements(function, body, &mut |statement| {
        for_each_statement_expression(function, statement, |expression| {
            collect_expression(function, expression, &mut variables);
        });
    });
    variables
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
mod tests {
    use crate::ir::hlil::tests::{Kit, ToyHlilBuilder, over_toys};
    use crate::ir::hlil::{FunctionBuilder, VariablePlacements};

    over_toys!(
        shared_branch_variable_anchors_before_if,
        branch_local_variable_stays_in_the_branch_body,
        loop_local_variable_stays_in_the_loop_body,
        for_clause_variable_anchors_before_the_loop,
        switch_arm_variables_anchor_in_their_arm,
        unreferenced_variables_have_no_anchor,
    );

    fn shared_branch_variable_anchors_before_if<D: Kit>() {
        let mut builder = ToyHlilBuilder::<D>::new();
        let value = builder.int_variable(0);
        let then_use = builder.use_of(value);
        let else_use = builder.use_of(value);
        let condition = builder.condition();
        let branch = builder.if_(condition, &[then_use], &[else_use]);
        let function = builder.finish(&[branch]).unwrap();

        let placements = VariablePlacements::compute(&function);

        assert_eq!(placements.placement(value), Some(branch));
        assert_eq!(placements.before(branch), [value]);
    }

    fn branch_local_variable_stays_in_the_branch_body<D: Kit>() {
        let mut builder = ToyHlilBuilder::<D>::new();
        let value = builder.int_variable(0);
        let unrelated = builder.noop();
        let first_use = builder.use_of(value);
        let second_use = builder.use_of(value);
        let condition = builder.condition();
        let branch = builder.if_(condition, &[unrelated, first_use, second_use], &[]);
        let function = builder.finish(&[branch]).unwrap();

        let placements = VariablePlacements::compute(&function);

        assert_eq!(placements.placement(value), Some(first_use));
        assert_eq!(placements.before(branch), []);
        assert_eq!(placements.before(first_use), [value]);
    }

    fn loop_local_variable_stays_in_the_loop_body<D: Kit>() {
        let mut builder = ToyHlilBuilder::<D>::new();
        let value = builder.int_variable(0);
        let first_use = builder.use_of(value);
        let second_use = builder.use_of(value);
        let condition = builder.condition();
        let looped = builder.while_(condition, &[first_use, second_use]);
        let function = builder.finish(&[looped]).unwrap();

        let placements = VariablePlacements::compute(&function);

        assert_eq!(placements.placement(value), Some(first_use));
        assert_eq!(placements.before(looped), []);
    }

    fn for_clause_variable_anchors_before_the_loop<D: Kit>() {
        let mut builder = ToyHlilBuilder::<D>::new();
        let value = builder.int_variable(0);
        let initializer = builder.use_of(value);
        let condition = builder.condition();
        let looped = builder.for_(&[initializer], Some(condition), &[], &[]);
        let function = builder.finish(&[looped]).unwrap();

        let placements = VariablePlacements::compute(&function);

        assert_eq!(placements.placement(value), Some(looped));
        assert_eq!(placements.before(looped), [value]);
        assert_eq!(placements.before(initializer), []);
    }

    fn switch_arm_variables_anchor_in_their_arm<D: Kit>() {
        use crate::ir::hlil::SwitchArm;

        let mut builder = ToyHlilBuilder::<D>::new();
        let local = builder.int_variable(0);
        let shared = builder.int_variable(1);
        let first_local = builder.use_of(local);
        let second_local = builder.use_of(local);
        let first_shared = builder.use_of(shared);
        let second_shared = builder.use_of(shared);
        let scrutinee = builder.number(0);
        let values = [D::constant(1)];
        let switch = builder.switch(
            scrutinee,
            &[SwitchArm {
                values: &values,
                body: &[first_local, second_local, first_shared],
            }],
            &[second_shared],
        );
        let function = builder.finish(&[switch]).unwrap();

        let placements = VariablePlacements::compute(&function);

        assert_eq!(placements.placement(local), Some(first_local));
        assert_eq!(placements.placement(shared), Some(switch));
        assert_eq!(placements.before(switch), [shared]);
    }

    fn unreferenced_variables_have_no_anchor<D: Kit>() {
        let mut builder = ToyHlilBuilder::<D>::new();
        let unused = builder.int_variable(0);
        let statement = builder.noop();
        let function = builder.finish(&[statement]).unwrap();

        let placements = VariablePlacements::compute(&function);

        assert_eq!(placements.placement(unused), None);
        assert_eq!(placements.before(statement), []);
    }
}
