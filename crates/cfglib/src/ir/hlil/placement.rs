//! Structural placement of presentation-level variable declarations.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;
use core::slice;

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
        anchor_variables(function, &mut placements);
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

/// Anchors the variables of every statement of `function`, the innermost
/// bodies first.
///
/// The analysis keeps its own stack of the statements that it is inside, so
/// the depth of the nesting costs no call frames. Each frame holds the one
/// child body that its statement is analyzing, and the frame below receives
/// the variables of the statement when it is done.
fn anchor_variables<H, F>(function: &F, placements: &mut [Option<F::Stmt>])
where
    H: Dialect,
    F: FunctionView<H>,
{
    let mut root = BodyFrame::new(function.body());
    let mut open: Vec<StatementFrame<'_, F::Stmt, F::Var>> = Vec::new();
    loop {
        let body = open.last_mut().map_or(&mut root, |frame| &mut frame.body);
        if let Some(&statement) = body.statements.next() {
            open.push(StatementFrame::new(function, statement));
            continue;
        }
        let Some(mut frame) = open.pop() else {
            break;
        };
        frame.take_in_body();
        if let Some(next) = frame.bodies.next() {
            frame.body = BodyFrame::new(next);
            open.push(frame);
            continue;
        }
        let statement = frame.statement;
        let variables = frame.finish(placements);
        let parent = open.last_mut().map_or(&mut root, |frame| &mut frame.body);
        parent.take_in_statement(statement, variables, placements);
    }
}

/// A body whose statements are being analyzed.
struct BodyFrame<'a, S, V> {
    statements: slice::Iter<'a, S>,
    /// Every variable that occurs in the statements analyzed so far.
    contained: BTreeSet<V>,
    /// The first statement that each of them occurs in.
    first_statement: BTreeMap<V, S>,
}

impl<'a, S: DenseId, V: DenseId> BodyFrame<'a, S, V> {
    fn new(body: &'a [S]) -> Self {
        Self {
            statements: body.iter(),
            contained: BTreeSet::new(),
            first_statement: BTreeMap::new(),
        }
    }

    /// Takes in the variables that occur in `statement`, the one just
    /// analyzed. A variable that a second statement shares cannot anchor in
    /// either, so it anchors before the first.
    fn take_in_statement(
        &mut self,
        statement: S,
        variables: BTreeSet<V>,
        placements: &mut [Option<S>],
    ) {
        for variable in variables {
            if let Some(&first) = self.first_statement.get(&variable) {
                place(placements, variable, first);
            } else {
                self.first_statement.insert(variable, statement);
            }
            self.contained.insert(variable);
        }
    }
}

/// A statement whose child bodies are being analyzed.
struct StatementFrame<'a, S, V> {
    statement: S,
    /// The variables that its own operands and its clauses mention. They
    /// anchor before the statement, since no narrower body holds them.
    owned: BTreeSet<V>,
    /// A variable's components are the places it occurs in the statement:
    /// its own operands, and each child body that mentions it.
    components: BTreeMap<V, usize>,
    /// The lexical bodies after the one being analyzed.
    bodies: vec::IntoIter<&'a [S]>,
    /// The lexical body being analyzed.
    body: BodyFrame<'a, S, V>,
}

impl<'a, S: DenseId, V: DenseId> StatementFrame<'a, S, V> {
    fn new<H, F>(function: &'a F, statement: S) -> Self
    where
        H: Dialect,
        F: FunctionView<H, Stmt = S, Var = V>,
    {
        let mut owned = BTreeSet::new();
        for_each_statement_expression(function, statement, |expression| {
            collect_expression(function, expression, &mut owned);
        });
        let mut components: BTreeMap<V, usize> =
            owned.iter().map(|&variable| (variable, 1)).collect();
        let mut bodies = Vec::new();
        for_each_child_body(function, statement, |body, kind| match kind {
            ChildBodyKind::Lexical => bodies.push(body),
            ChildBodyKind::Clause => {
                for variable in collect_body(function, body) {
                    owned.insert(variable);
                    *components.entry(variable).or_default() += 1;
                }
            }
        });
        let mut bodies = bodies.into_iter();
        let body = BodyFrame::new(bodies.next().unwrap_or_default());
        Self {
            statement,
            owned,
            components,
            bodies,
            body,
        }
    }

    /// Counts the variables of the lexical body that was just analyzed as
    /// one more component each.
    fn take_in_body(&mut self) {
        for variable in core::mem::take(&mut self.body.contained) {
            *self.components.entry(variable).or_default() += 1;
        }
    }

    /// Anchors the variables that no narrower body can hold, and returns
    /// every variable that occurs in the statement.
    fn finish(self, placements: &mut [Option<S>]) -> BTreeSet<V> {
        for (&variable, &component_count) in &self.components {
            if self.owned.contains(&variable) || component_count > 1 {
                place(placements, variable, self.statement);
            }
        }
        self.components.into_keys().collect()
    }
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
    use crate::graph::view::DenseId;
    use crate::ir::hlil::tests::{
        EVERY_WRAP, Kit, ToyHlilBuilder, VarId, Wrap, nest, on_small_stack, over_toys,
    };
    use crate::ir::hlil::{FunctionBuilder, VariablePlacements};

    over_toys!(
        shared_branch_variable_anchors_before_if,
        branch_local_variable_stays_in_the_branch_body,
        loop_local_variable_stays_in_the_loop_body,
        for_clause_variable_anchors_before_the_loop,
        switch_arm_variables_anchor_in_their_arm,
        a_deep_nest_anchors_without_recursion,
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

    /// The analysis keeps its own stack, so statements nested as deep as
    /// untrusted input can make them do not overflow the thread's.
    fn a_deep_nest_anchors_without_recursion<D: Kit>() {
        const DEPTH: usize = 100_000;
        for wraps in [&[Wrap::If][..], &[Wrap::While], &EVERY_WRAP] {
            let (root, core, placements) = on_small_stack(|| {
                let mut builder = ToyHlilBuilder::<D>::new();
                let shared = builder.int_variable(0);
                let local = builder.int_variable(1);
                let value = builder.read(shared);
                let core = builder.store(local, value);
                let root = nest(&mut builder, wraps, DEPTH, core);
                let later = builder.use_of(shared);
                let function = builder.finish_unchecked(&[root, later]);
                (root, core, VariablePlacements::compute(&function))
            });
            let (shared, local) = (VarId::from_index(0), VarId::from_index(1));

            // `local` stays at the bottom; `shared` is used after the nest,
            // so it is declared before the whole of it.
            assert_eq!(placements.placement(local), Some(core), "{wraps:?}");
            assert_eq!(placements.placement(shared), Some(root), "{wraps:?}");
            assert_eq!(placements.before(core), [local], "{wraps:?}");
            assert_eq!(placements.before(root), [shared], "{wraps:?}");
        }
    }
}
