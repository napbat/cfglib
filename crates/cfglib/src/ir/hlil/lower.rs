//! HLIL to flat lowering: control-flow flattening plus expression
//! linearization.
//!
//! [`lower`] is the downward mirror of `lift`: structured
//! statements become blocks and edges, expression trees flatten into typed
//! temporaries, and declared exception regions are registered on the flat
//! graph. The library owns block formation, joins, loop and switch wiring,
//! and the relation between the two functions. The consumer's
//! [`FlatBuilder`] decides what each flat instruction and edge is, from the
//! HLIL terms of [`Lowered`] and [`LowerEdge`].
//!
//! # Contract
//!
//! Every completed control path must end in a
//! [`Return`](super::StmtShape::Return) statement or iterate forever. A body
//! whose control falls off the end has no flat representation and is
//! rejected. Non-local transfers (`break`, `return`) out of a `try` do not
//! route through its `finally` body: a frontend with such semantics
//! duplicates the finally statements at each exit before lowering, as source
//! compilers do, keeping [`finally_body`](super::StmtShape::Try) for the
//! exceptional path.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::dataflow::ProgramPoint;
use crate::graph::view::DenseId;

mod control;

use super::flat_builder::{FlatBuilder, LowerEdge, Lowered, Occurrence, VariableOrigin};
use super::view::visit_expression_tree;
use super::{Dialect, Entity, ExprShape, FunctionView, LowerError};

/// The result of lowering one structured function.
#[derive(Debug, Clone)]
pub struct LoweredFunction<F, S, E, V, W> {
    /// The function that the consumer's builder sealed.
    pub function: F,
    /// The structured entity that each appended flat instruction was lowered
    /// from, in emission order. The rows are not sorted by point;
    /// [`Correspondence::instructions`](super::Correspondence) is the sorted
    /// form that a lift returns.
    ///
    /// An operation, a constant, or a copy that evaluates an expression
    /// carries [`Entity::Expression`]. The instruction that finishes a
    /// statement (the store or copy of an assignment, a branch, a dispatch, a
    /// return, a handler binding, a region enter or exit) carries
    /// [`Entity::Statement`]. A statement with several instructions has
    /// several rows, and a point has at most one.
    pub origins: Vec<(ProgramPoint, Entity<S, E, V>)>,
    /// The flat variable of each structured variable, by index, followed by
    /// the temporaries that lowering introduced.
    pub variables: Vec<W>,
}

/// Lowers one structured function into a flat one through `builder`.
///
/// The function is declared first: every variable one to one, in table
/// order, then the signature. The body then lowers statement by statement.
/// Control flow lowers through lazy join materialization: ending a construct
/// leaves its continuation as pending edges, and the next emitted
/// instruction materializes the join block and wires them. A path that
/// terminates (return, goto, break) leaves nothing pending.
///
/// # Errors
///
/// Returns [`LowerError::Unsupported`] when a statement has no flat
/// translation: control falling off the end of the function, a `break`,
/// `continue`, or `goto` with no matching target, a `continue` to a label
/// that does not name a loop, an assignment into a constant place, a handler
/// binding with no declared type, or a filter handler with an empty filter
/// body or one that does not end in a terminator. Returns
/// [`LowerError::Builder`] when the consumer's builder refuses an instruction
/// or the sealed function.
pub fn lower<H, S, B>(source: &S, mut builder: B) -> Lowering<H, S, B>
where
    H: Dialect,
    S: FunctionView<H>,
    B: FlatBuilder<H>,
{
    let mut variables = Vec::with_capacity(source.variable_count());
    for index in 0..source.variable_count() {
        let variable = S::Var::from_index(index);
        variables.push(builder.declare_variable(VariableOrigin::Source {
            variable: source.variable(variable),
            declared_type: source.declared_type(variable),
        }));
    }
    let parameters: Vec<B::Var> = source
        .parameters()
        .iter()
        .map(|parameter| variables[parameter.index()])
        .collect();
    builder.set_signature(&parameters, source.returns());

    let entry = builder.entry();
    let mut lowerer = Lowerer {
        source,
        builder,
        current: None,
        pending: alloc::vec![(entry, LowerEdge::Entry)],
        frames: Vec::new(),
        try_frames: Vec::new(),
        labels: BTreeMap::new(),
        forward_gotos: BTreeMap::new(),
        variables,
        created: Vec::new(),
        materialized: Vec::new(),
        origins: Vec::new(),
    };
    lowerer.lower_body(source.body())?;
    if lowerer.current.is_some() || !lowerer.pending.is_empty() {
        return Err(LowerError::Unsupported(
            "control falls off the end of the function; every path must end in a return or \
             iterate forever"
                .into(),
        ));
    }
    if let Some(label) = lowerer.forward_gotos.keys().next() {
        return Err(LowerError::Unsupported(alloc::format!(
            "goto targets the label {label}, which no statement defines"
        )));
    }
    let Lowerer {
        builder,
        variables,
        origins,
        ..
    } = lowerer;
    let function = builder.finish().map_err(LowerError::Builder)?;
    Ok(LoweredFunction {
        function,
        origins,
        variables,
    })
}

/// Edges into the block that the next instruction materializes.
type Sources<'a, C> = Vec<(BlockId, LowerEdge<'a, C>)>;

/// An occurrence of a flat variable at the type it is read or written at.
type Value<H, B> = Occurrence<<B as FlatBuilder<H>>::Var, <H as Dialect>::Type>;

/// The structured entity that a flat instruction is lowered from.
type Origin<H, S> =
    Entity<<S as FunctionView<H>>::Stmt, <S as FunctionView<H>>::Expr, <S as FunctionView<H>>::Var>;

/// What `lower` returns.
type Lowering<H, S, B> = Result<
    LoweredFunction<
        <B as FlatBuilder<H>>::Function,
        <S as FunctionView<H>>::Stmt,
        <S as FunctionView<H>>::Expr,
        <S as FunctionView<H>>::Var,
        <B as FlatBuilder<H>>::Var,
    >,
    LowerError<<B as FlatBuilder<H>>::Error>,
>;

/// One enclosing construct receiving break and continue transfers.
struct Frame<'a, C> {
    label: Option<&'a str>,
    kind: FrameKind,
    /// Sources waiting to be wired to the construct's continuation.
    break_sources: Sources<'a, C>,
}

enum FrameKind {
    /// A loop; `continue` transfers to `continue_target`.
    Loop {
        continue_target: BlockId,
    },
    Switch,
    /// A labeled non-loop statement; only labeled breaks reach it.
    Block,
}

/// One protected region being lowered. It collects the blocks that may
/// throw, which is where unwind edges leave from.
struct TryFrame {
    may_throw_blocks: BTreeSet<BlockId>,
}

struct Lowerer<'a, H: Dialect, S: FunctionView<H>, B: FlatBuilder<H>> {
    source: &'a S,
    builder: B,
    /// The open block receiving instructions, when one exists.
    current: Option<BlockId>,
    /// Edges into the next materialized block.
    pending: Sources<'a, H::Constant>,
    frames: Vec<Frame<'a, H::Constant>>,
    try_frames: Vec<TryFrame>,
    /// Materialized label targets.
    labels: BTreeMap<&'a str, BlockId>,
    /// Forward-goto sources waiting for their label to materialize.
    forward_gotos: BTreeMap<&'a str, Sources<'a, H::Constant>>,
    /// The flat variable of each structured variable, then the temporaries.
    variables: Vec<B::Var>,
    /// Every block this lowering allocated, in allocation order. A region
    /// protects the blocks allocated while its body lowered; the builder
    /// need not number blocks densely.
    created: Vec<BlockId>,
    /// The blocks that received pending edges, in order. The first one
    /// allocated while a loop body lowered is where the loop is entered.
    materialized: Vec<BlockId>,
    origins: Vec<(ProgramPoint, Origin<H, S>)>,
}

impl<'a, H, S, B> Lowerer<'a, H, S, B>
where
    H: Dialect,
    S: FunctionView<H>,
    B: FlatBuilder<H>,
{
    fn new_block(&mut self) -> BlockId {
        let block = self.builder.new_block(None);
        self.created.push(block);
        block
    }

    /// The open block, materializing one (and wiring the pending edges) when
    /// none is open.
    fn block(&mut self) -> BlockId {
        if let Some(block) = self.current {
            return block;
        }
        let block = self.new_block();
        self.materialized.push(block);
        for (from, role) in core::mem::take(&mut self.pending) {
            self.builder.add_edge(from, block, role);
        }
        self.current = Some(block);
        block
    }

    /// Ends the open block, deferring its continuation into the next
    /// materialized block through `role`.
    fn seal(&mut self, role: LowerEdge<'a, H::Constant>) {
        if let Some(block) = self.current.take() {
            self.pending.push((block, role));
        }
    }

    /// Wires the open block and every pending edge into `target`.
    fn connect_to(&mut self, target: BlockId) {
        if let Some(block) = self.current.take() {
            self.builder.add_edge(block, target, LowerEdge::Jump);
        }
        for (from, role) in core::mem::take(&mut self.pending) {
            self.builder.add_edge(from, target, role);
        }
    }

    /// Appends one instruction to the open block, recording its origin and
    /// the may-throw fact for every enclosing region.
    fn emit(
        &mut self,
        lowered: Lowered<'a, H>,
        uses: &[Value<H, B>],
        defs: &[Value<H, B>],
        origin: Origin<H, S>,
    ) -> Result<(), LowerError<B::Error>> {
        let block = self.block();
        let appended = self
            .builder
            .append(block, lowered, uses, defs)
            .map_err(LowerError::Builder)?;
        if appended.may_throw {
            for frame in &mut self.try_frames {
                frame.may_throw_blocks.insert(block);
            }
        }
        self.origins.push((appended.point, origin));
        Ok(())
    }

    /// Introduces a transient that holds a value of type `ty`.
    fn temporary(&mut self, ty: &H::Type) -> B::Var {
        let variable = self
            .builder
            .declare_variable(VariableOrigin::Temporary { ty });
        self.variables.push(variable);
        variable
    }

    /// The flat occurrence of the structured variable read by `expr`.
    fn read(&self, expr: S::Expr, variable: S::Var) -> Value<H, B> {
        Occurrence {
            variable: self.variables[variable.index()],
            ty: self.source.ty(expr).clone(),
        }
    }

    /// Flattens the tree under `root`, operands first, and returns the
    /// occurrence that holds its value. Only a variable read needs no
    /// instruction.
    fn flatten(&mut self, root: S::Expr) -> Result<Value<H, B>, LowerError<B::Error>> {
        if let ExprShape::Variable(variable) = self.source.expr(root) {
            return Ok(self.read(root, variable));
        }
        let mut values = Vec::with_capacity(1);
        self.flatten_onto(root, &mut values)?;
        Ok(values
            .pop()
            .expect("flattening a tree leaves the value of its root"))
    }

    /// Flattens `roots` in order into one occurrence each.
    fn flatten_all(&mut self, roots: &[S::Expr]) -> Result<Vec<Value<H, B>>, LowerError<B::Error>> {
        let mut values = Vec::with_capacity(roots.len());
        for &root in roots {
            self.flatten_onto(root, &mut values)?;
        }
        Ok(values)
    }

    /// Flattens the tree under `root` in postorder and pushes the occurrence
    /// that holds its value onto `values`.
    ///
    /// The walk keeps its own stack, so the depth of an expression costs no
    /// call frames. `values` holds the finished operands of the nodes that
    /// are still open, and a node takes its operands from its top.
    fn flatten_onto(
        &mut self,
        root: S::Expr,
        values: &mut Vec<Value<H, B>>,
    ) -> Result<(), LowerError<B::Error>> {
        let source = self.source;
        let mut failure = None;
        visit_expression_tree::<H, S>(source, root, &mut |expr| {
            if failure.is_none()
                && let Err(error) = self.flatten_node(expr, values)
            {
                failure = Some(error);
            }
        });
        failure.map_or(Ok(()), Err)
    }

    /// Flattens one node whose operands are already on top of `values`.
    fn flatten_node(
        &mut self,
        expr: S::Expr,
        values: &mut Vec<Value<H, B>>,
    ) -> Result<(), LowerError<B::Error>> {
        let source = self.source;
        let origin = Entity::Expression(expr);
        match source.expr(expr) {
            ExprShape::Variable(variable) => values.push(self.read(expr, variable)),
            ExprShape::Constant(constant) => {
                let ty = source.ty(expr).clone();
                let variable = self.temporary(&ty);
                let value = Occurrence { variable, ty };
                self.emit(
                    Lowered::Constant(constant),
                    &[],
                    core::slice::from_ref(&value),
                    origin,
                )?;
                values.push(value);
            }
            ExprShape::Operation { op, operands } => {
                let first = values.len() - operands.len();
                let ty = source.ty(expr).clone();
                let variable = self.temporary(&ty);
                let value = Occurrence { variable, ty };
                self.emit(
                    Lowered::Operation(op),
                    &values[first..],
                    core::slice::from_ref(&value),
                    origin,
                )?;
                values.truncate(first);
                values.push(value);
            }
        }
        Ok(())
    }

    /// Flattens `value` directly into `target`, which avoids a temporary for
    /// the root of the tree. The instruction that writes `target` has
    /// `origin`.
    fn flatten_into(
        &mut self,
        target: &Value<H, B>,
        value: S::Expr,
        origin: Origin<H, S>,
    ) -> Result<(), LowerError<B::Error>> {
        let target = core::slice::from_ref(target);
        match self.source.expr(value) {
            ExprShape::Variable(variable) => {
                let copied = self.read(value, variable);
                self.emit(Lowered::Copy, &[copied], target, origin)
            }
            ExprShape::Constant(constant) => {
                self.emit(Lowered::Constant(constant), &[], target, origin)
            }
            ExprShape::Operation { op, operands } => {
                let uses = self.flatten_all(operands)?;
                self.emit(Lowered::Operation(op), &uses, target, origin)
            }
        }
    }
}
