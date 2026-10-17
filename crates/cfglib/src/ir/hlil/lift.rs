//! Lifting a flat function into structured, expression-oriented HLIL:
//! control-flow structuring plus expression recovery.
//!
//! [`lift`] structures the control flow of a [`FlatFunction`] into a
//! [`ControlTree`] and translates each flat instruction list into statements
//! and expression trees. Single-use pure definitions inline into their
//! consumer, effectful definitions inline only when the evaluation order is
//! provably preserved, and everything else materializes as an assignment. The
//! per-instruction classification comes from [`LiftInstruction`], and the
//! library owns the ordering, the structuring, and the correspondence.
//!
//! The lift drives a consumer-implemented [`FunctionBuilder`], which is
//! write-only. A definition that is planned to inline therefore waits in a
//! [`Scratch`] arena until a statement consumes it, and only then does the
//! lift call the builder, children first.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::graph::view::DenseId;
use crate::{ControlTree, LiftReport, Liveness};
use crate::{InstrInfo, ProgramPoint};

use super::builder::FunctionBuilder;
use super::correspondence::Correspondence;
use super::dialect::LiftDialect;
use super::error::LiftError;
use super::flat::{FlatFunction, InstructionShape, LiftInstruction};
use super::view::Entity;

mod block;
mod control;
mod plan;
mod scratch;

use self::scratch::Slot;

/// The result of lifting one flat function.
#[derive(Debug, Clone)]
pub struct LiftedFunction<F, S, E, V> {
    /// The lifted function, sealed by the builder.
    pub function: F,
    /// The structuring fidelity of the control tree that the lift read.
    pub report: LiftReport,
    /// The tables that relate the flat function to the lifted one.
    pub correspondence: Correspondence<S, E, V>,
}

/// The metadata that a lift retains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LiftMetadata {
    /// Fill [`Correspondence::instructions`] with the entity that carries
    /// each flat instruction.
    #[default]
    Preserve,
    /// Leave [`Correspondence::instructions`] empty, for a consumer that
    /// needs only the HLIL semantics. The variable table is kept either way.
    Omit,
}

/// What a lift returns: the lifted function, or why it failed.
type Lifted<H, B> = Result<
    LiftedFunction<
        <B as FunctionBuilder<H>>::Function,
        <B as FunctionBuilder<H>>::Stmt,
        <B as FunctionBuilder<H>>::Expr,
        <B as FunctionBuilder<H>>::Var,
    >,
    LiftError<<B as FunctionBuilder<H>>::Error>,
>;

/// The entity of the lifted function that carries a flat instruction.
type Carrier<H, B> = Entity<
    <B as FunctionBuilder<H>>::Stmt,
    <B as FunctionBuilder<H>>::Expr,
    <B as FunctionBuilder<H>>::Var,
>;

/// A translation step that can fail only because the flat function has a
/// shape that HLIL cannot state. The builder reports its own failure when it
/// seals the function.
type Fallible<T> = Result<T, String>;

/// One flat instruction with the point it stands at.
type Entry<'a, I> = (ProgramPoint, &'a I);

/// Structures and lifts one flat function.
///
/// The control tree is computed here, with the blocks of
/// [`lift_trampolines`] treated as empty, so that a `break` or `continue`
/// that a compiler routes through an instruction-free jump block resolves
/// structurally instead of degrading to a `goto`. Use [`lift_with_tree`] to
/// reuse a tree that is already computed.
///
/// The lift declares every variable of `source` in order, sets the signature,
/// translates the tree, and seals the function with
/// [`finish`](FunctionBuilder::finish). It does not assume that the builder
/// issues handles equal to the flat indices:
/// [`Correspondence::variables`] states the map.
///
/// # Errors
///
/// Returns [`LiftError::Unsupported`] when the flat function has a shape with
/// no HLIL translation: an instruction that defines several results, a
/// conditional block whose terminator the instruction did not classify as a
/// branch, an instruction after the terminator of its block, a switch case
/// edge without values, or a parallel copy that overlaps without a
/// [temporary](LiftDialect::temporary_variable). Returns
/// [`LiftError::Builder`] when the builder refuses to seal the function.
pub fn lift<H, X, B>(source: &X, builder: B, metadata: LiftMetadata) -> Lifted<H, B>
where
    H: LiftDialect,
    X: FlatFunction<H>,
    B: FunctionBuilder<H>,
{
    let trampolines = lift_trampolines::<H, X>(source);
    let (tree, report) =
        ControlTree::compute_with_report_treating_empty(source.cfg(), &trampolines);
    lift_with_tree(source, &tree, &report, builder, metadata)
}

/// Lifts a flat function over a control tree and report that the caller
/// already computed for `source.cfg()`.
///
/// The caller owns the trampoline policy: pass
/// [`lift_trampolines`] to
/// [`ControlTree::compute_with_report_treating_empty`] to get the tree that
/// [`lift`] reads. The tree must hold whole-block spans, which is what every
/// constructor but the predicated one produces.
///
/// # Errors
///
/// Returns the same errors as [`lift`]. A tree with a
/// [`Guarded`](crate::ControlNode::Guarded) node is
/// [`LiftError::Unsupported`], because predicated regions have no HLIL
/// translation.
///
/// # Panics
///
/// Panics when `tree` was not computed from `source.cfg()`: a span then names
/// a block or an instruction that the graph does not have.
pub fn lift_with_tree<H, X, B>(
    source: &X,
    tree: &ControlTree,
    report: &LiftReport,
    builder: B,
    metadata: LiftMetadata,
) -> Lifted<H, B>
where
    H: LiftDialect,
    X: FlatFunction<H>,
    B: FunctionBuilder<H>,
{
    let mut lifter = Lifter::<H, X, B>::new(source, tree, builder, metadata);
    let mut body = Vec::new();
    lifter
        .translate_node(tree.root(), &mut body)
        .map_err(LiftError::Unsupported)?;
    let function = lifter.builder.finish(&body).map_err(LiftError::Builder)?;
    Ok(LiftedFunction {
        function,
        report: report.clone(),
        correspondence: Correspondence {
            instructions: lifter.rows.into_iter().collect(),
            variables: lifter.variables,
        },
    })
}

/// The blocks of `source` that carry no semantics beyond their outgoing
/// edges: every instruction lifts to
/// [`ControlFlow`](InstructionShape::ControlFlow), defines nothing, and
/// cannot throw.
///
/// Emission drops such instructions without consuming their uses, so treating
/// them as empty changes no lifted statement. It does change the structuring:
/// the control tree forwards `break` and `continue` through instruction-free
/// single-jump blocks (javac routes `break` through such trampolines), which
/// would otherwise degrade to gotos.
#[must_use]
pub fn lift_trampolines<H, X>(source: &X) -> Vec<BlockId>
where
    H: LiftDialect,
    X: FlatFunction<H>,
{
    let cfg = source.cfg();
    cfg.block_ids()
        .filter(|&block| {
            let instructions = cfg.block(block).instructions();
            !instructions.is_empty()
                && instructions.iter().all(|instruction| {
                    instruction.defs().is_empty()
                        && !instruction.may_throw()
                        && matches!(instruction.lift(), InstructionShape::ControlFlow)
                })
        })
        .collect()
}

/// One enclosing breakable construct during statement translation.
struct Context {
    /// The label of the loop, or `None` for a switch.
    loop_label: Option<String>,
    /// Set when an unlabeled break of the tree had to name this loop.
    needs_label: bool,
}

/// The state of one lift.
struct Lifter<'a, H, X, B>
where
    H: LiftDialect,
    X: FlatFunction<H>,
    B: FunctionBuilder<H>,
{
    cfg: &'a Cfg<X::Inst, X::Edge>,
    tree: &'a ControlTree,
    builder: B,
    /// The structured variable of each flat variable, by flat index.
    variables: Vec<B::Var>,
    liveness: Liveness<X::Var>,
    /// The entity that carries each flat instruction, sorted by point. Stays
    /// empty under [`LiftMetadata::Omit`].
    rows: BTreeMap<ProgramPoint, Carrier<H, B>>,
    metadata: LiftMetadata,
    contexts: Vec<Context>,
    /// Reused by the emission of an expression tree.
    values: Vec<B::Expr>,
    work: Vec<(Slot, bool)>,
    dialect: PhantomData<H>,
}

impl<'a, H, X, B> Lifter<'a, H, X, B>
where
    H: LiftDialect,
    X: FlatFunction<H>,
    B: FunctionBuilder<H>,
{
    fn new(source: &'a X, tree: &'a ControlTree, mut builder: B, metadata: LiftMetadata) -> Self {
        let variables: Vec<B::Var> = (0..source.variable_count())
            .map(|index| builder.declare_variable(source.variable(X::Var::from_index(index)), None))
            .collect();
        let parameters: Vec<B::Var> = source
            .parameters()
            .iter()
            .map(|parameter| variables[parameter.index()])
            .collect();
        builder.set_signature(&parameters, source.returns());
        let cfg = source.cfg();
        Self {
            cfg,
            tree,
            builder,
            variables,
            liveness: Liveness::compute(cfg),
            rows: BTreeMap::new(),
            metadata,
            contexts: Vec::new(),
            values: Vec::new(),
            work: Vec::new(),
            dialect: PhantomData,
        }
    }

    /// Records the entity that carries the flat instruction at `point`.
    fn record(&mut self, point: ProgramPoint, entity: Carrier<H, B>) {
        if self.metadata == LiftMetadata::Preserve {
            self.rows.insert(point, entity);
        }
    }

    /// The name of the label or goto target at `block`: its authored label,
    /// or `.bb{index}` when it has none. An empty label is an absent one.
    fn block_label(&self, block: BlockId) -> String {
        self.cfg
            .block(block)
            .label()
            .filter(|label| !label.is_empty())
            .map_or_else(|| format!(".bb{}", block.index()), String::from)
    }
}
