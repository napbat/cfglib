//! The flow substrate: how an operation transfers control, and how the
//! builders read it.
//!
//! Direct structural construction
//! ([`Cfg::new_block`](crate::Cfg::new_block) /
//! [`Cfg::add_edge`](crate::Cfg::add_edge)) needs no trait at all. Both
//! builder front doors read one vocabulary for how an operation leaves:
//!
//! - [`Flow`] states an operation's successors by target identity: a code
//!   address, a label, a function. A jump, a call, a return, a throw, and a
//!   trap are [`Flow`] variants for every reader. Every edge it produces
//!   carries an [`EdgeRole`], and [`EdgeRole::kind`] is the one mapping from
//!   a role to the structural [`EdgeKind`](crate::EdgeKind).
//! - [`build_address_cfg`](crate::build_address_cfg) reads the [`Flow`] of
//!   each instruction through
//!   [`AddressInstruction`](crate::AddressInstruction).
//! - [`CfgBuilder`](crate::CfgBuilder) reads [`FlowControl`]: a flat,
//!   structured stream (`if`/`else`/`endif`, loops, breaks) whose
//!   [`FlowEffect`] is either a structured marker the builder pairs, a label,
//!   or a [`Flow`]. [`resolve_jump_edges`](crate::resolve_jump_edges) wires
//!   the targets those flows name, and [`call_graph`](crate::call_graph)
//!   reads their callees.
//!
//! Whether an instruction may throw is not a flow: an instruction that
//! continues and may throw has the flow [`Flow::Next`] and reports
//! [`InstrInfo::may_unwind`](crate::InstrInfo::may_unwind).

pub mod transfer;

pub use transfer::{
    CallSite, EdgeRole, Flow, Transfer, Transfers, UnresolvedRole, UnresolvedTransfer,
};

/// How one instruction of a structured stream steers control.
///
/// The structured markers open, split, and close the regions that
/// [`CfgBuilder`](crate::CfgBuilder) pairs; every other instruction states
/// how it leaves as a [`Flow`]. `T` is the consumer's target identity: a
/// label token for a jump, a function key for a call. `K` is the
/// consumer's switch case key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlowEffect<T, K> {
    /// An unstructured transfer, or none ([`Flow::Next`]).
    ///
    /// [`CfgBuilder`](crate::CfgBuilder) **stores** the instruction in the
    /// current block. [`Flow::Next`] and a call that continues end no block,
    /// so a call and the instructions after it stay in one block. Every
    /// other flow ends the block, and one that continues gets an edge to the
    /// next block: [`ConditionalFalse`] after a conditional transfer or a
    /// conditional return, [`Fallthrough`] after an
    /// [`Exceptional`](Flow::Exceptional) transfer or a conditional trap. A
    /// conditional return also gets a [`ConditionalTrue`] edge to an empty
    /// exit block. The builder wires no call edge; the other named targets
    /// are wired afterwards by
    /// [`resolve_jump_edges`](crate::resolve_jump_edges).
    ///
    /// [`ConditionalFalse`]: crate::EdgeKind::ConditionalFalse
    /// [`ConditionalTrue`]: crate::EdgeKind::ConditionalTrue
    /// [`Fallthrough`]: crate::EdgeKind::Fallthrough
    Flow(Flow<T, K>),
    /// Opens a conditional region (`if`).
    ///
    /// The builder **stores** this instruction in the current block.
    ConditionalOpen,
    /// Alternate branch within a conditional region (`else`).
    ///
    /// The builder **stores** this instruction in the new else block.
    ConditionalAlternate,
    /// Closes a conditional region (`endif`).
    ///
    /// The builder does **not** store this instruction in any block;
    /// it only creates the merge block and wires up edges.
    ConditionalClose,
    /// Opens a switch region. `break` inside a switch exits the switch
    /// (unlike `break` in a loop which exits the loop).
    ///
    /// The builder **stores** this instruction in the current block.
    SwitchOpen,
    /// A case/default arm inside a switch.
    ///
    /// The builder **stores** this instruction in the new case block.
    SwitchCase,
    /// Closes a switch region (`endswitch`).
    ///
    /// The builder does **not** store this instruction in any block;
    /// it only creates the merge block and wires up edges.
    SwitchClose,
    /// Opens a loop region.
    ///
    /// The builder **stores** this instruction in the current block.
    LoopOpen,
    /// Closes a loop region.
    ///
    /// The builder does **not** store this instruction in any block;
    /// it only creates the back-edge and post-loop block.
    LoopClose,
    /// Unconditional break out of the innermost loop/switch.
    Break,
    /// Conditional break.
    ConditionalBreak,
    /// Unconditional continue to loop header.
    Continue,
    /// Conditional continue.
    ConditionalContinue,
    /// Defines the jump target `T`.
    ///
    /// The builder **stores** this instruction at the start of a new block,
    /// which the current block falls through to, so
    /// [`resolve_jump_edges`](crate::resolve_jump_edges) can find the block
    /// a target names.
    Label(T),
}

/// Trait that an instruction type implements so the CFG builder can classify
/// each instruction's effect on control flow.
///
/// Display rendering lives on [`DisplayInstr`](crate::DisplayInstr); this
/// trait is only the flow classification itself.
pub trait FlowControl {
    /// The consumer's target identity: a raw address, a label symbol, a
    /// function key, a CST node id — never a string imposed by the library.
    ///
    /// Jump targets and labels share it, and so do callees, which
    /// [`call_graph`](crate::call_graph) matches against its function keys.
    type Target: Clone + Ord;
    /// The consumer's switch case key.
    type CaseKey;

    /// Classify this instruction's control-flow effect.
    fn flow_effect(&self) -> FlowEffect<Self::Target, Self::CaseKey>;
}
