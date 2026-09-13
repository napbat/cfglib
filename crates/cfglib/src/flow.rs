//! The flow substrate: how an operation transfers control, and how the
//! builders read it.
//!
//! Direct structural construction
//! ([`Cfg::new_block`](crate::Cfg::new_block) /
//! [`Cfg::add_edge`](crate::Cfg::add_edge)) needs no trait at all. The two
//! builder front doors read one of two descriptions:
//!
//! - [`Flow`] states an operation's successors by target identity: a code
//!   address, a label, a block. [`build_address_cfg`](crate::build_address_cfg)
//!   reads it through [`AddressInstruction`](crate::AddressInstruction), and
//!   any walker that follows control from operation to operation reads it
//!   through [`Flow::transfers`]. Every edge it produces carries an
//!   [`EdgeRole`], and [`EdgeRole::kind`] is the one mapping from a role to
//!   the structural [`EdgeKind`](crate::EdgeKind).
//! - [`FlowControl`] classifies an operation in a flat, structured stream
//!   (`if`/`else`/`endif`, loops, breaks) so [`CfgBuilder`](crate::CfgBuilder)
//!   can pair the markers, and the opt-in companions ([`JumpTargets`],
//!   [`CallInfo`]) type the branch and call targets those operations carry.

pub mod transfer;

pub use transfer::{
    CallSite, EdgeRole, Flow, Transfer, Transfers, UnresolvedRole, UnresolvedTransfer,
};

/// Classification of an instruction's effect on control flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlowEffect {
    /// Normal instruction — execution falls through to the next.
    Fallthrough,
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
    /// Unconditional return / function terminator.
    Return,
    /// Conditional return.
    ConditionalReturn,
    /// Unconditional call to a label/address.
    Call,
    /// Conditional call.
    ConditionalCall,
    /// Terminates execution of the current invocation (e.g. `discard`).
    Terminate,
    /// A label/target that can be jumped to.
    Label,
    /// Declaration or metadata — skipped by the builder.
    Declaration,

    /// Unconditional explicit jump: a source `goto`, a machine `jmp`/`b`.
    ///
    /// The builder ends the block; the target edge is wired afterwards by
    /// [`resolve_jump_edges`](crate::resolve_jump_edges) when the
    /// instruction implements [`JumpTargets`].
    Jump,
    /// Conditional jump — falls through on false, jumps on true. A guarded
    /// `goto`, a machine `jcc`/`b.cond`.
    ///
    /// The builder wires the fallthrough edge; the taken edge is wired
    /// afterwards by [`resolve_jump_edges`](crate::resolve_jump_edges)
    /// when the instruction implements [`JumpTargets`].
    ConditionalJump,
    /// Computed / indirect jump (target unknown statically): a source
    /// computed goto or lowered `match` dispatch, a machine table jump.
    IndirectJump,
    /// Indirect call (target resolved at runtime): source dynamic dispatch
    /// or function-pointer call, a machine `call [vtable]`.
    IndirectCall,
    /// Instruction that may throw or trap: a source throwing expression or
    /// `raise`, a machine trapping `div` / `int 3`.
    MayThrow,
}

/// Trait that an instruction type implements so the CFG builder can classify
/// each instruction's effect on control flow.
///
/// Display rendering lives on [`DisplayInstr`](crate::DisplayInstr) and
/// explicit branch targets on [`JumpTargets`]; this trait is only the flow
/// classification itself.
pub trait FlowControl {
    /// Classify this instruction's control-flow effect.
    fn flow_effect(&self) -> FlowEffect;
}

/// Opt-in: instructions that transfer control to a callable.
///
/// The instruction performing a call is the authoritative source of its
/// target — the callee is consumer-typed (a raw address, a symbol id, a
/// mangled name, a CST node), never a library-imposed string or address
/// field. Used by
/// [`call_graph`](crate::call_graph) and explicit tail-call
/// detection.
pub trait CallInfo {
    /// Consumer-defined callee identity.
    type Callee: Clone + Ord;

    /// The callee, when statically known. `None` for non-call instructions
    /// and for unresolved indirect calls.
    fn callee(&self) -> Option<Self::Callee>;

    /// Whether this call site is a tail call (no return expected).
    fn is_tail_call(&self) -> bool {
        false
    }
}

/// Opt-in: instructions with explicit branch targets (gotos and labels).
///
/// The target token is consumer-typed — a raw address, a label symbol, a CST
/// node id — never a string imposed by the library. After
/// [`CfgBuilder::build`](crate::CfgBuilder::build), run
/// [`resolve_jump_edges`](crate::resolve_jump_edges) to wire the edges these
/// targets describe.
pub trait JumpTargets: FlowControl {
    /// Branch-target token: address, label symbol, CST node id.
    type Target: Clone + Ord;

    /// Target of a [`FlowEffect::Jump`] / [`FlowEffect::ConditionalJump`]
    /// instruction. `None` for every other instruction.
    fn jump_target(&self) -> Option<Self::Target>;

    /// Target token this instruction *defines* (for [`FlowEffect::Label`]
    /// instructions). `None` for every other instruction.
    fn label(&self) -> Option<Self::Target> {
        None
    }
}
