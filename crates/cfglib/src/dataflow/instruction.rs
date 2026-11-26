//! The instruction-side contract that a consumer's data-flow analyses read:
//! which variables an instruction reads and writes, and whether it may
//! unwind.

/// Trait that an instruction type implements to expose its data
/// dependencies (which variables it reads and writes).
///
/// This is the data-flow counterpart of [`FlowControl`](crate::FlowControl),
/// which classifies control-flow effects.
///
/// Returns slices rather than `Vec`s to avoid heap allocations in
/// hot-path fixpoint iteration. The instruction is expected to own (or
/// borrow from its own storage) the def/use sets it reports; adapters over
/// external structures typically materialize them once at construction.
/// Memory accesses are reported separately through the
/// [`MemoryEventInfo`](crate::MemoryEventInfo) extension so adapters without a
/// memory model keep this base contract minimal.
pub trait InstrInfo {
    /// IR variable identity used by this instruction representation: any
    /// ordered, cloneable type, such as a compact integer, a register enum,
    /// or a register-component structure.
    ///
    /// The identity describes the atomic unit the analyses track. For
    /// architectures with overlapping resources, such as x86 subregisters,
    /// the adapter exposes canonical units or all affected units.
    type Variable: Clone + Ord;

    /// Variables that this instruction **reads** (uses).
    fn uses(&self) -> &[Self::Variable];

    /// Variables that this instruction **writes** (defines).
    fn defs(&self) -> &[Self::Variable];

    /// Whether execution can unwind before this instruction defines its outputs.
    ///
    /// An [`ExceptionUnwind`](crate::EdgeKind::ExceptionUnwind) successor
    /// observes the values before this instruction, so its own definitions
    /// never reach the handler. Normal successors observe its completed
    /// definitions. Adapters with exceptional control flow must report every
    /// possible unwind point: a block with an unwind successor and no such
    /// instruction is treated as unwinding after its last instruction.
    ///
    /// A consumer's analyses read only this method. The
    /// [`FlowEffect::MayThrow`](crate::FlowEffect::MayThrow) classification
    /// the CFG builder sees is separate, so an adapter that classifies a
    /// throwing instruction reports it here as well.
    fn may_unwind(&self) -> bool {
        false
    }
}
