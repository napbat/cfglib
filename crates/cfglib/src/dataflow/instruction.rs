//! The instruction-side contract of the data-flow analyses: which variables
//! an instruction reads and writes, and the opt-in predicate and effect
//! extensions.

/// An IR-defined identity that participates in data-flow analysis.
///
/// This blanket trait accepts any ordered, cloneable type. An adapter can
/// therefore use a compact integer, an x86 register/flag enum, a shader
/// register-component structure, or another native identity directly.
///
/// The identity must describe the atomic unit tracked by the analyses. For
/// architectures with overlapping resources, such as x86 subregisters, the
/// adapter is responsible for exposing canonical units or all affected units.
pub trait VariableId: Clone + Ord {}

impl<T: Clone + Ord> VariableId for T {}

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
    /// IR variable identity used by this instruction representation.
    type Variable: VariableId;

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
    /// The analyses read only this method. The
    /// [`FlowEffect::MayThrow`](crate::FlowEffect::MayThrow) classification
    /// the CFG builder sees is separate, so an adapter that classifies a
    /// throwing instruction reports it here as well.
    fn may_unwind(&self) -> bool {
        false
    }
}

/// Opt-in: instructions whose execution is predicated on a condition
/// variable (ARM IT blocks, GPU wave predication, x86 CMOV, guarded
/// statements).
///
/// The predicate is expressed in the consumer's own variable domain.
///
/// # Contract
///
/// The predicate variable is a READ: it **must also appear in
/// [`uses`](InstrInfo::uses)**. The dataflow analyses consult only
/// `uses`/`defs` — an adapter that reports the guard here but omits it
/// from `uses` will see liveness call the predicate's definition dead and
/// dead-code elimination delete it while the guard still names it.
pub trait Predicated: InstrInfo {
    /// The guarding variable and the polarity under which the instruction
    /// executes. `None` for unconditionally executed instructions.
    fn predicate(&self) -> Option<(Self::Variable, bool)>;
}

/// Opt-in: instructions that declare side effects beyond their def/use sets.
///
/// The effect vocabulary is consumer-typed — machine memory and I/O for a
/// binary adapter, GC allocation / panics / channel sends for a source
/// language — the library imposes none. Purity analysis and dead-code
/// elimination consume this trait; requiring it there means a consumer must
/// state an effect vocabulary before code with possible side effects can be
/// classified or deleted.
pub trait EffectInfo: InstrInfo {
    /// Consumer effect vocabulary.
    type Effect: Clone + Ord;

    /// Side effects of this instruction. An empty slice means the
    /// instruction is pure.
    fn effects(&self) -> &[Self::Effect];
}
