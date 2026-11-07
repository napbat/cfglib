//! The instruction-side contract of the data-flow analyses: which variables
//! an instruction reads and writes, and the opt-in predicate and effect
//! extensions.
//!
//! Each fact comes in two forms. [`InstrInfo`] and [`EffectInfo`] answer from
//! the instruction alone. [`InstrOperands`] and [`EffectOperands`] answer
//! through a context the caller passes beside the instruction, for an
//! instruction that stores its operands as runs of storage its function owns.
//! The unit context `()` reads the first form through the second, so every
//! analysis that takes a context also takes an [`InstrInfo`] instruction.

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

/// The context-carrying form of [`InstrInfo`]: the variables an instruction
/// reads and writes, read through a context that owns them.
///
/// An instruction that stores its operands as a run of a pool its function
/// owns cannot lend them from `&self` alone, so it would otherwise hold a
/// pointer to the pool. The context is that pool, or whatever owns it, and an
/// analysis that takes one reads every operand through it. The answers follow
/// the same rules as [`InstrInfo`]: the same slices on every call, and every
/// possible unwind point reported by [`may_unwind`](Self::may_unwind).
///
/// `()` is the context of an instruction that implements [`InstrInfo`], so
/// [`CfgView`](crate::CfgView) over such instructions is read through it.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, InstrOperands, Liveness, WithOperands};
///
/// /// An instruction that is two runs of its function's operand pool.
/// #[derive(Clone)]
/// struct Inst { uses: (u32, u32), defs: (u32, u32) }
///
/// /// The pool every instruction of one function names runs of.
/// struct Pool(Vec<u16>);
///
/// impl InstrOperands<Inst> for Pool {
///     type Variable = u16;
///     fn uses<'a>(&'a self, inst: &'a Inst) -> &'a [u16] {
///         &self.0[inst.uses.0 as usize..inst.uses.1 as usize]
///     }
///     fn defs<'a>(&'a self, inst: &'a Inst) -> &'a [u16] {
///         &self.0[inst.defs.0 as usize..inst.defs.1 as usize]
///     }
/// }
///
/// let pool = Pool(vec![7, 7]);
/// let mut cfg = Cfg::<Inst>::new();
/// let (b0, b1) = (cfg.entry(), cfg.new_block());
/// cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
/// cfg.block_mut(b0).push(Inst { uses: (0, 0), defs: (0, 1) });
/// cfg.block_mut(b1).push(Inst { uses: (1, 2), defs: (2, 2) });
///
/// let live = Liveness::compute(&WithOperands::new(&cfg, &pool));
/// assert!(live.is_live_out(&7, b0));
/// ```
pub trait InstrOperands<I> {
    /// IR variable identity of the instructions this context answers for.
    type Variable: VariableId;

    /// Variables that `instruction` **reads** (uses).
    fn uses<'a>(&'a self, instruction: &'a I) -> &'a [Self::Variable];

    /// Variables that `instruction` **writes** (defines).
    fn defs<'a>(&'a self, instruction: &'a I) -> &'a [Self::Variable];

    /// Whether execution can unwind before `instruction` defines its outputs,
    /// with the meaning of [`InstrInfo::may_unwind`]. The default is `false`.
    fn may_unwind(&self, _instruction: &I) -> bool {
        false
    }
}

impl<I: InstrInfo> InstrOperands<I> for () {
    type Variable = I::Variable;

    fn uses<'a>(&'a self, instruction: &'a I) -> &'a [Self::Variable] {
        instruction.uses()
    }

    fn defs<'a>(&'a self, instruction: &'a I) -> &'a [Self::Variable] {
        instruction.defs()
    }

    fn may_unwind(&self, instruction: &I) -> bool {
        instruction.may_unwind()
    }
}

/// The context-carrying form of [`EffectInfo`]: the side effects of an
/// instruction, read through a context that owns them.
///
/// `()` is the context of an instruction that implements [`EffectInfo`].
pub trait EffectOperands<I>: InstrOperands<I> {
    /// Consumer effect vocabulary.
    type Effect: Clone + Ord;

    /// Side effects of `instruction`. An empty slice means the instruction
    /// is pure.
    fn effects<'a>(&'a self, instruction: &'a I) -> &'a [Self::Effect];
}

impl<I: EffectInfo> EffectOperands<I> for () {
    type Effect = I::Effect;

    fn effects<'a>(&'a self, instruction: &'a I) -> &'a [Self::Effect] {
        instruction.effects()
    }
}
