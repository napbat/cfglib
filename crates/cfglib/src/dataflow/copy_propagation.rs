//! Copy propagation.
//!
//! Identifies instructions that are simple copies (`dst = src`) and
//! replaces all uses of `dst` with `src`, then removes the dead copy.
//! [`alias_propagation`] applies the same guarded substitution to pairwise
//! value aliases whose types or other non-runtime metadata may differ.
//!
//! The consumer implements [`CopySource`] to tell the analysis which
//! instructions are copies and how to rewrite operands.
//!
//! This is a classic SSA/def-use chain optimization that simplifies
//! redundant moves and phi-resolved copies.

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::InstrInfo;
use super::def_use::DefUseChains;
use super::liveness::Liveness;
use crate::block::BlockId;
use crate::cfg::Cfg;

/// Borrowed pairwise alias definitions and their corresponding value sources.
pub type AliasPairs<'a, V> = (&'a [V], &'a [V]);

/// Trait for instructions that can be identified as copies.
///
/// A **copy** is an instruction with exactly one def and one use,
/// where the semantics are simply `def := use` with no computation.
/// Examples: `mov dst, src`, register-register copies, phi-resolved moves.
pub trait CopySource: InstrInfo {
    /// If this instruction is a simple copy, return `Some((dst, src))`.
    ///
    /// Return `None` if the instruction is not a copy (has side effects,
    /// multiple defs, computation, etc.).
    /// The returned destination and source must be the sole entries exposed by
    /// [`InstrInfo::defs`] and [`InstrInfo::uses`], respectively.
    fn as_copy(&self) -> Option<(Self::Variable, Self::Variable)>;

    /// Returns pairwise runtime-value aliases as `(definitions, uses)`.
    ///
    /// Each definition at position `i` must receive exactly the runtime value
    /// read from use `i`; value types and other non-runtime metadata may differ.
    /// All reads occur before any write. The default exposes an ordinary copy
    /// as a one-pair alias set.
    ///
    /// Consumers should override this for type refinements, parallel copy
    /// commits, or equivalent value-preserving operations. Returning `Some`
    /// promises that the instruction cannot throw, alter control flow, or have
    /// any observable effect beyond those definitions. Returning an empty or
    /// arity-mismatched pair is ignored.
    fn as_aliases(&self) -> Option<AliasPairs<'_, Self::Variable>> {
        (self.as_copy().is_some() && self.defs().len() == 1 && self.uses().len() == 1)
            .then(|| (self.defs(), self.uses()))
    }

    /// Rewrite a use of `old` to `new` in this instruction.
    ///
    /// Called during propagation to replace operands.
    fn rewrite_use(&mut self, old: &Self::Variable, new: &Self::Variable);
}

/// Result of copy propagation.
#[derive(Debug, Clone)]
pub struct CopyPropagationStats {
    /// Number of uses rewritten.
    pub uses_rewritten: usize,
    /// Number of copy instructions removed.
    pub copies_removed: usize,
}

/// Result of pairwise value-alias propagation.
#[derive(Debug, Clone)]
pub struct AliasPropagationStats {
    /// Number of uses rewritten.
    pub uses_rewritten: usize,
    /// Number of dead alias instructions removed.
    pub aliases_removed: usize,
}

#[derive(Clone, Copy)]
enum Propagation {
    Copies,
    Aliases,
}

#[derive(Clone)]
struct Substitution<V> {
    source: V,
    definition: super::ProgramPoint,
}

fn pairs<I: CopySource>(
    instruction: &I,
    propagation: Propagation,
) -> Option<AliasPairs<'_, I::Variable>> {
    let (definitions, uses) = match propagation {
        Propagation::Copies => {
            instruction.as_copy()?;
            (instruction.defs(), instruction.uses())
        }
        Propagation::Aliases => instruction.as_aliases()?,
    };
    (!definitions.is_empty() && definitions.len() == uses.len()).then_some((definitions, uses))
}

/// The provably value-preserving substitutions of the selected transfers,
/// with chains resolved: each admitted `dst → src` satisfies the
/// sole-definition, stable-source, and dominated-uses guards, and
/// soundness composes across links — each link's source is stable at and
/// below its transfer, and dominance is transitive.
fn sound_substitutions<I: CopySource, E>(
    cfg: &Cfg<I, E>,
    propagation: Propagation,
) -> BTreeMap<I::Variable, Substitution<I::Variable>> {
    let mut def_sites: BTreeMap<I::Variable, Vec<super::ProgramPoint>> = BTreeMap::new();
    let mut use_sites: BTreeMap<I::Variable, Vec<super::ProgramPoint>> = BTreeMap::new();
    for block_id in cfg.block_ids() {
        let block = cfg.block(block_id);
        for (inst_idx, inst) in block.instructions().iter().enumerate() {
            let point = super::ProgramPoint {
                block: block_id,
                inst_idx,
            };
            for def in inst.defs() {
                def_sites.entry(def.clone()).or_default().push(point);
            }
            for used in inst.uses() {
                use_sites.entry(used.clone()).or_default().push(point);
            }
        }
    }
    let dom = crate::DominatorTree::compute(cfg);
    let point_dominates = |a: super::ProgramPoint, b: super::ProgramPoint| {
        if a.block == b.block {
            a.inst_idx < b.inst_idx
        } else {
            dom.dominates(a.block, b.block)
        }
    };

    let mut substitutions = BTreeMap::new();
    for block_id in cfg.block_ids() {
        let block = cfg.block(block_id);
        for (inst_idx, inst) in block.instructions().iter().enumerate() {
            let Some((definitions, uses)) = pairs(inst, propagation) else {
                continue;
            };
            if !dom.is_reachable(block_id) {
                continue;
            }
            let alias_point = super::ProgramPoint {
                block: block_id,
                inst_idx,
            };
            for (dst, src) in definitions.iter().cloned().zip(uses.iter().cloned()) {
                if dst == src {
                    continue;
                }
                // The alias must be the sole definition of `dst`.
                if def_sites.get(&dst).is_none_or(|sites| sites.len() != 1) {
                    continue;
                }
                // `src` must hold one stable value wherever `dst` is read.
                match def_sites.get(&src).map(Vec::as_slice) {
                    None | Some([]) => {}
                    Some([site]) if point_dominates(*site, alias_point) => {}
                    Some(_) => continue,
                }
                // Every use of `dst` must see this alias. A same-instruction
                // use reads the pairwise transfer's pre-state and is left
                // untouched during rewriting below.
                let dominated = use_sites.get(&dst).is_none_or(|sites| {
                    sites
                        .iter()
                        .all(|&site| site == alias_point || point_dominates(alias_point, site))
                });
                if !dominated {
                    continue;
                }
                substitutions.insert(
                    dst,
                    Substitution {
                        source: src,
                        definition: alias_point,
                    },
                );
            }
        }
    }

    let targets: Vec<I::Variable> = substitutions.keys().cloned().collect();
    for dst in targets {
        let mut resolved = substitutions[&dst].source.clone();
        let definition = substitutions[&dst].definition;
        let mut seen = alloc::collections::BTreeSet::new();
        while let Some(next) = substitutions.get(&resolved) {
            if !seen.insert(next.source.clone()) {
                break; // cycle guard
            }
            resolved = next.source.clone();
        }
        substitutions.insert(
            dst,
            Substitution {
                source: resolved,
                definition,
            },
        );
    }
    substitutions
}

struct PropagationStats {
    uses_rewritten: usize,
    instructions_removed: usize,
}

fn propagate<I: CopySource + Clone, E>(
    cfg: &mut Cfg<I, E>,
    propagation: Propagation,
    live_out: impl Fn(BlockId) -> Vec<I::Variable>,
) -> PropagationStats {
    let substitutions = sound_substitutions(cfg, propagation);
    if substitutions.is_empty() {
        return PropagationStats {
            uses_rewritten: 0,
            instructions_removed: 0,
        };
    }
    let block_ids: Vec<BlockId> = cfg.block_ids().collect();
    let mut uses_rewritten = 0;
    for &bid in &block_ids {
        for (inst_idx, inst) in cfg.block_mut(bid).instructions_mut().iter_mut().enumerate() {
            let point = super::ProgramPoint {
                block: bid,
                inst_idx,
            };
            for (old, substitution) in &substitutions {
                if substitution.definition != point && inst.uses().contains(old) {
                    inst.rewrite_use(old, &substitution.source);
                    uses_rewritten += 1;
                }
            }
        }
    }

    let chains = DefUseChains::compute(cfg);
    let liveness = Liveness::compute_with_exits(cfg, live_out);
    let mut instructions_removed = 0;
    for &bid in &block_ids {
        let live_after = liveness.live_after_instructions(&*cfg, bid);
        let mut next_idx = 0usize;
        cfg.block_mut(bid).instructions_mut().retain(|inst| {
            let inst_idx = next_idx;
            next_idx += 1;
            let Some((definitions, _)) = pairs(inst, propagation) else {
                return true;
            };
            let def_site = super::ProgramPoint {
                block: bid,
                inst_idx,
            };
            if !chains.uses_of(def_site).is_empty() {
                return true;
            }
            // A definition the caller says leaves the function is read
            // by something this graph does not contain, so the transfer
            // that produces it stays even though nothing here reads it.
            // Its uses elsewhere were rewritten to the source all the
            // same.
            if definitions
                .iter()
                .any(|definition| live_after[inst_idx].contains(definition))
            {
                return true;
            }
            instructions_removed += 1;
            false
        });
    }
    PropagationStats {
        uses_rewritten,
        instructions_removed,
    }
}

/// Run copy propagation on the CFG.
///
/// 1. Build def and use site maps plus the dominator tree.
/// 2. Find copy instructions (`dst = src`) that are **provably
///    value-preserving**: the copy is `dst`'s only definition, it
///    dominates every use of `dst`, and `src` is stable — either never
///    defined in the graph (entry state: a parameter, an environment
///    value) or defined exactly once at a site dominating the copy.
/// 3. Replace the dominated uses of each such `dst` with `src`,
///    resolving copy chains (`a = b; c = a` → uses of `c` read `b`).
/// 4. Remove dead copies (whose defs have no remaining uses).
///
/// Multi-definition variables — reused storage slots, loop-carried
/// values — never propagate: the guards make the pass sound on any
/// well-formed graph, not only single-assignment form, at the cost of
/// leaving such copies in place. Single-assignment input satisfies every
/// guard, so SSA consumers see the previous behavior unchanged.
///
/// Dominance is judged at block granularity, which relies on the
/// standard well-formedness assumption that every use is reached only
/// after its definition executed (verifier-checked bytecode and
/// compiler-produced graphs guarantee this). A path that entered the
/// copy's block but left through a mid-block exceptional exit before the
/// copy cannot reach a use of the destination: the copy is the
/// destination's only definition, so such a use would read an
/// unassigned variable.
///
/// Returns the number of rewrites and removals.
pub fn copy_propagation<I: CopySource + Clone, E>(cfg: &mut Cfg<I, E>) -> CopyPropagationStats {
    copy_propagation_with_exits(cfg, |_| Vec::new())
}

/// [`copy_propagation`] told what leaves each block.
///
/// A graph says nothing about what outlives it, so a function whose
/// result is only ever a copy of one of its inputs loses that result: the
/// copy reads through to the source, nothing inside reads the copy, and
/// the removal takes the only definition of the place a caller reads
/// back. `live_out` states what leaves — a returned value, storage a
/// caller reads again, the state a non-returning exit hands on — and a
/// transfer whose definition is live there is never removed. Its uses
/// inside the function are still rewritten to the source, so the
/// propagation is not otherwise weakened.
///
/// A caller normally answers for the blocks with no successors and hands
/// back an empty vector everywhere else.
///
/// Returns the number of rewrites and removals.
pub fn copy_propagation_with_exits<I: CopySource + Clone, E>(
    cfg: &mut Cfg<I, E>,
    live_out: impl Fn(BlockId) -> Vec<I::Variable>,
) -> CopyPropagationStats {
    let stats = propagate(cfg, Propagation::Copies, live_out);
    CopyPropagationStats {
        uses_rewritten: stats.uses_rewritten,
        copies_removed: stats.instructions_removed,
    }
}

/// Propagates runtime-value aliases, including pairwise refinements.
///
/// This is the presentation-safe counterpart to [`copy_propagation`] for
/// instructions whose definitions retain each corresponding use's runtime
/// value while changing type or other analysis metadata. It applies the same
/// sole-definition, stable-source, and dominance guards, leaves simultaneous
/// pre-state reads untouched, and removes an alias instruction only when none
/// of its definitions remain live.
///
/// Returns the number of rewritten uses and removed alias instructions.
pub fn alias_propagation<I: CopySource + Clone, E>(cfg: &mut Cfg<I, E>) -> AliasPropagationStats {
    let stats = propagate(cfg, Propagation::Aliases, |_| Vec::new());
    AliasPropagationStats {
        uses_rewritten: stats.uses_rewritten,
        aliases_removed: stats.instructions_removed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::test_util::{DfInst, UdInst, df_copy, df_def, df_use, ud_inst};

    /// Whether the mock instruction is a pairwise alias set.
    #[derive(Debug, Clone, Copy)]
    struct IsAlias(bool);

    /// A non-copy instruction exposing pairwise aliases over the shared
    /// uses/defs mock.
    type AliasInst = UdInst<IsAlias>;

    impl CopySource for AliasInst {
        fn as_copy(&self) -> Option<(Self::Variable, Self::Variable)> {
            None
        }

        fn as_aliases(&self) -> Option<AliasPairs<'_, Self::Variable>> {
            self.payload.0.then_some((&self.defs, &self.uses))
        }

        fn rewrite_use(&mut self, old: &Self::Variable, new: &Self::Variable) {
            for used in &mut self.uses {
                if used == old {
                    *used = *new;
                }
            }
        }
    }

    #[test]
    fn simple_copy_propagation() {
        // def r0; copy r1 = r0; use r1 → use r0, remove copy.
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .extend([df_def("def_r0", 0), df_copy("mov", 1, 0)]);
        cfg.block_mut(exit)
            .instructions_mut()
            .push(df_use("use_r1", 1));
        cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);

        let result = copy_propagation(&mut cfg);
        assert_eq!(result.uses_rewritten, 1);
        assert_eq!(result.copies_removed, 1);

        // The use should now reference r0 instead of r1.
        let exit_inst = &cfg.block(exit).instructions()[0];
        assert_eq!(exit_inst.uses[0], 0);
    }

    #[test]
    fn copy_chain_propagation() {
        // def r0; copy r1 = r0; copy r2 = r1; use r2 → use r0.
        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_mut().extend([
            df_def("def_r0", 0),
            df_copy("mov1", 1, 0),
            df_copy("mov2", 2, 1),
            df_use("use_r2", 2),
        ]);

        let result = copy_propagation(&mut cfg);
        assert!(result.uses_rewritten >= 1);
        // The final use should reference r0.
        let insts = cfg.block(cfg.entry()).instructions();
        let last = insts.last().unwrap();
        assert_eq!(last.uses[0], 0);
    }

    #[test]
    fn a_redefined_source_never_propagates() {
        // def r0; copy r1 = r0; def r0; use r1 — the copy captured the
        // FIRST r0, so rewriting the use would read the second.
        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_mut().extend([
            df_def("def_r0", 0),
            df_copy("mov", 1, 0),
            df_def("redef_r0", 0),
            df_use("use_r1", 1),
        ]);

        let result = copy_propagation(&mut cfg);
        assert_eq!(result.uses_rewritten, 0);
        assert_eq!(result.copies_removed, 0);
        let insts = cfg.block(cfg.entry()).instructions();
        assert_eq!(insts.last().unwrap().uses[0], 1);
    }

    #[test]
    fn a_non_dominating_copy_never_propagates() {
        // entry branches; only one arm copies r1 = r0; the merge reads
        // r1 — the copy does not dominate the use.
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let arm = cfg.new_block();
        let other = cfg.new_block();
        let merge = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(df_def("def_r0", 0));
        cfg.block_mut(arm)
            .instructions_mut()
            .push(df_copy("mov", 1, 0));
        cfg.block_mut(merge)
            .instructions_mut()
            .push(df_use("use_r1", 1));
        cfg.add_edge(cfg.entry(), arm, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), other, EdgeKind::ConditionalFalse);
        cfg.add_edge(arm, merge, EdgeKind::Fallthrough);
        cfg.add_edge(other, merge, EdgeKind::Fallthrough);

        let result = copy_propagation(&mut cfg);
        assert_eq!(result.uses_rewritten, 0);
        let insts = cfg.block(merge).instructions();
        assert_eq!(insts[0].uses[0], 1);
    }

    #[test]
    fn an_undefined_source_is_entry_state_and_propagates() {
        // r7 has no definition in the graph (a parameter): the copy's
        // dominated uses may read it directly.
        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .extend([df_copy("mov", 1, 7), df_use("use_r1", 1)]);

        let result = copy_propagation(&mut cfg);
        assert_eq!(result.uses_rewritten, 1);
        assert_eq!(result.copies_removed, 1);
        let insts = cfg.block(cfg.entry()).instructions();
        assert_eq!(insts.last().unwrap().uses[0], 7);
    }

    /// `mov rax, rcx; add rax, rdx; ret` in variable form: the result is
    /// a copy of one of the inputs, one instruction inside reads it, and
    /// the place the caller reads back afterwards is the copy's
    /// destination.
    fn returned_copy_cfg() -> Cfg<DfInst> {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .extend([df_copy("mov", 1, 7), df_use("add", 1)]);
        cfg
    }

    #[test]
    fn an_unseeded_run_removes_the_copy_a_caller_would_read_back() {
        let mut cfg = returned_copy_cfg();
        let result = copy_propagation(&mut cfg);
        assert_eq!(result.uses_rewritten, 1);
        assert_eq!(result.copies_removed, 1);
        assert_eq!(
            cfg.block(cfg.entry()).instructions().len(),
            1,
            "nothing defines the result any more"
        );
    }

    #[test]
    fn a_definition_an_exit_observes_keeps_its_copy() {
        let mut cfg = returned_copy_cfg();
        let result = copy_propagation_with_exits(&mut cfg, |_| alloc::vec![1]);
        assert_eq!(
            result.uses_rewritten, 1,
            "the reader inside still reads through to the source"
        );
        assert_eq!(
            result.copies_removed, 0,
            "the definition the caller reads back stays"
        );
        let instructions = cfg.block(cfg.entry()).instructions();
        assert_eq!(instructions.len(), 2);
        assert_eq!(instructions[0].defs, [1]);
        assert_eq!(instructions[1].uses, [7]);
    }

    #[test]
    fn an_exit_seed_naming_nothing_leaves_the_propagation_alone() {
        let mut seeded = returned_copy_cfg();
        let mut plain = returned_copy_cfg();
        let seeded_result = copy_propagation_with_exits(&mut seeded, |_| Vec::new());
        let plain_result = copy_propagation(&mut plain);
        assert_eq!(seeded_result.uses_rewritten, plain_result.uses_rewritten);
        assert_eq!(seeded_result.copies_removed, plain_result.copies_removed);
    }

    #[test]
    fn pairwise_aliases_propagate_without_runtime_assignments() {
        let mut cfg: Cfg<AliasInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_mut().extend([
            ud_inst(&[7, 8], &[1, 2], IsAlias(true)),
            ud_inst(&[1, 2], &[], IsAlias(false)),
        ]);

        let result = alias_propagation(&mut cfg);

        assert_eq!(result.uses_rewritten, 2);
        assert_eq!(result.aliases_removed, 1);
        let instructions = cfg.block(cfg.entry()).instructions();
        assert_eq!(instructions.len(), 1);
        assert_eq!(instructions[0].uses, [7, 8]);
    }

    #[test]
    fn pairwise_aliases_preserve_same_instruction_pre_state_reads() {
        let mut cfg: Cfg<AliasInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_mut().extend([
            ud_inst(&[0, 1], &[1, 2], IsAlias(true)),
            ud_inst(&[2], &[], IsAlias(false)),
        ]);

        let result = alias_propagation(&mut cfg);

        assert_eq!(result.uses_rewritten, 0);
        assert_eq!(result.aliases_removed, 0);
        let alias = &cfg.block(cfg.entry()).instructions()[0];
        assert_eq!(alias.uses, [0, 1]);
    }
}
