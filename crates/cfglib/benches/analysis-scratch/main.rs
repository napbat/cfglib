//! What symtree's per-callable flow build allocates, phase by phase.
//!
//! The build runs four analyses once per callable — the dominator tree,
//! ordinary SSA, memory SSA (which constructs a shadow CFG and its own
//! dominators and SSA inside itself), and memory-carried value flow — and on a
//! whole-codebase corpus they dominate its allocation traffic, because every
//! call allocates all of its working storage from scratch. This benchmark
//! reports what one call costs at three procedure sizes, and, for the
//! scratch-taking entry points, what one call costs when the working storage
//! is already warm.
//!
//! The intermediate phases are here because the totals alone do not say where
//! the traffic is: reverse postorder is measured under the dominator tree,
//! dominance frontiers and phi placement under SSA, and the memory trace under
//! memory SSA.
//!
//! ```powershell
//! cargo bench -p cfglib --bench analysis-scratch
//! $env:RUSTFLAGS = "--cfg cfglib_bench_alloc"; cargo bench -p cfglib --bench analysis-scratch
//! ```

mod allocation;
mod fixtures;

use cfglib::{
    Cfg, DominanceFrontiers, DominatorScratch, DominatorTree, ExactMemoryAlias, MemorySSA,
    MemoryTrace, MemoryValueFlow, PhiPlacements, SsaForm, SsaScratch, TraversalDirection,
    reverse_postorder,
};

use allocation::case;
use fixtures::{FlowInst, LARGE_REGIONS, MEDIUM_REGIONS, SMALL_REGIONS, Slot, diamond_chain};

/// One analyzed procedure, with the inputs each phase is measured against
/// already built, so a case measures the phase and not its setup.
struct Subject {
    label: &'static str,
    cfg: Cfg<FlowInst>,
}

impl Subject {
    fn new(label: &'static str, regions: usize) -> Self {
        Self {
            label,
            cfg: diamond_chain(regions),
        }
    }

    fn name(&self, phase: &str) -> String {
        format!("{phase}/{}", self.label)
    }
}

/// One case list, filtered by the benchmark's single optional argument.
struct Cases<'filter> {
    filter: &'filter str,
}

impl Cases<'_> {
    fn run<T>(&self, subject: &Subject, phase: &str, operation: impl FnMut() -> T) {
        let name = subject.name(phase);
        if self.filter.is_empty() || name.contains(self.filter) {
            case(&name, operation);
        }
    }

    fn analyses(&self, subject: &Subject) {
        let cfg = &subject.cfg;
        let dominators = DominatorTree::compute(cfg);
        let ssa = SsaForm::compute(cfg, &dominators);
        let memory: MemorySSA<Slot, u32, ()> = MemorySSA::compute(cfg, &ExactMemoryAlias);

        self.run(subject, "reverse-postorder", || {
            reverse_postorder(cfg, cfg.entry(), TraversalDirection::Outgoing)
        });
        self.run(subject, "dominator-tree", || DominatorTree::compute(cfg));
        let mut dominator_scratch = DominatorScratch::new();
        self.run(subject, "dominator-tree-in", || {
            DominatorTree::compute_in(&mut dominator_scratch, cfg)
        });
        self.run(subject, "dominance-frontiers", || {
            DominanceFrontiers::compute(cfg, &dominators)
        });
        self.run(subject, "phi-placements", || {
            PhiPlacements::compute(cfg, &dominators)
        });
        self.run(subject, "ssa-form", || SsaForm::compute(cfg, &dominators));
        let mut ssa_scratch = SsaScratch::new();
        self.run(subject, "ssa-form-in", || {
            SsaForm::compute_in(&mut ssa_scratch, cfg, &dominators)
        });
        self.run(subject, "memory-trace", || {
            MemoryTrace::<Slot, u32, ()>::compute(cfg)
        });
        self.run(subject, "memory-ssa", || {
            MemorySSA::<Slot, u32, ()>::compute(cfg, &ExactMemoryAlias)
        });
        self.run(subject, "memory-value-flow", || {
            MemoryValueFlow::compute(&memory, &ssa).expect("fixture events match their SSA form")
        });
    }
}

fn main() {
    // Cargo passes `--bench` to a custom benchmark binary. Only a positional
    // argument is the optional case-name filter.
    let filter = std::env::args()
        .skip(1)
        .find(|argument| !argument.starts_with('-'))
        .unwrap_or_default();
    let subjects = [
        Subject::new("small", SMALL_REGIONS),
        Subject::new("medium", MEDIUM_REGIONS),
        Subject::new("large", LARGE_REGIONS),
    ];
    let cases = Cases { filter: &filter };
    for subject in &subjects {
        println!("{} blocks in {}", subject.cfg.block_count(), subject.label);
        cases.analyses(subject);
    }
}
