//! `PassPipeline` composing real transforms over a CFG target: ordering,
//! change reporting, failure identity, stateful passes, and payload-bearing
//! CFGs.

use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::convert::Infallible;

use cfglib::{
    BlockId, Cfg, Pass, PassChange, PassId, PassPipeline, SemanticValidator, dead_code_elimination,
    merge_blocks, pass_fn, remove_dead_code, remove_empty_blocks, remove_unreachable, simplify,
    split_critical_edges, split_critical_edges_mapped, verify_with,
};

use crate::oracles::{assert_surviving_payloads_unchanged, assert_valid, critical_edges};
use crate::programs::random_program;
use crate::toy::{Inst, Kind, Payload, Route, assert_same_behavior_observing};

#[derive(Debug, Clone, PartialEq, Eq)]
enum Rejected {
    Invalid(usize),
    Refused(&'static str),
}

impl std::fmt::Display for Rejected {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(count) => write!(formatter, "{count} structural errors"),
            Self::Refused(reason) => write!(formatter, "refused: {reason}"),
        }
    }
}

impl std::error::Error for Rejected {}

type Log = RefCell<Vec<(&'static str, bool)>>;

/// A transform as a pass. It also holds the transform to its own contract: it
/// reports a positive count exactly when the graph is no longer equal to the
/// snapshot taken before it ran, and records what it reported.
fn transform<'a, E: Payload + 'a>(
    name: &'static str,
    log: &'a Log,
    apply: impl Fn(&mut Cfg<Inst, E>) -> usize + 'a,
) -> impl Pass<Cfg<Inst, E>, Error = Rejected> + 'a {
    pass_fn(PassId::new(name), move |cfg: &mut Cfg<Inst, E>| {
        let before = cfg.clone();
        let count = apply(cfg);
        assert_eq!(
            count > 0,
            *cfg != before,
            "{name}: the count disagrees with what changed"
        );
        log.borrow_mut().push((name, count > 0));
        Ok(PassChange::from_changed(count > 0))
    })
}

/// The consumer's own rule: a block that ends in a branch has two exits.
struct TwoWayBranches;

impl<E> SemanticValidator<Inst, E> for TwoWayBranches {
    type Error = BlockId;

    fn validate_block(&self, cfg: &Cfg<Inst, E>, block: BlockId, errors: &mut Vec<BlockId>) {
        let branches = cfg
            .block(block)
            .instructions()
            .last()
            .is_some_and(|inst| inst.kind == Kind::Branch);
        if branches && cfg.outgoing(block).count() != 2 {
            errors.push(block);
        }
    }
}

fn verified<'a, E: 'a>(log: &'a Log) -> impl Pass<Cfg<Inst, E>, Error = Rejected> + 'a {
    pass_fn(
        PassId::new("check.verify"),
        move |cfg: &mut Cfg<Inst, E>| {
            let report = verify_with(cfg, &TwoWayBranches);
            log.borrow_mut().push(("check.verify", false));
            if report.is_ok() {
                Ok(PassChange::Unchanged)
            } else {
                Err(Rejected::Invalid(report.error_count()))
            }
        },
    )
}

#[test]
fn a_cleanup_pipeline_runs_in_schedule_order_and_reports_each_pass_honestly() {
    for seed in 0..60 {
        let before = random_program::<()>(seed, true);
        let mut cfg = before.clone();
        let log = Log::default();
        let mut pipeline = PassPipeline::<Cfg<Inst>, Rejected>::new();
        pipeline.push(transform("cleanup.remove-unreachable", &log, |cfg| {
            remove_unreachable(cfg)
        }));
        pipeline.push(transform("cleanup.remove-empty", &log, |cfg| {
            remove_empty_blocks(cfg)
        }));
        pipeline.push(transform("cleanup.merge", &log, merge_blocks));
        pipeline.push(transform("dce", &log, dead_code_elimination));
        pipeline.push(transform("ssa.split-critical", &log, |cfg| {
            split_critical_edges(cfg)
        }));
        pipeline.push(verified(&log));

        let scheduled: Vec<&str> = pipeline.ids().map(PassId::as_str).collect();
        let report = pipeline
            .run(&mut cfg)
            .expect("every pass succeeds on a valid program");

        let recorded = log.borrow();
        assert_eq!(scheduled.len(), pipeline.len());
        assert_eq!(
            report
                .executions()
                .iter()
                .map(|execution| execution.id().as_str())
                .collect::<Vec<_>>(),
            scheduled,
            "seed {seed}: executions follow the schedule"
        );
        assert_eq!(
            recorded.iter().map(|&(name, _)| name).collect::<Vec<_>>(),
            scheduled
        );
        for (execution, &(_, changed)) in report.executions().iter().zip(recorded.iter()) {
            assert_eq!(
                execution.change(),
                PassChange::from_changed(changed),
                "seed {seed}"
            );
        }
        let changes = recorded.iter().filter(|&&(_, changed)| changed).count();
        assert_eq!(report.changed_count(), changes);
        assert_eq!(report.is_changed(), changes > 0);
        assert_eq!(report.len(), 6);
        assert!(
            critical_edges(&cfg).is_empty(),
            "seed {seed}: the last structural pass was the split"
        );
        assert_valid(&cfg, &format!("seed {seed}"));
        assert_same_behavior_observing(&before, &cfg, &[], &format!("seed {seed}"));
    }
}

#[test]
fn the_schedule_decides_whether_critical_edges_survive() {
    let seed = (0..200)
        .find(|&seed| !critical_edges(&random_program::<()>(seed, false)).is_empty())
        .expect("some program has a critical edge");
    let before = random_program::<()>(seed, false);
    let log = Log::default();

    let mut split_last = before.clone();
    let mut pipeline = PassPipeline::<Cfg<Inst>, Rejected>::new();
    pipeline.push(transform("cleanup.simplify", &log, simplify));
    pipeline.push(transform("ssa.split-critical", &log, |cfg| {
        split_critical_edges(cfg)
    }));
    assert_eq!(pipeline.run(&mut split_last).unwrap().len(), 2);

    let mut split_first = before.clone();
    let mut pipeline = PassPipeline::<Cfg<Inst>, Rejected>::new();
    pipeline.push(transform("ssa.split-critical", &log, |cfg| {
        split_critical_edges(cfg)
    }));
    pipeline.push(transform("cleanup.simplify", &log, simplify));
    assert_eq!(pipeline.run(&mut split_first).unwrap().len(), 2);

    assert_eq!(critical_edges(&split_last), []);
    assert!(
        !critical_edges(&split_first).is_empty(),
        "simplify bypasses the empty blocks the split inserted"
    );
    assert_same_behavior_observing(&before, &split_last, &[], "split last");
    assert_same_behavior_observing(&before, &split_first, &[], "split first");
}

#[test]
fn a_failing_pass_stops_the_pipeline_and_the_failure_names_it() {
    let before = random_program::<()>(5, true);
    let mut cfg = before.clone();
    let log = Log::default();
    let mut pipeline = PassPipeline::<Cfg<Inst>, Rejected>::new();
    pipeline.push(transform("cleanup.simplify", &log, simplify));
    pipeline.push(pass_fn(
        PassId::new("test.reject"),
        |cfg: &mut Cfg<Inst>| {
            cfg.new_block();
            Err::<PassChange, _>(Rejected::Refused("no budget"))
        },
    ));
    pipeline.push(transform("never.runs", &log, simplify));

    let failure = pipeline.run(&mut cfg).expect_err("the second pass fails");

    assert_eq!(failure.pass(), PassId::new("test.reject"));
    assert_eq!(failure.error(), &Rejected::Refused("no budget"));
    assert_eq!(failure.completed().len(), 1, "only the prefix is reported");
    assert_eq!(
        failure.completed().executions()[0].id().as_str(),
        "cleanup.simplify"
    );
    assert_eq!(
        log.borrow().len(),
        1,
        "the pass after the failure never ran"
    );
    assert!(
        cfg.block_ids()
            .any(|block| block.index() >= before.block_bound()),
        "the failing pass's mutation stays visible: nothing is rolled back"
    );
    let text = failure.to_string();
    assert!(
        text.contains("test.reject") && text.contains("no budget"),
        "{text}"
    );

    let (pass, error, completed) = failure.into_parts();
    assert_eq!(
        (pass.as_str(), error),
        ("test.reject", Rejected::Refused("no budget"))
    );
    assert_eq!(completed.len(), 1);
}

#[test]
fn a_pass_failing_first_leaves_an_empty_completed_report_and_converts_to_a_boxed_error() {
    fn run_all(cfg: &mut Cfg<Inst>) -> Result<(), Box<dyn std::error::Error>> {
        let mut pipeline = PassPipeline::<Cfg<Inst>, Rejected>::new();
        pipeline.push(pass_fn(PassId::new("check.first"), |_: &mut Cfg<Inst>| {
            Err::<PassChange, _>(Rejected::Invalid(3))
        }));
        let report = pipeline.run(cfg)?;
        assert!(report.is_empty(), "the only pass failed");
        Ok(())
    }
    let mut cfg = random_program::<()>(1, false);
    let error = run_all(&mut cfg).expect_err("the only pass fails");
    assert!(error.to_string().contains("check.first"));
    assert!(
        error
            .source()
            .is_some_and(|source| source.to_string().contains("3 structural"))
    );
}

#[test]
fn a_verifying_pass_turns_a_violated_consumer_rule_into_a_failure() {
    let mut cfg = random_program::<()>(2, false);
    let log = Log::default();
    let mut pipeline = PassPipeline::<Cfg<Inst>, Rejected>::new();
    pipeline.push(pass_fn(PassId::new("corrupt"), |cfg: &mut Cfg<Inst>| {
        let branch = cfg
            .block_ids()
            .find(|&block| {
                cfg.block(block)
                    .instructions()
                    .last()
                    .is_some_and(|inst| inst.kind == Kind::Branch)
            })
            .expect("the program branches");
        let edge = cfg.outgoing(branch).next().unwrap();
        cfg.remove_edge(edge);
        Ok::<_, Rejected>(PassChange::Changed)
    }));
    pipeline.push(verified(&log));

    let failure = pipeline
        .run(&mut cfg)
        .expect_err("a one-way branch violates the rule");

    assert_eq!(failure.pass().as_str(), "check.verify");
    assert_eq!(failure.error(), &Rejected::Invalid(1));
    assert_eq!(failure.completed().changed_count(), 1);
}

#[test]
fn repeating_an_identity_expresses_a_fixpoint_schedule_that_ends_unchanged() {
    let mut converged = 0;
    for seed in 0..40 {
        let before = random_program::<()>(seed, true);
        let log = Log::default();
        let mut cfg = before.clone();
        let mut pipeline = PassPipeline::<Cfg<Inst>, Rejected>::new();
        for _ in 0..6 {
            pipeline.push(transform("dce", &log, dead_code_elimination));
            pipeline.push(transform("simplify", &log, simplify));
        }
        let ids: Vec<&str> = pipeline.ids().map(PassId::as_str).collect();
        assert_eq!(
            ids[..4],
            ["dce", "simplify", "dce", "simplify"],
            "duplicates are kept in order"
        );
        let report = pipeline.run(&mut cfg).unwrap();

        let tail = &report.executions()[report.len() - 2..];
        if tail
            .iter()
            .all(|execution| !execution.change().is_changed())
        {
            converged += 1;
            let mut expected = before.clone();
            remove_dead_code(&mut expected);
            assert_eq!(
                cfg, expected,
                "seed {seed}: same operations in the same order"
            );
        }
        assert_same_behavior_observing(&before, &cfg, &[], &format!("seed {seed}"));
    }
    assert!(
        converged > 30,
        "only {converged} schedules reached a fixpoint"
    );
}

struct CountingSimplify<'a> {
    blocks_removed: &'a Cell<usize>,
    runs: &'a Cell<usize>,
}

impl Pass<Cfg<Inst>> for CountingSimplify<'_> {
    type Error = Infallible;

    fn id(&self) -> PassId {
        PassId::new("stateful.simplify")
    }

    fn run(&mut self, cfg: &mut Cfg<Inst>) -> Result<PassChange, Infallible> {
        let removed = simplify(cfg);
        self.blocks_removed.set(self.blocks_removed.get() + removed);
        self.runs.set(self.runs.get() + 1);
        Ok(PassChange::from_changed(removed > 0))
    }
}

#[test]
fn passes_keep_their_own_state_and_borrowed_context_across_runs() {
    let mut cfg = random_program::<()>(11, true);
    let mut observed_sizes = Vec::new();
    let (blocks_removed, runs) = (Cell::new(0), Cell::new(0));
    {
        let mut pipeline = PassPipeline::<Cfg<Inst>, Infallible>::new().with_pass(pass_fn(
            PassId::new("observe.size"),
            |cfg: &mut Cfg<Inst>| {
                observed_sizes.push(cfg.block_count());
                Ok(PassChange::Unchanged)
            },
        ));
        pipeline.push(CountingSimplify {
            blocks_removed: &blocks_removed,
            runs: &runs,
        });
        let first = pipeline.run_infallible(&mut cfg);
        let second = pipeline.run_infallible(&mut cfg);

        assert!(first.is_changed());
        assert!(!second.is_changed(), "the second run finds nothing left");
        assert_eq!(second.changed_count(), 0);
    }
    assert_eq!(runs.get(), 2);
    assert!(blocks_removed.get() > 0);
    assert_eq!(observed_sizes.len(), 2);
    assert_eq!(
        observed_sizes[1],
        cfg.block_count(),
        "the second run saw the cleaned graph"
    );
}

#[test]
fn an_empty_schedule_succeeds_with_an_empty_report() {
    let mut cfg = random_program::<()>(0, false);
    let before = cfg.clone();
    let mut pipeline = PassPipeline::<Cfg<Inst>, Infallible>::default();
    assert!(pipeline.is_empty());
    assert_eq!(pipeline.len(), 0);
    let report = pipeline.run_infallible(&mut cfg);
    assert!(report.is_empty());
    assert_eq!(
        (report.len(), report.changed_count(), report.is_changed()),
        (0, 0, false)
    );
    assert_eq!(cfg, before);
}

#[test]
fn pass_identities_and_changes_have_the_documented_conversions() {
    assert_eq!(PassChange::default(), PassChange::Unchanged);
    assert_eq!(PassChange::from_changed(true), PassChange::Changed);
    assert_eq!(PassChange::from_changed(false), PassChange::Unchanged);
    assert!(PassChange::Changed.is_changed() && !PassChange::Unchanged.is_changed());

    let first = PassId::new("java.recover-loops");
    assert_eq!(first.as_str(), "java.recover-loops");
    assert_eq!(first.to_string(), "java.recover-loops");
    assert_eq!(first, PassId::new("java.recover-loops"));
    assert!(first < PassId::new("rust.recover-loops"));
    let distinct: BTreeSet<PassId> = [first, first, PassId::new("other")].into();
    assert_eq!(distinct.len(), 2);
}

fn payload_pipeline(log: &Log) -> PassPipeline<'_, Cfg<Inst, Route>, Rejected> {
    let mut pipeline = PassPipeline::new();
    pipeline.push(transform("dce", log, dead_code_elimination));
    pipeline.push(transform("cleanup.simplify", log, simplify));
    pipeline.push(transform("ssa.split-critical", log, |cfg| {
        split_critical_edges_mapped(cfg).0
    }));
    pipeline.push(verified(log));
    pipeline
}

#[test]
fn the_same_pipeline_runs_over_a_payload_bearing_cfg_and_keeps_every_edge_payload() {
    for seed in 0..40 {
        let before = random_program::<Route>(seed, true);
        let mut cfg = before.clone();
        let log = Log::default();
        let report = payload_pipeline(&log)
            .run(&mut cfg)
            .expect("valid at every step");
        assert_eq!(report.len(), 4, "seed {seed}");

        assert_surviving_payloads_unchanged(&before, &cfg, &format!("seed {seed}"));
        assert_eq!(critical_edges(&cfg), [], "seed {seed}");
        assert_same_behavior_observing(&before, &cfg, &[], &format!("seed {seed}"));
    }
}
