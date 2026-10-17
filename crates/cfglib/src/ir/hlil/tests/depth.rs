//! Statement trees nested as deep as untrusted input can make them, and the
//! small stack that a walk over them must fit on.
//!
//! Every algorithm over statements keeps its own stack, so the nesting depth
//! of a function costs no call frames. A walk under test runs on a thread
//! whose stack could not hold a few hundred recursive frames, so a walk that
//! spent one frame on each level of a nest would overflow it.

extern crate alloc;
extern crate std;

use alloc::format;
use std::thread;

use crate::ir::hlil::{FunctionBuilder, Handler, HandlerKind, SwitchArm};

use super::{Kit, Prim, StmtId, ToyHlil, ToyHlilBuilder};

/// The stack of the thread that runs a walk under test.
const STACK_BYTES: usize = 256 * 1024;

/// Runs `walk` on a thread with a small stack and returns what it returns. A
/// panic of the walk is a panic of the caller.
pub(crate) fn on_small_stack<T: Send>(walk: impl FnOnce() -> T + Send) -> T {
    thread::scope(|scope| {
        thread::Builder::new()
            .stack_size(STACK_BYTES)
            .spawn_scoped(scope, walk)
            .expect("a thread spawns")
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    })
}

/// A compound statement that nests another.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Wrap {
    If,
    While,
    DoWhile,
    Loop,
    For,
    Switch,
    Try,
    Labeled,
    Region,
}

/// Every compound statement. A `Labeled` follows a `While`, so a labeled loop
/// is among them.
pub(crate) const EVERY_WRAP: [Wrap; 9] = [
    Wrap::If,
    Wrap::While,
    Wrap::Labeled,
    Wrap::DoWhile,
    Wrap::Loop,
    Wrap::For,
    Wrap::Switch,
    Wrap::Try,
    Wrap::Region,
];

/// Adds `depth` statements, each wrapping the one before it, around `core`,
/// and returns the outermost. The wraps cycle through `wraps`, innermost
/// first.
pub(crate) fn nest<D: Kit>(
    builder: &mut ToyHlilBuilder<D>,
    wraps: &[Wrap],
    depth: usize,
    core: StmtId,
) -> StmtId {
    (0..depth).fold(core, |inner, level| {
        wrap(builder, wraps[level % wraps.len()], level, inner)
    })
}

/// A function whose body is `if (c) { nest } return;`, where the nest is
/// `depth` statements around the statement that `core` adds.
///
/// The function is sealed unchecked: the builder's own `verify` is one of the
/// walks that the tests run.
pub(crate) fn nested<D: Kit>(
    wraps: &[Wrap],
    depth: usize,
    core: impl FnOnce(&mut ToyHlilBuilder<D>) -> StmtId,
) -> ToyHlil<D> {
    let mut builder = ToyHlilBuilder::<D>::new();
    let core = core(&mut builder);
    let inner = nest(&mut builder, wraps, depth, core);
    let guard = builder.condition();
    let outer = builder.if_(guard, &[inner], &[]);
    let done = builder.return_(&[]);
    builder.finish_unchecked(&[outer, done])
}

fn wrap<D: Kit>(
    builder: &mut ToyHlilBuilder<D>,
    kind: Wrap,
    level: usize,
    inner: StmtId,
) -> StmtId {
    match kind {
        Wrap::If => {
            let condition = builder.condition();
            let otherwise = builder.noop();
            builder.if_(condition, &[inner], &[otherwise])
        }
        Wrap::While => {
            let condition = builder.condition();
            builder.while_(condition, &[inner])
        }
        Wrap::DoWhile => {
            let condition = builder.condition();
            builder.do_while(&[inner], condition)
        }
        Wrap::Loop => {
            // The exit keeps what follows the loop reachable.
            let condition = builder.condition();
            let leave = builder.break_(None);
            let exit = builder.if_(condition, &[leave], &[]);
            builder.loop_(&[exit, inner])
        }
        Wrap::For => {
            let initializer = builder.noop();
            let condition = builder.condition();
            let update = builder.noop();
            builder.for_(&[initializer], Some(condition), &[update], &[inner])
        }
        Wrap::Switch => {
            let scrutinee = builder.number(0);
            let values = [D::constant(1)];
            let fallback = builder.noop();
            builder.switch(
                scrutinee,
                &[SwitchArm {
                    values: &values,
                    body: &[inner],
                }],
                &[fallback],
            )
        }
        Wrap::Try => {
            let recovery = builder.noop();
            let cleanup = builder.noop();
            let handler = Handler {
                kind: HandlerKind::CatchAll,
                binding: None,
                caught_types: &[],
                body: &[recovery],
            };
            builder.try_(&[inner], &[handler], &[cleanup])
        }
        Wrap::Labeled => builder.labeled(&format!("l{level}"), &[inner]),
        Wrap::Region => builder.region(&D::op(Prim::Acquire), &[], &[inner]),
    }
}
