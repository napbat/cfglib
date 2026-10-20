//! Running a test on a thread with a small stack.
//!
//! The structurer reads untrusted input, so neither the nesting of a graph
//! nor its size may decide the call stack or make the time quadratic. A test
//! that builds such a graph runs on a thread with the stack of
//! [`SMALL_STACK`], where an overflow aborts the test binary.

extern crate std;
use std::sync::mpsc::{self, RecvError};
use std::thread;

/// The stack of the thread a test runs on: far less than one call frame per
/// level of nesting takes at the depths the tests use.
pub(super) const SMALL_STACK: usize = 256 * 1024;

/// Run `work` on a thread whose stack is [`SMALL_STACK`], and give back what it
/// returns. Panics when `work` panics; a thread that overflows the stack
/// aborts the test binary.
pub(super) fn on_small_stack<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> T {
    let (done, finished) = mpsc::channel();
    let worker = thread::Builder::new()
        .stack_size(SMALL_STACK)
        .spawn(move || {
            let value = work();
            done.send(value)
                .expect("the test waits for the thread until its limit");
        })
        .expect("a thread with a small stack can be spawned");
    match finished.recv() {
        Ok(value) => {
            worker
                .join()
                .expect("a thread that sent its value ended normally");
            value
        }
        Err(RecvError) => std::panic::resume_unwind(
            worker
                .join()
                .expect_err("a thread that sent no value panicked"),
        ),
    }
}
