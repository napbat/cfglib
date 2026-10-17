//! Control-flow shapes that the facts are checked on.
//!
//! The root names follow the neutral numbering: `R0..R3` are `0..=3`, the
//! flags `ZF..VF` are `4..=7`, `WIDE` is `8`, and `CSR` is `9`.

use crate::model::{
    Access::{
        Noise, RaiseBinding, Read, ReadIndexed, ReadSparse, WriteFull, WriteIndexed, WritePartial,
        WriteSparse,
    },
    Ending, Inst, Model, block, call, inst, removed,
};

pub const R0: u8 = 0;
pub const R1: u8 = 1;
pub const R2: u8 = 2;
pub const R3: u8 = 3;
pub const ZF: u8 = 4;
pub const CF: u8 = 6;
pub const VF: u8 = 7;
pub const WIDE: u8 = 8;
pub const CSR: u8 = 9;

/// A record that raises unconditionally.
fn trap(accesses: Vec<crate::model::Access>) -> Inst {
    Inst {
        accesses,
        ending: Ending::Trap,
    }
}

/// Two arms that meet. The left arm rewrites `R0`, the right arm is an empty
/// block, and the join reads both roots.
pub fn diamond() -> Model {
    Model::new(vec![
        block(vec![inst(vec![WriteFull(R0), WriteFull(R1)])], &[1, 2]),
        block(vec![inst(vec![WriteFull(R0)])], &[3]),
        block(vec![], &[3]),
        block(vec![inst(vec![Read(R0), Read(R1)])], &[]),
    ])
}

/// A loop with two exits, a block without instructions, a call, a removed
/// slot whose stale records name roots nothing else names, and an
/// unreachable block that flows into a reachable one.
///
/// Block 8 is unreachable. Slot 6 was removed: `VF` and `WIDE` appear only in
/// its stale records.
pub fn loop_with_exits() -> Model {
    Model::new(vec![
        block(
            vec![
                inst(vec![WriteFull(R0), WriteFull(R3)]),
                inst(vec![WriteFull(ZF)]),
            ],
            &[1],
        ),
        block(
            vec![inst(vec![Read(R0), ReadIndexed(vec![R1, R2])])],
            &[2, 5],
        ),
        block(vec![inst(vec![WriteFull(R1), Read(R0)])], &[3, 4]),
        block(
            vec![inst(vec![WritePartial(R0)]), call(vec![WriteSparse(CSR)])],
            &[7],
        ),
        block(vec![], &[1]),
        block(vec![inst(vec![Read(R1), Read(ZF), RaiseBinding(CF)])], &[]),
        removed(vec![inst(vec![Read(VF), WriteFull(WIDE)])]),
        block(vec![inst(vec![Read(R3)])], &[]),
        block(vec![inst(vec![WriteFull(R2), Read(R3)])], &[5]),
    ])
}

/// The entry block is also the target of a back edge, so it is in its own
/// dominance frontier.
pub fn entry_loop() -> Model {
    Model::new(vec![
        block(vec![inst(vec![Read(R0), WriteFull(R0)])], &[0, 1]),
        block(vec![inst(vec![Read(R0)])], &[]),
    ])
}

/// A cycle that can be entered at either of two blocks, so no block of the
/// cycle dominates the other.
pub fn irreducible() -> Model {
    Model::new(vec![
        block(vec![inst(vec![WriteFull(R0)])], &[1, 2]),
        block(vec![inst(vec![Read(R1), WriteFull(R1)])], &[2, 3]),
        block(vec![inst(vec![Read(R0), WriteFull(R2)])], &[1, 3]),
        block(vec![inst(vec![Read(R1), Read(R2)])], &[]),
    ])
}

/// Two nested loops with an exit from the outer header and a second exit
/// after the inner loop.
pub fn nested_loops() -> Model {
    Model::new(vec![
        block(vec![inst(vec![WriteFull(R0)])], &[1]),
        block(vec![inst(vec![Read(R0)])], &[2, 6]),
        block(vec![inst(vec![Read(R1)])], &[3, 4]),
        block(vec![inst(vec![WriteFull(R1)])], &[2]),
        block(vec![inst(vec![WriteFull(R0)])], &[1, 5]),
        block(vec![inst(vec![Read(R1)])], &[]),
        block(vec![inst(vec![Read(R0)])], &[]),
    ])
}

/// A block that branches back to itself.
pub fn self_loop() -> Model {
    Model::new(vec![
        block(vec![inst(vec![WriteFull(R1)])], &[1]),
        block(vec![inst(vec![Read(R0), WriteFull(R0)])], &[1, 2]),
        block(vec![inst(vec![Read(R0), Read(R1)])], &[]),
    ])
}

/// A four-way dispatch that joins again.
pub fn dispatch_fan() -> Model {
    Model::new(vec![
        block(vec![inst(vec![WriteFull(R0), WriteFull(R1)])], &[1, 2, 3]),
        block(vec![inst(vec![WriteFull(R0)])], &[4]),
        block(vec![inst(vec![WriteFull(R1)])], &[4]),
        block(vec![], &[4]),
        block(vec![inst(vec![Read(R0), Read(R1)])], &[]),
    ])
}

/// A dispatch whose cases 0 and 1 both go to block 1, so the entry has two
/// parallel edges to it.
pub fn parallel_dispatch() -> Model {
    Model::new(vec![
        block(vec![inst(vec![WriteFull(R0), WriteFull(R1)])], &[1, 1, 2]),
        block(vec![inst(vec![WriteFull(R0)])], &[3]),
        block(vec![], &[3]),
        block(vec![inst(vec![Read(R0), Read(R1)])], &[]),
    ])
}

/// The entry block holds no instruction and has no successor.
pub fn empty_entry() -> Model {
    Model::new(vec![block(vec![], &[])])
}

/// Everything after the entry was removed, and a live block that is left
/// reads a root that only the entry writes.
pub fn cut_off_block() -> Model {
    Model::new(vec![
        block(vec![inst(vec![WriteFull(R0)])], &[]),
        removed(vec![inst(vec![WriteFull(R0)])]),
        block(vec![inst(vec![Read(R0), ReadSparse(CSR)])], &[]),
    ])
}

/// Every record shape that a block can end with, including a trap.
pub fn endings() -> Model {
    Model::new(vec![
        block(vec![inst(vec![Noise, WriteFull(R0)])], &[1, 2]),
        block(vec![inst(vec![WriteIndexed(vec![R0, R1, R2])])], &[3]),
        block(vec![trap(vec![Read(R1), RaiseBinding(R2)])], &[]),
        block(vec![inst(vec![Read(R0)])], &[]),
    ])
}

/// Every named fixture, for the table-driven checks.
pub fn all() -> Vec<(&'static str, Model)> {
    vec![
        ("diamond", diamond()),
        ("loop_with_exits", loop_with_exits()),
        ("entry_loop", entry_loop()),
        ("irreducible", irreducible()),
        ("nested_loops", nested_loops()),
        ("self_loop", self_loop()),
        ("dispatch_fan", dispatch_fan()),
        ("parallel_dispatch", parallel_dispatch()),
        ("empty_entry", empty_entry()),
        ("cut_off_block", cut_off_block()),
        ("endings", endings()),
    ]
}
