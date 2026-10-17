//! Bare uses/defs mock instruction with a test-specific payload.

extern crate alloc;

use alloc::vec::Vec;

use crate::InstrInfo;

/// A mock instruction carrying only uses, defs, and a test-specific
/// payload `P`. Test modules attach their own semantics by implementing
/// the trait under test for `UdInst<TheirPayload>` — distinct payload
/// types keep those impls from colliding.
#[derive(Debug, Clone)]
pub struct UdInst<P> {
    /// Variables read by this instruction.
    pub uses: Vec<u16>,
    /// Variables written by this instruction.
    pub defs: Vec<u16>,
    /// Test-specific semantics.
    pub payload: P,
}

impl<P> InstrInfo for UdInst<P> {
    type Variable = u16;

    fn uses(&self) -> &[u16] {
        &self.uses
    }
    fn defs(&self) -> &[u16] {
        &self.defs
    }
}

/// Create a [`UdInst`] from use/def slices and a payload.
pub fn ud_inst<P>(uses: &[u16], defs: &[u16], payload: P) -> UdInst<P> {
    UdInst {
        uses: uses.to_vec(),
        defs: defs.to_vec(),
        payload,
    }
}
