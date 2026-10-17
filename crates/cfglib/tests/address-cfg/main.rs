//! A dasm/elevator-shaped consumer of `build_address_cfg`.
//!
//! A disassembler decodes sized, addressed instructions and hands them to the
//! builder with an exception table; the consumer then reads blocks, edges,
//! regions, unresolved transfers, and call sites back. These tests model that
//! consumer with two toy instruction sets, a stack bytecode over `u32`
//! addresses and a variable-length machine ISA over `u64` addresses above the
//! 32-bit range. Scenarios are written once in program positions and run on
//! both, so every expectation is independent of address width and encoding.
//!
//! Every graph the harness builds is also held to the oracles in
//! `oracles.rs`: `verify`, the address-to-block partition, block starts
//! against the documented leader rules, edge payloads, unwind edges, region
//! registration, dominators by node removal, and control-tree coverage.

/// Runs a scenario function generic over [`isa::Isa`] once per toy ISA, as
/// one test named after the scenario inside a module named after the ISA's
/// address width.
macro_rules! over_both_isas {
    ($($scenario:ident),+ $(,)?) => {
        $(
            mod $scenario {
                #[test]
                fn u32_bytecode() {
                    super::$scenario::<crate::bytecode::Isa32>();
                }

                #[test]
                fn u64_machine() {
                    super::$scenario::<crate::machine::Isa64>();
                }
            }
        )+
    };
}

mod bytecode;
mod calls;
mod consumer;
mod errors;
mod flows;
mod handlers;
mod harness;
mod isa;
mod leaders;
mod machine;
mod oracles;
mod placement;
mod random_programs;
mod regions;
mod transfers;
