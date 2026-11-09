//! Consumer model: an instruction-set generator (reify, dasm, elevator) that
//! emits RTL records into its own storage through the `cfglib::ir::rtl`
//! contract.
//!
//! Two unrelated dialects and two unrelated storages stand in for two
//! instruction sets. `kestrel` is a flag-setting register ISA with partial
//! width registers and pooled storage; `tessera` is a wide-slot machine with
//! owned, non-`Copy` vocabulary types and a different handle layout. The
//! tests check that both storages hold and read back the same records, so
//! neither vocabulary is privileged.

use cfglib::ir::rtl::Bits;

/// A bit-vector type whose width the test knows to be valid.
fn bits(width: u16) -> Bits {
    Bits::new(width).expect("the test width is in 1..=Bits::MAX_WIDTH")
}

/// Declares tests whose body is one scenario, generic over a dialect and the
/// storage that holds its functions, and runs it in both dialects.
macro_rules! in_both_dialects {
    ($(fn $name:ident<$dialect:ident, $store:ident>() $body:block)*) => {$(
        #[test]
        fn $name() {
            fn scenario<
                $dialect: crate::model::Target,
                $store: crate::model::Store<$dialect>,
            >() $body
            scenario::<crate::kestrel::Kestrel, crate::kestrel_store::KestrelFunction>();
            scenario::<crate::tessera::Tessera, crate::tessera::TesseraFunction>();
        }
    )*};
}

mod bit_vectors;
mod contract;
mod kestrel;
mod kestrel_store;
mod model;
mod record_roundtrip;
mod tessera;
