//! Tests of the structured contract over two toy dialects.
//!
//! [`Imp`] and [`Sym`] share nothing but the traits, and every test body that
//! is written over [`Kit`] runs over both. The storage behind them is
//! [`ToyHlil`], a `Vec`-backed function that implements
//! [`FunctionView`](super::FunctionView).

/// Generates one test per toy dialect for each function named, a body
/// written over [`Kit`](crate::ir::hlil::tests::Kit).
macro_rules! over_toys {
    ($($name:ident),+ $(,)?) => {
        $(
            mod $name {
                #[test]
                fn imp() {
                    super::$name::<$crate::ir::hlil::tests::Imp>();
                }

                #[test]
                fn sym() {
                    super::$name::<$crate::ir::hlil::tests::Sym>();
                }
            }
        )+
    };
}

pub(crate) use over_toys;

mod fixtures;
mod flat;
mod flat_storage;
mod imp;
mod kit;
mod lowering;
mod recovery;
mod rendering;
mod sketch;
mod sketch_build;
mod storage;
mod sym;
mod verification;
mod walks;

pub(crate) use imp::Imp;
pub(crate) use kit::{Kit, Prim, Ty};
pub(crate) use storage::{ExprId, StmtId, ToyHlil, ToyHlilBuilder, VarId};
pub(crate) use sym::Sym;
