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

/// Declares a dense handle of a toy storage, the way a consumer defines its
/// own.
macro_rules! dense_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub(crate) struct $name(pub(crate) u32);

        impl $crate::graph::view::DenseId for $name {
            fn from_index(index: usize) -> Self {
                Self(u32::try_from(index).unwrap())
            }

            fn index(self) -> usize {
                usize::try_from(self.0).unwrap()
            }
        }
    };
}

mod imp;
mod kit;
mod storage;
mod sym;
mod walks;

pub(crate) use imp::Imp;
pub(crate) use kit::{Kit, Prim, Ty};
pub(crate) use storage::{StmtId, ToyHlil, ToyHlilBuilder};
pub(crate) use sym::Sym;
