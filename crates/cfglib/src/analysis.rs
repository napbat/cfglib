//! Higher-level analysis passes over any
//! [`CfgView`](crate::dataflow::cfg_view::CfgView), which [`Cfg`](crate::Cfg)
//! implements.

pub mod alias;
pub mod dead_code;
pub mod expr;
pub mod metrics;
pub mod pattern;
pub mod profile;
pub mod purity;
pub mod switch_table;
pub mod tail_call;
pub mod value_numbering;
