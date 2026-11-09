//! Higher-level analysis passes over any
//! [`CfgView`](crate::dataflow::cfg_view::CfgView), which [`Cfg`](crate::Cfg)
//! implements.

pub mod control_tree;
pub mod metrics;
pub mod pattern;
