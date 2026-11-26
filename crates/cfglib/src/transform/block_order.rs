//! Block orders computed from facts about the graph, to hand to
//! [`linearize`](crate::linearize) as
//! [`BlockOrder::Custom`](crate::BlockOrder::Custom).
//!
//! [`BlockOrder`](crate::BlockOrder) names the orders that need nothing but
//! the graph. The orders here also read an analysis or caller data:
//!
//! - [`ext_tsp_order`] places blocks by a profile, to maximize the Ext-TSP
//!   score of fallthroughs and short jumps;
//!   [`falls_through`](crate::falls_through) gives it the fallthrough rule of
//!   the [`Emitter`](crate::Emitter) that will linearize the result.
//! - [`hot_cold_order`] partitions a base order by
//!   [`ColdnessClasses`](crate::ColdnessClasses), so the blocks that run
//!   normally are contiguous and the cold blocks form their own sections.
//!
//! The two compose: an Ext-TSP order is a base order for `hot_cold_order`.

mod ext_tsp;
mod hot_cold;

pub use ext_tsp::{ExtTspConfig, ext_tsp_order, ext_tsp_score};
pub use hot_cold::{SplitOrder, hot_cold_order};
