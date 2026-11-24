//! Block orders computed from facts about the graph, to hand to
//! [`linearize`](crate::linearize) as
//! [`BlockOrder::Custom`](crate::BlockOrder::Custom).
//!
//! [`BlockOrder`](crate::BlockOrder) names the orders that need nothing but
//! the graph. The orders here also read an analysis or caller data:
//! [`hot_cold_order`] partitions a base order by
//! [`ColdnessClasses`](crate::ColdnessClasses), so the blocks that run
//! normally are contiguous and the cold blocks form their own sections.

mod hot_cold;

pub use hot_cold::{SplitOrder, hot_cold_order};
