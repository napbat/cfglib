//! Event-stream walks over open node spaces — breadth-first and depth-first siblings.

mod breadth_first;
mod depth_first;

pub use breadth_first::{OpenBfsEvent, open_breadth_first_events};
pub use depth_first::{OpenDfsEvent, open_depth_first_events};
