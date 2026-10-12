//! Raw, disjoint byte extents. No archive or serving-engine semantics.
mod hydrate;
mod manifest;
mod pack;

pub use hydrate::{CompletedRange, ExtentHydration, ExtentOptions, ExtentProgress, ExtentReport};
pub use manifest::{Extent, ExtentManifest, ExtentTicket};
pub use pack::pack_file_ranges;
pub use pack::{pack_extents, pack_extents_with_prefixes};
