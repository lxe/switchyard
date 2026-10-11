//! Raw, disjoint byte extents. No archive or serving-engine semantics.
mod hydrate;
mod manifest;
mod pack;

pub use hydrate::{CompletedRange, ExtentHydration, ExtentOptions, ExtentReport};
pub use manifest::{Extent, ExtentManifest, ExtentTicket};
pub use pack::pack_extents;
