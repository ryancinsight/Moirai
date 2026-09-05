//! Provider-neutral GPU context and acquisition policy.

mod context;
mod preferences;

pub use context::GpuContext;
pub use preferences::DevicePreferences;
