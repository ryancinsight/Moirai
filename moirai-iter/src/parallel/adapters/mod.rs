pub mod blocks;
pub mod chunks;
pub mod pair;
pub mod position;
pub mod side_effect;
pub mod stride;
pub mod window;

pub mod filter;
pub mod flat;
pub mod map;
pub mod ref_ops;
pub mod slice_ops;

pub use blocks::{ExponentialBlocks, UniformBlocks};
pub use chunks::Chunks;
pub use pair::{Interleave, InterleaveShortest, Zip, ZipEq};
pub use position::{MapPositions, Positions};
pub use side_effect::{Inspect, PanicFuse};
pub use stride::StepBy;
pub use window::{SkipAnyWhile, TakeAnyWhile};

pub use filter::{Filter, FilterMap, WhileSome};
pub use flat::{FlatMap, Flatten};
pub use map::{Map, MapInit, MapWith, Update};
pub use ref_ops::{Cloned, Copied, Enumerate};
pub use slice_ops::{Chain, Intersperse, Rev, Skip, Take};
