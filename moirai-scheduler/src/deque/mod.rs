//! Double-ended queues (deques) and memory reclamation policies.

mod reclaim;
mod chase_lev;
mod block_based;
mod split;

#[cfg(test)]
mod tests;

pub use reclaim::{
    DequeReclaimPolicy, DequeReclaimState, QuiescentAccessGuard, QuiescentReclaim, QuiescentState,
    SharedEpochAccessGuard, SharedEpochReclaim, SharedEpochState,
};
pub use chase_lev::{ChaseLevDeque, StealResult};
pub use block_based::BlockBasedDeque;
pub use split::SplitDeque;
