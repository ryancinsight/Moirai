//! Double-ended queues (deques) and memory reclamation policies.

mod block_based;
mod chase_lev;
mod reclaim;
mod split;

#[cfg(test)]
mod tests;

pub use block_based::BlockBasedDeque;
pub use chase_lev::{ChaseLevDeque, StealResult};
pub use reclaim::{
    DequeReclaimPolicy, DequeReclaimState, QuiescentAccessGuard, QuiescentReclaim, QuiescentState,
    SharedEpochAccessGuard, SharedEpochReclaim, SharedEpochState,
};
pub use split::SplitDeque;
