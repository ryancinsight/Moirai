//! Double-ended queues (deques) and memory reclamation policies.

mod chase_lev;
mod reclaim;
mod split;

#[cfg(test)]
mod tests;

pub use chase_lev::{
    ChaseLevDeque, ChaseLevStealer, DequeCapacity, DequeCapacityError, StealResult, StolenBatch,
};
pub use reclaim::{
    DeferredAccessGuard, DeferredReclaim, DeferredState, DequeReclaimPolicy, DequeReclaimState,
    SharedEpochAccessGuard, SharedEpochReclaim, SharedEpochState,
};
pub use split::SplitDeque;
