//! Lock-free single-producer/single-consumer channel.
//!
//! One bounded ring with two ways to hand out its halves:
//!
//! - [`spsc(capacity)`](crate::channel::spsc()) returns `'static` halves backed
//!   by an `Arc`, for producer and consumer threads with independent lifetimes.
//! - [`SpscRing::split`] borrows a ring you own, for halves that live inside a
//!   scope. No `Arc`, so no refcount traffic and one fewer allocation.
//!
//! Both flavours drive the same primitives and implement the same
//! [`Producer`](crate::channel::Producer) and
//! [`Consumer`](crate::channel::Consumer) roles, so generic code accepts either
//! without a `'static` bound.
//!
//! Neither half is `Clone` or `Sync`, which is what keeps the channel
//! single-producer/single-consumer; the ring type itself stays crate-private
//! for the same reason (ADR-024, ADR-026).

mod borrowed;
mod ring;
mod shared;

pub use borrowed::{SpscConsumer, SpscProducer, SpscRing};
pub use shared::{SpscReceiver, SpscSender};

pub(crate) use ring::SpscChannel;
