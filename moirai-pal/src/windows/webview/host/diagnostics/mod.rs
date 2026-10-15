//! Private fixture-only observations; see the co-located contract.

mod cdp;
mod completion;
mod dom;
#[cfg(test)]
mod fixture;
mod input;
mod native;
mod protocol;
mod registration;
mod session;
mod state;

#[cfg(test)]
pub(super) use fixture::FixtureInput;
pub(super) use session::Startup;
