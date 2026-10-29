#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

//! Owned task-completion registrations and adversarial callback behavior.

mod callback_behavior;
mod notification;
mod owned_registration;

use super::state;
