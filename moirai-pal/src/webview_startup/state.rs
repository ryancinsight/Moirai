//! One callback owns a slot until its terminal completion, even after close.

use std::io;

/// Opaque within this module; the callback also retains its owning slot weakly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RequestId {
    generation: u64,
    sequence: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Start {
    Begun(RequestId),
    Busy,
    Closed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Finish {
    Current,
    Stale,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RequestSnapshot {
    pub(crate) generation: u64,
    pub(crate) pending: bool,
    pub(crate) closed: bool,
}

/// Generations never wrap; sequence identity is never reused within this slot.
#[derive(Debug)]
pub(crate) struct RequestSlot {
    generation: u64,
    next_sequence: Option<u64>,
    pending: Option<RequestId>,
    closed: bool,
}

impl Default for RequestSlot {
    fn default() -> Self {
        // The initial generation and sequence are nonzero domain identities,
        // so the structural zero-valued default cannot represent this state.
        Self {
            generation: 1,
            next_sequence: Some(1),
            pending: None,
            closed: false,
        }
    }
}

impl RequestSlot {
    pub(crate) fn start(&mut self) -> io::Result<Start> {
        if self.closed {
            return Ok(Start::Closed);
        }
        if self.pending.is_some() {
            return Ok(Start::Busy);
        }
        let sequence = self
            .next_sequence
            .ok_or_else(|| io::Error::other("Startup request sequence is exhausted"))?;
        let request = RequestId {
            generation: self.generation,
            sequence,
        };
        self.next_sequence = sequence.checked_add(1);
        self.pending = Some(request);
        Ok(Start::Begun(request))
    }

    pub(crate) fn is_current(&self, request: RequestId) -> bool {
        !self.closed && self.generation == request.generation && self.pending == Some(request)
    }

    pub(crate) fn finish(&mut self, request: RequestId) -> Finish {
        if self.pending != Some(request) {
            return Finish::Unknown;
        }
        let outcome = if self.is_current(request) {
            Finish::Current
        } else {
            Finish::Stale
        };
        self.pending = None;
        outcome
    }

    pub(crate) fn invalidate(&mut self) -> io::Result<()> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "Startup request slot is closed",
            ));
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| io::Error::other("Startup request generation is exhausted"))?;
        self.generation = generation;
        Ok(())
    }

    pub(crate) const fn close(&mut self) {
        self.closed = true;
    }

    pub(crate) const fn snapshot(&self) -> RequestSnapshot {
        RequestSnapshot {
            generation: self.generation,
            pending: self.pending.is_some(),
            closed: self.closed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_boundaries_admit_last_values_without_wrapping_or_reuse() {
        let _trace = crate::webview_startup::tests::trace();
        let _test = tracing::info_span!("startup_request_identity_boundaries").entered();
        let mut slot = tracing::info_span!("arrange").in_scope(|| RequestSlot {
            generation: u64::MAX - 1,
            next_sequence: Some(u64::MAX),
            pending: None,
            closed: false,
        });
        let request = tracing::info_span!("act").in_scope(|| {
            slot.invalidate().expect("last generation");
            let Start::Begun(request) = slot.start().expect("last sequence") else {
                panic!("empty slot admits the last identity")
            };
            request
        });
        let before = slot.snapshot();
        let generation = tracing::info_span!("act")
            .in_scope(|| slot.invalidate().expect_err("generation cannot wrap"));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(request.generation, u64::MAX);
            assert_eq!(request.sequence, u64::MAX);
            assert_eq!(
                before,
                RequestSnapshot {
                    generation: u64::MAX,
                    pending: true,
                    closed: false
                }
            );
            assert_eq!(slot.snapshot(), before);
            assert!(slot.is_current(request));
            assert_eq!(generation.kind(), io::ErrorKind::Other);
            assert_eq!(
                generation.to_string(),
                "Startup request generation is exhausted"
            );
            assert_eq!(slot.finish(request), Finish::Current);
        });
        let sequence = tracing::info_span!("act")
            .in_scope(|| slot.start().expect_err("sequence cannot be reused"));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(sequence.kind(), io::ErrorKind::Other);
            assert_eq!(
                sequence.to_string(),
                "Startup request sequence is exhausted"
            );
            assert_eq!(
                slot.snapshot(),
                RequestSnapshot {
                    generation: u64::MAX,
                    pending: false,
                    closed: false
                }
            );
            slot.close();
            assert_eq!(slot.start().expect("closed exhausted slot"), Start::Closed);
            assert_eq!(
                slot.snapshot(),
                RequestSnapshot {
                    generation: u64::MAX,
                    pending: false,
                    closed: true
                }
            );
        });
    }
}
