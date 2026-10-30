use std::sync::Arc;

use super::{
    super::TimerState,
    callbacks::{
        CallbackCounts, CallbackProbe, callback_waker, registered_waker_count,
        registered_waker_matches,
    },
};

const fn callback_counts(drops: usize) -> CallbackCounts {
    CallbackCounts {
        clones: 1,
        wakes: 0,
        drops,
    }
}

pub(super) struct CallbackFixture {
    probe: Arc<CallbackProbe>,
    waker: std::task::Waker,
}

impl CallbackFixture {
    pub(super) fn new(state: &Arc<TimerState>) -> Self {
        let probe = Arc::new(CallbackProbe::new(Arc::clone(state)));
        let waker = callback_waker(&probe);
        Self { probe, waker }
    }

    pub(super) fn waker(&self) -> &std::task::Waker {
        &self.waker
    }

    pub(super) fn into_registered_probe(self, state: &TimerState) -> Arc<CallbackProbe> {
        assert_eq!(registered_waker_count(state), 1);
        assert!(registered_waker_matches(state, &self.waker));
        assert_eq!(self.probe.counts(), callback_counts(0));
        assert_eq!(Arc::strong_count(&self.probe), 3);

        let Self { probe, waker } = self;
        drop(waker);
        assert_eq!(probe.counts(), callback_counts(1));
        assert_eq!(Arc::strong_count(&probe), 2);
        probe
    }

    pub(super) fn assert_cleared(state: &TimerState, probe: &Arc<CallbackProbe>) {
        assert_eq!(registered_waker_count(state), 0);
        assert_eq!(Arc::strong_count(probe), 1);
        assert_eq!(probe.counts(), callback_counts(2));
    }
}
