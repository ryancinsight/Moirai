//! Priority plane indices shared by the queue's owner and stealer sides.

use moirai_core::Priority;

/// One queue per priority level; indices come from [`Priority::index`] (SSOT).
pub(super) const PRIORITY_LEVELS: usize = Priority::Critical.index() + 1;
/// Pop scan order: highest [`Priority::index`] first.
pub(super) const PRIORITY_POP_ORDER: [usize; PRIORITY_LEVELS] = [
    Priority::Critical.index(),
    Priority::High.index(),
    Priority::Normal.index(),
    Priority::Low.index(),
];
