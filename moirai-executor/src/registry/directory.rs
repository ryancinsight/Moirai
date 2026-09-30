use std::{collections::VecDeque, sync::Arc};

use super::state::TaskStateBlock;

/// Where a block index sits relative to the directory's retirement watermark.
pub(super) enum BlockLookup<'directory> {
    /// The block was retired: every task it ever held completed and its
    /// storage was released.
    Retired,
    /// The block is resident.
    Live(&'directory Arc<TaskStateBlock>),
    /// No block has been created at this index yet.
    Absent,
}

/// Dense directory of task-state blocks with a retirement watermark.
///
/// Block indices below `base` are retired without an entry, and a retired
/// block above `base` (retired while an older block is still pinned by a
/// running task) is a `None` entry. Retirement therefore costs one directory
/// word per retired block until the oldest resident block also retires, at
/// which point the leading run of `None` entries collapses into `base`.
///
/// The retirement sweep does not walk `entries`, whose span a pinned block
/// stretches without bound. It rotates through `sweep_queue`, which holds only
/// resident block indices, so the work of a sweep step depends on the resident
/// count and never on the span.
#[derive(Debug, Default)]
pub(super) struct BlockDirectory {
    base: usize,
    entries: VecDeque<Option<Arc<TaskStateBlock>>>,
    resident: usize,
    /// Resident blocks no sweep has checked out, in examination order: creation
    /// order, with each examined block that stayed resident moved behind the
    /// blocks created while it was examined.
    sweep_queue: VecDeque<usize>,
}

impl BlockDirectory {
    pub(super) const fn new() -> Self {
        Self {
            base: 0,
            entries: VecDeque::new(),
            resident: 0,
            sweep_queue: VecDeque::new(),
        }
    }

    /// One past the highest block index the directory has created.
    pub(super) fn end(&self) -> usize {
        self.base + self.entries.len()
    }

    /// Number of blocks whose storage is still resident, including those a
    /// sweep has checked out.
    pub(super) fn resident(&self) -> usize {
        self.resident
    }

    /// Directory words currently held: the retirement span above the watermark.
    #[cfg(test)]
    pub(super) fn span(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn lookup(&self, block_index: usize) -> BlockLookup<'_> {
        let Some(offset) = block_index.checked_sub(self.base) else {
            return BlockLookup::Retired;
        };
        match self.entries.get(offset) {
            Some(Some(block)) => BlockLookup::Live(block),
            Some(None) => BlockLookup::Retired,
            None => BlockLookup::Absent,
        }
    }

    /// Resident blocks in index order.
    pub(super) fn resident_blocks(&self) -> impl Iterator<Item = &Arc<TaskStateBlock>> {
        self.entries.iter().flatten()
    }

    /// Create every block up to and including `block_index` and return it.
    ///
    /// Returns `None` when the block was already retired, and whether the call
    /// created a block otherwise.
    pub(super) fn ensure(&mut self, block_index: usize) -> Option<(Arc<TaskStateBlock>, bool)> {
        let offset = block_index.checked_sub(self.base)?;
        let mut created = false;
        while self.entries.len() <= offset {
            self.sweep_queue.push_back(self.end());
            self.entries
                .push_back(Some(Arc::new(TaskStateBlock::new())));
            self.resident += 1;
            created = true;
        }
        let block = self.entries[offset].as_ref()?;
        Some((Arc::clone(block), created))
    }

    /// Blocks awaiting examination: the most a sweep pass can check out.
    pub(super) fn queued(&self) -> usize {
        self.sweep_queue.len()
    }

    /// Check out up to `limit` queued blocks, at most `N`, for one sweep step.
    ///
    /// A checked-out block stays resident and readable but belongs to the
    /// caller until it is settled with [`Self::retire`] or [`Self::requeue`], so
    /// concurrent sweeps never examine the same block. Each checkout costs one
    /// queue pop however many retired entries the directory span holds.
    pub(super) fn check_out<const N: usize>(&mut self, limit: usize) -> SweepWindow<N> {
        let mut blocks = [const { None }; N];
        for slot in blocks.iter_mut().take(limit) {
            let Some(index) = self.sweep_queue.pop_front() else {
                break;
            };
            let block = self
                .entries
                .get(index - self.base)
                .and_then(Option::as_ref)
                .expect("invariant: a queued or checked-out index names a resident block");
            *slot = Some((index, Arc::clone(block)));
        }
        SweepWindow {
            blocks,
            resident: self.resident,
        }
    }

    /// Return a checked-out block to the back of the sweep queue.
    pub(super) fn requeue(&mut self, block_index: usize) {
        self.sweep_queue.push_back(block_index);
    }

    /// Retire a checked-out block, returning it so the caller drops it outside
    /// the directory lock.
    pub(super) fn retire(&mut self, block_index: usize) -> Arc<TaskStateBlock> {
        let entry = block_index
            .checked_sub(self.base)
            .and_then(|offset| self.entries.get_mut(offset))
            .expect("invariant: a checked-out index lies inside the directory");
        let retired = entry
            .take()
            .expect("invariant: a checked-out block is resident until its sweep retires it");
        self.resident -= 1;
        while matches!(self.entries.front(), Some(None)) {
            self.entries.pop_front();
            self.base += 1;
        }
        retired
    }
}

/// A bounded run of queued blocks checked out for one sweep step.
pub(super) struct SweepWindow<const N: usize> {
    /// The checked-out blocks with their block indices.
    pub(super) blocks: [Option<(usize, Arc<TaskStateBlock>)>; N],
    /// Resident blocks across the whole directory at checkout.
    pub(super) resident: usize,
}
