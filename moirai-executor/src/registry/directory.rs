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
#[derive(Debug, Default)]
pub(super) struct BlockDirectory {
    base: usize,
    entries: VecDeque<Option<Arc<TaskStateBlock>>>,
    resident: usize,
}

impl BlockDirectory {
    pub(super) const fn new() -> Self {
        Self {
            base: 0,
            entries: VecDeque::new(),
            resident: 0,
        }
    }

    /// One past the highest block index the directory has created.
    pub(super) fn end(&self) -> usize {
        self.base + self.entries.len()
    }

    /// Number of blocks whose storage is still resident.
    #[cfg(test)]
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
            self.entries
                .push_back(Some(Arc::new(TaskStateBlock::new())));
            self.resident += 1;
            created = true;
        }
        let block = self.entries[offset].as_ref()?;
        Some((Arc::clone(block), created))
    }

    /// The resident blocks among the next `N` entries from `cursor`.
    ///
    /// A cursor outside the directory restarts at the watermark. Retired
    /// entries inside the window cost one step each, so the window bounds the
    /// work however long a pinned block's retirement span grows.
    pub(super) fn window<const N: usize>(&self, cursor: usize) -> SweepWindow<N> {
        let end = self.end();
        let start = if cursor < self.base || cursor >= end {
            self.base
        } else {
            cursor
        };
        let mut blocks = [const { None }; N];
        let stop = start.saturating_add(N).min(end);
        for (slot, index) in blocks.iter_mut().zip(start..stop) {
            if let Some(Some(block)) = self.entries.get(index - self.base) {
                *slot = Some((index, Arc::clone(block)));
            }
        }
        let wrapped = stop >= end;
        SweepWindow {
            next: if wrapped { self.base } else { stop },
            wrapped,
            blocks,
            resident: self.resident,
        }
    }

    /// Retire `block_index` if it still holds `expected`, returning the block so
    /// the caller drops it outside the directory lock.
    pub(super) fn retire(
        &mut self,
        block_index: usize,
        expected: &Arc<TaskStateBlock>,
    ) -> Option<Arc<TaskStateBlock>> {
        let offset = block_index.checked_sub(self.base)?;
        let entry = self.entries.get_mut(offset)?;
        if !entry
            .as_ref()
            .is_some_and(|block| Arc::ptr_eq(block, expected))
        {
            return None;
        }
        let retired = entry.take();
        self.resident -= 1;
        while matches!(self.entries.front(), Some(None)) {
            self.entries.pop_front();
            self.base += 1;
        }
        retired
    }
}

/// A bounded run of directory entries examined by one sweep step.
pub(super) struct SweepWindow<const N: usize> {
    /// Cursor for the following window: the watermark once this window reached
    /// the end of the directory.
    pub(super) next: usize,
    /// Whether this window reached the end of the directory.
    pub(super) wrapped: bool,
    /// The resident blocks in the window, with their block indices.
    pub(super) blocks: [Option<(usize, Arc<TaskStateBlock>)>; N],
    /// Resident blocks across the whole directory when the window was taken.
    pub(super) resident: usize,
}
