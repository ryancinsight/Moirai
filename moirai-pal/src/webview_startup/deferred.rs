//! Fixed callback storage, drained only after the native callback returns.
//!
//! For capacity N, 0 <= len <= N. Insertion preserves the first N records in
//! order, counts excess attempts, and never overwrites admitted evidence.
//! Taking a batch restores the empty state without allocating.

#[derive(Debug)]
pub(crate) struct Deferred<T: Copy, const N: usize> {
    records: [Option<T>; N],
    len: usize,
    omitted: u64,
}

impl<T: Copy, const N: usize> Deferred<T, N> {
    pub(crate) const fn new() -> Self {
        Self {
            records: [None; N],
            len: 0,
            omitted: 0,
        }
    }

    pub(crate) fn push(&mut self, record: T) {
        if self.len == N {
            self.omitted = self.omitted.saturating_add(1);
        } else {
            self.records[self.len] = Some(record);
            self.len += 1;
        }
    }

    pub(crate) fn take(&mut self) -> ([Option<T>; N], u64) {
        let records = std::mem::replace(&mut self.records, [None; N]);
        self.len = 0;
        (records, std::mem::take(&mut self.omitted))
    }
}

#[cfg(test)]
mod tests;
