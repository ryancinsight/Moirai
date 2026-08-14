#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

/// CSR-shaped chunked buffer: one contiguous flat allocation plus a
/// chunk-offset table.
///
/// Replaces the jagged `Vec<Vec<T>>` layout previously used by the collective
/// operations. `offsets[i]..offsets[i+1]` is chunk `i`, so element traversal
/// is a single contiguous pass over `flat` instead of a per-chunk pointer
/// chase, and [`ChunkedVec::into_flat`] hands the storage back with no
/// re-flatten pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkedVec<T> {
    flat: Vec<T>,
    offsets: Vec<usize>,
}

impl<T> ChunkedVec<T> {
    /// Number of chunks in the buffer.
    #[must_use]
    pub fn num_chunks(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Total number of elements across all chunks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.flat.len()
    }

    /// True when the buffer holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.flat.is_empty()
    }

    /// Iterate the chunk slices in order.
    pub fn chunks(&self) -> impl Iterator<Item = &[T]> + '_ {
        self.offsets
            .windows(2)
            .map(move |window| &self.flat[window[0]..window[1]])
    }

    /// Consume the buffer, returning the contiguous element storage.
    #[must_use]
    pub fn into_flat(self) -> Vec<T> {
        self.flat
    }
}

/// Efficient collective operations for group communication
pub struct CollectiveOps;

impl CollectiveOps {
    /// All-reduce operation: combine values from all participants
    pub fn all_reduce<T, F>(values: Vec<T>, op: F) -> Vec<T>
    where
        T: Clone + Send,
        F: Fn(T, T) -> T + Sync,
    {
        if values.is_empty() {
            return vec![];
        }

        let result_len = values.len();

        // Tree reduction for efficiency
        let mut current = values;
        while current.len() > 1 {
            let mut next = Vec::with_capacity(current.len().div_ceil(2));

            for chunk in current.chunks(2) {
                if chunk.len() == 2 {
                    next.push(op(chunk[0].clone(), chunk[1].clone()));
                } else {
                    next.push(chunk[0].clone());
                }
            }

            current = next;
        }

        // Broadcast result to all
        vec![current[0].clone(); result_len]
    }

    /// Scatter operation: distribute data into one CSR-shaped chunked buffer
    /// with one chunk per participant.
    ///
    /// The result allocates once (`flat`) plus the chunk-offset table, instead
    /// of one `Vec` per participant. Empty input and a zero participant count
    /// produce an empty buffer rather than the historical `chunks(0)` panic.
    pub fn scatter<T: Clone>(data: Vec<T>, num_participants: usize) -> ChunkedVec<T> {
        let num_participants = num_participants.max(1);
        let chunk_size = data.len().div_ceil(num_participants).max(1);
        let mut offsets = Vec::with_capacity(num_participants + 1);
        offsets.push(0);
        let mut flat = Vec::with_capacity(data.len());
        for chunk in data.chunks(chunk_size) {
            flat.extend_from_slice(chunk);
            offsets.push(flat.len());
        }
        ChunkedVec { flat, offsets }
    }

    /// Gather operation: collect the chunked buffer back into one contiguous
    /// `Vec`. This is O(1): the flat buffer is returned directly, with no
    /// re-flatten pass over per-chunk allocations.
    pub fn gather<T>(chunks: ChunkedVec<T>) -> Vec<T> {
        chunks.into_flat()
    }

    /// All-to-all communication pattern: transpose the chunked buffer so
    /// result chunk `j` holds element `j` of every input chunk. Columns at or
    /// beyond the chunk count are dropped, matching the historical contract.
    pub fn all_to_all<T: Clone>(data: ChunkedVec<T>) -> ChunkedVec<T> {
        let columns = data.num_chunks();
        if columns == 0 {
            return ChunkedVec {
                flat: Vec::new(),
                offsets: vec![0],
            };
        }
        let mut offsets = Vec::with_capacity(columns + 1);
        offsets.push(0);
        for column in 0..columns {
            let count = data.chunks().filter(|chunk| chunk.len() > column).count();
            offsets.push(offsets.last().unwrap() + count);
        }
        let mut flat = Vec::with_capacity(*offsets.last().unwrap());
        for column in 0..columns {
            for chunk in data.chunks() {
                if let Some(item) = chunk.get(column) {
                    flat.push(item.clone());
                }
            }
        }
        ChunkedVec { flat, offsets }
    }

    /// Zero-copy scatter operation using slices
    #[deprecated(
        since = "0.5.0",
        note = "use [`CollectiveOps::scatter`] which returns a CSR-shaped `ChunkedVec`\
                (flat buffer + offset table) with the same chunk tiling; this slice-array\
                form is superseded by the flat layout."
    )]
    pub fn scatter_zero_copy<T>(data: &[T], num_chunks: usize) -> Vec<&[T]> {
        let chunk_size = data.len() / num_chunks;
        let mut chunks = Vec::with_capacity(num_chunks);

        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = if i == num_chunks - 1 {
                data.len()
            } else {
                (i + 1) * chunk_size
            };
            chunks.push(&data[start..end]);
        }

        chunks
    }

    /// Zero-copy gather operation using iterators
    #[deprecated(
        since = "0.5.0",
        note = "use [`CollectiveOps::gather`] which hands back the contiguous `ChunkedVec`\
                storage in O(1); iteration over the flat buffer replaces this slice iterator."
    )]
    pub fn gather_zero_copy<'a, T, I>(chunks: I) -> impl Iterator<Item = &'a T>
    where
        I: IntoIterator<Item = &'a [T]>,
        T: 'a,
    {
        chunks.into_iter().flat_map(|chunk| chunk.iter())
    }

    /// Zero-copy all-reduce operation
    pub fn all_reduce_zero_copy<T, F>(data: &[T], op: F) -> T
    where
        T: Clone,
        F: Fn(&T, &T) -> T,
    {
        data.iter()
            .skip(1)
            .fold(data[0].clone(), |acc, item| op(&acc, item))
    }
}
