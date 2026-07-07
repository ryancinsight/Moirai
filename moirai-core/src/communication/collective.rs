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

    /// Scatter operation: distribute data chunks to participants
    pub fn scatter<T: Clone>(data: Vec<T>, num_participants: usize) -> Vec<Vec<T>> {
        let chunk_size = data.len().div_ceil(num_participants);
        data.chunks(chunk_size).map(<[T]>::to_vec).collect()
    }

    /// Gather operation: collect data from all participants
    pub fn gather<T>(chunks: Vec<Vec<T>>) -> Vec<T> {
        chunks.into_iter().flatten().collect()
    }

    /// All-to-all communication pattern
    pub fn all_to_all<T: Clone>(data: Vec<Vec<T>>) -> Vec<Vec<T>> {
        let n = data.len();
        let mut result = vec![Vec::new(); n];

        for row in &data {
            for (j, item) in row.iter().enumerate() {
                if j < n {
                    result[j].push(item.clone());
                }
            }
        }

        result
    }

    /// Zero-copy scatter operation using slices
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
