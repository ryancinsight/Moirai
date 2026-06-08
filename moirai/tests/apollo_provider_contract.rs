//! Apollo-facing provider contracts for Moirai's public crate surface.

use moirai::{
    for_each_chunk_mut_enumerated_with, Adaptive, IndexedParallelIterator, IntoParallelIterator,
};

#[test]
fn apollo_chunked_mutation_covers_each_element_once() {
    const CHUNK: usize = 7;
    let mut data = vec![0usize; 31];

    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(&mut data, CHUNK, |chunk_index, chunk| {
        let base = chunk_index * CHUNK;
        for (offset, slot) in chunk.iter_mut().enumerate() {
            *slot = base + offset + 1;
        }
    });

    assert_eq!(data, (1..=31).collect::<Vec<_>>());
}

#[test]
fn apollo_collect_into_vec_moves_non_clone_values_into_existing_storage() {
    struct Sample {
        value: usize,
    }

    let input = vec![
        Sample { value: 3 },
        Sample { value: 5 },
        Sample { value: 8 },
    ];
    let mut output = Vec::with_capacity(8);
    output.push(Sample { value: 999 });
    let capacity = output.capacity();

    input.into_par_iter().collect_into_vec(&mut output);

    assert_eq!(output.capacity(), capacity);
    assert_eq!(
        output.iter().map(|sample| sample.value).collect::<Vec<_>>(),
        vec![3, 5, 8]
    );
}
