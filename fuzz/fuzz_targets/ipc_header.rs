//! Fuzz target for shared-queue header parsing and layout arithmetic.
//!
//! Throws peer-controlled bytes at `parse_header_capacity` and arbitrary
//! size combinations at `layout_total` - the two pure checks
//! `SharedQueue::{create,open}` rely on. A panic or overflow here would
//! mean a hostile peer can wedge or corrupt queue attachment; both must be
//! typed `IpcError`s instead.

#![no_main]

use moirai_core::ipc::__fuzz_ipc_header;

libfuzzer_sys::fuzz_target!(|data: (u64, u64, &[u8])| {
    // Sizes are clamped to realistic magnitudes so the target explores
    // boundary values without spending time on absurd multiplies.
    let elem_size = (data.0 % (1 << 16)) as usize;
    let capacity = (data.1 % (1 << 20)) as usize;
    let (header, layout) = __fuzz_ipc_header(data.2, elem_size, capacity);
    match header {
        Ok(value) => {
            std::hint::black_box(value);
        }
        Err(error) => {
            std::hint::black_box(error);
        }
    }
    match layout {
        Ok(value) => {
            std::hint::black_box(value);
        }
        Err(error) => {
            std::hint::black_box(error);
        }
    }
});
