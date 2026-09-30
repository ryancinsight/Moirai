//! Fuzz target for shared-queue layout arithmetic.
//!
//! Throws arbitrary size combinations at `layout_total`, the pure check
//! `SharedQueue::{create,open}` rely on. A panic or overflow here would mean a
//! hostile peer can wedge or corrupt queue attachment; both must be typed
//! `IpcError`s instead.

#![no_main]

use moirai_core::ipc::__fuzz_ipc_layout;

libfuzzer_sys::fuzz_target!(|data: (u64, u64)| {
    // Sizes are clamped to realistic magnitudes so the target explores
    // boundary values without spending time on absurd multiplies.
    let elem_size = (data.0 % (1 << 16)) as usize;
    let capacity = (data.1 % (1 << 20)) as usize;
    match __fuzz_ipc_layout(elem_size, capacity) {
        Ok(value) => {
            std::hint::black_box(value);
        }
        Err(error) => {
            std::hint::black_box(error);
        }
    }
});
