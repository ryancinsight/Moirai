//! Pure bounds for browser-selected file reads.

use std::io;

pub(crate) const MAX_READ_BYTES: usize = 1_048_576;

pub(crate) fn validate_buffer_length(length: usize) -> io::Result<u64> {
    if length > MAX_READ_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "browser file read exceeds the bounded chunk size",
        ));
    }
    u64::try_from(length).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "browser file read length cannot be represented",
        )
    })
}

pub(crate) fn advance_cursor(position: u64, amount: u64) -> io::Result<u64> {
    position.checked_add(amount).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "browser file cursor arithmetic overflowed",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{MAX_READ_BYTES, advance_cursor, validate_buffer_length};
    use std::io::ErrorKind;

    #[test]
    fn read_buffer_bound_accepts_empty_and_maximum_chunks() {
        assert_eq!(validate_buffer_length(0).expect("empty read is valid"), 0);
        assert_eq!(
            validate_buffer_length(MAX_READ_BYTES).expect("maximum chunk is valid"),
            MAX_READ_BYTES as u64
        );
        assert_eq!(
            validate_buffer_length(MAX_READ_BYTES + 1)
                .expect_err("oversized chunks must be rejected")
                .kind(),
            ErrorKind::InvalidInput
        );
    }

    #[test]
    fn cursor_advance_is_checked() {
        assert_eq!(advance_cursor(4, 8).expect("bounded cursor"), 12);
        assert_eq!(
            advance_cursor(u64::MAX, 1)
                .expect_err("cursor overflow must be rejected")
                .kind(),
            ErrorKind::InvalidInput
        );
    }
}
