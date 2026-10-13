# ADR 0076: Exact WebView2 capture byte admission

Status: Accepted

Date: 2026-10-13

Change class: [patch]

## Context

The WebView2 host admits a preview stream when its reported PNG byte count is
at most `MAX_WEBVIEW_CAPTURE_BYTES`. The presented-frame response encodes its
PNG bytes as canonical base64. Charging three bytes per encoded quartet
without subtracting terminal padding rejects a valid response at the same
inclusive bound. At 64 MiB, the byte count is congruent to one modulo three;
its canonical encoding ends with two padding characters.

The existing public capture byte bound and Windows host behavior remain the
contract. COM, JSON response interpretation and PNG integrity verification
remain in the host. The portable base64 byte law needs executable coverage on
each development platform, without a WebView2 runtime.

## Decision and invariants

The capture module owns the bound, derived from the existing maximum frame
pixel count multiplied by four RGBA8 channels. The existing WebView2 public
module exposes that same constant. There is one owner and no duplicate bound.

For a nonempty canonical string with `q` quartets and `p` terminal padding
characters, `p` is in `{0, 1, 2}` and the decoded length is `L = 3q - p`.
Every complete quartet contributes three bytes; the terminal quartet omits
the bytes represented only by padding. The decoder rejects `L > LIMIT`
before requesting output storage and requests exactly `L` bytes otherwise.
The limit is a const generic over the same decoder, so boundary tests execute
the production algorithm at both small exact budgets and the full capture
budget.

The decoder accepts the alphabet defined in
[RFC 4648 sections 3.5 and 4](https://www.rfc-editor.org/rfc/rfc4648), requires
padding only at the terminal quartet and rejects nonzero discarded bits.
Empty, misaligned, malformed and over-budget data retain their concrete I/O
error context. Length admission precedes alphabet decoding, so an oversized
encoded response fails in constant time without requesting output storage.
Before the terminal quartet, the decoder produces at most `3(q - 1) <= L`
bytes; it validates terminal grammar before appending terminal bytes.
Malformed padding therefore cannot force output growth beyond the admitted
length.

The Windows host imports the portable decoder directly. Test-only portable
module loading uses that source on platforms where the Windows provider is
absent. Windows tests with the provider enabled load its normal module once.
No forwarding decoder, compatibility alias or alternate algorithm remains.

## Alternatives and failure behavior

Raising the capture limit would alter the resource contract and would leave
the same padding defect at the next boundary. Reserving the rounded three-byte
capacity would charge bytes absent from the payload. A browser-side decoder
would duplicate the native law and prevent portable Rust verification.

Alphabet and padding violations return `InvalidData`; allocation failure
returns `OutOfMemory` with the presented-frame context. PNG CRC, image
dimensions, decompression and trailer validation retain their existing host
checks. This change does not alter navigation, permissions, callback ownership
or the presented-frame wait deadline.

## Verification

The unchanged extracted decoder reproduces the defect: three of five tests
fail, including the complete 67,108,864-byte payload. The corrected decoder
passes those five portable tests under the 60-second nextest timeout. They
verify published RFC vectors, every possible byte value, both terminal padding
lengths, alphabet/padding/trailing-bit rejection and concrete budget errors.

The full resource test generates the nonconstant byte sequence
`byte(i) = i mod 256`, independently encodes it, and compares every decoded
byte at 67,108,862, 67,108,863 and 67,108,864 bytes. It rejects 67,108,865 bytes.
These lengths cover one padding character, no padding, two padding characters
and the first disallowed byte count. Output allocation follows the admission
check by source construction. Production Windows compilation and provider CI
verify the host imports and feature-gated PNG dependency; an installed runtime
capture remains covered by the existing Windows provider integration tests.
