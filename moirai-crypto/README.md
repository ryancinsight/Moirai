# moirai-crypto

[![crates.io](https://img.shields.io/crates/v/moirai-crypto.svg)](https://crates.io/crates/moirai-crypto)
[![docs.rs](https://docs.rs/moirai-crypto/badge.svg)](https://docs.rs/moirai-crypto)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![No C dependencies](https://img.shields.io/badge/C%20deps-none-brightgreen.svg)](#zero-c-dependencies)

A **pure-Rust** [`rustls 0.23`](https://docs.rs/rustls) `CryptoProvider` backed entirely by the
[RustCrypto](https://github.com/RustCrypto) ecosystem.

**No `cc`, no NASM, no cmake, no `ring`, no `aws-lc-rs`.** Every cryptographic
operation is implemented in safe Rust and compiles with the standard Rust
toolchain on any target.

## Why?

`rustls` is a clean, audited TLS library but its two built-in crypto providers
both have C/assembly dependencies:

| Provider | C dependency | Consequence |
|----------|-------------|-------------|
| `ring` (default until 0.23) | `cc` build script + inline asm | Requires C compiler, fails on MUSL/cross targets without extra setup |
| `aws-lc-rs` (new default) | `aws-lc-sys` (NASM on Windows, cmake elsewhere) | Requires NASM on Windows, cmake everywhere |

`moirai-crypto` uses `rustls`'s stable **`CryptoProvider` extension point** to
supply the same functionality with zero non-Rust build steps.

## Supported algorithms

### Cipher suites

| ID | Suite | AEAD | Hash |
|----|-------|------|------|
| `0x1303` | `TLS_CHACHA20_POLY1305_SHA256` | ChaCha20-Poly1305 | SHA-256 |
| `0x1301` | `TLS_AES_128_GCM_SHA256` | AES-128-GCM | SHA-256 |
| `0x1302` | `TLS_AES_256_GCM_SHA384` | AES-256-GCM | SHA-384 |
| `0xC02B` | `TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256` (TLS 1.2) | AES-128-GCM | SHA-256 |
| `0xC02F` | `TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256` (TLS 1.2) | AES-128-GCM | SHA-256 |
| `0xC02C` | `TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384` (TLS 1.2) | AES-256-GCM | SHA-384 |
| `0xC030` | `TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384` (TLS 1.2) | AES-256-GCM | SHA-384 |
| `0xCCA9` | `TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256` (TLS 1.2) | ChaCha20-Poly1305 | SHA-256 |
| `0xCCA8` | `TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256` (TLS 1.2) | ChaCha20-Poly1305 | SHA-256 |

### Key exchange groups

| Group | Algorithm | Crate |
|-------|-----------|-------|
| `X25519` | Curve25519 ECDH (RFC 7748) | `x25519-dalek` |
| `secp256r1` (P-256) | NIST P-256 ECDH | `p256` |
| `secp384r1` (P-384) | NIST P-384 ECDH | `p384` |

### Certificate signature verification

| Scheme | Algorithm | Crate |
|--------|-----------|-------|
| `ecdsa_secp256r1_sha256` | ECDSA + P-256 + SHA-256 | `p256` |
| `ecdsa_secp384r1_sha384` | ECDSA + P-384 + SHA-384 | `p384` |
| `rsa_pkcs1_sha256` | RSA-PKCS1 + SHA-256 | `rsa` |
| `rsa_pkcs1_sha384` | RSA-PKCS1 + SHA-384 | `rsa` |
| `rsa_pkcs1_sha512` | RSA-PKCS1 + SHA-512 | `rsa` |
| `rsa_pss_rsae_sha256` | RSA-PSS + SHA-256 | `rsa` |
| `rsa_pss_rsae_sha384` | RSA-PSS + SHA-384 | `rsa` |
| `rsa_pss_rsae_sha512` | RSA-PSS + SHA-512 | `rsa` |
| `ed25519` | Ed25519 | `ed25519-dalek` |

## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
moirai-crypto = "0.5"
rustls = { version = "0.23", default-features = false, features = ["std", "tls12"] }
```

### As a process-wide default provider

```rust
fn main() {
    // Install moirai-crypto as the process-wide default before any TLS operations.
    moirai_crypto::provider()
        .install_default()
        .expect("failed to install moirai-crypto as default CryptoProvider");

    // All subsequent rustls configs pick it up automatically.
}
```

### Explicit per-config provider

```rust
use std::sync::Arc;
use rustls::{ClientConfig, RootCertStore};

let mut roots = RootCertStore::empty();
roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

let config = ClientConfig::builder_with_provider(Arc::new(moirai_crypto::provider()))
    .with_safe_default_protocol_versions()?
    .with_root_certificates(roots)
    .with_no_client_auth();
```

### With moirai-tls (built-in integration)

[`moirai-tls`](https://crates.io/crates/moirai-tls) uses `moirai-crypto` automatically:

```rust
use moirai_tls::{TlsConnector, ServerName};

let connector = TlsConnector::with_webpki_roots(); // uses moirai-crypto internally
let domain = ServerName::try_from("example.com")?;
let tls_stream = connector.connect(domain, tcp_stream).await?;
```

## Zero C dependencies

Verify with `cargo tree`:

```sh
cargo tree -p moirai-crypto --edges normal | grep -v '^moirai-crypto'
# All lines will be pure-Rust crates — no cc, cmake, or nasm entries.
```

To audit all transitive C build scripts:

```sh
cargo build -p moirai-crypto 2>&1 | grep -i 'running.*build.rs\|cc\|cmake\|nasm'
# (no output expected)
```

## RustCrypto component versions

| Component | Crate | Version |
|-----------|-------|---------|
| AES-GCM | `aes-gcm` | 0.10 |
| ChaCha20-Poly1305 | `chacha20poly1305` | 0.10 |
| X25519 | `x25519-dalek` | 2 |
| P-256 | `p256` | 0.13 |
| P-384 | `p384` | 0.13 |
| RSA | `rsa` | 0.9 |
| Ed25519 | `ed25519-dalek` | 2 |
| HMAC | `hmac` | 0.12 |
| HKDF | `hkdf` | 0.12 |
| SHA-2 | `sha2` | 0.10 |
| Entropy | `getrandom` | 0.2 |

## Security notes

- All cryptographic implementations come from the [RustCrypto](https://github.com/RustCrypto)
  project, which undergoes regular security audits.
- `X25519` key exchange rejects low-order (non-contributory) points.
- Private keys are zeroed on drop via `zeroize`.
- The AEAD confidentiality limit for AES-GCM is set to 2^23 messages per key
  (NIST SP 800-38D recommendation for random IV construction).

## MSRV

Rust **1.95** (matches the moirai workspace `rust-version`).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.
