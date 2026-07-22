//! Registers configuration names emitted by Melinoe's TLS macro.

/// Publishes the macro-owned configuration name to rustc's check-cfg lint.
fn main() {
    println!("cargo:rustc-check-cfg=cfg(nightly_tls_active)");
}
