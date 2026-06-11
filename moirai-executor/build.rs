//! Build script to dynamically detect toolchain capabilities for the moirai-executor crate.
//!
//! Specifically, this detects whether we are on a nightly toolchain
//! to determine if we should enable unstable features like `#[thread_local]`.

use std::{env, process::Command};

/// Main entry point of the build script.
fn main() {
    println!("cargo:rustc-check-cfg=cfg(nightly_tls_active)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");

    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let Ok(output) = Command::new(rustc).arg("-vV").output() else {
        return;
    };
    if !output.status.success() {
        return;
    }

    let version = String::from_utf8_lossy(&output.stdout);
    let is_nightly_compiler = version.lines().any(|line| {
        line.strip_prefix("release: ")
            .is_some_and(|release| release.contains("nightly"))
    });

    if is_nightly_compiler {
        println!("cargo:rustc-cfg=nightly_tls_active");
    }
}
