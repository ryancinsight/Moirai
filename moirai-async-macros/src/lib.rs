//! Attribute macro providing an async `main` for Moirai.
//!
//! Rust's entry point cannot itself be `async`, so something has to build
//! a runtime and block on the future. This crate supplies that shim for
//! Moirai, keeping executor construction out of user code.

#![deny(missing_docs)]

extern crate proc_macro;

use proc_macro::TokenStream;

/// Marks an `async fn main` as the program entry point.
///
/// Rewrites the annotated function into a synchronous `main` that builds
/// an `AsyncExecutor` and blocks on the original body, preserving the
/// function's attributes and visibility.
///
/// The attribute takes no arguments and the executor is constructed with
/// its defaults; a caller needing a configured executor should build one
/// directly and call `block_on` instead.
///
/// # Panics
///
/// The generated `main` panics if the executor cannot be created. That is
/// deliberate: it runs before any user code, so there is no state to
/// unwind and nothing useful to recover to.
#[proc_macro_attribute]
pub fn main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input: syn::ItemFn = syn::parse(item).expect("expected a function");

    let fn_attrs = &input.attrs;
    let fn_vis = &input.vis;
    let fn_block = &input.block;

    let result = quote::quote! {
        #(#fn_attrs)*
        #fn_vis fn main() {
            let executor = moirai_async::executor::AsyncExecutor::new()
                .expect("Failed to create Moirai async executor");
            executor.block_on(async #fn_block);
        }
    };

    result.into()
}
