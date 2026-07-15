extern crate proc_macro;

use proc_macro::TokenStream;

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
