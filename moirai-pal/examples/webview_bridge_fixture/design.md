# Installed-runtime Rust bridge regression

This verification payload compiles as one Cargo `cdylib` example. Cargo's
[target contract](https://doc.rust-lang.org/cargo/reference/cargo-targets.html#the-crate-type-field)
admits library examples; Cargo JSON identifies the actual WASM artifact. The
locked matching wasm-bindgen compiler produces the original web glue and WASM.
The only authored JavaScript imports that glue and awaits its initializer.
All callback, protocol, computation, close and remount behavior resides in Rust.

The native fixture owns one actual WebView2 surface, document and fresh profile.
It uses the public `WebViewBridge::current`, `listen` and `post_json` APIs inside
that installed runtime. Ordinary Chrome tests cannot establish native bridge
admission. No fake `chrome`, event dispatch, evaluated script, checked generated
snapshot or runtime Cargo invocation participates.

The protocol has one outstanding native request. The WASM callback increments
its mount's actual invocation counter and hands the received value to one
bounded Rust channel. One sender's reserved slot in `mpsc::channel(0)` admits
one pending message; the sender is never cloned. The asynchronous coordinator
consumes the command after the callback returns, so explicit close never drops
the currently executing callback. Unexpected commands or queue pressure fail.

Mount 1 reports ready, receives a Unicode patient and maximum-u64 decimal string,
and sums three input integers with checked arithmetic. The native oracle derives
the result from powers of two. A second request has exactly the shared inclusive
JSON byte limit: its Unicode prefix and ASCII padding derive from the measured
serialization overhead. The identical received JSON travels back through the
actual native bridge. Adding one trailing space preserves valid JSON but exceeds
the byte limit by one. Both host and WASM APIs must reject that value before
delivery, preserving the actual error context.

Explicit close retires the first listener. The coordinator attaches a second
listener in the same document and reports ready. A real subsequent native probe
must reach the new listener while the old counter remains unchanged; another
Unicode/string-identity computation and explicit close follow. The native
observer also rejects actual Runtime exceptions, covering a stale freed callback
thunk that would throw instead of incrementing its retired Rust counter. This
proves cancellation for the exercised subsequent messages, not arbitrary future
browser scheduling. The final counted cleanup guard is zero.

The original 30-second operation deadline includes host construction and every
roundtrip. The existing nextest 60-second fixture bound, 120-second nextest CLI
bound and 20-minute workflow bound remain unchanged. Compilation uses the
existing browser producer's separate 600-second bound. Generated output has a
source/tool/lock/command/file-hash receipt; the native test copies the original
payload into its owned package. Successful completion requires real Destroyed,
surface drop and actual package/profile removal. Source compilation cannot
substitute for execution on the installed Windows runtime.
