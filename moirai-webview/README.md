# Moirai WebView

Apple application lifecycle and bounded packaged WebKit presentation. Native
objects remain on the main thread; the host supplies the packaged resource root
and exact HTTPS destinations it authorizes. Apple host APIs compile on macOS and
iOS. Resource and destination policy types also compile on other native targets.

```rust
use moirai_webview::ExternalLinks;

let allowed = ExternalLinks::new(["https://example.com/help"])?;
let repeated = ExternalLinks::new([
    "https://example.com/help",
    "https://example.com/help",
])?;
assert_eq!(allowed, repeated);
# Ok::<(), std::io::Error>(())
```

Destinations are retained verbatim. Authorizing one path does not authorize
another query, fragment, spelling or origin. An empty policy denies external
handoffs. The native host applies this policy to new-window navigation while
retaining packaged frames.

This README supplies the crate's Rustdoc overview. Publication serves that
reference at [docs.rs](https://docs.rs/moirai-webview). Source and workspace
documentation are in the [Moirai repository](https://github.com/ryancinsight/Moirai).
