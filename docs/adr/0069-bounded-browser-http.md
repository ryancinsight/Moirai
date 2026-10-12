# 0069 — Bounded browser HTTP and page lifecycle

Status: Accepted

Date: 2026-10-12

## Contract

Metis removes its authored JavaScript transport and page policy. Moirai owns
browser mechanisms; Metis owns protocol, authority and presentation decisions.
The additive PAL surface targets workspace 0.8.0 and leaves the independently
versioned Apple WebView crate unchanged.

`WebHttp::request` admits HTTP(S) URLs without credentials, GET or POST, a
finite nonzero deadline, and request/response bounds no larger than the existing
1 MiB browser-read bound. Requests omit credentials, suppress redirects and
disable caching. A declared Content-Length must be decimal and within the
response bound. Actual streamed bytes must also fit it, including one extra
chunk after the exact bound: declared length never substitutes for counted I/O.

For counted bytes b, next chunk c and bound B, admission is c <= B - b. Only
then may b become b + c. This subtraction form prevents overflow. EOF is
observed for HTTP; selected-file slices may finish when their requested buffer
fills. One statically dispatched stream reader implements both termination
policies. Future drop aborts Fetch, cancels an unfinished reader and releases
its lock; expiry covers headers and body under one deadline.

The document exposes its current URL, bounded URLSearchParams access, owned
document listeners and element closest/attribute reads. These return browser
observations without granting navigation or native authority. Metis decides
which configuration and destinations are admitted.

## Alternatives and verification

Direct web-sys imports in Metis duplicate the platform boundary. Retaining a
JavaScript HTTP facade duplicates the protocol and cancellation owner. Both
are rejected. No native shell, network sandbox or browser permission bypass
is added.

Native policy tests cover exact/excess bounds, malformed decimal lengths and
deadline limits with computed values. WASM compilation and real loopback
browser journeys cover Fetch, binary bodies, cancellation and DOM teardown.
Browser CORS/CSP refusal remains an explicit transport error.
