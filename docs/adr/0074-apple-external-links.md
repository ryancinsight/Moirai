# ADR 0074: Host-authorized Apple external links

- Status: Accepted
- Date: 2026-10-13
- Change class: [minor]; target: moirai-webview 0.2.0

## Contract and decision

The packaged starter's target="_blank" logos request a WebKit popup. Without
a WKUIDelegate popup callback, the native host cannot complete that request.
The provider reads the host-owned MoiraiExternalLinks string array from the
application's signed Info.plist before creating its WebView. Missing metadata
grants no external authority; malformed metadata fails startup with context.
ExternalLinks validates at most 64 absolute HTTPS destinations of at most
2048 bytes each. These allocation bounds match the consumer's opener budget.
Credentials, invalid percent encoding, invalid ports and non-URI bytes fail
validation. The entire declared URI is the identity, including query and
fragment; neither prefix nor origin matching expands authority.

Let P(u) mean ResourceRoot admits the packaged URI, E(u) mean the exact host
allowlist contains u, and N mean the target frame is absent (a new window).
Existing packaged frames are allowed iff P(destination) and not N.
ExternalPopup requires N, E(destination), a packaged main source frame,
LinkActivated classification, and no download. Every other navigation is denied.
The navigation delegate admits ExternalPopup to the UI delegate without
opening it. The UI delegate independently checks the same contract, performs
one OS URL handoff, and returns no WebView. Remote content never replaces the
packaged frame. AppKit uses NSWorkspace; UIKit uses UIApplication's URL open
completion. OS refusal remains a structured error rather than an embedded
network fallback.

LinkActivated classifies navigation; it does not prove Event.isTrusted.
WebKit's automatic JavaScript window prohibition remains enabled, but public
WebKit APIs provide no unforgeable user-gesture witness for this callback.
This grants bounded host-declared link destinations, not the consumer's typed
OPEN_EXTERNAL broker capability. Provider code does not depend on Metis and
the existing run(resources, title) surface remains unchanged.

## Alternatives and failure modes

A JavaScript click bridge duplicates native policy and adds page authority.
An unrestricted popup view admits remote code into the packaged application.
Hardcoding starter domains in the provider violates host ownership. Opening
in both delegates duplicates OS actions. Each alternative is rejected.
Unknown schemes, undeclared links, subframe sources, existing external targets,
downloads, forms and other navigation classifications are rejected. The OS
owns subsequent browser redirects. Bundle signing protects installed metadata;
this policy does not establish trust in page code or browser destinations.
Apple UI and WebKit frameworks remain required dynamic system dependencies.

## Verification plan

Check exact identity and adversarial URI validation, inclusive bounds, empty
policy, packaged routing and the Cartesian product of source, target, action
and download facts. Build warning-denying macOS and both iOS ABIs. With the
real packaged starter, click each logo and inspect the OS browser destination;
verify the packaged URI and input-sensitive Rust greeting remain intact.
Compilation establishes ABI consistency, not native click acceptance.

## Provider evidence

Twenty tests pass against real Foundation arrays and real filesystem
resources. The source/action/target/download matrix checks 168 combinations.
Warning-denying macOS diagnostics and both iOS ABI checks pass. A minor-class
comparison against 0.1.2 passes all 196 applicable cargo-semver-checks checks.
The real native preview records ExternalPopup in both delegates and exactly
one successful OS handoff after clicking the Metis logo. Canonical consumer
and visual browser confirmation follow provider delivery. Protocol callbacks
live in their actual protocol implementations, so required-selector checking
also succeeds with objc2 0.6.5; the canonical lock retains 0.6.4.
