# ADR 0073: Rust-owned AppKit application and window commands

- Status: Accepted
- Date: 2026-10-12
- Change class: [patch]; target: moirai-webview 0.1.2

## Contract and decision

A packaged desktop application has a usable native exit and window-control
surface while its WebView holds keyboard focus. The existing window already
has Titled, Closable, Miniaturizable and Resizable style flags. An AppKit window
does not create an application main menu: the Metis starter has an empty app
menu, and a real Command-Q press leaves the process and window alive.

The AppKit adapter installs native application, Edit and Window menus before
entering its event loop. Application commands target NSApplication; editing
and window commands use the key window's responder chain. Command-Q invokes
terminate:, Command-W invokes performClose:, and Command-M invokes
performMiniaturize:. Zoom and Control-Command-F use the native window actions.
The existing last-window delegate retains its termination contract. AppKit owns
menu item/submenu retention; weak application targets remain alive through the
singleton. Menus and delegates detach if the event loop returns.
The same retained delegate observes native resize, minimize/restore, full-screen
and close notifications. Structured events record the actual window frame,
state flags and standard-button visibility/enabled state; no page text enters
these diagnostics. This distinguishes an AppKit transition from a capture that
re-raises the window or keeps the previous image dimensions.

This uses the existing native provider boundary. HTML buttons, JavaScript
window.close callbacks and frontend IPC termination requests are rejected:
they duplicate native behavior or require page authority for a host action.
No public API, page permission, resource origin or mobile lifecycle changes.
It does not establish a general programmable macOS window API or Tauri parity.

## Behavior verification

Given the real packaged Rust starter with its name field focused, Command-Q
terminates its native process. After relaunch, Command-M minimizes the same
window without terminating; restoring it retains the entered text and greeting.
Command-W and the native close button terminate the last-window host. Window
Zoom changes its geometry; normal native controls remain AppKit-owned.
Native menu-click and keyboard journeys exercise the actual responder chain;
source inspection or compilation alone cannot close this acceptance.

Compile the provider with warning-denying macOS diagnostics and both iOS ABIs;
run the unchanged bounded resource tests. Rebuild the frozen Metis consumer,
verify its signature and exact Unicode greeting, then run the native journeys
and record process exit. The remaining native permission, custom window API
and physical-device acceptance stay in their existing owning items.

## Local evidence

The real starter returns `Hello, Anaë — 東京! You've been greeted from Rust!`.
Command-Q with the name field focused and Command-W each leave no installed
starter process. Native traces report minimized=true, then restored=false for
the minimized flag, with the same greeting retained. Zoom changes 800×632 to
1728×1034 and back; full-screen entry records fullscreen=true. The installed
app's Control-Command-F enters and exits full screen, restoring its native
button surface. The supervised development session exits with status zero.
All three standard buttons have nonzero frames, hidden=false and enabled=true
in normal mode. Recordings show a purple capture badge over their title-bar
area, so a recording alone does not establish their unobscured appearance.
Eight real resource/response tests pass within the 60-second bound; native
warning-denying diagnostics and both iOS ABI checks pass. Frozen published-source
consumer verification follows provider delivery.
