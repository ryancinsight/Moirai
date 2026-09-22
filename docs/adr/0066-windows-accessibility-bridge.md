# ADR 0066: Windows accessibility bridge at the PAL boundary

Status: Accepted

Date: 2026-09-21

Driver: [MOI-WINDOW-ACCESSIBILITY-2026-09-21](../backlog.md#MOI-WINDOW-ACCESSIBILITY-2026-09-21),
[Metis accessibility](../../metis/backlog.md#METIS-A11Y-001)

## Context

The Win32 window provider exposes a software framebuffer and bounded input, but
its custom-rendered controls are invisible to Windows UI Automation. Metis
already derives a validated, format-neutral semantic tree. The operating-system
bridge must therefore live at the PAL boundary and preserve the tree's identity,
focus and action contract without putting Windows or AccessKit types in the
renderer or UI-language crates.

The HWND must be subclassed before it is shown or focused. UI Automation actions
may arrive on a thread other than the window thread, while application state and
frame presentation remain thread-affine. The crossing needs a bounded queue and
a wake message so the owning event loop can apply actions without blocking the
UI Automation callback.

## Decision

`moirai-pal` owns a Windows-only `AccessibilityTree` contract with bounded
validated nodes, stable nonzero identities, source order, focus, states and
typed actions. `WindowsAccessibilityAdapter` translates that contract to the
AccessKit Windows UI Automation adapter. AccessKit is the maintained provider
for this OS role; no first-party Atlas accessibility implementation exists, and
the dependency remains confined to the PAL Windows boundary.

`NativeWindow::new` constructs the HWND without showing it, installs the
adapter when requested, then shows the window. Existing callers retain the
visible constructor behavior. Tree updates validate connectivity and focus
before AccessKit receives them. AccessKit action callbacks copy only bounded
typed requests into a 256-entry queue and post a scalar wake message; the window
thread drains the queue as `WindowEvent::AccessibilityAction`. Overflow, queue
poisoning or a failed wake is surfaced as an I/O error instead of dropping a
request silently.

The PAL contract carries no application action policy, browser DOM, DICOM
semantics or screen-reader automation. Consumers map their own semantic tree to
the contract and apply requests through their existing state transitions.

The role vocabulary includes navigation and complementary landmarks, toolbars,
menus and menu items. These roles stay format-neutral and map directly to the
corresponding AccessKit roles so consumers can preserve command-surface
semantics without importing AccessKit into their own crates.

## Alternatives

- Implement UI Automation directly with Windows COM interfaces. Rejected because
  it duplicates a large provider implementation, expands the unsafe surface and
  would still require an independent tree consumer and event model.
- Depend on AccessKit from `metis-ui-lang`. Rejected because the renderer's
  validated semantics must remain host-neutral and WASM-compatible.
- Invoke the consumer callback synchronously from the UI Automation thread.
  Rejected because native application state and presentation are owned by the
  window thread and callback latency would become an unbounded host dependency.

## Invariants and failure behavior

- Tree identities are nonzero and unique; every node is reachable exactly once
  from the root, and focus is either the root or an enabled focusable node.
- Hidden or disabled nodes cannot expose focus or actions. Text and node counts
  remain within the PAL bounds before conversion to AccessKit.
- The adapter is installed only while the HWND is hidden. Its destructor removes
  the subclass before window destruction.
- The action queue has a fixed capacity. A full queue or failed wake poisons the
  next drain with a typed error, so the host cannot claim that every request was
  delivered when it was not.
- AccessKit's UI Automation and Windows policy remain external evidence. This
  provider proves translation and delivery to the OS bridge; a consumer must
  still record a real supported screen-reader or UI Automation traversal.

## Verification

Native Windows lifecycle tests cover hidden construction, adapter installation
before `show`, tree validation, stable updates and typed action translation,
including both UI Automation text replacement and `ValuePattern.SetValue`.
The locked Windows build, strict Clippy, Rustdoc and lockfile gates compile
the AccessKit provider. Metis owns the consumer conversion and records
host-level screen-reader evidence under `METIS-A11Y-001`.
