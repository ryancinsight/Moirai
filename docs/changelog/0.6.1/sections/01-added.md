### Added

- **Tokio I/O compatibility** (`TokioCompat`, `MoiraiCompat`). Both wrappers
  now map `AsyncBufRead` (`poll_fill_buf`, `consume`) and vectored writes, and
  assert at compile time that they add no size or alignment to the wrapped
  type. `moirai_async::io::AsyncWrite` gains defaulted `poll_write_vectored`
  (the first non-empty slice through `poll_write`) and `is_write_vectored`
  (`false`); existing writers compile unchanged (ADR 0006).

- **Worker pinning** (`WorkerPlacement::{Unbound, Pinned}`,
  `ExecutorConfig::worker_placement`, and `worker_placement(..)` on
  `MoiraiBuilder`, `ExecutorBuilder`, and `moirai_executor::ExecutorBuilder`).
  Under `Pinned`, worker `i` binds its own thread to the `i % n`-th logical
  processor of the detected topology (`themis::bind_current_thread`) before it
  takes work, and the per-worker NUMA table the steal tier reads is populated
  from that enforced assignment. Construction fails closed: a refused binding
  returns `ExecutorError::WorkerPlacementFailed` for the lowest-numbered
  failing worker after every started worker is joined, and an undetectable
  topology is `InvalidConfiguration`. The default, `Unbound`, claims nothing
  (ADR 0037).

- **Browser history** (`moirai_pal::wasm::WebHistory`, `HistoryListener`).
  A client-side router reads the current path, pushes or replaces history
  entries and hears back/forward navigation. Only printable same-origin
  paths up to `MAX_HISTORY_PATH_BYTES` are accepted: a leading `//`, a `\`
  or a scheme is refused, so an entry can never leave the page's origin,
  and dropping the listener removes it before its closure is released.

- **Native menu bars** (`NativeWindow::set_menu_bar`, `take_menu_commands`,
  `MenuBar`, `MenuCommand`, and the same methods on `WebViewHost`). A
  window attaches, replaces or removes a bar of up to `MAX_MENU_BAR_MENUS`
  titled `PopupMenu`s. Chosen items queue as menu and item indexes apart
  from `WindowEvent`; accelerator and control notifications, and commands
  from a replaced bar, are dropped. The window owns the attached menus and
  destroys them with itself.

- **Launch-at-login entries** (`moirai_pal::windows::startup`:
  `set_startup_command`, `remove_startup_command`, `startup_command`). One
  application's command line is written, read or removed as a named value
  under `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`, the Windows
  half of an autostart service; no other registry location is reachable.
  Names are at most 64 UTF-16 units without `\`, commands at most 2048, and
  neither may contain control characters.

- **Single-instance coordination** (`moirai_pal::instance::claim`,
  `InstanceName`, `PrimaryInstance`, `SecondaryInstance`). The first
  process to claim a name becomes primary; a later one forwards one
  length-framed message of at most `MAX_INSTANCE_MESSAGE_BYTES`, such as
  its arguments or deep link, and exits. On Unix the primary holds an
  advisory lock and listens on a socket in an owner-only directory under
  `XDG_RUNTIME_DIR` or `/tmp`; on Windows it owns the first instance of a
  session-scoped named pipe that rejects remote clients, and clients connect
  at identification level so the pipe cannot impersonate them. Either claim
  is released by the operating system when the primary exits.

- **Tray icon and notifications** (`NativeWindow::show_tray_icon`,
  `show_notification`, `remove_tray_icon`, `take_tray_events`,
  `TrayIconImage`, `TrayEvent`, and the same methods on `WebViewHost`). A
  window shows one notification-area icon from a 16 or 32 pixel ARGB image
  with a bounded tooltip. Clicks, keyboard selection, context-menu
  requests (with their screen anchor) and notification clicks queue apart
  from `WindowEvent`. Notifications are shell balloons, which Windows 10
  and later present as toasts without a packaged application identity.
  Closing the window removes the icon and destroys its handle.
  `NativeWindow::show_popup_menu` (and `WebViewHost::show_popup_menu`)
  opens a bounded `PopupMenu` of labelled, separator and disabled items at a
  screen position, such as a tray context request's anchor, and returns the
  chosen item's index.

- **System-wide hotkeys** (`NativeWindow::register_hotkey`,
  `unregister_hotkey`, `take_hotkey_presses`, `GlobalHotkey`, `HotkeyId`,
  and the same methods on `WebViewHost`). A chord registered through
  `RegisterHotKey` fires while other applications have focus. It must hold
  at least one modifier, auto-repeat is suppressed, and a window holds at
  most `MAX_GLOBAL_HOTKEYS`. Presses queue apart from `WindowEvent`, so
  existing exhaustive event matches are unaffected; at most
  `MAX_PENDING_HOTKEY_PRESSES` wait unread, and closing the window
  releases every registration.

- **Restorable window placement** (`NativeWindow::placement`,
  `NativeWindow::set_placement`, `WindowPlacement`). A window reports and
  accepts its restored outer rectangle and maximized state, the value
  desktop applications persist between sessions. A hidden window stays
  hidden and shows maximized when its placement asks for it; corners are
  bounded by `MAX_PLACEMENT_COORDINATE` and sizes by the frame limits.
  `WindowConfig::with_placement` restores a saved placement before the
  window first appears, and `WebViewHost::window_placement` reads the
  parent window's placement for saving.

- **Keyboard shortcuts in the Windows accessibility tree**
  (`AccessibilityNode::set_keyboard_shortcut`). A node's bounded
  `aria-keyshortcuts`-style value reaches AccessKit's `keyboard_shortcut`
  property, so UI Automation clients announce the accelerator that activates
  a command.

- **Process-tree containment on Linux, Android and Apple targets**
  (`ProcessSpec::tree_containment`). The child leads a new process group, so
  termination and terminate-on-drop signal every normally created
  descendant, including those left after the root exits. The root's exit is
  observed with `waitid(WNOWAIT)` and it is reaped only after the group is
  signalled, so the group ID cannot be reused in between. The Windows job
  regressions for descendant cleanup now run on these targets too.

- **Select values** (`WebElement::set_value`). A `<select>` now takes the
  value of one of its options; a value no option carries is rejected and the
  previous selection kept, rather than silently clearing the selection.

- **Region presents** (`NativeWindow::present_argb8888_region`, `FrameRegion`).
  A frame that differs from the retained one only inside a rectangle copies
  that rectangle's rows and invalidates only it, so a damage-limited software
  repaint no longer pays for the whole window: a 300×40 present with its
  paint measured 0.025 ms against 0.53 ms whole at 640×480, and 0.05 ms against
  1.32 ms at 1280×960. New dimensions or a stretched client area present whole.

- **WebView2 folder hosting** (`WebViewConfig::folder`). A local folder is
  served as `https://{host}/` through WebView2's virtual host mapping, so a
  page built for a web server (ES modules, `fetch`, WebAssembly streaming
  compilation) runs unchanged, which a `file:///` page cannot. The host must
  lie under `.example`, `.invalid`, `.localhost` or `.test` so it cannot
  shadow a real site, navigation is confined to it, and other origins are
  denied its resources.
  A canonicalized (verbatim `\\?\`) folder is mapped by its plain spelling,
  since WebView2 does not normalize a verbatim path when it appends a request
  path, which left files below the folder's top level unreachable.

- **`LockFreeQueue::len`.** The bounded MPMC queue reports its current depth as
  the difference of its `head` and `tail` cursors, best-effort in the same sense
  as `is_empty`/`is_full`. `UnifiedChannel::len` and its `current_length`
  statistic needed it once the channel moved onto this core, and the cursors
  already carried the value.

- **Unit-task ranges** (`for_each_unit_task_range_with`). The unit-task
  operators (ADR 0059) all hand out runs of a dense slice, so a pass whose
  units are not one contiguous buffer — a strided row walk, a tiled block
  pass, a reduction writing one output per axis index — had no way to ask for
  a task sized by the bytes a unit moves, and kept picking chunk sizes by
  hand. This runs a closure over consecutive runs of unit *indices* through
  the same planner the slice operators use, so both make the same decision for
  the same bytes. It owns no data and therefore carries no `unsafe`: the
  caller keeps the disjointness proof for what its indices address.

- **Root-confined native file opening.** `moirai_pal::fs::open_file_within_root`
  resolves native path components from a directory handle, rejects traversal
  and link components, and returns the handle opened under the selected root.
  WebAssembly reports unsupported so browser file entries remain on the DOM
  provider.

- **Worker idle hooks** (`register_idle_hook`, `run_idle_hooks`). Executor
  workers are long-lived, so thread-local state a job builds up — scratch
  buffers, plan caches, allocator arenas — stays resident for the process
  lifetime. Consumers can now register a plain function pointer to run on
  every worker thread at the one point that is both on the owning thread and
  quiescent by construction: the worker exhausted its spin budget with no work
  found and is about to block. The registry lock is held only to snapshot the
  pointers, never across a hook call, so hooks cannot deadlock against
  registration. On this mechanism the apollo FFT stack will release its
  ~7.2 MB high-water worker scratch at quiescence
  (`ATLAS-APOLLO-WORKER-RETENTION-2026-09-03`).

[Version 0.6.1 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)
