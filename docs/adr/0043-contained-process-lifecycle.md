# ADR 0043: Contained process lifecycle

Status: Accepted

Date: 2026-09-05

Driver: [MOI-PROCESS-2026-09-05](../backlog.md#MOI-PROCESS-2026-09-05).

## Contract and ownership

ADR 0008 places operating-system process lifecycle in moirai-transport.
A downstream Metis session requires owned stdin/stdout pipes, a cleared
child environment, deadline-aware cleanup, and containment of descendants
that could otherwise hold IPC handles open. These mechanisms belong here;
Metis retains only its protocol handler and session policy.

Windows creation uses CreateProcessW with STARTUPINFOEX, an explicit
inherited HANDLE_LIST, and JOB_LIST assigning the child before execution.
KILL_ON_JOB_CLOSE applies to terminate-on-drop jobs. Assigning a job after
ordinary spawn is rejected because the child could spawn descendants first.
No breakaway flags are enabled. Detach-on-drop jobs omit kill-on-close.

Explicit waits and termination use finite deadlines. Drop closes owned
Windows handles without waiting. Portable process waits use bounded polling;
non-Windows implementations reject requested tree containment until they own
an equivalent primitive. A deadline failure is surfaced, never treated as
successful cleanup. Pipes remain caller-owned and need their own I/O policy.

Explicit Windows termination confirms that the job has zero active processes,
including when its original process has already exited. Portable Drop retains
only a best-effort direct-child kill request and never waits for reaping.

## Migration

This is a major public-contract change: `ProcessStatus` now preserves the
platform `ExitStatus` in a private field. Callers obtain statuses from lifecycle
operations instead of constructing struct literals, and inspect the platform
result with `exit_status()`. The existing `wait()` and `terminate()` entry
points use 30-second and one-second budgets; callers needing different finite
budgets use `wait_timeout()` and `terminate_timeout()`. Exhaustive error matches
must handle the new deadline, invalid-specification, containment, and OS cases.

## Threat boundary and limits

The job contains normally created descendants and narrows inherited handles.
It does not reduce the child's token privileges, filesystem/network access,
or ability to attack another same-user process. It is lifecycle containment,
not a security sandbox. OS calls and kernel termination are trusted system
boundaries; no synchronous Rust handler can be forcibly preempted safely.

## Verification

Native tests exercise actual processes, argument quoting, inherited pipe
communication, successful exit, deadline expiration, and job termination.
ABI layouts are const-checked on the Windows host. Windows FFI is not Miri
coverage; operating-system integration tests provide behavioral evidence.

## References

- [CreateProcessW](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessw), extended startup and mutable command line.
- [UpdateProcThreadAttribute](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-updateprocthreadattribute), JOB_LIST and HANDLE_LIST.
- [Job basic limits](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_limit_information), KILL_ON_JOB_CLOSE and breakaway.
- [WaitForSingleObject](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitforsingleobject), finite wait and signaled process handles.
- [C argument parsing](https://learn.microsoft.com/en-us/cpp/c-language/parsing-c-command-line-arguments), backslash and quote rules.
