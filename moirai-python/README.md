# moirai-python

A Python handle on the [Moirai](https://github.com/ryancinsight/Moirai) Rust
concurrency runtime: construct it, ask what it is doing, wait for it, shut it
down.

## Install

```sh
pip install moirai-python
```

Wheels are built for CPython 3.10 through 3.13 on Linux, Windows, and macOS.
There are no runtime dependencies.

## Use

```python
from moirai_python import MoiraiPython

runtime = MoiraiPython(workers=4)

print(runtime.worker_count())   # 4
print(runtime.has_work())       # False

runtime.join()                  # wait for queued and active work
runtime.shutdown()
```

The worker count is validated when the runtime is constructed, so an invalid
value fails there rather than at first use.

## What this package deliberately does not do

It does not submit work. There is no task-submission API, no chunk planner, no
workload kernel, and no benchmark harness on the Python side — the facade
covers the runtime *lifecycle* and nothing else.

That is a boundary, not an omission in progress. Scheduling decisions belong
where the scheduler is, and a Python-side planner would be a second
implementation of policy that Rust already owns — divergent the moment either
changed. Comparative performance work belongs in Moirai's own Rust benchmarks
for the same reason: a harness measuring through a binding measures the
binding.

So `has_work()` reports on work submitted from the Rust side. Called on a
runtime this package constructed and never fed, it returns `False`.

## Typing

The package ships `py.typed` and is fully annotated, so `mypy` and IDE
completion see the surface without stubs.

## Links

- [Source and issues](https://github.com/ryancinsight/Moirai)

## Licence

MIT or Apache-2.0, at your option.
