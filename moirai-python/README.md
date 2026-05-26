# moirai-python

`moirai-python` provides PyO3 bindings over the Rust `moirai` crate. The
importable package does not implement its own scheduler, chunk planner, workload
kernels, benchmark harness, or execution backend; it exposes a native
`moirai::Moirai` runtime lifecycle facade.

## Contracts

- `MoiraiPython` wraps a native `moirai::Moirai` runtime.
- Worker count validation occurs at construction.
- `worker_count`, `has_work`, `join`, and `shutdown` forward to the wrapped
  runtime.
- Workload functions and benchmark-specific processing helpers are excluded
  from the Python package. Comparative performance coverage belongs in
  scheduler-level Rust benchmarks or external harnesses that do not expand the
  public Python API.

## Usage

```bash
py -3.13 -m pip install -e moirai-python
py -3.13 -m unittest discover moirai-python\tests
```
