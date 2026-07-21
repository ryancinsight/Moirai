# moirai-python

`moirai-python` provides PyO3 bindings over the Rust `moirai` crate. The
importable package does not implement its own scheduler, chunk planner, workload
kernels, benchmark harness, or execution backend; it exposes a native
`moirai::Moirai` runtime lifecycle facade.

## Releases

GitHub Releases tagged `moirai-python-v<version>` build locked Linux, Windows,
and universal macOS wheels for CPython 3.10 through 3.13. The release workflow
installs and exercises each wheel, validates its distribution name and version,
generates SHA-256 checksums and build provenance, attaches those artifacts to
the GitHub Release, and publishes the same wheels to PyPI through OIDC Trusted
Publishing. The tag version must equal the `moirai-python` Cargo package
version; Cargo is the sole version source.

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
py -3.13 -m pip install moirai-python

# Source checkout verification
py -3.13 -m pip install -e moirai-python
py -3.13 -m unittest discover moirai-python\tests
```
