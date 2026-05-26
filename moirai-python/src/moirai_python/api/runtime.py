"""Python facade over the PyO3 wrapper for `moirai::Moirai`."""

from moirai_python._native import Runtime as _NativeRuntime


class MoiraiPython:
    """Facade over the native `moirai::Moirai` runtime."""

    def __init__(self, workers: int) -> None:
        self._runtime = _NativeRuntime(workers)

    def worker_count(self) -> int:
        """Return the wrapped Moirai runtime's worker count."""

        return int(self._runtime.worker_count())

    def has_work(self) -> bool:
        """Return whether wrapped Moirai work is queued or active."""

        return bool(self._runtime.has_work())

    def join(self) -> None:
        """Wait for queued and active wrapped Moirai work to complete."""

        self._runtime.join()

    def shutdown(self) -> None:
        """Shut down the wrapped Moirai runtime."""

        self._runtime.shutdown()
