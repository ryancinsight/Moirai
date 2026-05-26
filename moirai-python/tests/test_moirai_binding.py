import unittest

from moirai_python.api.runtime import MoiraiPython


class MoiraiBindingTests(unittest.TestCase):
    def test_facade_exposes_native_runtime_lifecycle(self) -> None:
        runtime = MoiraiPython(workers=2)
        self.assertEqual(runtime.worker_count(), 2)
        self.assertFalse(runtime.has_work())
        runtime.join()
        runtime.shutdown()

    def test_invalid_worker_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MoiraiPython(0)


if __name__ == "__main__":
    unittest.main()
