"""Budget accounting of the local benchmark runner."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bench_suite  # noqa: E402


def python(source: str) -> list[str]:
    return [sys.executable, "-c", source]


def run(commands: dict[str, list[str]], budget: float) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with tempfile.TemporaryDirectory() as workdir, contextlib.redirect_stdout(
        out
    ), contextlib.redirect_stderr(err):
        status = bench_suite.run_suite(commands, budget, Path(workdir))
    return status, out.getvalue(), err.getvalue()


class BudgetTests(unittest.TestCase):
    def test_a_suite_inside_its_budget_reports_each_target_and_the_total(self):
        status, out, err = run({"a": python("pass"), "b": python("pass")}, 60.0)
        self.assertEqual(status, 0)
        self.assertEqual(err, "")
        lines = out.splitlines()
        self.assertTrue(lines[0].startswith("a ") and lines[0].endswith("ok"))
        self.assertTrue(lines[1].startswith("b ") and lines[1].endswith("ok"))
        self.assertTrue(lines[2].startswith("total") and "of 60 s" in lines[2])

    def test_a_target_that_outlives_the_budget_is_named_and_stops_the_suite(self):
        status, out, err = run(
            {
                "fast": python("pass"),
                "slow": python("import time; time.sleep(30)"),
                "zz_never_started": python("pass"),
            },
            1.0,
        )
        self.assertEqual(status, 1)
        self.assertIn("BUDGET BREACH: slow ", err)
        self.assertIn("slow", out)
        self.assertNotIn("zz_never_started", out)

    def test_a_failing_target_is_named(self):
        status, _, err = run({"broken": python("raise SystemExit(3)")}, 60.0)
        self.assertEqual(status, 1)
        self.assertIn("FAILED: broken ", err)

    def test_targets_share_one_remaining_budget(self):
        # Each target alone fits the budget; together they cannot.
        status, _, err = run(
            {
                "first": python("import time; time.sleep(1.2)"),
                "second": python("import time; time.sleep(30)"),
            },
            2.0,
        )
        self.assertEqual(status, 1)
        self.assertIn("BUDGET BREACH: second ", err)


if __name__ == "__main__":
    unittest.main()
