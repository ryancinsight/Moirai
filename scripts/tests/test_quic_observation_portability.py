"""Patch inventory and bytes must not depend on host Git/path conventions."""
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quic_observation.prepare import patch_source, registry_source, relative_patch_path


class ObservationPortability(unittest.TestCase):
    def test_patch_inventory_uses_posix_paths_on_both_platforms(self):
        for path_type, root in [(PureWindowsPath, "C:/registry/quinn-proto"),
                                (PurePosixPath, "/registry/quinn-proto")]:
            with self.subTest(platform=path_type.__name__):
                base = path_type(root)
                self.assertEqual(relative_patch_path(base / "src/connection/mod.rs", base),
                                 "src/connection/mod.rs")

    def test_real_patch_bytes_ignore_inherited_autocrlf(self):
        root = Path(__file__).resolve().parents[2]
        source, _ = registry_source(root)
        reports = []
        for setting in ("false", "true", "input"):
            with self.subTest(autocrlf=setting), tempfile.TemporaryDirectory() as temporary:
                # Process-local configuration only; never write the user's Git config.
                environment = dict(os.environ, GIT_CONFIG_COUNT="1",
                                   GIT_CONFIG_KEY_0="core.autocrlf",
                                   GIT_CONFIG_VALUE_0=setting)
                with patch.dict(os.environ, environment, clear=True):
                    destination = Path(temporary) / "engine"
                    report = patch_source(source, destination, root)
                reports.append(report)
        self.assertEqual(reports[1], reports[0])
        self.assertEqual(reports[2], reports[0])


if __name__ == "__main__":
    unittest.main()
