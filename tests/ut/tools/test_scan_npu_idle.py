#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Regression tests for tools/scan_npu_idle.sh npu-smi info parsing."""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tools" / "scan_npu_idle.sh"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _run(args, check=True):
    return subprocess.run(
        [str(SCRIPT), *args],
        check=check,
        capture_output=True,
        text=True,
    )


class TestScanNpuIdle(unittest.TestCase):

    def test_self_test_covers_a5_and_legacy_formats(self):
        result = _run(["--self-test"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("A5 all idle", result.stdout)
        self.assertIn("self-test passed", result.stdout)

    def test_parse_a5_idle_fixture_reports_eight_free_npus(self):
        fixture = FIXTURES / "npu_smi_a5_idle.txt"
        result = _run(["--parse-file", str(fixture)])
        self.assertEqual(result.returncode, 0)
        self.assertIn("[OK]", result.stdout)
        self.assertIn("共 8 个 NPU", result.stdout)
        self.assertIn("无进程", result.stdout)
        self.assertNotIn("[BUSY]", result.stdout)

    def test_parse_a5_partial_busy_fixture_lists_busy_npu(self):
        fixture = FIXTURES / "npu_smi_a5_partial_busy.txt"
        result = _run(["--parse-file", str(fixture)])
        self.assertEqual(result.returncode, 0)
        self.assertIn("[BUSY]", result.stdout)
        self.assertIn("NPU0", result.stdout)
        self.assertIn("1 张卡", result.stdout)

    def test_missing_parse_file_fails(self):
        result = _run(["--parse-file", str(FIXTURES / "missing.txt")],
                      check=False)
        self.assertNotEqual(result.returncode, 0)
        combined = result.stdout + result.stderr
        self.assertIn("文件不存在", combined)


if __name__ == "__main__":
    unittest.main()
