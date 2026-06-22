#!/usr/bin/env python3
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BuildConfigurationTest(unittest.TestCase):
    def test_root_cmake_does_not_force_debug_or_o0(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()

        self.assertNotRegex(
            cmake,
            r"SET\s*\(\s*CMAKE_BUILD_TYPE\s+\"Debug\"\s*\)",
        )
        self.assertNotRegex(
            cmake,
            r"set\s*\(\s*CMAKE_CXX_FLAGS\s+\"[^\"]*-O0[^\"]*\"\s*\)",
        )

    def test_roptlib_external_project_applies_local_compatibility_patch(self):
        roptlib_cmake = (ROOT / "cmake" / "roptlib.cmake").read_text()

        self.assertIn("PATCH_COMMAND", roptlib_cmake)
        self.assertIn("patch_roptlib.cmake", roptlib_cmake)
        self.assertTrue((ROOT / "cmake" / "patch_roptlib.cmake").exists())


if __name__ == "__main__":
    unittest.main()
