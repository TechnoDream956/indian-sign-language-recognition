"""
Unit tests for the centralized path resolution helper (src/core/paths.py).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.paths import (
    DATA_DIR,
    DOCS_DIR,
    MODELS_DIR,
    PACKAGING_DIR,
    PRODUCTION_LABEL_MAP,
    PRODUCTION_MODEL_PATH,
    PROJECT_ROOT,
    SRC_DIR,
    STATIC_DIR,
    TEMPLATES_DIR,
    TESTS_DIR,
    _find_project_root,
    _is_valid_project_root,
    get_project_root,
    get_static_dir,
    get_templates_dir,
    resolve_path,
    verify_project_structure,
)


class TestPathsHelper(unittest.TestCase):
    """Test suite verifying robust project root and resource path resolution."""

    def test_project_root_exists_and_is_directory(self):
        """PROJECT_ROOT must resolve to an existing directory."""
        self.assertTrue(PROJECT_ROOT.exists(), "PROJECT_ROOT does not exist on disk")
        self.assertTrue(PROJECT_ROOT.is_dir(), "PROJECT_ROOT is not a directory")

    def test_project_root_contains_markers(self):
        """PROJECT_ROOT must contain definitive anchor files."""
        self.assertTrue((PROJECT_ROOT / "requirements.txt").is_file(), "Missing requirements.txt at root")
        self.assertTrue((PROJECT_ROOT / "README.md").is_file(), "Missing README.md at root")

    def test_get_project_root_matches_constant(self):
        """get_project_root() must return identical path to PROJECT_ROOT constant."""
        self.assertEqual(get_project_root(), PROJECT_ROOT)

    def test_core_subdirectories_exist(self):
        """Core top-level directories must exist at root."""
        self.assertTrue(MODELS_DIR.is_dir(), f"MODELS_DIR does not exist: {MODELS_DIR}")
        self.assertTrue(SRC_DIR.is_dir(), f"SRC_DIR does not exist: {SRC_DIR}")
        self.assertTrue(DATA_DIR.is_dir(), f"DATA_DIR does not exist: {DATA_DIR}")
        self.assertTrue(DOCS_DIR.is_dir(), f"DOCS_DIR does not exist: {DOCS_DIR}")
        self.assertTrue(TESTS_DIR.is_dir(), f"TESTS_DIR does not exist: {TESTS_DIR}")
        self.assertTrue(PACKAGING_DIR.is_dir(), f"PACKAGING_DIR does not exist: {PACKAGING_DIR}")

    def test_production_model_and_label_map_paths(self):
        """Production model checkpoint and label map must exist at their canonical paths."""
        self.assertTrue(PRODUCTION_MODEL_PATH.is_file(), f"Production model not found at {PRODUCTION_MODEL_PATH}")
        self.assertTrue(PRODUCTION_LABEL_MAP.is_file(), f"Production label map not found at {PRODUCTION_LABEL_MAP}")

    def test_resolve_path_relative(self):
        """resolve_path() with relative string should resolve relative to PROJECT_ROOT."""
        resolved = resolve_path("models/alphabet_model_26class.h5")
        self.assertEqual(resolved, PRODUCTION_MODEL_PATH)
        self.assertTrue(resolved.is_file())

    def test_resolve_path_absolute(self):
        """resolve_path() with absolute Path should return equivalent resolved absolute Path."""
        abs_path = PRODUCTION_MODEL_PATH.resolve()
        resolved = resolve_path(abs_path)
        self.assertEqual(resolved, abs_path)

    def test_get_templates_and_static_dirs(self):
        """get_templates_dir() and get_static_dir() must return existing directories."""
        templates = get_templates_dir()
        static = get_static_dir()
        self.assertTrue(templates.is_dir(), f"Templates dir does not exist: {templates}")
        self.assertTrue(static.is_dir(), f"Static dir does not exist: {static}")
        self.assertTrue((templates / "index.html").is_file(), "index.html missing from templates dir")
        self.assertTrue((static / "app.js").is_file(), "app.js missing from static dir")
        self.assertTrue((static / "style.css").is_file(), "style.css missing from static dir")

    def test_verify_project_structure(self):
        """verify_project_structure() must return True for all core components."""
        status = verify_project_structure()
        for key, value in status.items():
            self.assertTrue(value, f"verify_project_structure failed on: {key}")

    def test_deep_start_path_resolution(self):
        """Root resolver must successfully find root from deep subpaths."""
        deep_path = PROJECT_ROOT / "src" / "core" / "paths.py"
        found = _find_project_root(start_path=deep_path)
        self.assertEqual(found, PROJECT_ROOT)

    def test_environment_override(self):
        """Resolver must respect ISL_PROJECT_ROOT environment override if valid."""
        test_dir = PROJECT_ROOT
        old_env = os.environ.get("ISL_PROJECT_ROOT")
        try:
            os.environ["ISL_PROJECT_ROOT"] = str(test_dir)
            found = _find_project_root()
            self.assertEqual(found, test_dir)
        finally:
            if old_env is None:
                os.environ.pop("ISL_PROJECT_ROOT", None)
            else:
                os.environ["ISL_PROJECT_ROOT"] = old_env

    def test_mocked_pyinstaller_frozen_with_valid_meipass(self):
        """Mock PyInstaller frozen environment with bundled models and src in _MEIPASS."""
        with tempfile.TemporaryDirectory() as fake_meipass:
            p = Path(fake_meipass).resolve()
            (p / "models").mkdir()
            (p / "src").mkdir()
            with patch.object(sys, "frozen", True, create=True), patch.object(sys, "_MEIPASS", str(p), create=True):
                resolved = _find_project_root()
                self.assertEqual(resolved, p)

    def test_mocked_pyinstaller_frozen_without_meipass(self):
        """Mock PyInstaller frozen environment where _MEIPASS is None, falling back to executable ancestor."""
        with tempfile.TemporaryDirectory() as fake_root:
            root = Path(fake_root).resolve()
            (root / "requirements.txt").touch()
            (root / "README.md").touch()
            fake_exe = root / "dist" / "bin" / "app_binary"
            fake_exe.parent.mkdir(parents=True)
            fake_exe.touch()
            with patch.object(sys, "frozen", True, create=True), patch.object(sys, "_MEIPASS", None, create=True), patch.object(sys, "executable", str(fake_exe)):
                resolved = _find_project_root()
                self.assertEqual(resolved, root)

    def test_environment_override_invalid(self):
        """Resolver must ignore invalid non-existent ISL_PROJECT_ROOT paths."""
        with patch.dict(os.environ, {"ISL_PROJECT_ROOT": "/non/existent/directory/path/that/does/not/exist"}):
            self.assertEqual(_find_project_root(), PROJECT_ROOT)

    def test_environment_override_empty(self):
        """Resolver must ignore empty ISL_PROJECT_ROOT string."""
        with patch.dict(os.environ, {"ISL_PROJECT_ROOT": ""}):
            self.assertEqual(_find_project_root(), PROJECT_ROOT)

    def test_environment_override_with_spaces(self):
        """Resolver must correctly preserve and resolve paths containing spaces."""
        with tempfile.TemporaryDirectory(prefix="isl project with spaces ") as space_dir:
            sp = Path(space_dir).resolve()
            with patch.dict(os.environ, {"ISL_PROJECT_ROOT": str(sp)}):
                self.assertEqual(_find_project_root(), sp)

    def test_unrelated_cwd_with_only_readme(self):
        """An unrelated directory with only README.md must never be accepted as project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = Path(tmpdir).resolve()
            (tp / "README.md").touch()
            self.assertFalse(_is_valid_project_root(tp))
            old_cwd = os.getcwd()
            try:
                os.chdir(str(tp))
                self.assertEqual(_find_project_root(tp), PROJECT_ROOT)
            finally:
                os.chdir(old_cwd)

    def test_unrelated_cwd_with_only_requirements(self):
        """An unrelated directory with only requirements.txt must never be accepted as project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = Path(tmpdir).resolve()
            (tp / "requirements.txt").touch()
            self.assertFalse(_is_valid_project_root(tp))
            old_cwd = os.getcwd()
            try:
                os.chdir(str(tp))
                self.assertEqual(_find_project_root(tp), PROJECT_ROOT)
            finally:
                os.chdir(old_cwd)

    def test_modular_template_and_static_fallback_mocked(self):
        """Test get_templates_dir() and get_static_dir() fallback in temporary directory without workspace mutations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_src = Path(tmpdir).resolve() / "src"
            (mock_src / "templates").mkdir(parents=True)
            (mock_src / "static").mkdir(parents=True)
            (mock_src / "web" / "templates").mkdir(parents=True)
            (mock_src / "web" / "static").mkdir(parents=True)
            
            with patch("src.core.paths.SRC_DIR", mock_src):
                self.assertEqual(get_templates_dir(), mock_src / "web" / "templates")
                self.assertEqual(get_static_dir(), mock_src / "web" / "static")
                
            (mock_src / "web" / "templates").rmdir()
            (mock_src / "web" / "static").rmdir()
            (mock_src / "web").rmdir()
            with patch("src.core.paths.SRC_DIR", mock_src):
                self.assertEqual(get_templates_dir(), mock_src / "templates")
                self.assertEqual(get_static_dir(), mock_src / "static")


if __name__ == "__main__":
    unittest.main()

