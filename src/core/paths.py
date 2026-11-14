"""
Centralized filesystem path resolution for the Indian Sign Language Recognition project.

This module provides a robust, marker-based project root resolver that locates the
repository root regardless of current working directory, directory nesting depth,
or packaging mode (standard Python execution vs PyInstaller frozen bundle).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# Markers used to unambiguously identify the project root directory
PROJECT_ROOT_MARKERS: List[str] = [
    "requirements.txt",
    "README.md",
    ".git",
]


def _is_valid_project_root(path: Path) -> bool:
    """
    Check if a candidate directory satisfies compound project root marker requirements.
    
    A directory is accepted as project root ONLY if it satisfies at least one compound condition:
    1. Both 'requirements.txt' AND 'README.md' exist as files.
    2. Both '.git' (version control anchor) AND 'src/' (source tree) exist.
    """
    try:
        resolved = path.resolve()
        if not resolved.is_dir():
            return False
        if (resolved / "requirements.txt").is_file() and (resolved / "README.md").is_file():
            return True
        if (resolved / ".git").exists() and (resolved / "src").is_dir():
            return True
    except (OSError, PermissionError):
        return False
    return False


def _find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Search upward from start_path to find the directory containing project root markers.
    
    If running inside a PyInstaller frozen application bundle, resolves to the unpacked
    bundle directory or workspace root appropriately.
    """
    # 1. Handle PyInstaller frozen application
    if getattr(sys, "frozen", False):
        # In PyInstaller, sys._MEIPASS holds the extracted bundle root
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass and Path(meipass).exists():
            # Check if meipass contains markers or standard project folders
            meipass_path = Path(meipass).resolve()
            if _is_valid_project_root(meipass_path):
                return meipass_path
            if (meipass_path / "models").exists() and (meipass_path / "src").exists():
                return meipass_path
        # Fallback to executable location
        executable_dir = Path(sys.executable).resolve().parent
        for candidate in [executable_dir, executable_dir.parent, executable_dir.parent.parent]:
            if _is_valid_project_root(candidate):
                return candidate

    # 2. Handle environment variable override if specified
    env_root = os.environ.get("ISL_PROJECT_ROOT")
    if env_root and Path(env_root).resolve().exists():
        return Path(env_root).resolve()

    # 3. Standard upward traversal starting from current file location
    current = (start_path or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    # Search upward through all directory ancestors
    for candidate in [current, *current.parents]:
        if _is_valid_project_root(candidate):
            return candidate

    # 4. Fallback to current working directory ONLY if it satisfies compound root markers
    cwd = Path.cwd().resolve()
    if _is_valid_project_root(cwd):
        return cwd

    # 5. Last-resort fallback: three levels up from src/core/paths.py
    return Path(__file__).resolve().parent.parent.parent


# Immutable project-wide root path
PROJECT_ROOT: Path = _find_project_root()

# Core top-level directories
MODELS_DIR: Path = PROJECT_ROOT / "models"
DATA_DIR: Path = PROJECT_ROOT / "data"
SRC_DIR: Path = PROJECT_ROOT / "src"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
PACKAGING_DIR: Path = PROJECT_ROOT / "packaging"
RESEARCH_DIR: Path = PROJECT_ROOT / "research"
TOOLS_DIR: Path = PROJECT_ROOT / "tools"

# Canonical production model and label map paths
PRODUCTION_MODEL_PATH: Path = MODELS_DIR / "alphabet_model_26class.h5"
PRODUCTION_LABEL_MAP: Path = MODELS_DIR / "alphabet_label_map.json"
PRODUCTION_MODEL_SUMMARY: Path = MODELS_DIR / "alphabet_model_26class_summary.txt"

# Canonical template and static asset locations (with backward-compatible fallbacks)
TEMPLATES_DIR: Path = SRC_DIR / "templates"
STATIC_DIR: Path = SRC_DIR / "static"


def get_project_root() -> Path:
    """Return the resolved absolute project root path."""
    return PROJECT_ROOT


def resolve_path(relative_or_absolute_path: Union[str, Path]) -> Path:
    """
    Resolve a path relative to the project root if it is not already absolute.
    
    Args:
        relative_or_absolute_path: Relative subpath or absolute path.
        
    Returns:
        Resolved absolute Path.
    """
    path_obj = Path(relative_or_absolute_path)
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (PROJECT_ROOT / path_obj).resolve()


def get_templates_dir() -> Path:
    """
    Return the active templates directory.
    Checks `src/web/templates` first (modular structure), then falls back to `src/templates`.
    """
    modular_templates = SRC_DIR / "web" / "templates"
    if modular_templates.is_dir():
        return modular_templates
    return SRC_DIR / "templates"


def get_static_dir() -> Path:
    """
    Return the active static assets directory.
    Checks `src/web/static` first (modular structure), then falls back to `src/static`.
    """
    modular_static = SRC_DIR / "web" / "static"
    if modular_static.is_dir():
        return modular_static
    return SRC_DIR / "static"


def verify_project_structure() -> Dict[str, bool]:
    """
    Sanity check verifying that critical project anchor files and directories exist.
    
    Returns:
        Dictionary mapping path descriptions to existence status booleans.
    """
    return {
        "project_root": PROJECT_ROOT.is_dir(),
        "models_dir": MODELS_DIR.is_dir(),
        "production_model": PRODUCTION_MODEL_PATH.is_file(),
        "production_label_map": PRODUCTION_LABEL_MAP.is_file(),
        "data_dir": DATA_DIR.is_dir(),
        "src_dir": SRC_DIR.is_dir(),
        "templates_dir": get_templates_dir().is_dir(),
        "static_dir": get_static_dir().is_dir(),
        "tests_dir": TESTS_DIR.is_dir(),
    }

