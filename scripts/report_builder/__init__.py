"""Report builder package for ISL College Project Report."""

from .diagram_generator import generate_all_diagrams
from .xml_helpers import enable_update_fields_on_open
from .cover_page import build_cover_page
from .front_matter import build_front_matter
from .chapter1_introduction import build_chapter1
from .chapter2_background import build_chapter2
from .chapter3_methodology import build_chapter3
from .chapter4_conclusion import build_chapter4
from .chapter5_references import build_chapter5
from .todo_extractor import extract_todos

__all__ = [
    "generate_all_diagrams",
    "enable_update_fields_on_open",
    "build_cover_page",
    "build_front_matter",
    "build_chapter1",
    "build_chapter2",
    "build_chapter3",
    "build_chapter4",
    "build_chapter5",
    "extract_todos"
]
