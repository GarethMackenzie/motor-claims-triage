"""Repository hygiene checks for accidental local or sensitive content."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".json", ".csv", ".txt", ".ipynb"}
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*['\"][^'\"]+"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]
SLASH = "/"
BACKSLASH = "\\"
LOCAL_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:" + re.escape(BACKSLASH) + "Users" + re.escape(BACKSLASH)),
    re.compile(re.escape(SLASH) + "Users" + re.escape(SLASH) + r"[^/]+/"),
    re.compile(re.escape(SLASH) + "home" + re.escape(SLASH) + r"[^/]+/"),
]


def repository_text() -> str:
    parts: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in {".git", ".venv", "venv", "__pycache__"} for part in path.parts):
            continue
        parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def test_repository_contains_no_obvious_credentials() -> None:
    content = repository_text()
    assert not any(pattern.search(content) for pattern in SECRET_PATTERNS)


def test_repository_contains_no_absolute_user_paths() -> None:
    content = repository_text()
    assert not any(pattern.search(content) for pattern in LOCAL_PATH_PATTERNS)
