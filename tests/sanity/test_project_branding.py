from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("graph" + "gpo",)
SKIP_DIRS = {".git", ".venv", "__pycache__"}
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def iter_repo_paths():
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def test_repository_does_not_reference_old_project_name():
    hits = []
    for path in iter_repo_paths():
        rel_path = path.relative_to(ROOT).as_posix()
        lower_path = rel_path.lower()
        for term in FORBIDDEN:
            if term in lower_path:
                hits.append(rel_path)

        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for term in FORBIDDEN:
                if term in text:
                    hits.append(rel_path)
                    break

    assert hits == []
