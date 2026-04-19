from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        return (REPO_ROOT / path).resolve(strict=False)

    resolved = path.resolve(strict=False)
    try:
        if resolved.exists():
            return resolved
    except OSError:
        pass

    for anchor in ("data", "_output", "_figures", "fig"):
        if anchor in resolved.parts:
            suffix = Path(*resolved.parts[resolved.parts.index(anchor) :])
            return (REPO_ROOT / suffix).resolve(strict=False)
    return resolved


def repo_relative_path(path_value: str | Path) -> str:
    return (
        (REPO_ROOT / Path(path_value))
        .resolve(strict=False)
        .relative_to(REPO_ROOT.resolve(strict=False))
        .as_posix()
    )
