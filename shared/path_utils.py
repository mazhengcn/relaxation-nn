from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path_value: str | Path) -> Path:
    return (REPO_ROOT / Path(path_value)).resolve(strict=False)


def repo_relative_path(path_value: str | Path) -> str:
    return (
        (REPO_ROOT / Path(path_value))
        .resolve(strict=False)
        .relative_to(REPO_ROOT.resolve(strict=False))
        .as_posix()
    )
