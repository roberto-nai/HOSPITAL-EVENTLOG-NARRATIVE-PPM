from pathlib import Path

def ensure_dir_with_gitkeep(dir_path: Path):
    """
    Ensure that the directory exists and contains a .gitkeep file.
    Args:
        dir_path (Path): The directory to create/check.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    gitkeep = dir_path / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()
        print(f"Created .gitkeep in {dir_path}")