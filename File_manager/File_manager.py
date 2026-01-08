from pathlib import Path


class FileManager:
    """Simple file manager for saving text files."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)

    def save(self, filename: str, content: str, encoding: str = "utf-8", append: bool = False) -> Path:
        """
        Save text content to file.

        Args:
            filename: File path relative to base_dir
            content: Text content to save
            encoding: File encoding
            append: If True, append to file instead of overwriting

        Returns:
            Full path to saved file
        """
        filepath = self.base_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        with open(filepath, mode, encoding=encoding) as f:
            f.write(content)

        return filepath

    def read(self, filename: str, encoding: str = "utf-8") -> str:
        """Read text file content."""
        return (self.base_dir / filename).read_text(encoding=encoding)

    def exists(self, filename: str) -> bool:
        """Check if file exists."""
        return (self.base_dir / filename).exists()


if __name__ == "__main__":
    fm = FileManager("output")

    fm.save("log.txt", "Line 1\n")
    fm.save("log.txt", "Line 2\n", append=True)
    fm.save("log.txt", "Line 3\n", append=True)

    print(fm.read("log.txt"))
    # Line 1
    # Line 2
    # Line 3