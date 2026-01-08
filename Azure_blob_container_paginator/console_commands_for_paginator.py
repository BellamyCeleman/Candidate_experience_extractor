import argparse
from typing import Any, Optional


class ConsoleArgs:
    """Configurable command line argument parser."""

    def __init__(self, description: str = "", epilog: str = ""):
        self._parser = argparse.ArgumentParser(
            description=description,
            epilog=epilog,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        self._args: Optional[argparse.Namespace] = None

    def add(
        self,
        name: str,
        short: str = None,
        type: type = str,
        default: Any = None,
        help: str = "",
        flag: bool = False
    ) -> "ConsoleArgs":
        """
        Add an argument.

        Args:
            name: Argument name (e.g. "page-size" → --page-size, access as args.page_size)
            short: Short flag (e.g. "p" → -p)
            type: Value type (int, str, float)
            default: Default value
            help: Help text
            flag: If True, argument is a boolean flag (store_true)

        Returns:
            self for chaining
        """
        names = [f"--{name}"]
        if short:
            names.insert(0, f"-{short}")

        if flag:
            self._parser.add_argument(*names, action="store_true", default=default or False, help=help)
        else:
            self._parser.add_argument(*names, type=type, default=default, help=help)

        return self

    def parse(self) -> "ConsoleArgs":
        """Parse arguments from command line."""
        self._args = self._parser.parse_args()
        return self

    def __getattr__(self, name: str) -> Any:
        """Access parsed arguments as attributes."""
        if name.startswith("_"):
            return super().__getattribute__(name)
        if self._args is None:
            raise RuntimeError("Call parse() first")
        return getattr(self._args, name.replace("-", "_"))

    def __repr__(self) -> str:
        if self._args is None:
            return "ConsoleArgs(not parsed)"
        return f"ConsoleArgs({vars(self._args)})"


if __name__ == "__main__":
    args = (
        ConsoleArgs(description="Resume processing tool")
        .add("page-size", short="s", type=int, default=5, help="Resumes per page")
        .add("start-page", type=int, default=0, help="Start page")
        .add("end-page", type=int, default=3, help="End page")
        .add("test-files", short="t", flag=True, help="Enable PII verification")
        .add("token", help="Continuation token")
        .parse()
    )

    print(args)
    print(f"Pages: {args.start_page} - {args.end_page}")
    print(f"Test mode: {args.test_files}")