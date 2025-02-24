"""Command line interface: ``loyalty-spend <command>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

Handler = Callable[[argparse.Namespace], int]
_COMMANDS: dict[
    str, tuple[str, Callable[[argparse.ArgumentParser], None], Handler]
] = {}


def command(name: str, help: str, arguments: Callable[[argparse.ArgumentParser], None]):
    """Register a subcommand with its argument builder."""

    def decorator(handler: Handler) -> Handler:
        _COMMANDS[name] = (help, arguments, handler)
        return handler

    return decorator


def _config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="YAML configuration file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loyalty-spend")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, arguments, handler) in _COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        arguments(p)
        p.set_defaults(func=handler)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
