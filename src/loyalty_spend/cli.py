"""Command line interface: ``loyalty-spend <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence

from loyalty_spend.config import load_config

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


def _clean_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument("--input", default=None, help="raw CSV, defaults to loyalty.csv")
    p.add_argument("--output", required=True)


@command("clean", "clean a raw extract to the data dictionary", _clean_args)
def _clean(args: argparse.Namespace) -> int:
    from loyalty_spend.data.io import load_clean

    cfg = load_config(args.config)
    path = args.input or cfg.paths.raw(cfg.paths.loyalty_file)
    clean, report = load_clean(path)
    clean.to_csv(args.output, index=False)
    print(report.to_frame().to_string(index=False))
    print(f"wrote {len(clean)} rows to {args.output}")
    return 0


def _drivers_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument("--group", default="loyalty_years")


@command("drivers", "test whether a feature drives spend", _drivers_args)
def _drivers(args: argparse.Namespace) -> int:
    from loyalty_spend.analysis.drivers import effect_sizes, spend_by_group, welch_anova
    from loyalty_spend.data.io import load_clean

    cfg = load_config(args.config)
    clean, _ = load_clean(cfg.paths.raw(cfg.paths.loyalty_file), impute=False)
    print(spend_by_group(clean, args.group).round(2).to_string(index=False))
    print(
        json.dumps(
            {**welch_anova(clean, args.group), **effect_sizes(clean, args.group)}
        )
    )
    return 0


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
