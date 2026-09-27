"""Startup credential checks for the entry points.

The engine reaches two external services, each gated by an environment
variable. Without them the underlying clients raise deep, cryptic
validation errors; these helpers turn a missing key into one clear,
actionable message instead.
"""

import os

HELP = {
    "GOOGLE_API_KEY": "https://aistudio.google.com/app/apikey",
    "HUGGINGFACEHUB_API_TOKEN": "https://huggingface.co/settings/tokens",
}


def missing(*names: str) -> list[str]:
    """Return the names whose environment variable is unset or empty."""
    return [name for name in names if not os.environ.get(name)]


def describe(names: list[str]) -> str:
    lines = ["Missing required environment variable(s); add them to your .env file:"]
    for name in names:
        hint = HELP.get(name)
        lines.append(f"  - {name}" + (f"  ({hint})" if hint else ""))
    return "\n".join(lines)


def require(*names: str) -> None:
    """Exit with a clear message if any required variable is missing.

    Raises SystemExit so a command-line entry point stops with a readable
    message and a non-zero status, rather than a library traceback.
    """
    absent = missing(*names)
    if absent:
        raise SystemExit(describe(absent))
