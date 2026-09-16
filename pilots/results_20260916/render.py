"""Render an existing-results bundle as a self-contained, offline HTML report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_TEMPLATE = Path(__file__).with_name("dashboard.html")
_PLACEHOLDER = "__RESULTS_JSON__"


def render(data: dict, output: Path) -> None:
    """Embed schema-v1 results without computing or changing their statistics."""
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Expected a results object with schema_version: 1.")
    if not isinstance(data.get("studies"), dict):
        raise ValueError("The results object must contain a studies object.")

    output = Path(output)
    if output.resolve() in {_TEMPLATE.resolve(), Path(__file__).resolve()}:
        raise ValueError("The output must not overwrite a dashboard source file.")

    template = _TEMPLATE.read_text(encoding="utf-8")
    if template.count(_PLACEHOLDER) != 1:
        raise ValueError("The dashboard template must contain exactly one JSON placeholder.")

    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # A JSON string must not be able to close its HTML script element.
    for character, escaped in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        payload = payload.replace(character, escaped)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace(_PLACEHOLDER, payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Collected schema-v1 JSON")
    parser.add_argument("--out", required=True, type=Path, help="Standalone HTML destination")
    args = parser.parse_args()
    if args.data.resolve() == args.out.resolve():
        parser.error("--out must not overwrite --data.")
    try:
        data = json.loads(args.data.read_text(encoding="utf-8"))
        render(data, args.out)
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"Could not render results: {exc}\n")
    print(f"Offline dashboard written to {args.out}")


if __name__ == "__main__":
    main()
