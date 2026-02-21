"""CSV and JSON export functionality."""

import csv
import io
import json
import sys


def export_data(
    data: list[dict] | dict, fmt: str, output: str | None = None
) -> None:
    """Export data as CSV or JSON.

    Args:
        data: List of dicts (or single dict) to export.
        fmt: 'csv' or 'json'.
        output: File path, or None for stdout.
    """
    if isinstance(data, dict):
        data = [data]

    if fmt == "json":
        text = json.dumps(data, indent=2, default=str)
    elif fmt == "csv":
        if not data:
            text = ""
        else:
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
            text = buf.getvalue()
    else:
        print(f"Error: Unknown export format '{fmt}'")
        return

    if output:
        with open(output, "w") as f:
            f.write(text)
        print(f"Exported {len(data)} records to {output}")
    else:
        sys.stdout.write(text)
