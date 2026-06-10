"""Generate a static local dashboard from run records."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

from src.common.experiment_registry import (
    RunRecord,
    collect_run_records,
    format_run_score,
    rank_run_records,
)


def _score(record: RunRecord) -> str:
    return format_run_score(record)


def _commands(record: RunRecord) -> tuple[str, str]:
    checkpoint = record.artifacts.get("best_checkpoint_path") or "<checkpoint>"
    if record.task_name == "classification":
        export_cmd = f"dnn-export-classification {checkpoint} model.onnx"
        serve_cmd = f"dnn-serve-classification {checkpoint} image.jpg"
    elif record.task_name == "segmentation":
        export_cmd = f"dnn-export-segmentation {checkpoint} model.onnx"
        serve_cmd = f"dnn-serve-segmentation {checkpoint} image.jpg"
    else:
        export_cmd = ""
        serve_cmd = ""
    return export_cmd, serve_cmd


def build_studio_html(records: list[RunRecord]) -> str:
    rows = []
    for index, record in enumerate(rank_run_records(records), start=1):
        export_cmd, serve_cmd = _commands(record)
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{escape(record.task_name)}</td>"
            f"<td>{escape(_score(record))}</td>"
            f"<td><code>{escape(str(record.run_dir))}</code></td>"
            f"<td><code>{escape(str(record.artifacts.get('best_checkpoint_path') or ''))}</code></td>"
            f"<td><code>{escape(export_cmd)}</code><br><code>{escape(serve_cmd)}</code></td>"
            "</tr>"
        )

    body = "\n".join(rows) if rows else "<tr><td colspan=\"6\">No runs found.</td></tr>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>dnn-pytorch Experiment Studio</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    code {{ white-space: nowrap; }}
  </style>
</head>
<body>
  <h1>Experiment Studio</h1>
  <p>Local run leaderboard generated from <code>run.json</code> records.</p>
  <table>
    <thead>
      <tr>
        <th>Rank</th>
        <th>Task</th>
        <th>Score</th>
        <th>Run</th>
        <th>Best Checkpoint</th>
        <th>Next Commands</th>
      </tr>
    </thead>
    <tbody>
      {body}
    </tbody>
  </table>
</body>
</html>
"""


def write_studio(records: list[RunRecord], output_path: str | Path) -> Path:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(build_studio_html(records), encoding="utf-8")
    return resolved_output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a local experiment studio.")
    parser.add_argument("paths", nargs="+", help="Run directories or parent directories.")
    parser.add_argument(
        "--output",
        default="experiment-studio.html",
        help="HTML file to write.",
    )
    args = parser.parse_args(argv)
    output_path = write_studio(collect_run_records(args.paths), args.output)
    print(f"Wrote experiment studio to {output_path}")


if __name__ == "__main__":
    main()
