from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


SENSITIVE_PATTERNS = (
    re.compile(r"(CLEARML_API_ACCESS_KEY=)[^\s]+"),
    re.compile(r"(CLEARML_API_SECRET_KEY=)[^\s]+"),
    re.compile(r"(api\.credentials\.access_key\s*=\s*)[^\s]+"),
    re.compile(r"(api\.credentials\.secret_key\s*=\s*)[^\s]+"),
)

EVIDENCE_PATTERNS = (
    "Cache experiment trace",
    "Controlled cache experiment",
    "Running task id",
    "Using Cached",
    "Using cached",
    "Process completed",
    "failed",
    "Traceback",
    "Exception",
    "RuntimeError",
)


def sanitize_line(line: str) -> str:
    sanitized = line
    for pattern in SENSITIVE_PATTERNS:
        sanitized = pattern.sub(r"\1********", sanitized)
    return sanitized.replace("\r", "")


def compact(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("|", "\\|").strip()
    return text


def task_script_summary(task) -> dict[str, str]:
    script = getattr(task.data, "script", None)
    return {
        "repo": compact(getattr(script, "repository", "")),
        "branch": compact(getattr(script, "branch", "")),
        "commit": compact(getattr(script, "version_num", "")),
        "entry": compact(getattr(script, "entry_point", "")),
    }


def interesting_console_lines(task, *, limit: int) -> list[str]:
    try:
        lines = task.get_reported_console_output(number_of_reports=limit) or []
    except Exception as exc:  # pragma: no cover - network/server dependent
        return [f"Could not fetch console output: {exc}"]

    selected: list[str] = []
    for raw_line in lines:
        for piece in str(raw_line).splitlines():
            sanitized = sanitize_line(piece).strip()
            if not sanitized:
                continue
            if any(pattern.lower() in sanitized.lower() for pattern in EVIDENCE_PATTERNS):
                selected.append(sanitized)
    return selected[-limit:]


def artifact_names(task) -> str:
    artifacts = task.artifacts or {}
    return ", ".join(sorted(artifacts.keys()))


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    header_list = list(headers)
    lines = [
        "| " + " | ".join(header_list) + " |",
        "| " + " | ".join("---" for _ in header_list) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(compact(cell) for cell in row) + " |")
    return "\n".join(lines)


def snapshot_pipeline(pipeline_id: str, *, console_limit: int) -> str:
    from clearml import Task

    controller = Task.get_task(task_id=pipeline_id)
    children = Task.get_tasks(
        task_filter={"parent": pipeline_id},
        allow_archived=True,
    )
    children.sort(
        key=lambda task: (
            getattr(task.data, "created", None) or datetime.min.replace(tzinfo=timezone.utc),
            task.name,
        )
    )

    controller_script = task_script_summary(controller)
    params = controller.get_parameters() or {}
    controller_rows = [
        ("Pipeline ID", controller.id),
        ("Name", controller.name),
        ("Project", controller.project),
        ("Status", controller.get_status()),
        ("Tags", ", ".join(controller.get_tags() or [])),
        ("Created", getattr(controller.data, "created", "")),
        ("Started", getattr(controller.data, "started", "")),
        ("Completed", getattr(controller.data, "completed", "")),
        ("Repository", controller_script["repo"]),
        ("Branch", controller_script["branch"]),
        ("Commit", controller_script["commit"]),
        ("Entry point", controller_script["entry"]),
        ("pipeline/default_queue", params.get("pipeline/default_queue", "")),
    ]

    child_rows = []
    evidence_blocks = []
    for child in children:
        script = task_script_summary(child)
        child_rows.append(
            (
                child.name,
                child.id,
                child.get_status(),
                script["entry"],
                script["branch"],
                script["commit"],
                getattr(child.data, "created", ""),
                getattr(child.data, "started", ""),
                getattr(child.data, "completed", ""),
                artifact_names(child),
            )
        )
        evidence = interesting_console_lines(child, limit=console_limit)
        if evidence:
            evidence_blocks.append(
                "### Evidence: "
                + child.name
                + f" `{child.id}`\n\n"
                + "\n".join(f"- `{compact(line)}`" for line in evidence)
            )

    output = [
        f"## Pipeline Snapshot: `{pipeline_id}`",
        "",
        f"Captured at `{datetime.now().astimezone().isoformat(timespec='seconds')}`.",
        "",
        markdown_table(("Field", "Value"), controller_rows),
        "",
        markdown_table(
            (
                "Step",
                "Task ID",
                "Status",
                "Entry",
                "Branch",
                "Commit",
                "Created",
                "Started",
                "Completed",
                "Artifacts",
            ),
            child_rows,
        ),
    ]
    if evidence_blocks:
        output.extend(["", *evidence_blocks])
    return "\n".join(output).strip() + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a markdown evidence snapshot for a ClearML pipeline task."
    )
    parser.add_argument("pipeline_id", help="ClearML pipeline controller task ID")
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the project .env with ClearML connection settings.",
    )
    parser.add_argument(
        "--console-limit",
        type=int,
        default=20,
        help="Maximum recent console evidence lines to keep per child task.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional markdown file path. Prints to stdout when omitted.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    load_dotenv(args.env_file)
    markdown = snapshot_pipeline(args.pipeline_id, console_limit=args.console_limit)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(markdown, end="")


if __name__ == "__main__":
    main()
