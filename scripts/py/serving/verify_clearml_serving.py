"""Verify a deposit ClearML Serving endpoint."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Any

from config import (
    CLEARML_SERVING_BASE_URL,
    CLEARML_SERVING_ENDPOINT,
    CLEARML_SERVING_ENDPOINT_PREFIX,
    CLEARML_SERVING_ENDPOINT_VERSION,
    FORECAST_HORIZON,
)
from scripts.py.serving.deploy_clearml_serving import endpoint_name


def read_url(url: str, payload: bytes | None = None) -> tuple[int, str]:
    headers = {}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=payload, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.read().decode("utf-8")


def endpoint_url(args) -> str:
    endpoint = endpoint_name(args.endpoint_prefix, args.endpoint, horizon=args.horizon)
    base = args.base_url.rstrip("/")
    url = f"{base}/serve/{endpoint}"
    if args.version:
        url = f"{url}/{args.version}"
    return url


def run_check(args) -> dict[str, Any]:
    url = endpoint_url(args)
    docs_url = f"{args.base_url.rstrip('/')}/docs"
    report: dict[str, Any] = {
        "ok": False,
        "base_url": args.base_url,
        "docs_url": docs_url,
        "endpoint_url": url,
        "endpoint": endpoint_name(args.endpoint_prefix, args.endpoint, horizon=args.horizon),
        "endpoint_version": args.version,
    }
    try:
        if args.payload_json:
            with open(args.payload_json, "rb") as payload_file:
                payload = payload_file.read()
            status, body = read_url(url, payload=payload)
            report["status"] = status
            report["ok"] = 200 <= status < 300
            report["response_preview"] = body[:1000]
        else:
            status, body = read_url(docs_url)
            report["status"] = status
            report["ok"] = 200 <= status < 300
            report["response_preview"] = body[:300]
            report["hint"] = "Provide --payload_json to verify an actual model request."
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        report["error"] = str(exc)
        report["hint"] = "ClearML Serving inference runtime is not reachable or returned an error."
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_url", default=CLEARML_SERVING_BASE_URL)
    parser.add_argument("--endpoint_prefix", default=CLEARML_SERVING_ENDPOINT_PREFIX)
    parser.add_argument("--endpoint", default=CLEARML_SERVING_ENDPOINT)
    parser.add_argument("--version", default=CLEARML_SERVING_ENDPOINT_VERSION)
    parser.add_argument("--payload_json", default=os.getenv("CLEARML_SERVING_VERIFY_PAYLOAD", ""))
    parser.add_argument("--horizon", type=int, default=FORECAST_HORIZON)
    return parser


def main() -> None:
    report = run_check(build_parser().parse_args())
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
