"""Deploy the promoted deposit champion model to ClearML Serving."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any

from clearml import Model

from config import (
    AUTO_CREATE_CLEARML_SERVING_SERVICE,
    CLEARML_BASE_SERVING_URL,
    CLEARML_SERVING_ALIAS,
    CLEARML_SERVING_ENDPOINT,
    CLEARML_SERVING_ENDPOINT_PREFIX,
    CLEARML_SERVING_ENDPOINT_VERSION,
    CLEARML_SERVING_ENGINE,
    CLEARML_SERVING_METRIC_LOG_FREQ,
    CLEARML_SERVING_PREPROCESS,
    CLEARML_SERVING_PROJECT,
    CLEARML_SERVING_SERVICE_ID,
    CLEARML_SERVING_SERVICE_NAME,
    FORECAST_HORIZON,
    forecast_horizon_tag,
    forecast_horizon_endpoint_suffix,
)


def truthy(value: Any) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "y"}


def run_command(command: list[str], *, capture_output: bool = False) -> str:
    print(" ".join(command))
    result = subprocess.run(
        command,
        check=True,
        capture_output=capture_output,
        text=capture_output,
    )
    if not capture_output:
        return ""
    return "\n".join(value for value in [result.stdout, result.stderr] if value)


def parse_service_id(output: str) -> str | None:
    for pattern in [
        r"(?:service|task)[\s_-]*id\s*[:=]\s*([a-f0-9]{32})",
        r"\bid\s*[:=]\s*([a-f0-9]{32})",
        r"\b([a-f0-9]{32})\b",
    ]:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def endpoint_name(
    endpoint_prefix: str,
    endpoint: str = "",
    *,
    horizon: int | str | None = None,
) -> str:
    base = endpoint.strip("/") if endpoint else endpoint_prefix.strip("/")
    name = base.replace("/", "_")
    if horizon is None:
        return name
    suffix = forecast_horizon_endpoint_suffix(horizon)
    return name if name.endswith(f"_{suffix}") else f"{name}_{suffix}"


def endpoint_version(model: Model, explicit_version: str = "") -> str:
    if explicit_version:
        return explicit_version
    metadata = model.get_all_metadata()
    promoted_time = metadata.get("promoted_time", {}).get("value", "")
    if promoted_time:
        return promoted_time.replace(" ", "_").replace(":", "").replace(".", "_")
    return model.id


def resolve_model(
    model_id: str = "",
    alias: str = "champion",
    horizon: int | str | None = None,
) -> Model:
    if model_id:
        return Model(model_id=model_id)

    tags = [alias]
    if horizon is not None:
        tags.append(forecast_horizon_tag(horizon))
    models = Model.query_models(tags=tags, only_published=True, max_results=1)
    if not models:
        raise ValueError(f"No published ClearML model found with tags {tags}.")
    return models[0]


def create_serving_service(args) -> str:
    command = [
        "clearml-serving",
        "--yes",
        "create",
        "--name",
        args.service_name,
        "--project",
        args.service_project,
        "--tags",
        "deposit",
        "cashflow",
    ]
    output = run_command(command, capture_output=True)
    service_id = parse_service_id(output)
    if not service_id:
        raise RuntimeError(
            "Created ClearML Serving Service, but could not parse the service task id. "
            "Set CLEARML_SERVING_SERVICE_ID manually and rerun deployment."
        )
    return service_id


def ensure_service_id(args) -> str:
    if args.service_id:
        return args.service_id
    if not args.auto_create_service:
        raise ValueError("CLEARML_SERVING_SERVICE_ID or --service_id is required.")
    args.service_id = create_serving_service(args)
    return args.service_id


def configure_service(args) -> None:
    if not args.base_serving_url and not args.metric_log_freq:
        return
    command = ["clearml-serving", "--yes", "--id", args.service_id, "config"]
    if args.base_serving_url:
        command.extend(["--base-serving-url", args.base_serving_url])
    if args.metric_log_freq:
        command.extend(["--metric-log-freq", str(args.metric_log_freq)])
    run_command(command)


def model_metadata_value(model: Model, key: str, default: str = "") -> str:
    return str(model.get_all_metadata().get(key, {}).get("value", default) or default)


def set_model_metadata(model: Model, values: dict[str, Any]) -> None:
    for key, value in values.items():
        model.set_metadata(key, str(value))


def deploy_model(args) -> dict[str, Any]:
    if not shutil.which("clearml-serving"):
        raise RuntimeError(
            "clearml-serving CLI is not installed in this environment. "
            "Install clearml-serving on the deployment agent or run deployment elsewhere."
        )

    model = resolve_model(args.model_id, args.alias, args.horizon)
    args.service_id = ensure_service_id(args)
    configure_service(args)

    endpoint = endpoint_name(args.endpoint_prefix, args.endpoint, horizon=args.horizon)
    version = endpoint_version(model, args.endpoint_version)

    command = [
        "clearml-serving",
        "--yes",
        "--id",
        args.service_id,
        "model",
        "add",
        "--engine",
        args.engine,
        "--endpoint",
        endpoint,
        "--version",
        version,
        "--model-id",
        model.id,
        "--preprocess",
        args.preprocess,
        "--tags",
        args.alias,
        "deposit",
        "cashflow",
        f"horizon:{args.horizon}",
    ]
    run_command(command)

    metrics_command = [
        "clearml-serving",
        "--yes",
        "--id",
        args.service_id,
        "metrics",
        "add",
        "--endpoint",
        f"{endpoint}/*",
        "--log-freq",
        str(args.metric_log_freq or 1.0),
        "--variable-scalar",
        "horizon=1,7,14,30,60",
        "history_rows=1,30,60,90,180,365",
        "prediction_mean=0,1000000,10000000,50000000,100000000,500000000,1000000000",
        "prediction_min=0,1000000,10000000,50000000,100000000,500000000,1000000000",
        "prediction_max=0,1000000,10000000,50000000,100000000,500000000,1000000000",
    ]
    run_command(metrics_command)

    deployed_at = datetime.now(timezone.utc).isoformat()
    deployment = {
        "deployed": True,
        "service_id": args.service_id,
        "endpoint": endpoint,
        "endpoint_version": version,
        "endpoint_url": f"{args.inference_base_url.rstrip('/')}/serve/{endpoint}/{version}",
        "model_id": model.id,
        "model_name": model.name,
        "alias": args.alias,
        "horizon": int(args.horizon),
        "preprocess": args.preprocess,
        "engine": args.engine,
        "feature_dataset_id": model_metadata_value(model, "feature_dataset_id"),
        "deployed_at": deployed_at,
    }
    set_model_metadata(
        model,
        {
            "clearml_serving_service_id": args.service_id,
            "clearml_serving_endpoint": endpoint,
            "clearml_serving_endpoint_version": version,
            "clearml_serving_endpoint_url": deployment["endpoint_url"],
            "clearml_serving_alias": args.alias,
            "clearml_serving_deployed_at": deployed_at,
        },
    )
    return deployment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default=os.getenv("CLEARML_SERVING_MODEL_ID", ""))
    parser.add_argument("--alias", default=CLEARML_SERVING_ALIAS)
    parser.add_argument("--service_id", default=CLEARML_SERVING_SERVICE_ID)
    parser.add_argument(
        "--auto_create_service",
        action="store_true",
        default=AUTO_CREATE_CLEARML_SERVING_SERVICE,
    )
    parser.add_argument("--service_name", default=CLEARML_SERVING_SERVICE_NAME)
    parser.add_argument("--service_project", default=CLEARML_SERVING_PROJECT)
    parser.add_argument("--base_serving_url", default=CLEARML_BASE_SERVING_URL)
    parser.add_argument("--inference_base_url", default=os.getenv("CLEARML_SERVING_BASE_URL", "http://127.0.0.1:8082"))
    parser.add_argument("--metric_log_freq", type=float, default=CLEARML_SERVING_METRIC_LOG_FREQ)
    parser.add_argument("--endpoint_prefix", default=CLEARML_SERVING_ENDPOINT_PREFIX)
    parser.add_argument("--endpoint", default=CLEARML_SERVING_ENDPOINT)
    parser.add_argument("--endpoint_version", default=CLEARML_SERVING_ENDPOINT_VERSION)
    parser.add_argument("--preprocess", default=CLEARML_SERVING_PREPROCESS)
    parser.add_argument("--engine", default=CLEARML_SERVING_ENGINE)
    parser.add_argument("--horizon", type=int, default=FORECAST_HORIZON)
    return parser


def main() -> None:
    deployment = deploy_model(build_parser().parse_args())
    print(json.dumps(deployment, indent=2))


if __name__ == "__main__":
    main()
