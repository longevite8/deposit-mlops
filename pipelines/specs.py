"""Declarative ClearML pipeline step definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import config


@dataclass(frozen=True)
class PipelineStepSpec:
    """Configuration for one template-backed ClearML pipeline step."""

    name: str
    template_id_name: str
    parents: tuple[str, ...] = ()
    parameter_override: dict[str, Any] = field(default_factory=dict)
    monitor_metrics: tuple[tuple[str, str], ...] = ()
    cache_executed_step: bool = True
    execution_queue_name: str = "CPU_QUEUE"

    def template_id(self, config_module: ModuleType = config) -> str:
        return getattr(config_module, self.template_id_name, "")

    def execution_queue(self, config_module: ModuleType = config) -> str:
        return getattr(config_module, self.execution_queue_name)

    def add_to_pipeline(self, pipeline, config_module: ModuleType = config) -> None:
        """Attach this step to a ClearML PipelineController."""

        kwargs = {
            "name": self.name,
            "base_task_id": self.template_id(config_module),
            "execution_queue": self.execution_queue(config_module),
            "cache_executed_step": self.cache_executed_step,
        }
        if self.parents:
            kwargs["parents"] = list(self.parents)
        if self.parameter_override:
            kwargs["parameter_override"] = dict(self.parameter_override)
        if self.monitor_metrics:
            kwargs["monitor_metrics"] = list(self.monitor_metrics)
        pipeline.add_step(**kwargs)

    def manifest_entry(self, config_module: ModuleType = config) -> dict[str, Any]:
        """Return a serializable description of this step for controller artifacts."""

        return {
            "name": self.name,
            "template_id_name": self.template_id_name,
            "template_id": self.template_id(config_module),
            "parents": list(self.parents),
            "parameter_override": dict(self.parameter_override),
            "monitor_metrics": [list(metric) for metric in self.monitor_metrics],
            "cache_executed_step": self.cache_executed_step,
            "execution_queue": self.execution_queue(config_module),
        }


def validate_pipeline_specs(
    specs: tuple[PipelineStepSpec, ...],
    config_module: ModuleType = config,
) -> None:
    """Fail fast when a pipeline references missing template IDs."""

    required_ids = {
        spec.template_id_name: spec.template_id(config_module) for spec in specs
    }
    config_module.validate_config(required_ids=required_ids)


def add_specs_to_pipeline(
    pipeline,
    specs: tuple[PipelineStepSpec, ...],
    config_module: ModuleType = config,
) -> None:
    """Attach all declarative step specs to a ClearML PipelineController."""

    for spec in specs:
        spec.add_to_pipeline(pipeline, config_module=config_module)


def build_pipeline_manifest(
    *,
    pipeline_type: str,
    pipeline_name: str,
    deployment_version: str,
    run_mode: str | None,
    timestamp: str,
    specs: tuple[PipelineStepSpec, ...],
    config_module: ModuleType = config,
) -> dict[str, Any]:
    """Build a compact controller-level manifest for observability."""

    return {
        "pipeline_type": pipeline_type,
        "pipeline_name": pipeline_name,
        "deployment_version": deployment_version,
        "run_mode": run_mode,
        "timestamp": timestamp,
        "project": config_module.PROJECT_PIPELINE,
        "default_execution_queue": config_module.SERVICES_QUEUE,
        "steps": [spec.manifest_entry(config_module) for spec in specs],
    }


TRAINING_STEPS: tuple[PipelineStepSpec, ...] = (
    PipelineStepSpec(
        name="extract",
        template_id_name="TEMPLATE_EXTRACT_ID",
        cache_executed_step=False,
    ),
    PipelineStepSpec(
        name="feature",
        template_id_name="TEMPLATE_FEATURE_ID",
        parents=("extract",),
        parameter_override={"General/extract_task_id": "${extract.id}"},
    ),
    PipelineStepSpec(
        name="validate",
        template_id_name="TEMPLATE_VALIDATE_ID",
        parents=("feature",),
        parameter_override={"General/feature_task_id": "${feature.id}"},
    ),
    PipelineStepSpec(
        name="drift",
        template_id_name="TEMPLATE_DRIFT_ID",
        parents=("validate",),
        parameter_override={"General/feature_task_id": "${feature.id}"},
        cache_executed_step=False,
    ),
    PipelineStepSpec(
        name="hpo",
        template_id_name="TEMPLATE_HPO_ID",
        parents=("drift",),
        parameter_override={"General/feature_task_id": "${feature.id}"},
    ),
    PipelineStepSpec(
        name="train",
        template_id_name="TEMPLATE_TRAIN_ID",
        parents=("hpo",),
        parameter_override={
            "General/feature_task_id": "${feature.id}",
            "General/hpo_task_id": "${hpo.id}",
        },
    ),
    PipelineStepSpec(
        name="evaluate",
        template_id_name="TEMPLATE_EVALUATE_ID",
        parents=("train",),
        parameter_override={
            "General/feature_task_id": "${feature.id}",
            "General/train_task_id": "${train.id}",
        },
        monitor_metrics=(("MAPE", "mape"), ("R2", "r2")),
    ),
    PipelineStepSpec(
        name="register",
        template_id_name="TEMPLATE_REGISTER_ID",
        parents=("evaluate",),
        parameter_override={
            "General/train_task_id": "${train.id}",
            "General/evaluate_task_id": "${evaluate.id}",
        },
    ),
    PipelineStepSpec(
        name="explain_model",
        template_id_name="TEMPLATE_EXPLAIN_ID",
        parents=("register",),
        parameter_override={
            "General/feature_task_id": "${feature.id}",
            "General/train_task_id": "${train.id}",
            "General/n_samples": config.N_SHAP_SAMPLES,
        },
    ),
    PipelineStepSpec(
        name="compare_champion",
        template_id_name="TEMPLATE_COMPARE_CHAMPION_ID",
        parents=("register",),
        parameter_override={"General/register_task_id": "${register.id}"},
    ),
    PipelineStepSpec(
        name="promote_champion",
        template_id_name="TEMPLATE_PROMOTE_CHAMPION_ID",
        parents=("compare_champion",),
        parameter_override={"General/compare_task_id": "${compare_champion.id}"},
    ),
    PipelineStepSpec(
        name="deploy_serving",
        template_id_name="TEMPLATE_DEPLOY_SERVING_ID",
        parents=("promote_champion",),
        parameter_override={
            "General/promote_task_id": "${promote_champion.id}",
            "General/service_id": config.CLEARML_SERVING_SERVICE_ID,
            "General/auto_create_service": config.AUTO_CREATE_CLEARML_SERVING_SERVICE,
            "General/service_name": config.CLEARML_SERVING_SERVICE_NAME,
            "General/service_project": config.CLEARML_SERVING_PROJECT,
            "General/base_serving_url": config.CLEARML_BASE_SERVING_URL,
            "General/inference_base_url": config.CLEARML_SERVING_BASE_URL,
            "General/metric_log_freq": config.CLEARML_SERVING_METRIC_LOG_FREQ,
            "General/endpoint_prefix": config.CLEARML_SERVING_ENDPOINT_PREFIX,
            "General/endpoint": config.CLEARML_SERVING_ENDPOINT,
            "General/endpoint_version": config.CLEARML_SERVING_ENDPOINT_VERSION,
            "General/preprocess": config.CLEARML_SERVING_PREPROCESS,
            "General/engine": config.CLEARML_SERVING_ENGINE,
            "General/alias": config.CLEARML_SERVING_ALIAS,
            "General/horizon": config.FORECAST_HORIZON,
        },
        cache_executed_step=False,
        execution_queue_name="SERVICES_QUEUE",
    ),
    PipelineStepSpec(
        name="verify_endpoint",
        template_id_name="TEMPLATE_VERIFY_ENDPOINT_ID",
        parents=("deploy_serving",),
        parameter_override={
            "General/deploy_serving_task_id": "${deploy_serving.id}",
            "General/base_url": config.CLEARML_SERVING_BASE_URL,
            "General/endpoint_prefix": config.CLEARML_SERVING_ENDPOINT_PREFIX,
            "General/endpoint": config.CLEARML_SERVING_ENDPOINT,
            "General/version": config.CLEARML_SERVING_ENDPOINT_VERSION,
            "General/horizon": config.FORECAST_HORIZON,
        },
        cache_executed_step=False,
        execution_queue_name="SERVICES_QUEUE",
    ),
)


PRODUCTION_STEPS: tuple[PipelineStepSpec, ...] = (
    PipelineStepSpec(
        name="extract",
        template_id_name="TEMPLATE_EXTRACT_ID",
        cache_executed_step=False,
    ),
    PipelineStepSpec(
        name="feature",
        template_id_name="TEMPLATE_FEATURE_ID",
        parents=("extract",),
        parameter_override={"General/extract_task_id": "${extract.id}"},
        cache_executed_step=False,
    ),
    PipelineStepSpec(
        name="drift",
        template_id_name="TEMPLATE_DRIFT_ID",
        parents=("feature",),
        parameter_override={"General/feature_task_id": "${feature.id}"},
        cache_executed_step=False,
    ),
    PipelineStepSpec(
        name="inference",
        template_id_name="TEMPLATE_INFERENCE_ID",
        parents=("feature",),
        parameter_override={"General/feature_task_id": "${feature.id}"},
        cache_executed_step=False,
    ),
    PipelineStepSpec(
        name="monitoring",
        template_id_name="TEMPLATE_MONITORING_ID",
        parents=("inference", "drift"),
        parameter_override={
            "General/feature_task_id": "${feature.id}",
            "General/inference_task_id": "${inference.id}",
            "General/drift_task_id": "${drift.id}",
        },
        monitor_metrics=(
            ("MAPE", "MAPE"),
            ("R2", "R2"),
            ("drift_ratio", "drift_ratio"),
            ("need_retraining", "need_retraining"),
        ),
        cache_executed_step=False,
    ),
    PipelineStepSpec(
        name="alerting",
        template_id_name="TEMPLATE_ALERTING_ID",
        parents=("monitoring",),
        parameter_override={"General/monitoring_task_id": "${monitoring.id}"},
        cache_executed_step=False,
    ),
    PipelineStepSpec(
        name="auto_retraining",
        template_id_name="TEMPLATE_AUTO_RETRAINING_ID",
        parents=("alerting",),
        parameter_override={"General/alert_task_id": "${alerting.id}"},
        cache_executed_step=False,
    ),
)
