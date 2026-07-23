import types
import unittest

from pipelines.specs import (
    PRODUCTION_STEPS,
    TRAINING_STEPS,
    add_specs_to_pipeline,
    build_pipeline_manifest,
    build_training_steps_for_horizons,
    validate_pipeline_specs,
)


def fake_config(**overrides):
    values = {
        "CPU_QUEUE": "cpu",
        "SERVICES_QUEUE": "services",
        "PROJECT_PIPELINE": "Deposit/Pipelines",
        "TEMPLATE_EXTRACT_ID": "extract-template",
        "TEMPLATE_FEATURE_ID": "feature-template",
        "TEMPLATE_VALIDATE_ID": "validate-template",
        "TEMPLATE_DRIFT_ID": "drift-template",
        "TEMPLATE_HPO_ID": "hpo-template",
        "TEMPLATE_TRAIN_ID": "train-template",
        "TEMPLATE_EVALUATE_ID": "evaluate-template",
        "TEMPLATE_REGISTER_ID": "register-template",
        "TEMPLATE_COMPARE_CHAMPION_ID": "compare-template",
        "TEMPLATE_PROMOTE_CHAMPION_ID": "promote-template",
        "TEMPLATE_EXPLAIN_ID": "explain-template",
        "TEMPLATE_DEPLOY_SERVING_ID": "deploy-serving-template",
        "TEMPLATE_VERIFY_ENDPOINT_ID": "verify-endpoint-template",
        "TEMPLATE_DEPLOY_CANDIDATE_SERVING_ID": "deploy-candidate-template",
        "TEMPLATE_VERIFY_CANDIDATE_ENDPOINT_ID": "verify-candidate-template",
        "TEMPLATE_INFERENCE_ID": "inference-template",
        "TEMPLATE_MONITORING_ID": "monitoring-template",
        "TEMPLATE_ALERTING_ID": "alerting-template",
        "TEMPLATE_AUTO_RETRAINING_ID": "auto-retraining-template",
        "FORECAST_HORIZON": 30,
        "FORECAST_HORIZONS": (30,),
    }
    values.update(overrides)
    module = types.SimpleNamespace(**values)

    def validate_config(required_ids=None):
        missing = [name for name, value in (required_ids or {}).items() if not value]
        if missing:
            raise ValueError(f"Missing template IDs: {missing}")

    module.validate_config = validate_config
    return module


class FakePipeline:
    def __init__(self):
        self.steps = []

    def add_step(self, **kwargs):
        self.steps.append(kwargs)


class PipelineSpecsTest(unittest.TestCase):
    def test_training_specs_define_expected_order_and_dependencies(self):
        self.assertEqual(
            [spec.name for spec in TRAINING_STEPS],
            [
                "extract",
                "feature",
                "validate",
                "drift",
                "hpo",
                "train",
                "evaluate",
                "register",
                "deploy_candidate_serving",
                "verify_candidate_endpoint",
                "explain_model",
                "compare_champion",
                "promote_champion",
                "deploy_serving",
                "verify_endpoint",
            ],
        )

        train_step = next(spec for spec in TRAINING_STEPS if spec.name == "train")
        self.assertEqual(train_step.parents, ("hpo",))
        self.assertEqual(
            train_step.parameter_override,
            {
                "General/feature_task_id": "${feature.id}",
                "General/hpo_task_id": "${hpo.id}",
                "General/horizon": 30,
            },
        )
        candidate_deploy_step = next(
            spec for spec in TRAINING_STEPS if spec.name == "deploy_candidate_serving"
        )
        candidate_verify_step = next(
            spec for spec in TRAINING_STEPS if spec.name == "verify_candidate_endpoint"
        )
        deploy_step = next(
            spec for spec in TRAINING_STEPS if spec.name == "deploy_serving"
        )
        verify_step = next(
            spec for spec in TRAINING_STEPS if spec.name == "verify_endpoint"
        )
        compare_step = next(
            spec for spec in TRAINING_STEPS if spec.name == "compare_champion"
        )
        self.assertEqual(candidate_deploy_step.parents, ("register",))
        self.assertEqual(candidate_verify_step.parents, ("deploy_candidate_serving",))
        self.assertEqual(compare_step.parents, ("register",))
        self.assertEqual(deploy_step.parents, ("promote_champion",))
        self.assertEqual(verify_step.parents, ("deploy_serving",))
        evaluate_step = next(spec for spec in TRAINING_STEPS if spec.name == "evaluate")
        self.assertEqual(evaluate_step.parameter_override["General/eval_horizon"], 30)
        self.assertEqual(candidate_deploy_step.execution_queue(fake_config()), "services")
        self.assertEqual(candidate_verify_step.execution_queue(fake_config()), "services")
        self.assertEqual(deploy_step.execution_queue(fake_config()), "services")
        self.assertEqual(verify_step.execution_queue(fake_config()), "services")
        self.assertFalse(candidate_deploy_step.cache_executed_step)
        self.assertFalse(candidate_verify_step.cache_executed_step)
        self.assertFalse(deploy_step.cache_executed_step)
        self.assertFalse(verify_step.cache_executed_step)

    def test_production_specs_define_expected_uncached_runtime_steps(self):
        self.assertEqual(
            [spec.name for spec in PRODUCTION_STEPS],
            [
                "extract",
                "feature",
                "drift",
                "inference",
                "monitoring",
                "alerting",
                "auto_retraining",
            ],
        )

        self.assertTrue(all(not spec.cache_executed_step for spec in PRODUCTION_STEPS))
        monitoring = next(spec for spec in PRODUCTION_STEPS if spec.name == "monitoring")
        self.assertEqual(monitoring.parents, ("inference", "drift"))
        self.assertIn(("need_retraining", "need_retraining"), monitoring.monitor_metrics)
        inference = next(spec for spec in PRODUCTION_STEPS if spec.name == "inference")
        self.assertEqual(inference.parameter_override["General/horizon"], 30)

    def test_build_training_steps_for_horizons_expands_horizon_branches(self):
        steps = build_training_steps_for_horizons((7, 14))
        names = [spec.name for spec in steps]

        self.assertIn("train_h7", names)
        self.assertIn("train_h14", names)
        self.assertIn("deploy_serving_h7", names)
        self.assertIn("verify_endpoint_h14", names)

        train_h7 = next(spec for spec in steps if spec.name == "train_h7")
        evaluate_h14 = next(spec for spec in steps if spec.name == "evaluate_h14")
        deploy_h7 = next(spec for spec in steps if spec.name == "deploy_serving_h7")

        self.assertEqual(train_h7.parents, ("hpo",))
        self.assertEqual(train_h7.parameter_override["General/horizon"], 7)
        self.assertEqual(
            evaluate_h14.parameter_override["General/train_task_id"],
            "${train_h14.id}",
        )
        self.assertEqual(evaluate_h14.parameter_override["General/eval_horizon"], 14)
        self.assertEqual(deploy_h7.parents, ("promote_champion_h7",))
        self.assertEqual(deploy_h7.parameter_override["General/horizon"], 7)

    def test_build_training_steps_for_single_nondefault_horizon_preserves_names(self):
        steps = build_training_steps_for_horizons((7,))
        names = [spec.name for spec in steps]
        train = next(spec for spec in steps if spec.name == "train")
        evaluate = next(spec for spec in steps if spec.name == "evaluate")

        self.assertIn("deploy_serving", names)
        self.assertNotIn("deploy_serving_h7", names)
        self.assertEqual(train.parameter_override["General/horizon"], 7)
        self.assertEqual(evaluate.parameter_override["General/eval_horizon"], 7)

    def test_add_specs_to_pipeline_resolves_template_ids_and_queues(self):
        pipeline = FakePipeline()

        add_specs_to_pipeline(
            pipeline,
            TRAINING_STEPS[:2],
            config_module=fake_config(),
        )

        self.assertEqual(len(pipeline.steps), 2)
        self.assertEqual(pipeline.steps[0]["base_task_id"], "extract-template")
        self.assertEqual(pipeline.steps[0]["execution_queue"], "cpu")
        self.assertFalse(pipeline.steps[0]["cache_executed_step"])
        self.assertEqual(pipeline.steps[1]["parents"], ["extract"])
        self.assertEqual(
            pipeline.steps[1]["parameter_override"],
            {"General/extract_task_id": "${extract.id}"},
        )

    def test_validate_pipeline_specs_fails_fast_for_missing_template_id(self):
        with self.assertRaises(ValueError) as raised:
            validate_pipeline_specs(
                TRAINING_STEPS,
                config_module=fake_config(TEMPLATE_HPO_ID=""),
            )

        self.assertIn("TEMPLATE_HPO_ID", str(raised.exception))

    def test_build_pipeline_manifest_is_serializable_step_summary(self):
        manifest = build_pipeline_manifest(
            pipeline_type="training",
            pipeline_name="Training Pipeline",
            deployment_version="1.0.0",
            run_mode="MANUAL",
            timestamp="20260716_101010",
            specs=TRAINING_STEPS[:1],
            config_module=fake_config(),
        )

        self.assertEqual(manifest["pipeline_type"], "training")
        self.assertEqual(manifest["default_execution_queue"], "services")
        self.assertEqual(manifest["steps"][0]["name"], "extract")
        self.assertEqual(manifest["steps"][0]["template_id"], "extract-template")


if __name__ == "__main__":
    unittest.main()
