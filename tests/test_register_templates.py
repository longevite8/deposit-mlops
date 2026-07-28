import unittest

from register_templates import (
    build_templates,
    build_task_create_kwargs,
    select_templates,
    should_pin_template_commit,
    split_selector_values,
)


class RegisterTemplatesTest(unittest.TestCase):
    def setUp(self):
        self.templates = build_templates(
            light_requirements_file="requirements-tasks.txt",
            forecast_requirements_file="requirements-forecast.txt",
        )

    def test_select_templates_returns_all_without_selectors(self):
        self.assertEqual(select_templates(self.templates, []), self.templates)

    def test_select_templates_accepts_short_config_alias(self):
        selected = select_templates(self.templates, ["train"])

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][3], "TEMPLATE_TRAIN_ID")

    def test_select_templates_accepts_config_var_and_script_aliases(self):
        selected = select_templates(
            self.templates,
            ["TEMPLATE_TRAIN_ID", "tasks/extract_data.py"],
        )

        self.assertEqual(
            [template[3] for template in selected],
            ["TEMPLATE_EXTRACT_ID", "TEMPLATE_TRAIN_ID"],
        )

    def test_split_selector_values_supports_commas_and_repeated_options(self):
        self.assertEqual(
            split_selector_values(["train, evaluate", "TEMPLATE_HPO_ID"]),
            ["train", "evaluate", "TEMPLATE_HPO_ID"],
        )

    def test_select_templates_rejects_unknown_selector(self):
        with self.assertRaises(ValueError) as raised:
            select_templates(self.templates, ["not-a-real-template"])

        self.assertIn("not-a-real-template", str(raised.exception))

    def test_template_registration_pins_commit_by_default(self):
        self.assertTrue(
            should_pin_template_commit(branch_head=False, env_branch_head=False)
        )

    def test_template_registration_can_use_branch_head(self):
        self.assertFalse(
            should_pin_template_commit(branch_head=True, env_branch_head=False)
        )
        self.assertFalse(
            should_pin_template_commit(branch_head=False, env_branch_head=True)
        )

    def test_build_task_create_kwargs_includes_commit_when_pinned(self):
        kwargs = build_task_create_kwargs(
            project_name="project",
            task_name="task",
            task_type="training",
            repo="repo",
            branch="branch",
            commit="abc123",
            script="tasks/train_model.py",
            requirements_file="requirements.txt",
        )

        self.assertEqual(kwargs["commit"], "abc123")
        self.assertEqual(kwargs["working_directory"], ".")

    def test_build_task_create_kwargs_omits_commit_in_branch_head_mode(self):
        kwargs = build_task_create_kwargs(
            project_name="project",
            task_name="task",
            task_type="training",
            repo="repo",
            branch="branch",
            commit=None,
            script="tasks/train_model.py",
            requirements_file="requirements.txt",
        )

        self.assertNotIn("commit", kwargs)


if __name__ == "__main__":
    unittest.main()
