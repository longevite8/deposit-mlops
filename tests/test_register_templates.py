import unittest

from register_templates import (
    build_templates,
    select_templates,
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


if __name__ == "__main__":
    unittest.main()
