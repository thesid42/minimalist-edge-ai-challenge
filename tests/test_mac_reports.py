import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import record_mac_results as report


class MacReportTests(unittest.TestCase):
    def test_report_replacement_preserves_existing_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "README.md"
            target.write_text("Original Windows results\n\n## Models and data\nKeep this too\n")
            with patch.object(report, "ROOT", root):
                report.insert_block("README.md", "First Mac results", "## Models and data")
                report.insert_block("README.md", "Updated Mac results", "## Models and data")
            text = target.read_text()
            self.assertIn("Original Windows results", text)
            self.assertIn("Keep this too", text)
            self.assertIn("Updated Mac results", text)
            self.assertNotIn("First Mac results", text)
            self.assertEqual(text.count(report.BEGIN), 1)
            self.assertLess(text.index(report.END), text.index("## Models and data"))

    def test_missing_anchor_does_not_change_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "README.md"
            target.write_text("Preserve me\n")
            with patch.object(report, "ROOT", root), self.assertRaises(ValueError):
                report.insert_block("README.md", "Mac results", "Missing anchor")
            self.assertEqual(target.read_text(), "Preserve me\n")

    def test_json_report_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(report, "ROOT", Path(directory)):
                value = {"records": [{"name": "mac_candidate_spin", "median_ms": 1.9}]}
                report.write_json("report.json", value)
                self.assertEqual(report.read("report.json"), value)

    def test_named_blocks_keep_full_and_subset_results_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("Existing Windows evidence\n")
            with patch.object(report, "ROOT", root):
                report.insert_block("README.md", "Subset measurements")
                report.insert_block("README.md", "Full measurements", marker="MAC FULL CPU RESULTS")
                report.insert_block("README.md", "Updated full measurements", marker="MAC FULL CPU RESULTS")
            text = (root / "README.md").read_text()
            self.assertIn("Existing Windows evidence", text)
            self.assertIn("Subset measurements", text)
            self.assertIn("Updated full measurements", text)
            self.assertEqual(text.count("<!-- BEGIN MAC FULL CPU RESULTS -->"), 1)

    def test_full_dataset_is_training_set_not_diagnostic_subset(self):
        from scripts.evaluate_mac_sample import evaluation_details
        result = evaluation_details(27000, 27000)
        self.assertTrue(result["full_dataset_evaluated"])
        self.assertEqual(result["evaluation_scope"], "training_set")
        self.assertNotIn("subset", result["accuracy_caveat"])

    def test_partial_dataset_is_diagnostic_and_invalid_counts_fail(self):
        from scripts.evaluate_mac_sample import evaluation_details
        self.assertEqual(evaluation_details(15000, 27000)["evaluation_scope"], "training_pool_diagnostic_subset")
        self.assertFalse(evaluation_details(15000, 27000)["full_dataset_evaluated"])
        with self.assertRaises(ValueError):
            evaluation_details(27001, 27000)


if __name__ == "__main__":
    unittest.main()
