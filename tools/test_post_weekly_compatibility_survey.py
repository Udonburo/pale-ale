import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SURVEY_JSON = REPO_ROOT / "tools" / "post_weekly_compatibility_survey.json"


class PostWeeklyCompatibilitySurveyTest(unittest.TestCase):
    def test_survey_has_expected_schema_and_targets(self) -> None:
        payload = json.loads(SURVEY_JSON.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_id"],
            "pale-ale.eval_factory.post_weekly_compatibility_survey.v1",
        )
        self.assertEqual(
            payload["current_weekly_mainline_closed_targets"],
            ["qwen2_5_3b", "llama3_2_3b", "qwen3_4b"],
        )

    def test_candidate_rows_cover_expected_candidates_and_fields(self) -> None:
        payload = json.loads(SURVEY_JSON.read_text(encoding="utf-8"))
        rows = payload["candidate_matrix"]
        candidates = {row["candidate"] for row in rows}

        expected_candidates = {
            "Qwen/Qwen3-4B-Instruct-2507",
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3.6-27B",
            "Qwen/Qwen3.6-35B-A3B",
            "google/gemma-4-E2B-it",
            "google/gemma-4-E4B-it",
            "google/gemma-4-31B-it",
            "microsoft/Phi-4-mini-instruct",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        }
        self.assertEqual(candidates, expected_candidates)

        valid_support = {"yes", "no", "uncertain"}
        valid_lanes = {
            "current_mainline_candidate",
            "compatibility_lane",
            "auth_only_lane",
            "protocol_expanding_lane",
            "defer",
        }
        valid_risks = {"low", "high", "uncertain"}

        for row in rows:
            self.assertIn(row["likely_supported_by_current_stack"], valid_support)
            self.assertIn(row["recommended_lane"], valid_lanes)
            self.assertIn(row["auth_gated_risk"], valid_risks)
            self.assertIn(row["trust_remote_code_risk"], valid_risks)
            self.assertTrue(row["likely_blockers"])
            self.assertTrue(row["source_urls"])


if __name__ == "__main__":
    unittest.main()
