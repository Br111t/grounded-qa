"""Checkpoint-free unit tests for V5D logit ranking and joint scoring."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments.balanced_boundary_v5d.diagnose_logits import (
    boundary_rank, choose_cases, ranked_candidates, top_joint_spans, run,
)


class LogitDiagnosticTests(unittest.TestCase):
    def test_raw_ranks_only_permitted_evidence_positions(self):
        logits = [999, 1, 4, 3, 900]
        self.assertEqual(ranked_candidates(logits, [1, 2, 3], 2),
                         [{"position": 2, "score": 4.0},
                          {"position": 3, "score": 3.0}])
        self.assertEqual(boundary_rank(logits, [1, 2, 3], 1), 3)

    def test_joint_decoder_can_recover_wrong_raw_end(self):
        starts = [-100, 10, 0, 0]
        ends = [-100, 7, 5, 9]
        # Raw end 3; however max span length=1 makes (1, 1) best.
        self.assertEqual(ranked_candidates(ends, [1, 2, 3], 1)[0]["position"], 3)
        self.assertEqual(top_joint_spans(starts, ends, [1, 2, 3], 1, 1)[0],
                         {"start": 1, "end": 1, "score": 17.0})

    def test_joint_span_respects_max_length(self):
        starts = [0, 10, 1, 0]
        ends = [0, 0, 2, 100]
        spans = top_joint_spans(starts, ends, [1, 2, 3], 2, 10)
        self.assertFalse(any(s["start"] == 1 and s["end"] == 3 for s in spans))

    def test_deterministic_tie_break(self):
        starts = [0, 1, 1]
        ends = [0, 1, 1]
        self.assertEqual(top_joint_spans(starts, ends, [1, 2], 2, 1)[0],
                         {"start": 1, "end": 1, "score": 2.0})

    def test_representative_cases_are_deterministic(self):
        cases = [
            {"group": "length4_vowel", "error_category": "exact",
             "example_id": "b"},
            {"group": "length4_vowel", "error_category": "exact",
             "example_id": "a"},
        ]
        self.assertEqual([r["example_id"] for r in choose_cases(cases, 1)], ["a"])


    def test_primary_report_matches_metrics_and_recovers_exact_answers(self):
        """Regression: an exact recovery must not disappear from the summary."""
        def fake_examine(model, row, name, checkpoint_hash):
            index = row["index"]
            recovery = name == "v5d" and index < 11
            exact = recovery or (name == "v5d" and 11 <= index < 21)
            raw_correct = exact and not recovery
            return {
                "model": name, "example_id": f"case_{index:03d}",
                "group": "length4_consonant", "exact": exact,
                "selected_span_correct": exact,
                "raw_end_correct": raw_correct,
                "selected_start": 1, "gold_end_rank": 2 if recovery else 1,
                "best_end_score": 3.0, "gold_end_score": 2.0,
                "gold_start_score": 1.0,
                "top_joint_spans": [{"score": 3.0}],
                "error_category": "exact" if exact else "raw_and_selected_wrong",
                "expected": "Leader: Demo.",
                "predicted": "Leader: Demo." if exact else "Leader: Wrong.",
            }

        with TemporaryDirectory() as temp:
            base = Path(temp)
            train = base / "train.jsonl"
            dev = base / "dev.jsonl"
            train.write_text("train", encoding="utf-8")
            dev.write_text("dev", encoding="utf-8")
            checkpoints = {}
            for name in ("v5", "v5b", "v5d"):
                path = base / (name + ".pt")
                path.write_bytes(name.encode("utf-8"))
                checkpoints[name] = path

            def load_mock(path):
                name = path.stem
                info = {"answerability_threshold": 0.5}
                if name == "v5d":
                    import hashlib
                    info.update({
                        "experiment": "v5d_balanced_boundaries",
                        "extra_training_sha256": hashlib.sha256(
                            train.read_bytes()).hexdigest(),
                    })
                return object(), info

            out = base / "reports"
            with (patch("experiments.balanced_boundary_v5d.diagnose_logits.read_split",
                        return_value=[{"index": i} for i in range(96)]),
                  patch("experiments.balanced_boundary_v5d.diagnose_logits.PATHS",
                        {"train": train, "dev": dev}),
                  patch("experiments.balanced_boundary_v5d.diagnose_logits.load_span_checkpoint",
                        side_effect=load_mock),
                  patch("experiments.balanced_boundary_v5d.diagnose_logits.examine",
                        side_effect=fake_examine)):
                run(checkpoints, out)
            summary = (out / "summary.md").read_text(encoding="utf-8")
            metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            cases = json.loads((out / "all_cases.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["counts"]["v5d"]["recovery"], 11)
            self.assertEqual(metrics["counts"]["v5d"]["exact"], 21)
            self.assertEqual(
                sum(r["recovered_wrong_raw_end"] for r in cases
                    if r["model"] == "v5d"), 11)
            self.assertIn("| V5D | 85/96 | 96/96 | 11 | 0 |", summary)
            self.assertEqual(metrics["counts"]["v5d"]["recovery"],
                             sum(r["recovered_wrong_raw_end"] for r in cases
                                 if r["model"] == "v5d"))
            self.assertIn("## Score margins", summary)
            self.assertIn("| V5D | length4_consonant | True | True |", summary)



if __name__ == "__main__":
    unittest.main()
