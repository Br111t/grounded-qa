"""V5G invariants; no training or evaluation checkpoints required."""
import unittest

import torch

from experiments.question_aware_v5g.data_design import (
    EVAL_QUESTIONS, QUESTIONS, build_partition,
)
from experiments.question_aware_v5g.train import target_for
from experiments.question_aware_v5g.evaluate import enriched, summarize
from grounded_qa.models.span_model_v5 import prepare_span_input


class V5GTests(unittest.TestCase):
    def test_training_dev_eval_identities_disjoint(self):
        sets = {p: build_partition(p) for p in ("train", "dev", "eval")}
        self.assertEqual({k: len(v) for k,v in sets.items()},
                         {"train":48, "dev":48, "eval":192})
        identities = [{r["information"] for r in v} for v in sets.values()]
        for i in range(3):
            for j in range(i+1,3):
                self.assertFalse(identities[i] & identities[j])

    def test_training_only_supervises_two_canonical_fields(self):
        rows = build_partition("train")
        self.assertEqual({r["target_field"] for r in rows}, {"leader", "location"})
        self.assertEqual({r["question"] for r in rows}, set(QUESTIONS.values()))
        counts = {(f,a): sum(r["target_field"] == f and r["answerability"] == a
                              for r in rows)
                  for f in QUESTIONS for a in ("answerable","unanswerable")}
        self.assertEqual(counts, {
            ("leader","answerable"):18, ("leader","unanswerable"):6,
            ("location","answerable"):18, ("location","unanswerable"):6,
        })

    def test_gold_spans_are_grounded_and_question_conditional(self):
        rows = build_partition("train")
        for r in rows:
            ids, mask = prepare_span_input(r["information"], r["question"])
            target = target_for(r, ids, mask)
            if r["answerability"] == "unanswerable":
                self.assertEqual(target, (0.0, None, None))
            else:
                self.assertEqual(target[0], 1.0)
                self.assertTrue(mask[0, target[1]:target[2]+1].all().item())
        pairs = {}
        for r in rows:
            pairs.setdefault(r["source_id"], {})[r["target_field"]] = r
        self.assertEqual(len(pairs), 24)
        self.assertTrue(all(len(p)==2 and len({r["information"] for r in p.values()})==1
                            for p in pairs.values()))

    def test_reserved_eval_categories_never_enter_training(self):
        train_questions = {r["question"] for r in build_partition("train")}
        eval_rows = build_partition("eval")
        reserved = {r["question"] for r in eval_rows
                    if r["question_variant"] != "canonical"}
        self.assertFalse(train_questions & reserved)
        self.assertEqual(sum(r["target_field"] == "topic" for r in eval_rows),24)
        self.assertEqual(sum(r["target_field"] == "unrelated" for r in eval_rows),24)
        self.assertEqual(sum(r["question_variant"].startswith("paraphrase")
                             for r in eval_rows),96)

    def test_metrics_differentiate_abstention_and_field_accuracy(self):
        rows = build_partition("dev")
        samples=[]
        for r in rows:
            predicted = {
                "predicted_field":r["target_field"],
                "predicted_value":r["expected_value"],
                "answerability_probability": 0.9,
                "field_scores":{"leader":2.0, "location":1.0},
                "candidate_fields":[],
            }
            if r["answerability"]=="unanswerable":
                predicted["predicted_field"]=None
                predicted["predicted_value"]=None
                predicted["answerability_probability"]=0.1
            samples.append(enriched(r,predicted))
        metrics = summarize(samples)
        self.assertEqual(metrics["canonical"]["correct"],48)
        self.assertEqual(metrics["paired"]["count"],24)
        self.assertEqual(metrics["paired"]["both_correct"],24)
        self.assertEqual(metrics["paired"]["correct_switches"],24)


if __name__ == "__main__":
    unittest.main()
