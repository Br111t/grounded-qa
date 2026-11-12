"""Tests for V5G single-change uniform-question sampling experiment."""
import collections
import unittest

from experiments.question_aware_v5g.data_design import build_partition
from experiments.question_aware_v5g.train_uniform_sampling import uniform_schedule


class V5GUniformSamplingTests(unittest.TestCase):
    def test_every_epoch_sees_every_question_once(self):
        schedule = uniform_schedule(48, 1200, 42)
        self.assertEqual(len(schedule), 1200)
        self.assertEqual(collections.Counter(schedule), collections.Counter({
            index: 25 for index in range(48)
        }))
        for offset in range(0, 1200, 48):
            self.assertEqual(sorted(schedule[offset:offset + 48]), list(range(48)))
        self.assertEqual(schedule, uniform_schedule(48, 1200, 42))
        self.assertNotEqual(schedule, uniform_schedule(48, 1200, 43))

    def test_natural_answerability_ratio_is_preserved(self):
        rows = build_partition("train")
        schedule = uniform_schedule(len(rows), 1200, 42)
        seen = collections.Counter(
            (rows[index]["target_field"], rows[index]["answerability"])
            for index in schedule
        )
        self.assertEqual(seen, {
            ("leader", "answerable"): 450,
            ("leader", "unanswerable"): 150,
            ("location", "answerable"): 450,
            ("location", "unanswerable"): 150,
        })

    def test_partial_epoch_and_invalid_arguments(self):
        schedule = uniform_schedule(48, 50, 42)
        self.assertEqual(len(schedule), 50)
        self.assertEqual(sorted(schedule[:48]), list(range(48)))
        for n, steps in ((0, 48), (48, 0), (-1, 48)):
            with self.assertRaises(ValueError):
                uniform_schedule(n, steps, 42)


if __name__ == "__main__":
    unittest.main()
