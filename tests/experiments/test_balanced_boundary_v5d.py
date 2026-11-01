"""V5D dataset invariants, independent of model training."""
import unittest

from experiments.balanced_boundary_v5d.generate_data_v5d import (
    GROUPS, build_split, validate_disjoint, validate_split,
)


class V5DDataTests(unittest.TestCase):
    def test_balanced_train_and_dev(self):
        self.assertEqual(len(build_split("train")), 96)
        self.assertEqual(len(build_split("dev")), 96)

    def test_repeatable_generation(self):
        self.assertEqual(build_split("train"), build_split("train"))
        self.assertEqual(build_split("dev"), build_split("dev"))

    def test_existing_partitions_disjoint(self):
        validate_disjoint(build_split("train"), build_split("dev"))

    def test_corrupted_label_rejected(self):
        rows = build_split("train")
        rows[0] = {**rows[0], "answer": "Leader: Wrong."}
        with self.assertRaises(ValueError):
            validate_split("train", rows)

    def test_partition_prefix_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            validate_split("train", build_split("dev"))

    def test_unknown_partition_rejected(self):
        with self.assertRaises(ValueError):
            build_split("holdout")


if __name__ == "__main__":
    unittest.main()
