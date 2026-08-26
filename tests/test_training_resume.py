import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from ar_opd.train_toy import ToyTrainConfig, run_training


def assert_nested_equal(
    test: unittest.TestCase,
    first: object,
    second: object,
) -> None:
    if isinstance(first, torch.Tensor):
        test.assertIsInstance(second, torch.Tensor)
        test.assertTrue(torch.equal(first, second))
    elif isinstance(first, dict):
        test.assertIsInstance(second, dict)
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            assert_nested_equal(test, first[key], second[key])
    elif isinstance(first, list | tuple):
        test.assertIs(type(second), type(first))
        test.assertEqual(len(first), len(second))
        for left, right in zip(first, second, strict=True):
            assert_nested_equal(test, left, right)
    else:
        test.assertEqual(first, second)


class TrainingResumeTest(unittest.TestCase):
    def test_exact_resume_matches_uninterrupted_and_rejects_config_mismatch(
        self,
    ) -> None:
        config = ToyTrainConfig(
            seed=23,
            updates=2,
            episodes_per_update=2,
            evaluation_episodes=1,
            hidden_size=8,
            learning_rate=0.003,
            goal_position=4,
            trap_positions=(1,),
            max_steps=8,
            ppo_epochs=1,
            local_sft_epochs=2,
            local_sft_learning_rate=0.02,
            local_sft_replay_capacity_per_kind=16,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uninterrupted = run_training(
                config,
                output_dir=root / "uninterrupted",
            )
            partial = run_training(
                replace(config, updates=1),
                output_dir=root / "partial",
            )
            partial_checkpoint_path = Path(partial["checkpoint"])
            partial_checkpoint = torch.load(
                partial_checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(partial_checkpoint["completed_updates"], 1)
            self.assertEqual(len(partial_checkpoint["metrics"]), 1)
            partial_replay = partial_checkpoint["local_sft_replay"]
            self.assertGreater(
                len(partial_replay["corrective"]) + len(partial_replay["fallback"]),
                0,
            )

            resumed = run_training(
                config,
                output_dir=root / "resumed",
                resume_from=partial_checkpoint_path,
            )
            self.assertEqual(resumed["start_update"], 1)
            self.assertEqual(resumed["resumed_from"], str(partial_checkpoint_path))
            self.assertEqual(uninterrupted["start_update"], 0)

            uninterrupted_checkpoint = torch.load(
                uninterrupted["checkpoint"],
                map_location="cpu",
                weights_only=True,
            )
            resumed_checkpoint = torch.load(
                resumed["checkpoint"],
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(uninterrupted_checkpoint["completed_updates"], 2)
            self.assertEqual(resumed_checkpoint["completed_updates"], 2)
            for field in (
                "model_state_dict",
                "ppo_optimizer_state_dict",
                "local_sft_replay",
                "rng_state",
            ):
                with self.subTest(checkpoint_field=field):
                    assert_nested_equal(
                        self,
                        uninterrupted_checkpoint[field],
                        resumed_checkpoint[field],
                    )

            uninterrupted_metrics_path = Path(uninterrupted["metrics"])
            resumed_metrics_path = Path(resumed["metrics"])
            self.assertEqual(
                uninterrupted_metrics_path.read_bytes(),
                resumed_metrics_path.read_bytes(),
            )
            uninterrupted_metrics = [
                json.loads(line)
                for line in uninterrupted_metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            resumed_metrics = [
                json.loads(line)
                for line in resumed_metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(uninterrupted_metrics), 2)
            assert_nested_equal(self, uninterrupted_metrics, resumed_metrics)
            assert_nested_equal(self, uninterrupted["updates"], resumed["updates"])
            assert_nested_equal(
                self,
                uninterrupted_checkpoint["metrics"],
                resumed_checkpoint["metrics"],
            )
            assert_nested_equal(
                self,
                uninterrupted["local_sft_evaluations"],
                resumed["local_sft_evaluations"],
            )
            self.assertEqual(len(resumed["local_sft_evaluations"]), 2)
            assert_nested_equal(
                self,
                uninterrupted_checkpoint["local_sft_evaluations"],
                resumed_checkpoint["local_sft_evaluations"],
            )

            with self.assertRaisesRegex(
                ValueError,
                "immutable fields: gamma",
            ):
                run_training(
                    replace(config, gamma=0.91),
                    output_dir=root / "mismatched",
                    resume_from=partial_checkpoint_path,
                )


if __name__ == "__main__":
    unittest.main()
