import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from ar_opd.train_toy import ToyTrainConfig, run_training


class LocalSFTTrainingSmokeTest(unittest.TestCase):
    def test_training_reports_executed_only_sft_and_strict_student_eval(self) -> None:
        config = ToyTrainConfig(
            seed=17,
            updates=2,
            episodes_per_update=4,
            evaluation_episodes=2,
            goal_position=4,
            trap_positions=(1,),
            max_steps=8,
            ppo_epochs=2,
            local_sft_epochs=3,
            local_sft_learning_rate=0.02,
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = run_training(config, output_dir=directory)
            metric_lines = Path(summary["metrics"]).read_text(
                encoding="utf-8"
            ).splitlines()
            updates = [json.loads(line) for line in metric_lines]
            self.assertEqual(len(updates), 2)
            cumulative_corrective = 0.0
            cumulative_fallback = 0.0
            for update in updates:
                self.assertTrue(all(math.isfinite(value) for value in update.values()))
                self.assertEqual(
                    update["new_corrective_sft_examples"]
                    + update["new_fallback_sft_examples"],
                    update["teacher_executed_steps"],
                )
                cumulative_corrective += update["new_corrective_sft_examples"]
                cumulative_fallback += update["new_fallback_sft_examples"]
                self.assertEqual(update["trained_corrective_sft_examples"], cumulative_corrective)
                self.assertEqual(update["trained_fallback_sft_examples"], cumulative_fallback)
                self.assertEqual(
                    update["replay_corrective_sft_examples"], cumulative_corrective
                )
                self.assertEqual(
                    update["replay_fallback_sft_examples"], cumulative_fallback
                )
                self.assertGreater(update["teacher_executed_steps"], 0.0)
                self.assertEqual(update["local_sft_optimizer_steps"], 3.0)
                self.assertEqual(update["ppo_actor_optimizer_states_cleared"], 2.0)

            self.assertEqual(len(summary["local_sft_evaluations"]), 2)
            for evaluation in summary["local_sft_evaluations"]:
                for stage in ("student_only_before", "student_only_after"):
                    metrics = evaluation[stage]
                    self.assertEqual(metrics["teacher_probe_count"], 0.0)
                    self.assertEqual(metrics["teacher_query_count"], 0.0)
                    self.assertEqual(metrics["teacher_generated_steps"], 0.0)
                    self.assertEqual(metrics["teacher_executed_steps"], 0.0)
                    self.assertEqual(metrics["teacher_query_cost"], 0.0)
                    self.assertEqual(metrics["teacher_execution_cost"], 0.0)

            checkpoint = torch.load(summary["checkpoint"], weights_only=True)
            self.assertEqual(checkpoint["config"]["local_sft_epochs"], 3)
            self.assertEqual(checkpoint["completed_updates"], 2)
            self.assertIn("local_sft_replay", checkpoint)
            self.assertIn("rng_state", checkpoint)


if __name__ == "__main__":
    unittest.main()
