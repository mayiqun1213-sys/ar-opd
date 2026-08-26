import json
import math
import tempfile
import unittest
from pathlib import Path

from ar_opd.train_toy import ToyTrainConfig, run_training


class ToyTrainingSmokeTest(unittest.TestCase):
    def test_short_training_writes_finite_metrics_and_student_only_eval(self) -> None:
        config = ToyTrainConfig(
            seed=11,
            updates=1,
            episodes_per_update=4,
            evaluation_episodes=2,
            goal_position=2,
            trap_positions=(1,),
            max_steps=6,
            ppo_epochs=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = run_training(config, output_dir=directory)
            checkpoint = Path(summary["checkpoint"])
            metrics = Path(summary["metrics"])
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(metrics.is_file())
            update = json.loads(metrics.read_text(encoding="utf-8").strip())
            self.assertTrue(all(math.isfinite(value) for value in update.values()))
            self.assertEqual(summary["student_only_eval"]["teacher_query_count"], 0.0)
            self.assertEqual(summary["student_only_eval"]["teacher_executed_steps"], 0.0)
            self.assertEqual(summary["student_only_eval"]["teacher_query_cost"], 0.0)
            self.assertEqual(summary["student_only_eval"]["teacher_execution_cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
