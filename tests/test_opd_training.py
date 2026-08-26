import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch

from ar_opd.train_toy import (
    ToyTrainConfig,
    _validate_resume_configuration,
    run_training,
)


class OPDTrainingIntegrationTest(unittest.TestCase):
    def test_fresh_student_only_opd_metrics_and_ephemeral_checkpoint_state(
        self,
    ) -> None:
        config = ToyTrainConfig(
            seed=41,
            updates=2,
            episodes_per_update=2,
            evaluation_episodes=1,
            hidden_size=8,
            learning_rate=0.003,
            goal_position=4,
            trap_positions=(1,),
            max_steps=8,
            probe_probability=1.0,
            ppo_epochs=1,
            local_sft_epochs=0,
            opd_episodes_per_update=2,
            opd_epochs=3,
            opd_learning_rate=0.1,
            opd_annotation_query_cost=0.125,
        )

        with tempfile.TemporaryDirectory() as directory:
            summary = run_training(config, output_dir=directory)
            self.assertEqual(len(summary["updates"]), config.updates)

            for update_index, metrics in enumerate(summary["updates"]):
                with self.subTest(update=update_index + 1):
                    opd_examples = metrics["opd_examples"]
                    annotation_queries = metrics["opd_annotation_query_count"]
                    rollout_actor_rows = metrics["opd_rollout_actor_rows"]
                    self.assertGreater(opd_examples, 0.0)
                    self.assertEqual(annotation_queries, opd_examples)
                    self.assertEqual(rollout_actor_rows, opd_examples)
                    self.assertEqual(
                        metrics["opd_annotation_scored_actions"],
                        2.0 * opd_examples,
                    )
                    self.assertAlmostEqual(
                        metrics["opd_annotation_query_cost"],
                        config.opd_annotation_query_cost * annotation_queries,
                    )

                    self.assertEqual(metrics["opd_enabled"], 1.0)
                    self.assertEqual(
                        metrics["opd_collection_id"],
                        float(update_index),
                    )
                    self.assertEqual(
                        metrics["opd_rollout_episodes"],
                        float(config.opd_episodes_per_update),
                    )
                    self.assertAlmostEqual(
                        metrics["opd_rollout_mean_steps"]
                        * metrics["opd_rollout_episodes"],
                        rollout_actor_rows,
                    )
                    for key in (
                        "opd_rollout_teacher_probe_count",
                        "opd_rollout_teacher_query_count",
                        "opd_rollout_teacher_executed_steps",
                        "opd_rollout_teacher_cost",
                    ):
                        self.assertEqual(metrics[key], 0.0, key)

                    self.assertAlmostEqual(
                        metrics["teacher_query_cost"],
                        config.teacher_query_cost * metrics["teacher_query_count"],
                    )
                    self.assertAlmostEqual(
                        metrics["total_teacher_resource_cost"],
                        metrics["teacher_query_cost"]
                        + metrics["teacher_execution_cost"]
                        + metrics["opd_annotation_query_cost"],
                    )
                    self.assertLess(
                        metrics["opd_kl_after"],
                        metrics["opd_kl_before"],
                    )
                    self.assertEqual(
                        metrics["opd_optimizer_steps"],
                        float(config.opd_epochs),
                    )
                    self.assertEqual(
                        metrics["ppo_actor_optimizer_states_cleared"],
                        2.0,
                    )

            checkpoint = torch.load(
                Path(summary["checkpoint"]),
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(len(checkpoint["metrics"]), config.updates)
            self.assertNotIn("opd_dataset", checkpoint)
            self.assertNotIn("opd_replay", checkpoint)
            self.assertFalse(
                any(key.startswith("opd_") for key in checkpoint),
                checkpoint.keys(),
            )

    def test_m2a_config_migration_allows_only_default_opd_fields(self) -> None:
        current = ToyTrainConfig(updates=2)
        saved_m2a = asdict(current)
        opd_fields = {
            name for name in tuple(saved_m2a) if name.startswith("opd_")
        }
        self.assertTrue(opd_fields)
        for name in opd_fields:
            del saved_m2a[name]

        _validate_resume_configuration(
            saved_m2a,
            current,
            completed_updates=1,
        )

        with self.assertRaisesRegex(
            ValueError,
            "opd_episodes_per_update, opd_epochs",
        ):
            _validate_resume_configuration(
                saved_m2a,
                replace(
                    current,
                    opd_episodes_per_update=1,
                    opd_epochs=1,
                ),
                completed_updates=1,
            )


if __name__ == "__main__":
    unittest.main()
