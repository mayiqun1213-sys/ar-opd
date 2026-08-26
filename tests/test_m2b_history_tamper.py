import tempfile
import unittest
from copy import deepcopy

from ar_opd.distillation import LocalSFTDataset
from ar_opd.train_toy import (
    ToyTrainConfig,
    _METRIC_SCHEMA_VERSION,
    _disabled_opd_metric_values,
    _validate_restored_history,
    run_training,
)


class M2BHistoryRateTamperTest(unittest.TestCase):
    def test_disabled_history_rejects_enabled_only_opd_trace(self) -> None:
        valid_row = {
            "update": 1.0,
            "metric_schema_version": float(_METRIC_SCHEMA_VERSION),
            **_disabled_opd_metric_values(),
            "teacher_query_cost": 0.0,
            "teacher_execution_cost": 0.0,
            "total_teacher_resource_cost": 0.0,
            "replay_corrective_sft_examples": 0.0,
            "replay_fallback_sft_examples": 0.0,
        }
        config = ToyTrainConfig(updates=1)
        _validate_restored_history(
            1,
            [valid_row],
            [],
            LocalSFTDataset(),
            config=config,
        )

        tampered = {
            **valid_row,
            "student_only_success_before_opd": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "enabled-stage metrics"):
            _validate_restored_history(
                1,
                [tampered],
                [],
                LocalSFTDataset(),
                config=config,
            )

    def test_enabled_history_rejects_out_of_range_success_and_disagreement(
        self,
    ) -> None:
        config = ToyTrainConfig(
            seed=157,
            updates=1,
            episodes_per_update=1,
            evaluation_episodes=1,
            hidden_size=4,
            goal_position=3,
            trap_positions=(1,),
            max_steps=5,
            ppo_epochs=1,
            opd_episodes_per_update=1,
            opd_epochs=1,
            opd_learning_rate=0.1,
        )
        with tempfile.TemporaryDirectory() as directory:
            valid_row = run_training(config, output_dir=directory)["updates"][0]

        for key in (
            "student_only_success_before_opd",
            "opd_disagreement_before",
        ):
            with self.subTest(metric=key):
                tampered = deepcopy(valid_row)
                tampered[key] = 2.0
                with self.assertRaisesRegex(ValueError, "rates must lie"):
                    _validate_restored_history(
                        1,
                        [tampered],
                        [],
                        LocalSFTDataset(),
                        config=config,
                    )


if __name__ == "__main__":
    unittest.main()
