import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import torch

from ar_opd.distillation import LocalSFTDataset
from ar_opd.opd import (
    ActionDistributionAnnotation,
    OPDAnnotationLedger,
    OPDConfig,
    OPDExample,
    ToyOracleDistributionAnnotator,
    validate_probability_distribution,
)
from ar_opd.train_toy import (
    ToyTrainConfig,
    _BASE_OPD_METRIC_FIELDS,
    _METRIC_SCHEMA_VERSION,
    _OPD_CONFIG_FIELDS,
    _disabled_opd_metric_values,
    _migrate_m2a_metric_rows,
    _validate_resume_configuration,
    _validate_restored_history,
    run_training,
)


class M2BResumeMigrationBoundaryTest(unittest.TestCase):
    @staticmethod
    def _legacy_config(config: ToyTrainConfig) -> dict[str, object]:
        saved = asdict(config)
        for name in _OPD_CONFIG_FIELDS:
            del saved[name]
        return saved

    @staticmethod
    def _enabled_config() -> ToyTrainConfig:
        return ToyTrainConfig(
            seed=149,
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
            opd_annotation_query_cost=0.125,
        )

    @staticmethod
    def _recursive_payload_keys(value: object) -> set[str]:
        keys: set[str] = set()
        if isinstance(value, dict):
            for key, nested in value.items():
                if isinstance(key, str):
                    keys.add(key)
                keys.update(
                    M2BResumeMigrationBoundaryTest._recursive_payload_keys(nested)
                )
        elif isinstance(value, list | tuple):
            for nested in value:
                keys.update(
                    M2BResumeMigrationBoundaryTest._recursive_payload_keys(nested)
                )
        return keys

    def test_all_seven_missing_opd_fields_identify_default_config_as_m2a(self) -> None:
        current = ToyTrainConfig(updates=2)
        self.assertEqual(len(_OPD_CONFIG_FIELDS), 7)

        is_legacy_m2a = _validate_resume_configuration(
            self._legacy_config(current),
            current,
            completed_updates=1,
        )

        self.assertTrue(is_legacy_m2a)

    def test_missing_non_opd_field_is_not_accepted_as_m2a(self) -> None:
        current = ToyTrainConfig(updates=2)
        saved = self._legacy_config(current)
        del saved["gamma"]

        with self.assertRaisesRegex(ValueError, "gamma"):
            _validate_resume_configuration(saved, current, completed_updates=1)

    def test_partially_missing_opd_schema_is_rejected(self) -> None:
        current = ToyTrainConfig(updates=2)
        saved = asdict(current)
        del saved["opd_epochs"]

        with self.assertRaisesRegex(ValueError, "opd_epochs"):
            _validate_resume_configuration(saved, current, completed_updates=1)

    def test_legacy_metric_backfill_has_complete_disabled_schema_and_cost(self) -> None:
        legacy_row = {
            "update": 1.0,
            "teacher_query_cost": 0.125,
            "teacher_execution_cost": 0.375,
            "unrelated_metric": 7.0,
        }

        migrated = _migrate_m2a_metric_rows([legacy_row])

        self.assertEqual(len(migrated), 1)
        row = migrated[0]
        self.assertIsNot(row, legacy_row)
        self.assertNotIn("metric_schema_version", legacy_row)
        self.assertTrue(_BASE_OPD_METRIC_FIELDS <= row.keys())
        self.assertEqual(
            row["metric_schema_version"],
            float(_METRIC_SCHEMA_VERSION),
        )
        self.assertEqual(
            {key: row[key] for key in _disabled_opd_metric_values()},
            _disabled_opd_metric_values(),
        )
        self.assertEqual(row["opd_enabled"], 0.0)
        self.assertEqual(row["opd_collection_id"], -1.0)
        self.assertEqual(row["total_teacher_resource_cost"], 0.5)

    def test_legacy_metric_rows_with_any_m2b_trace_are_rejected(self) -> None:
        traces = (
            {"metric_schema_version": float(_METRIC_SCHEMA_VERSION)},
            {"total_teacher_resource_cost": 0.0},
            {"opd_enabled": 0.0},
            {"student_only_success_before_opd": 0.0},
        )
        for trace in traces:
            with self.subTest(trace=next(iter(trace))):
                row = {
                    "update": 1.0,
                    "teacher_query_cost": 0.0,
                    "teacher_execution_cost": 0.0,
                    **trace,
                }
                with self.assertRaisesRegex(ValueError, "M2b OPD fields"):
                    _migrate_m2a_metric_rows([row])

    def test_opd_integer_counts_reject_float_and_bool_early(self) -> None:
        invalid_factories = (
            (
                "toy episodes float",
                lambda: ToyTrainConfig(opd_episodes_per_update=1.0),
            ),
            (
                "toy episodes bool",
                lambda: ToyTrainConfig(opd_episodes_per_update=True),
            ),
            (
                "toy epochs float",
                lambda: ToyTrainConfig(
                    opd_episodes_per_update=1,
                    opd_epochs=1.0,
                ),
            ),
            (
                "toy epochs bool",
                lambda: ToyTrainConfig(
                    opd_episodes_per_update=1,
                    opd_epochs=True,
                ),
            ),
            ("OPDConfig epochs float", lambda: OPDConfig(epochs=1.0)),
            ("OPDConfig epochs bool", lambda: OPDConfig(epochs=True)),
        )
        for label, factory in invalid_factories:
            with self.subTest(case=label):
                with self.assertRaisesRegex(TypeError, "integer"):
                    factory()

    def test_public_opd_types_reject_bool_and_non_integer_inputs_early(self) -> None:
        for proposal in (1.0, True):
            with self.subTest(field="student_proposal_action", value=proposal):
                with self.assertRaisesRegex(TypeError, "integer"):
                    OPDExample(
                        observation=(0.0, 0.0, 1.0),
                        teacher_probabilities=(0.25, 0.75),
                        student_proposal_action=proposal,
                        collection_id=0,
                        episode_index=0,
                        decision_id=0,
                    )

        for action_size in (2.0, True):
            with self.subTest(field="action_size", value=action_size):
                with self.assertRaisesRegex(TypeError, "integer"):
                    validate_probability_distribution(
                        (0.25, 0.75),
                        action_size=action_size,
                    )

        invalid_query_cost_factories = (
            (
                "annotation",
                lambda: ActionDistributionAnnotation(
                    (0.25, 0.75), query_cost=True
                ),
            ),
            ("ledger", lambda: OPDAnnotationLedger(query_cost=True)),
            ("annotator", lambda: ToyOracleDistributionAnnotator(query_cost=True)),
        )
        for label, factory in invalid_query_cost_factories:
            with self.subTest(query_cost_type=label):
                with self.assertRaisesRegex(TypeError, "real number"):
                    factory()

        invalid_config_factories = (
            ("learning_rate", lambda: OPDConfig(learning_rate=True)),
            ("target_temperature", lambda: OPDConfig(target_temperature=True)),
            ("max_grad_norm", lambda: OPDConfig(max_grad_norm=True)),
        )
        for label, factory in invalid_config_factories:
            with self.subTest(config_field=label):
                with self.assertRaisesRegex(TypeError, "real number"):
                    factory()

    def test_enabled_opd_history_rejects_semantic_tampering(self) -> None:
        config = self._enabled_config()
        with tempfile.TemporaryDirectory() as directory:
            summary = run_training(config, output_dir=directory)

        valid_row = summary["updates"][0]
        _validate_restored_history(
            1,
            [deepcopy(valid_row)],
            [],
            LocalSFTDataset(),
            config=config,
        )

        for key in (
            "student_only_success_before_opd",
            "student_only_return_after_opd",
        ):
            with self.subTest(tamper=f"missing {key}"):
                row = deepcopy(valid_row)
                del row[key]
                with self.assertRaisesRegex(ValueError, "lack enabled OPD fields"):
                    _validate_restored_history(
                        1,
                        [row],
                        [],
                        LocalSFTDataset(),
                        config=config,
                    )

        for key in (
            "opd_examples",
            "opd_annotation_query_count",
            "opd_rollout_actor_rows",
        ):
            with self.subTest(tamper=f"inconsistent {key}"):
                row = deepcopy(valid_row)
                row[key] += 1.0
                with self.assertRaisesRegex(ValueError, "collection metrics"):
                    _validate_restored_history(
                        1,
                        [row],
                        [],
                        LocalSFTDataset(),
                        config=config,
                    )

        row = deepcopy(valid_row)
        row["opd_rollout_teacher_query_count"] = 1.0
        with self.assertRaisesRegex(ValueError, "used Teacher resources"):
            _validate_restored_history(
                1,
                [row],
                [],
                LocalSFTDataset(),
                config=config,
            )

        invalid_costs = (
            ("opd_annotation_query_cost", float("nan"), "finite real number"),
            ("opd_rollout_teacher_cost", -1.0, "non-negative"),
        )
        for key, value, message in invalid_costs:
            with self.subTest(tamper=f"invalid {key}"):
                row = deepcopy(valid_row)
                row[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    _validate_restored_history(
                        1,
                        [row],
                        [],
                        LocalSFTDataset(),
                        config=config,
                    )

    def test_enabled_checkpoint_recursively_excludes_ephemeral_opd_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = run_training(self._enabled_config(), output_dir=directory)
            checkpoint = torch.load(
                summary["checkpoint"],
                map_location="cpu",
                weights_only=True,
            )

        payload_keys = self._recursive_payload_keys(checkpoint)
        forbidden_keys = {
            "teacher_probabilities",
            "opd_dataset",
            "opd_replay",
        }
        self.assertTrue(
            payload_keys.isdisjoint(forbidden_keys),
            payload_keys & forbidden_keys,
        )

    def test_real_m2a_like_checkpoint_resumes_with_uniform_metric_schema(self) -> None:
        partial_config = ToyTrainConfig(
            seed=137,
            updates=1,
            episodes_per_update=1,
            evaluation_episodes=1,
            hidden_size=4,
            goal_position=3,
            trap_positions=(1,),
            max_steps=5,
            ppo_epochs=1,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = run_training(partial_config, output_dir=root / "partial")
            payload = torch.load(
                partial["checkpoint"],
                map_location="cpu",
                weights_only=True,
            )
            for name in _OPD_CONFIG_FIELDS:
                del payload["config"][name]
            for row in payload["metrics"]:
                for key in tuple(row):
                    if (
                        key == "metric_schema_version"
                        or key == "total_teacher_resource_cost"
                        or key.startswith("opd_")
                        or key.endswith("_opd")
                    ):
                        del row[key]
            legacy_checkpoint = root / "m2a-like.pt"
            torch.save(payload, legacy_checkpoint)

            resumed = run_training(
                replace(partial_config, updates=2),
                output_dir=root / "resumed",
                resume_from=legacy_checkpoint,
            )

            self.assertEqual(resumed["start_update"], 1)
            self.assertEqual(len(resumed["updates"]), 2)
            restored_row, fresh_row = resumed["updates"]
            self.assertEqual(restored_row.keys(), fresh_row.keys())
            self.assertTrue(_BASE_OPD_METRIC_FIELDS <= restored_row.keys())
            self.assertEqual(restored_row["metric_schema_version"], 2.0)
            self.assertEqual(restored_row["opd_enabled"], 0.0)
            self.assertEqual(restored_row["opd_collection_id"], -1.0)
            self.assertEqual(
                restored_row["total_teacher_resource_cost"],
                restored_row["teacher_query_cost"]
                + restored_row["teacher_execution_cost"],
            )


if __name__ == "__main__":
    unittest.main()
