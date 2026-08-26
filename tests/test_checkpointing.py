import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from ar_opd.checkpointing import load_training_checkpoint, save_training_checkpoint
from ar_opd.distillation import LocalSFTDataset, LocalSFTExample, LocalSFTKind
from ar_opd.models import ActorCritic


def replay_example(kind: LocalSFTKind, decision_id: int) -> LocalSFTExample:
    return LocalSFTExample(
        observation=(float(decision_id), 1.0, 0.5),
        target_action=decision_id % 2,
        kind=kind,
        episode_index=2,
        decision_id=decision_id,
        primitive_offset=0,
        segment_length=1,
        weight=1.0,
        collection_id=3,
    )


def populated_model_and_optimizer() -> tuple[ActorCritic, torch.optim.Optimizer]:
    model = ActorCritic(3, 2, hidden_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    observations = torch.tensor([[0.0, 1.0, 0.5], [0.4, 0.0, 0.7]])
    logits, values = model(observations)
    loss = logits.square().mean() + values.square().mean()
    loss.backward()
    optimizer.step()
    return model, optimizer


def assert_nested_equal(test: unittest.TestCase, first, second) -> None:
    if isinstance(first, torch.Tensor):
        test.assertTrue(torch.equal(first, second))
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            assert_nested_equal(test, first[key], second[key])
    elif isinstance(first, list | tuple):
        test.assertEqual(len(first), len(second))
        for left, right in zip(first, second, strict=True):
            assert_nested_equal(test, left, right)
    else:
        test.assertEqual(first, second)


class TrainingCheckpointTest(unittest.TestCase):
    def test_weights_only_round_trip_restores_training_and_rng_state(self) -> None:
        random.seed(29)
        torch.manual_seed(31)
        generator = torch.Generator().manual_seed(37)
        model, optimizer = populated_model_and_optimizer()
        replay = LocalSFTDataset(
            corrective=(replay_example(LocalSFTKind.CORRECTIVE, 4),),
            fallback=(replay_example(LocalSFTKind.FALLBACK, 5),),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "checkpoint.pt"
            saved_path = save_training_checkpoint(
                path,
                model=model,
                ppo_optimizer=optimizer,
                completed_updates=7,
                config={"seed": 29, "trap_positions": (1, 3)},
                metrics=[{"update": 7.0, "loss": 0.25}],
                local_sft_evaluations=[
                    {"update": 7, "student_only_after": {"success_rate": 1.0}}
                ],
                local_sft_replay=replay,
                generator=generator,
            )
            self.assertEqual(saved_path, path)
            raw = torch.load(path, weights_only=True)
            self.assertEqual(raw["format_version"], 1)
            self.assertIn("rng_state", raw)
            self.assertEqual(raw["rng_state"]["cuda"], [])

            expected_python = random.random()
            expected_torch = torch.rand(4)
            expected_generator = torch.rand(4, generator=generator)
            expected_model = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            expected_optimizer = optimizer.state_dict()

            random.seed(101)
            torch.manual_seed(103)
            generator.manual_seed(107)
            restored_model = ActorCritic(3, 2, hidden_size=8)
            restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=9.0)
            with mock.patch(
                "ar_opd.checkpointing.torch.cuda.is_available",
                return_value=False,
            ):
                loaded = load_training_checkpoint(
                    path,
                    model=restored_model,
                    ppo_optimizer=restored_optimizer,
                    generator=generator,
                )

            self.assertEqual(loaded.completed_updates, 7)
            self.assertEqual(loaded.config["trap_positions"], (1, 3))
            self.assertEqual(loaded.metrics[0]["loss"], 0.25)
            self.assertEqual(loaded.local_sft_evaluations[0]["update"], 7)
            self.assertEqual(loaded.local_sft_replay, replay)
            assert_nested_equal(self, restored_model.state_dict(), expected_model)
            assert_nested_equal(
                self, restored_optimizer.state_dict(), expected_optimizer
            )
            self.assertEqual(random.random(), expected_python)
            self.assertTrue(torch.equal(torch.rand(4), expected_torch))
            self.assertTrue(
                torch.equal(torch.rand(4, generator=generator), expected_generator)
            )

    def test_restore_rng_false_leaves_all_rng_streams_untouched(self) -> None:
        model, optimizer = populated_model_and_optimizer()
        saved_generator = torch.Generator().manual_seed(11)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_training_checkpoint(
                path,
                model=model,
                ppo_optimizer=optimizer,
                completed_updates=0,
                config={},
                metrics=[],
                local_sft_evaluations=[],
                local_sft_replay=LocalSFTDataset(),
                generator=saved_generator,
            )

            random.seed(41)
            torch.manual_seed(43)
            target_generator = torch.Generator().manual_seed(47)
            expected_python_state = random.getstate()
            expected_torch_state = torch.get_rng_state().clone()
            expected_generator_state = target_generator.get_state().clone()
            load_training_checkpoint(
                path,
                model=model,
                ppo_optimizer=optimizer,
                generator=target_generator,
                restore_rng=False,
            )
            self.assertEqual(random.getstate(), expected_python_state)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch_state))
            self.assertTrue(
                torch.equal(target_generator.get_state(), expected_generator_state)
            )

    def test_rejects_unsafe_metadata_before_writing(self) -> None:
        model, optimizer = populated_model_and_optimizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            with self.assertRaisesRegex(TypeError, "unsupported checkpoint value"):
                save_training_checkpoint(
                    path,
                    model=model,
                    ppo_optimizer=optimizer,
                    completed_updates=1,
                    config={"unsafe": object()},
                    metrics=[],
                    local_sft_evaluations=[],
                    local_sft_replay=LocalSFTDataset(),
                    generator=torch.Generator(),
                )
            self.assertFalse(path.exists())

    def test_rejects_invalid_replay_before_writing_checkpoint(self) -> None:
        model, optimizer = populated_model_and_optimizer()
        valid = replay_example(LocalSFTKind.CORRECTIVE, 4)
        cases = (
            (
                replace(
                    valid,
                    observation=(float("nan"), *valid.observation[1:]),
                ),
                "observation|finite",
            ),
            (
                replace(valid, target_action=-1),
                "target",
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (invalid, message) in enumerate(cases):
                with self.subTest(message=message):
                    path = root / f"checkpoint-{index}.pt"
                    with self.assertRaisesRegex(ValueError, message):
                        save_training_checkpoint(
                            path,
                            model=model,
                            ppo_optimizer=optimizer,
                            completed_updates=1,
                            config={},
                            metrics=[],
                            local_sft_evaluations=[],
                            local_sft_replay=LocalSFTDataset(
                                corrective=(invalid,)
                            ),
                            generator=torch.Generator(),
                        )
                    self.assertFalse(path.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_failed_save_preserves_previous_checkpoint_and_cleans_temp(self) -> None:
        model, optimizer = populated_model_and_optimizer()
        generator = torch.Generator().manual_seed(53)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            arguments = {
                "model": model,
                "ppo_optimizer": optimizer,
                "completed_updates": 1,
                "config": {},
                "metrics": [],
                "local_sft_evaluations": [],
                "local_sft_replay": LocalSFTDataset(),
                "generator": generator,
            }
            save_training_checkpoint(path, **arguments)
            previous = path.read_bytes()

            with mock.patch(
                "ar_opd.checkpointing.torch.save",
                side_effect=RuntimeError("interrupted save"),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted save"):
                    save_training_checkpoint(path, **arguments)

            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
