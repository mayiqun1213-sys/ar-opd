import json
import unittest
from dataclasses import replace

from ar_opd.distillation import LocalSFTDataset, LocalSFTExample, LocalSFTKind
from ar_opd.distillation_replay import (
    append_local_sft_replay,
    load_local_sft_replay_state_dict,
    local_sft_replay_state_dict,
)


def example(
    kind: LocalSFTKind,
    decision_id: int,
    *,
    collection_id: int = 0,
    primitive_offset: int = 0,
    segment_length: int = 1,
    weight: float | None = None,
) -> LocalSFTExample:
    return LocalSFTExample(
        observation=(float(decision_id), float(primitive_offset), 1.0),
        target_action=(decision_id + primitive_offset) % 2,
        kind=kind,
        collection_id=collection_id,
        episode_index=0,
        decision_id=decision_id,
        primitive_offset=primitive_offset,
        segment_length=segment_length,
        weight=1.0 / segment_length if weight is None else weight,
    )


def fallback_segment(
    decision_id: int,
    length: int,
    *,
    collection_id: int = 0,
) -> tuple[LocalSFTExample, ...]:
    return tuple(
        example(
            LocalSFTKind.FALLBACK,
            decision_id,
            collection_id=collection_id,
            primitive_offset=offset,
            segment_length=length,
        )
        for offset in range(length)
    )


class LocalSFTReplayTest(unittest.TestCase):
    def test_retains_rare_corrections_in_a_separate_bounded_bucket(self) -> None:
        replay = LocalSFTDataset(
            corrective=(example(LocalSFTKind.CORRECTIVE, 0),),
            fallback=(example(LocalSFTKind.FALLBACK, 1),),
        )
        fresh = LocalSFTDataset(
            fallback=(
                example(LocalSFTKind.FALLBACK, 2),
                example(LocalSFTKind.FALLBACK, 3),
            )
        )
        merged = append_local_sft_replay(replay, fresh, capacity_per_kind=2)

        self.assertEqual([row.decision_id for row in merged.corrective], [0])
        self.assertEqual([row.decision_id for row in merged.fallback], [2, 3])

    def test_small_capacity_retains_a_multistep_fallback_segment_in_full(self) -> None:
        old = LocalSFTDataset(
            fallback=(example(LocalSFTKind.FALLBACK, 1, collection_id=0),)
        )
        newest_segment = fallback_segment(2, 3, collection_id=1)
        merged = append_local_sft_replay(
            old,
            LocalSFTDataset(fallback=newest_segment),
            capacity_per_kind=1,
        )

        self.assertEqual(merged.fallback, newest_segment)
        self.assertGreater(len(merged.fallback), 1)
        self.assertEqual(
            [row.primitive_offset for row in merged.fallback],
            [0, 1, 2],
        )
        self.assertAlmostEqual(sum(row.weight for row in merged.fallback), 1.0)

    def test_rejects_incomplete_offsets_and_invalid_segment_weights(self) -> None:
        valid = fallback_segment(4, 2)
        cases = (
            (valid[1:], "incomplete"),
            ((valid[1], valid[0]), "offsets"),
            ((valid[0], replace(valid[1], segment_length=3)), "segment_length"),
            (
                (replace(valid[0], weight=0.4), replace(valid[1], weight=0.4)),
                "sum to one",
            ),
        )
        for rows, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    append_local_sft_replay(
                        LocalSFTDataset(),
                        LocalSFTDataset(fallback=rows),
                        capacity_per_kind=4,
                    )

    def test_state_dict_json_round_trip_preserves_complete_segments(self) -> None:
        replay = LocalSFTDataset(
            corrective=(
                example(LocalSFTKind.CORRECTIVE, 7, collection_id=3),
            ),
            fallback=fallback_segment(8, 3, collection_id=3),
        )

        state = local_sft_replay_state_dict(replay)
        json_state = json.loads(json.dumps(state))
        restored = load_local_sft_replay_state_dict(json_state)

        self.assertEqual(restored, replay)
        self.assertEqual(json_state["version"], 1)

    def test_state_dict_rejects_invalid_observation_and_target_action(self) -> None:
        valid = example(LocalSFTKind.CORRECTIVE, 9)
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

        for invalid, message in cases:
            with self.subTest(message=message):
                replay = LocalSFTDataset(corrective=(invalid,))
                with self.assertRaisesRegex(ValueError, message):
                    local_sft_replay_state_dict(replay)

    def test_rejects_zero_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            append_local_sft_replay(
                LocalSFTDataset(), LocalSFTDataset(), capacity_per_kind=0
            )


if __name__ == "__main__":
    unittest.main()
