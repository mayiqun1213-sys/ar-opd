"""Small in-memory replay utilities for local Teacher supervision."""

from __future__ import annotations

import math

from ar_opd.distillation import (
    LocalSFTDataset,
    LocalSFTExample,
    LocalSFTKind,
)


_STATE_VERSION = 1
_ROW_FIELDS = {
    "observation",
    "target_action",
    "kind",
    "collection_id",
    "episode_index",
    "decision_id",
    "primitive_offset",
    "segment_length",
    "weight",
}

SegmentKey = tuple[int, int, int, LocalSFTKind]
Segment = tuple[LocalSFTExample, ...]


def _segment_key(example: LocalSFTExample) -> SegmentKey:
    return (
        example.collection_id,
        example.episode_index,
        example.decision_id,
        example.kind,
    )


def _validated_segments(
    examples: tuple[LocalSFTExample, ...],
    *,
    expected_kind: LocalSFTKind,
) -> tuple[Segment, ...]:
    """Split a replay bucket into complete, contiguous Teacher segments."""

    segments: list[Segment] = []
    seen: set[SegmentKey] = set()
    index = 0
    while index < len(examples):
        key = _segment_key(examples[index])
        if key in seen:
            raise ValueError("a local SFT segment must be contiguous and unique")
        seen.add(key)

        stop = index + 1
        while stop < len(examples) and _segment_key(examples[stop]) == key:
            stop += 1
        segment = examples[index:stop]
        if key[-1] is not expected_kind:
            raise ValueError("local SFT example kind does not match its replay bucket")

        segment_length = segment[0].segment_length
        if any(row.segment_length != segment_length for row in segment):
            raise ValueError("rows in one local SFT segment disagree on segment_length")
        if len(segment) != segment_length:
            raise ValueError("local SFT replay contains an incomplete segment")
        offsets = tuple(row.primitive_offset for row in segment)
        if offsets != tuple(range(segment_length)):
            raise ValueError("local SFT segment offsets must be contiguous and ordered")
        if expected_kind is LocalSFTKind.CORRECTIVE and segment_length != 1:
            raise ValueError("a corrective local SFT segment must contain exactly one row")
        if not math.isclose(
            sum(row.weight for row in segment),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("local SFT segment weights must sum to one")

        segments.append(segment)
        index = stop
    return tuple(segments)


def _bounded_segment_suffix(
    segments: tuple[Segment, ...],
    *,
    capacity: int,
) -> tuple[LocalSFTExample, ...]:
    """Keep the newest complete segment suffix under a primitive-row soft cap."""

    kept_reversed: list[Segment] = []
    retained_rows = 0
    for segment in reversed(segments):
        if kept_reversed and retained_rows + len(segment) > capacity:
            break
        kept_reversed.append(segment)
        retained_rows += len(segment)
    kept_reversed.reverse()
    return tuple(row for segment in kept_reversed for row in segment)


def _merge_bucket(
    replay: tuple[LocalSFTExample, ...],
    fresh: tuple[LocalSFTExample, ...],
    *,
    kind: LocalSFTKind,
    capacity: int,
) -> tuple[LocalSFTExample, ...]:
    replay_segments = _validated_segments(replay, expected_kind=kind)
    fresh_segments = _validated_segments(fresh, expected_kind=kind)
    replay_keys = {_segment_key(segment[0]) for segment in replay_segments}
    fresh_keys = {_segment_key(segment[0]) for segment in fresh_segments}
    if replay_keys & fresh_keys:
        raise ValueError("fresh local SFT segments must have unique collection provenance")
    return _bounded_segment_suffix(
        replay_segments + fresh_segments,
        capacity=capacity,
    )


def append_local_sft_replay(
    replay: LocalSFTDataset,
    fresh: LocalSFTDataset,
    *,
    capacity_per_kind: int,
) -> LocalSFTDataset:
    """Append fresh complete segments with a primitive-row soft cap per kind.

    Eviction always removes whole oldest segments. If the newest segment alone
    exceeds the configured capacity, it is retained in full rather than sliced.
    """

    if capacity_per_kind < 1:
        raise ValueError("local SFT replay capacity must be positive")
    corrective = _merge_bucket(
        replay.corrective,
        fresh.corrective,
        kind=LocalSFTKind.CORRECTIVE,
        capacity=capacity_per_kind,
    )
    fallback = _merge_bucket(
        replay.fallback,
        fresh.fallback,
        kind=LocalSFTKind.FALLBACK,
        capacity=capacity_per_kind,
    )
    return LocalSFTDataset(corrective=corrective, fallback=fallback)


def _example_state(example: LocalSFTExample) -> dict[str, object]:
    return {
        "observation": [float(value) for value in example.observation],
        "target_action": example.target_action,
        "kind": example.kind.value,
        "collection_id": example.collection_id,
        "episode_index": example.episode_index,
        "decision_id": example.decision_id,
        "primitive_offset": example.primitive_offset,
        "segment_length": example.segment_length,
        "weight": example.weight,
    }


def local_sft_replay_state_dict(replay: LocalSFTDataset) -> dict[str, object]:
    """Return a versioned, JSON-safe replay payload containing only primitives."""

    _validated_segments(replay.corrective, expected_kind=LocalSFTKind.CORRECTIVE)
    _validated_segments(replay.fallback, expected_kind=LocalSFTKind.FALLBACK)
    state: dict[str, object] = {
        "version": _STATE_VERSION,
        "corrective": [_example_state(row) for row in replay.corrective],
        "fallback": [_example_state(row) for row in replay.fallback],
    }
    # Validate with the same schema used on load. A successful save must never
    # create replay state that this version cannot subsequently restore.
    load_local_sft_replay_state_dict(state)
    return state


def _required_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"local SFT replay field {field!r} must be an integer")
    return value


def _required_non_negative_int(value: object, *, field: str) -> int:
    converted = _required_int(value, field=field)
    if converted < 0:
        raise ValueError(f"local SFT replay field {field!r} must be non-negative")
    return converted


def _required_float(value: object, *, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"local SFT replay field {field!r} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"local SFT replay field {field!r} must be finite")
    return converted


def _load_bucket(
    value: object,
    *,
    expected_kind: LocalSFTKind,
) -> tuple[LocalSFTExample, ...]:
    if not isinstance(value, list):
        raise ValueError("local SFT replay buckets must be lists")
    examples: list[LocalSFTExample] = []
    for row_index, raw_row in enumerate(value):
        if not isinstance(raw_row, dict) or set(raw_row) != _ROW_FIELDS:
            raise ValueError("local SFT replay row has unexpected fields")
        raw_observation = raw_row["observation"]
        if not isinstance(raw_observation, list) or not raw_observation:
            raise ValueError("local SFT replay observation must be a non-empty list")
        observation = tuple(
            _required_float(item, field=f"observation[{item_index}]")
            for item_index, item in enumerate(raw_observation)
        )
        raw_kind = raw_row["kind"]
        if not isinstance(raw_kind, str):
            raise ValueError("local SFT replay kind must be a string")
        try:
            kind = LocalSFTKind(raw_kind)
        except ValueError as error:
            raise ValueError(f"unknown local SFT replay kind at row {row_index}") from error
        examples.append(
            LocalSFTExample(
                observation=observation,
                target_action=_required_non_negative_int(
                    raw_row["target_action"], field="target_action"
                ),
                kind=kind,
                collection_id=_required_int(
                    raw_row["collection_id"], field="collection_id"
                ),
                episode_index=_required_int(
                    raw_row["episode_index"], field="episode_index"
                ),
                decision_id=_required_int(
                    raw_row["decision_id"], field="decision_id"
                ),
                primitive_offset=_required_int(
                    raw_row["primitive_offset"], field="primitive_offset"
                ),
                segment_length=_required_int(
                    raw_row["segment_length"], field="segment_length"
                ),
                weight=_required_float(raw_row["weight"], field="weight"),
            )
        )
    loaded = tuple(examples)
    _validated_segments(loaded, expected_kind=expected_kind)
    return loaded


def load_local_sft_replay_state_dict(state: object) -> LocalSFTDataset:
    """Load and validate replay state created by the matching state-dict helper."""

    if not isinstance(state, dict) or set(state) != {
        "version",
        "corrective",
        "fallback",
    }:
        raise ValueError("local SFT replay state has unexpected fields")
    if _required_int(state["version"], field="version") != _STATE_VERSION:
        raise ValueError("unsupported local SFT replay state version")
    return LocalSFTDataset(
        corrective=_load_bucket(
            state["corrective"], expected_kind=LocalSFTKind.CORRECTIVE
        ),
        fallback=_load_bucket(
            state["fallback"], expected_kind=LocalSFTKind.FALLBACK
        ),
    )
