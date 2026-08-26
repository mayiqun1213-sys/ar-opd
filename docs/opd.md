# Strict Student-only OPD

## State-distribution boundary

The first OPD implementation uses a separate state collection for every
training update:

```text
hybrid rollout -> PPO -> optional local SFT
               -> fresh stochastic Student-only rollout
               -> full-action Teacher annotation
               -> immediate forward-KL update -> discard OPD dataset
```

The OPD rollout sets probe probability to zero. Every actual transition must
come from `S`, all Teacher rollout counters and costs must remain zero, and the
general completed-rollout validator must pass before the first annotation
query occurs. A hybrid rollout is never relabeled as Student-only occupancy:
executed `T` or `F` actions would change later states.

Each ephemeral dataset has a `collection_id` matching the current update.
`opd_update` rejects another id. The dataset is neither replayed nor placed in
the checkpoint; after its configured local epochs it is discarded. Checkpoints
retain the model, PPO optimizer, metrics, configuration, and RNG stream, so a
resumed run recollects the next fresh Student-only distribution exactly. Metric
rows use schema version 2. When OPD is disabled, `opd_enabled` is zero,
`opd_collection_id` is the explicit `-1` sentinel, and all OPD-stage metrics
are zero. Resume accepts a pre-OPD M2a checkpoint only when the entire known
seven-field OPD config extension is absent and the current OPD settings remain
at their defaults; its historical rows are backfilled to the same disabled
schema before training continues.

## Full-distribution objective

The toy annotator returns a smoothed categorical distribution over every
environment action. In a jam it prefers `REPAIR`; otherwise it prefers
`ADVANCE`. The loss is the forward KL

```text
L_OPD = mean_s KL(q_teacher,target_temperature(. | s) || pi_student(. | s))
```

`opd_target_temperature` changes only the frozen Teacher target distribution.
Student logits remain at their execution temperature, so this parameter is not
the bilateral temperature used by some knowledge-distillation formulations.
The default `1.0` leaves Teacher probabilities unchanged. There is no DPO-style
chosen/rejected pair, reference policy, PPO old log-probability, or advantage in
this loss.

Like local SFT, OPD uses stateless actor-head-only SGD. Encoder and value-head
parameters remain bitwise unchanged. If OPD changes the actor, PPO Adam state
for the actor head is cleared before the next hybrid rollout.

## Cost accounting and failure behavior

Annotation resources live in `OPDAnnotationLedger`, separate from the hybrid
rollout ledger and PPO rewards:

- `opd_annotation_query_count` counts full-distribution queries;
- `opd_annotation_scored_actions` records how many action scores were returned;
- `opd_annotation_query_cost` records annotation cost only;
- `opd_rollout_teacher_*` metrics must remain zero;
- `total_teacher_resource_cost` sums hybrid query/execution cost and OPD
  annotation cost without inserting annotation cost into PPO rewards.

The annotator action dimension is checked before any query. All episodes are
validated in a query-free first pass. If annotation later fails,
`OPDAnnotationError.partial_ledger` preserves every successfully completed
query accounted before the failure.

## Running the smoke loop

The checked configuration deliberately disables local SFT so the run verifies
that OPD alone can distill the toy Oracle:

```bash
PYTHONPATH=src python -m ar_opd.train_toy \
  --config configs/toy_opd_smoke.json \
  --output-dir /tmp/ar-opd-opd-smoke
```

For a combined run, set both `local_sft_epochs` and `opd_epochs` above zero.
The order remains local SFT first, then a newly collected on-policy OPD batch.
