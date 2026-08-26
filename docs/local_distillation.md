# Local teacher distillation

## Executed-only data contract

Corrective-SFT and Fallback-SFT are intentionally extracted only from the
trajectory that the environment actually executed:

- a selected `T` decision contributes its one Teacher transition to
  Corrective-SFT;
- a selected `F` decision contributes each actually executed primitive
  transition to Fallback-SFT, using that transition's own observation;
- a selected `S` decision contributes no local SFT row;
- unselected Teacher candidates and an unexecuted recovery suffix never become
  SFT targets.

Before extraction, the rollout validator checks terminal placement, decision
slice bounds, contiguity, action source, selected option, candidate action
prefixes and terminal flags, decision-boundary observations, probe metadata,
and the Teacher cost ledger. Extraction fails closed if any invariant is
broken. Every executed Teacher step maps to exactly one fresh local SFT row.

Steps inside one recovery segment receive weights summing to one, so a longer
`F` does not dominate merely because it contains more primitive actions. Each
segment has globally unambiguous provenance
`(collection_id, episode_index, decision_id, kind)`. Corrective and Fallback
examples are retained in separate bounded replay buckets. Eviction removes
whole oldest segments; if one newest recovery segment exceeds the primitive-row
capacity, the complete segment is retained as a documented soft-cap exception.

## Optimization boundary

The update order is:

```text
fresh hybrid rollout -> PPO -> replay append -> local SFT -> fresh rollout
```

Local SFT trains only `actor_head` with stateless SGD. Encoder and value-head
parameters are bitwise unchanged, so the value predictions used by the gate do
not silently drift between PPO collections. After any local-SFT step, the
training loop clears PPO Adam state for the actor-head parameters. This avoids
carrying momentum computed for pre-distillation actor weights into the next
PPO update. PPO batches still contain Student proposals only, while the local
SFT dataset contains actual Teacher action targets only.

When local SFT is enabled, each update evaluates the same deterministic
student-only episodes immediately before and after distillation. Both paths
disable probing and assert that Teacher query, generation, execution, and cost
are all zero. Success improvement is reported but is not a unit-test condition:
a short stochastic run can already solve the toy task before distillation.

Metrics distinguish new rows from replay occupancy:

- `new_corrective_sft_examples` and `new_fallback_sft_examples` belong to the
  current hybrid rollout and sum to its `teacher_executed_steps`;
- `trained_*_sft_examples`, `replay_*_sft_examples`, and `replay_*_segments`
  describe the dataset used for the current local-SFT update;
- `replay_*_evicted_examples`, `replay_*_evicted_segments`, and
  `replay_*_soft_cap_ratio` expose bounded whole-segment replay behavior;
- `ppo_actor_optimizer_states_cleared` records the optimizer-state boundary.

## Checkpoint and resume

Every completed update writes an atomic, `weights_only=True`-loadable
checkpoint. It contains model and PPO optimizer state, complete local-SFT
replay, completed metrics and evaluations, plus Python, Torch, and rollout
generator RNG state. GPU runs also retain CUDA RNG state. A failed replacement
preserves the previous checkpoint.

Resume accepts an increased total update count but rejects changes to training
semantics such as seeds, environment, optimizer, or distillation settings. The
metrics file is reconstructed from checkpoint history before new rows are
appended, so a resumed run has the same model, optimizer, replay, RNG stream,
and JSONL history as an uninterrupted run.

Run the local-SFT smoke loop with:

```bash
PYTHONPATH=src python -m ar_opd.train_toy \
  --config configs/toy_local_sft_smoke.json \
  --output-dir /tmp/ar-opd-local-sft-smoke
```

Continue it after raising `updates` in the configuration:

```bash
PYTHONPATH=src python -m ar_opd.train_toy \
  --config configs/toy_local_sft_smoke.json \
  --resume /tmp/ar-opd-local-sft-smoke/checkpoint.pt \
  --output-dir /tmp/ar-opd-local-sft-resumed
```

## OPD boundary

Strict on-policy distillation is a separate stage. Hybrid rollouts cannot be
relabeled as student-only occupancy because T/F changes later states. OPD must
therefore execute Student actions only, ask the Teacher for a full action
distribution at each resulting decision state without executing it, keep
annotation costs separate from PPO rewards, and minimize forward KL on that
independent dataset. It does not reuse local-SFT targets as a proxy for a
Teacher distribution and does not introduce a DPO reference policy.
