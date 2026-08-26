# Implementation notes

## Milestone 1: runnable toy loop

The first closed loop deliberately uses no external environment, model, or
service. `JammedChainEnv` makes recovery observable: advancing onto a trap jams
the agent, a repair clears it, and an oracle teacher can either repair one step
(`T`) or repair and continue for a short horizon (`F`).

The student is a shared PyTorch actor/critic. The value gate has no trainable
parameters and is not a router. On a probed step, one teacher query produces
the correction and recovery candidates. The toy candidate evaluator previews
each S/T/F action sequence on a cloned deterministic environment, adds the
same PPO critic as its tail value, subtracts query and per-step execution
costs, and takes the highest net score. Exact ties prefer `S`, then `T`, then
`F`.

Environment cloning is isolated inside `CounterfactualCandidateEvaluator`.
It is a test scaffold, not an assumption of the final method. Real environments
will replace it with action-conditioned value estimates and sparse controlled
counterfactual replay.

PPO and value learning use decision-boundary semi-Markov transitions. A
correction has duration one; a recovery folds its primitive rewards and costs
into a duration-aware macro reward and bootstraps only after control returns to
the gate. Primitive Teacher steps remain in the rollout for accounting and
future distillation, but are not separate critic states. This avoids assigning
one `V(s)` to both a state committed to continue `F` and the same environment
state at which S/T/F would be reconsidered. Because the toy observation embeds
the finite-horizon clock, both success and time-limit truncation bootstrap to
zero.

The Student proposal sampled before gating is the PPO actor action at every
decision, including decisions where T or F overrides its environment action.
The proposal influences both the gate and the resulting net return, so dropping
overridden proposals would train on the biased conditional sample
`policy(action | gate selected S)`. Executed Teacher actions are structurally
separate primitive records and can never be passed to the PPO actor loss.

Teacher accounting separates probes, model queries, generated candidate
steps, executed teacher steps, query cost, and execution cost. A query is paid
once on the first executed step even when the gate ultimately selects `S`,
because candidate generation has already occurred. The CLI always reports both
hybrid and strict student-only evaluation; the latter runs with probing fully
disabled.

Run the loop without downloading anything:

```bash
PYTHONPATH=src python -m ar_opd.train_toy \
  --config configs/toy_smoke.json \
  --output-dir /tmp/ar-opd-toy-smoke
```

Generated metrics and checkpoints are excluded from Git.

## Environment runtime boundary

The online rollout collector now depends on framework-neutral environment,
Teacher, gate, and candidate-evaluator protocols. A context-managed adapter
creates fresh seeded episode resources and owns their cleanup. Exact state
branching is an optional capability used by the deterministic toy evaluator,
not a requirement of every online environment. Student-only collection does
not instantiate a Teacher. The toy path remains numerically identical and the
checkpoint schema is unchanged.

TextWorldExpress will remain optional rather than a base dependency. This
machine has neither Java nor TextWorldExpress, so a real JVM backend cannot yet
be smoke-tested. Its dynamic string actions require a separate policy-facing
codec, and its replay-based clone is too expensive for one JVM per candidate.
The next milestone will use a fake backend to establish dynamic action mapping,
explicit episode traces, replay boundary fingerprints, and independent
termination/truncation classification before adding the real lazy-imported
backend. See [`environment_adapter.md`](environment_adapter.md) for the tested
M3a contract and the M3b integration constraints.
