# Environment adapter boundary

## M3a scope

The first adapter milestone changes dependency direction and resource
ownership without changing the toy algorithm or checkpoint schema:

```text
EnvironmentAdapter.open_episode(seed, require_teacher)
    -> fresh online environment
    -> optional Teacher policy
    -> candidate evaluator / scratch owner
    -> RolloutCollector
    -> close evaluator and environment on success or failure
```

`EnvironmentSpec` is the policy-facing contract: a fixed-length finite vector
and a dense integer action range. `RolloutEnvironment` deliberately does not
require cloning. Exact branching is a separate optional capability used only
by `CounterfactualCandidateEvaluator`; an online-only environment can inject a
different evaluator.

M3a evaluators must still be exact for the current episode state: the
previewed action prefix and terminal/truncated metadata must agree with real
execution, and candidate construction must not mutate the online environment.
A branch must not share mutable backend state with that environment.
Approximate or stochastic candidate semantics require a different candidate
record contract and are intentionally outside this milestone.

The collector keeps raw and encoded observations separate. The Student and
value model receive only encoded tuples. The Teacher receives the current raw
observation. The next encoded observation is derived from the exact object
returned by `step`, not from a later mutable environment property.

The adapter boundary is fail-closed. Every `step` result must be an
`EnvironmentStep`, so finite rewards and terminal-state invariants cannot be
bypassed by a structural lookalike. Accepted specs, steps, proposals, and
candidates are copied into exact base-record snapshots before use. Evaluator
output must contain exactly S for an unprobed decision or S/T/F for a probed
decision; its action sequences must match the sampled Student and Teacher
proposals, and its query/execution costs and preview metadata must match the
eventual execution. This prevents a malformed evaluator from making the
environment execute one action while PPO silently trains another.

`open_episode` is the sole episode-seed and ownership boundary. Every episode
gets fresh components, the collector calls `reset` exactly once, and the
adapter closes resources even if Student inference, Teacher generation,
candidate evaluation, or environment stepping raises. Student-only collection
sets `require_teacher=False`, so it does not construct a Teacher merely to
leave it unused.

The deterministic toy runtime implements this contract while preserving the
existing S/T/F scores, probe RNG order, Torch generator stream, Teacher costs,
termination behavior, and exact-resume results. Toy branches are closed after
each preview and cannot mutate the online environment.

## TextWorldExpress boundary for M3b

This machine currently has neither Java nor TextWorldExpress, so M3a does not
claim a working JVM integration. The future backend must remain a raw
environment layer underneath a text/action codec rather than pretending its
dynamic action strings are the fixed dense action space used by the current
toy actor.

The official TextWorldExpress wrapper starts a JVM through Py4J when
`TextWorldExpressEnv` is constructed. Its `reset` accepts seed/game parameters,
`step` accepts an action string, and `infos["validActions"]` changes with the
state. Its public `done` combines task outcomes and an internal step limit.
Therefore the AR-OPD backend must:

- pass a complete explicit episode seed/spec on every reset;
- validate actions against the current dynamic valid-action set;
- maintain its own action trace, cumulative score, and boundary fingerprint;
- classify task success/failure as termination and enforce the project step
  cap separately as truncation;
- use idempotent explicit close around every JVM resource;
- use a bounded scratch environment plus reset/replay for counterfactual
  branches, rather than launching the upstream replay-based `clone` once per
  candidate;
- treat generated gold paths as initial-state walkthroughs, not as a local
  correction oracle after Student divergence.

The next implementation step is a dependency-free fake TextWorldExpress
backend that tests dynamic action mapping, replay cursors, boundary mismatch
failure, and termination/truncation classification. A real backend and smoke
test follow only after a compatible Java/Python runtime is available.

Primary upstream references:

- [TextWorldExpress repository and usage](https://github.com/cognitiveailab/TextWorldExpress)
- [Python wrapper implementation](https://github.com/cognitiveailab/TextWorldExpress/blob/main/textworld_express/textworld_express.py)
- [Official wrapper tests](https://github.com/cognitiveailab/TextWorldExpress/blob/main/tests/test_textworld_express.py)
