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

## Replayable TextWorld boundary in M3b

M3b implements the raw backend boundary underneath a fixed policy-facing
action codec. `TextWorldRuntimeConfig` binds a backend/revision identity,
canonical game/fold/parameter configuration, signed-32-bit episode seed,
project action cap, encoder ABI, and ordered global command vocabulary. Dynamic
backend menus become Boolean masks over that vocabulary; local menu positions
never become policy action IDs. The upstream-only `help` command is rejected.

`BackendBoundary` carries the official observable fields needed for exact
replay: observation, look, inventory, valid actions, raw and normalized scores,
task description, and raw task success/failure flags. The runtime rejects
conflicting flags and derives failure, success (`tasksuccess` or normalized
`score >= 1.0`), or active centrally. Raw `done` must agree with that natural
classification. Only afterward does AR-OPD apply its own action cap as
truncation, with natural termination taking precedence.

Each online step records the stable ID, exact decoded command, score delta,
outcome, and before/after fingerprints. One independent scratch backend is
reused for every S/T/F preview. It resets with the complete episode spec,
replays the online command prefix, compares every trace row, then executes the
candidate. A mismatch fails closed and the online cursor is checked before and
after evaluation. The context-managed adapter closes scratch then online on
normal exit and every partial-construction or runtime exception.

The dependency-free fake backend and CLI test these contracts, including
dynamic masks, command/score drift, opaque `done`, terminal-versus-cap priority,
Student-only Teacher elision, `S -> F`, and one PPO update:

```bash
PYTHONPATH=src python -m ar_opd.fake_textworld_smoke
```

This machine still has neither Java nor TextWorldExpress, so no real JVM
integration is claimed. A future concrete backend must wrap upstream `close`
with its own idempotence guard and construct TextWorldExpress with an internal
step limit strictly above the AR-OPD cap, so upstream truncation cannot appear
as opaque `done`. Generated gold paths remain initial-state walkthroughs, not
local correction oracles after Student divergence. The full pinned upstream
analysis and source links live in [`textworld_replay.md`](textworld_replay.md).

The current Student interface also has no mask input. Until the next hard-mask
milestone, only deliberately valid fake policies are safe on this adapter.
