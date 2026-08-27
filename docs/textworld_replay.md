# TextWorldExpress replay contract

## Scope and pinned upstream

M3b targets the official TextWorldExpress `main` revision
`db0e6c09802a9e66ed535fc81155f27b4cd3cbac`. This machine currently has
neither Java nor the TextWorldExpress Python package. The dependency-free fake
backend validates AR-OPD contracts; it does not claim that the real JVM
integration works. Run its complete rollout/PPO smoke with:

```bash
PYTHONPATH=src python -m ar_opd.fake_textworld_smoke
```

The pinned wrapper has the following behavior:

- Constructing `TextWorldExpressEnv(serverPath=None, envStepLimit=100)` starts
  one JVM and one Py4J gateway immediately. Separate instances own separate
  processes and ports.
- `reset(seed=None, gameFold=None, gameName=None, gameParams=None,
  generateGoldPath=False)` returns `(observation, infos)`. The wrapper caches
  the game name, integer-valued parameter string, fold, and seed. Omitting a
  seed after an explicit seeded reset reuses that seed. AR-OPD must always pass
  a complete explicit episode specification.
- `step(command)` returns `(observation, reward, done, infos)`. Commands are
  strings checked against the current `validActions`. Invalid commands do not
  advance the game state, but the wrapper still records a move. The special
  `help` command bypasses the simulator and move history, so it is not a policy
  action.
- Simulator infos contain `observation`, `look`, `inventory`, `validActions`,
  `scoreRaw`, normalized cumulative `score`, `tasksuccess`, and `taskfailure`.
  The wrapper adds `reward`, `done`, `numMoves`, and `taskDescription`; a normal
  step also adds `lastActionStr`. `reward` is the change in normalized score.
- `done` combines `tasksuccess`, `taskfailure`, `score >= 1.0`, and the wrapper
  step cap. The cap test is `numMoves > envStepLimit`, with reset stored as the
  first history entry. It is not a usable terminated/truncated classification.
- `clone()` serializes the episode specification and command history, creates
  another `TextWorldExpressEnv` and JVM, resets it, and replays every command.
  It is not an in-memory snapshot and must not be called once per candidate.
- `close()` shuts down the gateway and writes a newline to the Java process if
  it is still alive. The wrapper also calls `close()` from `__del__`, without an
  idempotence guard. AR-OPD must own explicit, idempotent cleanup.

## Policy action identity

The policy action space is a fixed global command vocabulary. Each command
string has one immutable integer ID for the entire run, checkpoint, and replay
history. The vocabulary and its digest are checkpointed configuration.

A state-local ordinal is not a policy action ID. For example, local ID 2 could
mean `open fridge` in one state and `move north` in another. Reusing that ID
would attach PPO log probabilities, value-gate decisions, and OPD Teacher
targets to different commands. Sorting `validActions` is useful for canonical
comparison, but it must not redefine model output coordinates.

Every state supplies a Boolean valid-action mask over the fixed vocabulary.
The adapter rejects empty commands, duplicates, unknown commands, and states
whose valid set cannot be represented. The wrapper-only `help` command is
rejected as a policy action. Sampling, deterministic selection,
log-probability calculation, and OPD distributions must all apply the same
mask. Hard policy action masking is the next milestone; M3b only establishes
and tests the backend, vocabulary, mask, and replay contracts.

## Episode and replay identity

An immutable episode specification contains at least:

- game name, canonical game parameters, and game fold;
- explicit signed-32-bit-compatible seed;
- AR-OPD maximum action count;
- fixed vocabulary digest; and
- the required backend/upstream revision identity.

The fold is one of `train`, `dev`, or `test`. Integer-valued game parameters
are parsed, deduplicated, sorted by key, and serialized canonically. The
wrapper's internal cap and cached defaults are not part of control flow. The
adapter owns the local action count and passes the full specification on every
backend reset. A real wrapper must set its internal cap strictly above the
project cap so it cannot emit an upstream-limit `done` first.

A replay trace stores decoded command strings, not only action IDs. Each
boundary also stores a canonical fingerprint of the observation, look text,
inventory, global valid-action mask, raw and normalized scores, task flags,
task description, and episode identity. Replay fails closed on the first
fingerprint mismatch.

Backend calls are transactional only at the wrapper boundary. Once a backend
`reset` or `step` begins, any exception, malformed result, opaque `done`, or
replay mismatch permanently marks that environment instance as faulted. Reads,
steps, resets, and further restores then fail closed; only idempotent `close`
remains available. In contrast, rejecting an invalid local action before the
backend is called does not poison the environment.

`state_token` is an optional backend debugging/fingerprint field; official
TextWorldExpress does not expose one. It is deliberately excluded from the
Student encoder and the fake Teacher does not use it.

Counterfactual evaluation owns one reusable scratch environment. For each S,
T, or F candidate it resets that scratch environment, replays the common
command prefix, verifies the boundary fingerprint, and then executes the
candidate. Sequential reuse avoids launching a JVM per candidate while keeping
the online environment untouched. The fake backend must exercise the same
reset/replay protocol.

## Boundary classification

AR-OPD treats upstream `done` only as a consistency check. It derives a natural
outcome centrally from raw `tasksuccess`, `taskfailure`, and the upstream
`score >= 1.0` completion rule, then applies its own action limit. Natural
termination takes precedence when it coincides with the limit.

| Natural outcome | Local limit reached | terminated | truncated | success |
| --- | --- | --- | --- | --- |
| ongoing | no | false | false | false |
| ongoing | yes | false | true | false |
| success | either | true | false | true |
| failure | either | true | false | false |

Conflicting success and failure flags are an adapter error. Active `done` is an
opaque backend terminal and fails closed; a natural terminal with `done=False`
is inconsistent too. A terminal reset, empty active action set, or malformed
info is raised before collection, so the wrapper's reset-time `done=False`
placeholder cannot hide it.

## Ownership

The episode adapter owns the online environment and its single scratch
environment. Neither is shared concurrently. Both are closed exactly once in
reverse construction order on normal exit, inference failure, replay mismatch,
or partial construction failure. Cleanup never relies on garbage collection.

## Pinned primary sources

- [Python wrapper](https://github.com/cognitiveailab/TextWorldExpress/blob/db0e6c09802a9e66ed535fc81155f27b4cd3cbac/textworld_express/textworld_express.py)
- [Official wrapper tests](https://github.com/cognitiveailab/TextWorldExpress/blob/db0e6c09802a9e66ed535fc81155f27b4cd3cbac/tests/test_textworld_express.py)
- [Scala step-result JSON](https://github.com/cognitiveailab/TextWorldExpress/blob/db0e6c09802a9e66ed535fc81155f27b4cd3cbac/simulator/src/main/scala/textworldexpress/struct/StepResult.scala)
- [Scala Python interface](https://github.com/cognitiveailab/TextWorldExpress/blob/db0e6c09802a9e66ed535fc81155f27b4cd3cbac/simulator/src/main/scala/textworldexpress/runtime/PythonInterface.scala)
- [Official repository and runtime notes](https://github.com/cognitiveailab/TextWorldExpress/tree/db0e6c09802a9e66ed535fc81155f27b4cd3cbac)
