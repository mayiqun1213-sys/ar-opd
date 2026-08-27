# AR-OPD

AR-OPD is a research prototype for value-gated teacher intervention in
step-level agent rollouts. At each probed environment state, a shared value
model compares three options:

- `S`: the student takes the next environment step;
- `T`: the teacher corrects one step and returns control;
- `F`: the teacher performs a short recovery and returns control.

The gate ranks predicted task value minus explicit teacher cost. Teacher
actions are retained for value learning and local distillation, but are never
treated as student actions in the PPO actor objective.

## Development setup

The implementation targets Python 3.10+ and PyTorch. Once the declared runtime
dependency is installed, tests need no model downloads or optional services:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

For an editable development install with optional lint/test tooling, use
`python -m pip install -e '.[dev]'`.

The deterministic toy loop covers value-gated S/T/F rollout, decision-boundary
PPO, executed-only local SFT, and strict Student-only OPD. A dependency-free
TextWorld-like smoke now covers fixed global action IDs, dynamic valid-action
masks, exact reset/replay candidate evaluation, and one PPO update:

```bash
PYTHONPATH=src python -m ar_opd.fake_textworld_smoke
```

Neither path downloads models or calls external services. See
[`docs/implementation_notes.md`](docs/implementation_notes.md),
[`docs/local_distillation.md`](docs/local_distillation.md),
[`docs/opd.md`](docs/opd.md), and
[`docs/environment_adapter.md`](docs/environment_adapter.md) for the enforced
data and optimizer boundaries. The exact TextWorld replay contract and current
real-JVM limitation are recorded in
[`docs/textworld_replay.md`](docs/textworld_replay.md).

## Repository policy

Source, tests, small configurations, and concise experiment metadata belong in
Git. Secrets, model weights, datasets, generated checkpoints, caches, and
large run logs do not.
