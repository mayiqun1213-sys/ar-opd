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

The current deterministic toy loop covers value-gated S/T/F rollout,
decision-boundary PPO, executed-only local SFT, and strict Student-only OPD
without model downloads or external services. See
[`docs/implementation_notes.md`](docs/implementation_notes.md),
[`docs/local_distillation.md`](docs/local_distillation.md), and
[`docs/opd.md`](docs/opd.md) for the enforced data and optimizer boundaries.
TextWorldExpress support will be added behind an environment adapter.

## Repository policy

Source, tests, small configurations, and concise experiment metadata belong in
Git. Secrets, model weights, datasets, generated checkpoints, caches, and
large run logs do not.
