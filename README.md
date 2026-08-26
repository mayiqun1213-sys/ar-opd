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

The initial implementation targets Python 3.10+ and PyTorch. A no-download
test run only needs the standard library:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

For an editable development install with optional lint/test tooling, use
`python -m pip install -e '.[dev]'`.

The first development stage uses a small deterministic environment so the
rollout, accounting, and optimization invariants can be tested without model
downloads or external services. TextWorldExpress support will be added behind
an environment adapter.

## Repository policy

Source, tests, small configurations, and concise experiment metadata belong in
Git. Secrets, model weights, datasets, generated checkpoints, caches, and
large run logs do not.
