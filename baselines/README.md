# Baselines

IPPO and MAPPO training and rollout collection for MPE, SMAX, Overcooked V1/V2,
Hanabi, and MaBrax. Inspired by [JaxMARL](https://github.com/FLAIROx/JaxMARL).

## Install

From the repository root, create a Python 3.12 environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r baselines/requirements-lock.txt
pip install jaxmarl==0.1.0 --no-deps
pip install -e ".[dev]"
```

The lock file contains the tested Linux CPU dependencies. JaxMARL is installed
separately because its dependency metadata requires older JAX and SciPy versions;
`pip check` will report those conflicts. Keep MuJoCo and MuJoCo-MJX at 3.1.3 for
Brax 0.10.3. GPU execution has not been tested.

## Test the pipeline

```bash
JAX_PLATFORMS=cpu python -m baselines.smoke --output outputs/smoke --steps 1024
JAX_PLATFORMS=cpu python -m baselines.smoke --suite hanabi --output outputs/smoke-hanabi --steps 1024
JAX_PLATFORMS=cpu python -m baselines.smoke --suite mabrax --output outputs/smoke-mabrax --steps 1024
```

These commands train eight MPE variants, four Hanabi variants, and four MaBrax
variants for 1,024 joint environment steps each. Each run reloads its checkpoint,
checks that the rollouts match, and computes all five diagnostics with two
permutation nulls. MaBrax uses Hopper with a shortened 25-step horizon here.
Logs and `smoke_report.json` are saved in each output directory, which must be
new. These short runs do not reproduce the paper's performance results.

Regression tests and formatting checks:

```bash
JAX_PLATFORMS=cpu pytest -q
ruff check baselines tests/test_baselines.py
ruff format --check baselines tests/test_baselines.py
```

## Train and collect

```bash
JAX_PLATFORMS=cpu python -m baselines.run \
  --algorithm ippo_rnn_nps --config ippo_rnn_mpe \
  --output outputs/ippo-rnn-seed0 \
  SEED=0 NUM_ENVS=4 NUM_STEPS=32 TOTAL_TIMESTEPS=1024 \
  NUM_MINIBATCHES=2 UPDATE_EPOCHS=2 FC_DIM_SIZE=16 GRU_HIDDEN_DIM=16 \
  TEST_NUM_ENVS=4 TEST_NUM_STEPS=100 TEST_DURING_TRAINING=True
```

Choose `ippo_ff_nps`, `ippo_rnn_nps`, `mappo_ff_nps`, or `mappo_rnn_nps`.
Append `_het` for MPE speaker/listener and tag. `--config` accepts a name from
`config/` or a YAML path; trailing `KEY=VALUE` arguments override settings.
Use a new output directory for each seed.

Recurrent runs require `FC_DIM_SIZE == GRU_HIDDEN_DIM` and `NUM_ENVS` divisible
by `NUM_MINIBATCHES`. Feed-forward rollout batches must also divide evenly into
minibatches. Training rounds the step budget down to complete batches of
`NUM_ENVS * NUM_STEPS`; `report.json` records the actual count.

Each run saves `config.json`, actor parameters in `policy.msgpack`, losses and
returns in `training_metrics.npz`, evaluation data in `rollouts.pkl`, and a
`report.json`. Checkpoints support evaluation only. Evaluation uses greedy
discrete actions or Gaussian means. Set `TEST_NUM_STEPS` long enough to finish
an episode; the reported return averages completed episodes only.

To collect fresh rollouts and compute metrics:

```bash
python -m baselines.collect --run outputs/ippo-rnn-seed0 \
  --output outputs/fresh-rollouts.pkl --seed 123 --num-envs 32 --num-steps 100

dec-pomdp-metrics --input outputs/fresh-rollouts.pkl \
  --output outputs/fresh_metrics.csv --history-k 3 --cmi-k 25 --null-reps 5
```

For MaBrax, add `--force-continuous-A` to the metrics command. Omitting collection
overrides reproduces the training run's evaluation. `--seed` changes only the
evaluation seed. Only load pickle datasets from trusted sources.

Rollouts include episode IDs and timesteps that track resets. Recurrent hidden
states are the GRU outputs used to choose each recorded action. MaBrax actions
contain only the agent's controlled actuators; unused padding is removed.

## Experiment settings

`experiments.json` contains 126 recipes. Resolve comma-separated seed and map
sweeps to one value each before passing overrides to the runner. The 24
Hanabi/MaBrax recipes marked `paper_defaults` use Appendix Tables 6–7's defaults,
not scenario-specific tuning. MPE Tag appears in Appendix Table 8's tuned
settings, but not Table 14's 37-scenario results in the
[paper](https://arxiv.org/pdf/2602.20804).

MaBrax supports `ant_4x2`, `halfcheetah_6x1`, `hopper_3x1`, `humanoid_9|8`, and
`walker2d_2x3`. Use the standard trainers with `CONTINUOUS_ACTIONS` enabled and
the padding mode specified in each recipe.
