# SafeRL-Drive: MetaDrive MVP with PPO and SAC

SafeRL-Drive is a focused Phase-1 autonomous-driving reinforcement-learning project. It
trains continuous-control PPO and SAC agents in MetaDrive, evaluates them on held-out
procedural scenarios, and compares them with MetaDrive's rule-based IDM controller.

The project uses vector observations and records success, collision, off-road, timeout,
route-completion, return, cost, speed, and episode-length metrics. Every experiment keeps
its resolved config, detailed logs, metadata, and artifacts in one run directory.

## Project structure

```text
safedrive/
├── configs/
│   ├── ppo_mvp.yaml
│   ├── sac_mvp.yaml
│   └── smoke_test.yaml
├── notebooks/
│   ├── colab_smoke_test.ipynb
│   └── phase1_colab_driver.ipynb
├── reports/
│   ├── main.tex
│   ├── surrogate_notes.tex
│   └── references.bib
├── saferl_drive/
│   ├── algorithms.py
│   ├── config.py
│   ├── envs.py
│   ├── evaluation.py
│   └── utils.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_baseline.py
│   ├── record_video.py
│   ├── plot_results.py
│   └── compare_runs.py
├── pyproject.toml
└── requirements.txt
```

## Installation

Python 3.10 or 3.11 is the safest local choice for simulation packages. Current Colab
runtimes use Python 3.12, so MetaDrive is pinned to an upstream revision that supports it.

```bash
conda create -n saferl-drive python=3.10 -y
conda activate saferl-drive
cd safedrive
pip install -e .
```

SafeRL-Drive uses Gymnasium. The Colab notebooks remove the obsolete `gym` distribution
before installation so Stable-Baselines3 does not print Gym's maintenance warning.

## Phase-1 experiment plan

Run only this limited matrix, in order:

1. **Smoke test** — verify MetaDrive, Stable-Baselines3, training, evaluation, plotting,
   and artifact creation with `configs/smoke_test.yaml`. This is not a report result.
2. **Native-policy gates** — require the expert to succeed on at least 80% of 10 fixed
   validation trials, then repeat IDM and require reproducible outcomes.
3. **Control pilots** — run PPO and SAC for 100,000 steps without traffic. Each must reach
   at least 10% validation success and 50% route completion before a full run is allowed.
4. **PPO** — train one continuous-control vector-observation run for 500,000 timesteps
   using four subprocess environments.
5. **SAC** — train one comparable run for 500,000 timesteps using one environment.
6. **Held-out evaluation** — after both learned configurations are frozen, evaluate IDM
   and each frozen validation winner once on 100 held-out test scenarios.
7. **Videos** — record one top-down best-model rollout for PPO and one for SAC.
8. **Final comparison** — compare IDM, PPO, and SAC and write the Phase-1 CSV, JSON, and
   plot.
9. **Report generation** — fill the result placeholders and compile the LaTeX report.

Training scenarios begin at seed 0. Fixed validation scenarios begin at seed 1000 and are
used for checkpoint selection. The reported test scenarios begin at seed 3000 and are not
consulted until the checkpoint is frozen. Seeds 2000--2099 were inspected during the July
diagnosis and are no longer considered held out. Validation and test traffic generation is
deterministic, while full training keeps randomized traffic for diversity. Phase 1 uses a
three-block road and traffic density 0.05 as a deliberately learnable control benchmark;
harder maps and traffic are difficulty extensions, not prerequisites for proving the
training pipeline works.

## Google Colab driver notebook

Use [`notebooks/phase1_colab_driver.ipynb`](notebooks/phase1_colab_driver.ipynb) from VS
Code connected to a Google Colab runtime. The notebook uses this layout:

```text
VS Code --git push--> GitHub --clone/pull--> /content/safedrive
                                                   |
                                                   +--> /content/drive/MyDrive/SafeDrive
                                                        persistent artifacts only
```

GitHub and `/content/safedrive` are the source of truth for code. Do not run the repository
directly from mounted Drive; Drive is slower for many small files and is used only to back
up completed run directories, comparison outputs, and reports.

Notebook run order:

1. Define paths and experiment constants.
2. Check the runtime, GPU, and PyTorch CUDA status.
3. Mount Drive and approve Google's authentication prompt.
4. Clone or fast-forward pull the public GitHub repository.
5. Remove legacy Gym and install the repository with `pip install -e .`.
6. Run the smoke test.
7. Run the expert task-sanity and deterministic IDM reproducibility gates.
8. Run both short learning gates and copy their artifacts to Drive.
9. Run gated PPO and copy it to Drive.
10. Run gated SAC and copy it to Drive.
11. Run the IDM, best PPO, and best SAC held-out tests and copy them to Drive.
12. Record one video for each learned agent.
13. Build and display the Phase-1 comparison.
14. Compile the report if `latexmk` is available, then perform the final artifact sync.

The long experiment sections are independent. In a later Colab session, rerun the setup
sections and then continue from the required experiment. `FULL_TIMESTEPS` defaults to
500,000 and can be changed to 1,000,000 at the top of the notebook for an intentionally
longer run. The smaller `notebooks/colab_smoke_test.ipynb` remains available as a quick
VS Code, Colab, GitHub, Drive, GPU, and headless-MetaDrive connection check.

## Command-line workflow

Run the smoke test first:

```bash
python -m scripts.train --config configs/smoke_test.yaml
```

Run the rule-based reproducibility gate:

```bash
python -m scripts.evaluate_baseline \
  --config configs/ppo_mvp.yaml \
  --split validation \
  --episodes 10 \
  --prefix idm_repro \
  --verify-repeat
```

Before a full run, execute the 100,000-step no-traffic control pilots used by the Colab
notebook. They use separate latest pointers, so they cannot replace report-quality runs:

```bash
python -m scripts.train --config configs/ppo_mvp.yaml --run-name ppo_control_pilot \
  experiment.latest_name=ppo_pilot train.total_timesteps=100000 \
  train.checkpoint_freq=25000 train.eval_freq=25000 validation.episodes=20 \
  metadrive.traffic_density=0.0 metadrive.random_traffic=false \
  validation.traffic_density=0.0

python -m scripts.train --config configs/sac_mvp.yaml --run-name sac_control_pilot \
  experiment.latest_name=sac_pilot train.total_timesteps=100000 \
  train.checkpoint_freq=25000 train.eval_freq=25000 validation.episodes=20 \
  metadrive.traffic_density=0.0 metadrive.random_traffic=false \
  validation.traffic_density=0.0
```

The notebook enforces the numerical gates. If either pilot fails, stop and inspect its
validation history and `training_diagnostics.json`; do not launch the corresponding full
run. Once both pass, training selects and freezes a validation winner without inspecting
the held-out test split:

```bash
python -m scripts.train --config configs/ppo_mvp.yaml
python -m scripts.train --config configs/sac_mvp.yaml
```

Dotlist overrides remain available:

```bash
python -m scripts.train --config configs/ppo_mvp.yaml \
  train.total_timesteps=1000000 \
  test.episodes=100
```

The successful runs update `runs/latest_idm.txt`, `runs/latest_ppo.txt`, and
`runs/latest_sac.txt`. The smoke test uses `runs/latest_smoke.txt`, so it cannot replace the
latest report-quality PPO pointer.

MetaDrive owns one process-global simulation engine. During training, validation
runs in a separate subprocess even when PPO, SAC, or the smoke test uses one training
environment. Do not configure more than one MetaDrive environment with `DummyVecEnv`; use
`subproc` or set `n_envs: 1`.

The PPO configuration explicitly keeps its MLP policy on CPU and collects experience in
four MetaDrive subprocesses. Each update receives 4,096 samples and uses batches of 256,
giving more frequent updates while retaining 16 minibatches per epoch. More
workers are not automatically better: keep `n_envs` at or below the Colab runtime's
logical CPU count and leave capacity for the learner and evaluation subprocess.

SAC keeps `device: auto`. On a GPU Colab runtime this selects CUDA; on a machine without
CUDA it falls back to CPU. SAC performs actor and critic gradient updates after experience
collection, so the GPU is more useful than it is for PPO's small MLP. The selected device
is printed at startup and saved in `run_metadata.json`. To compare SAC wall-clock speed on
a particular runtime, run a short trial with `algorithm.kwargs.device=cpu`; GPU speedups
are workload-dependent. The corrected SAC setup also reduces the learning rate, uses a
larger replay buffer and batch, and fixes the entropy coefficient at 0.05 because the prior
automatic coefficient and critic grew without bound together.

MetaDrive's vector observation is already bounded in `[0, 1]`, so the current configs do
not add `VecNormalize` observation statistics. A SafeDrive reward wrapper makes the
reported objective explicit: slow driving, lateral lane deviation, unstable steering, and timeout are penalized;
terminal success and safety outcomes dominate dense progress. `truncate_as_terminate` is
enabled because a horizon timeout is a task failure, not a state from which the value
function should bootstrap.

After both learned configurations are frozen, run the IDM test and evaluate each trained
best model on the same held-out split:

```bash
python -m scripts.evaluate_baseline \
  --config configs/ppo_mvp.yaml \
  --split test \
  --episodes 100 \
  --prefix idm_test

python -m scripts.evaluate \
  --run-dir runs/<ppo-run> \
  --model best \
  --split test \
  --episodes 100 \
  --prefix best_test
```

If an interrupted short run did not create `best_model.zip`, the evaluation command gives
a clear error and can be rerun with `--model final`. Video recording falls back from best
to final automatically with a warning:

```bash
python -m scripts.record_video \
  --run-dir runs/<ppo-run> \
  --model best \
  --seed 3007 \
  --steps 1000
```

## Interpreting a weak PPO result

The smoke test is intentionally easier than the Phase-1 benchmark: it has no traffic, two
simple scenarios, and only checks wiring. It is not a performance baseline. Compare PPO
with IDM on the same held-out seeds instead.

Always evaluate `best_model.zip` before judging a completed run. The final PPO update can
be worse than an earlier checkpoint. The best checkpoint is now selected first by
validation success and then by route completion and lower failure rates, instead of mean
reward alone. New runs save `best_vecnormalize.pkl` beside the best model so that checkpoint
is evaluated with the matching observation statistics. Older runs fall back to final
normalization statistics with a warning.

To distinguish poor learning from poor generalization, evaluate the same model on its
training seed range under a separate prefix:

```bash
python -m scripts.evaluate \
  --run-dir runs/<ppo-run> \
  --model best \
  --split train \
  --episodes 50 \
  --prefix best_train_seeds
```

High training-seed success with low validation success indicates overfitting. Low success on
both ranges indicates an optimization, reward, termination, or task-difficulty problem. The
Phase-1 configs now use a learnable three-block road, align terminal and timeout incentives
with the reported metrics, reduce the pure progress incentive, penalize unstable control,
and use a larger policy network.

Manual summary comparison is still supported:

```bash
python -m scripts.compare_runs \
  --summaries \
    runs/<ppo-run>/eval/best_test_summary.json \
    runs/<sac-run>/eval/best_test_summary.json \
  --output runs/ppo_vs_sac_eval_summary.png
```

After the IDM, PPO, and SAC latest pointers exist, generate the complete comparison with:

```bash
python -m scripts.compare_runs --phase1
```

## Expected outputs

Each training run contains:

```text
run_dir/
├── resolved_config.yaml
├── run_metadata.json
├── logs/
│   ├── train.log
│   ├── train_monitor/*.monitor.csv
│   ├── training_diagnostics.json
│   └── tensorboard/                 # when enabled
├── models/
│   ├── final_model.zip
│   ├── best_model.zip               # success-first validation winner
│   ├── vecnormalize.pkl              # only when normalization is enabled
│   ├── best_vecnormalize.pkl         # only when normalization is enabled
│   └── replay_buffer.pkl             # SAC
├── checkpoints/
├── eval/
│   ├── validation_history.json
│   ├── best_validation_episodes.csv
│   ├── best_validation_summary.json
│   ├── best_test_episodes.csv
│   └── best_test_summary.json
├── plots/
│   ├── training_returns.png
│   ├── eval_route_completion.png
│   └── eval_outcome_rates.png
└── videos/
    ├── <algorithm>_<best-or-final>_seed<seed>_topdown.mp4
    └── <algorithm>_<best-or-final>_seed<seed>_topdown.json
```

The IDM run uses the same evaluation CSV, summary JSON, plots, log, and metadata layout,
without models or training. The project-level final outputs are:

```text
runs/phase1_manifest.jsonl
runs/phase1_comparison.csv
runs/phase1_comparison.json
runs/phase1_comparison.png
runs/phase1_training_returns.png       # when both monitor logs are available
runs/phase1_compare.log
reports/main.pdf                       # when LaTeX is compiled
```

Console output is intentionally concise. Detailed arguments, system and package versions,
Git commit, CUDA/GPU information, resolved configuration, paths, and exception stack traces
live in each operation's log file.

## Metrics

Per-episode CSV files include the episode number, scenario seed, shaped and base return,
reward-shaping penalty, length, success, collision, off-road, timeout, cumulative cost,
route completion, mean speed, steering/throttle/brake behavior, action variation, and
serialized final MetaDrive `info` data.

Summary JSON files include:

- episode count;
- mean and standard deviation of return;
- mean episode length;
- success, collision, off-road, and timeout rates;
- a 95% Wilson confidence interval for success rate;
- mean cost;
- mean route completion;
- mean speed when MetaDrive exposes it;
- mean base return and shaping penalty;
- steering magnitude and saturation, throttle/brake rates, and action variation.

## Report

`reports/main.tex` is the concise portfolio report. It compiles before results exist and
uses explicit TODO fields until the final values are copied from
`runs/phase1_comparison.csv`. `reports/surrogate_notes.tex` holds exact commands,
hyperparameters, runtime metadata, failed attempts, implementation notes, and deferred
ideas.

Build from the repository root:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error reports/main.tex
```
