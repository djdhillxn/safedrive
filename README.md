# SafeRL-Drive: MetaDrive MVP with PPO and SAC

SafeRL-Drive is a focused Phase-1 autonomous-driving reinforcement-learning project. It
trains continuous-control PPO and SAC agents in MetaDrive, evaluates them on unseen
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
2. **IDM baseline** — evaluate MetaDrive's `IDMPolicy` without training for 50 unseen
   episodes beginning at seed 1000.
3. **PPO** — train one continuous-control vector-observation run for 500,000 timesteps
   using four subprocess environments.
4. **SAC** — train one comparable run for 500,000 timesteps using one environment.
5. **Best-model evaluation** — explicitly evaluate the PPO and SAC best checkpoints on
   50 unseen scenarios.
6. **Videos** — record one top-down best-model rollout for PPO and one for SAC.
7. **Final comparison** — compare IDM, PPO, and SAC and write the Phase-1 CSV, JSON, and
   plot.
8. **Report generation** — fill the result placeholders and compile the LaTeX report.

Training scenarios begin at seed 0. Evaluation scenarios begin at seed 1000, so the
reported evaluation roads are disjoint from the training range. Phase 1 intentionally has
one training run for PPO and one for SAC; it does not run seed sweeps.

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
7. Run the IDM baseline and copy it to Drive.
8. Run PPO and copy it to Drive.
9. Run SAC and copy it to Drive.
10. Evaluate the best PPO and SAC models.
11. Record one video for each learned agent.
12. Build and display the Phase-1 comparison.
13. Compile the report if `latexmk` is available.
14. Perform the final `runs/` and `reports/` sync to Drive.

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

Evaluate the rule-based IDM baseline:

```bash
python -m scripts.evaluate_baseline \
  --config configs/ppo_mvp.yaml \
  --episodes 50 \
  --prefix idm_unseen
```

Train PPO and SAC:

```bash
python -m scripts.train --config configs/ppo_mvp.yaml
python -m scripts.train --config configs/sac_mvp.yaml
```

Dotlist overrides remain available:

```bash
python -m scripts.train --config configs/ppo_mvp.yaml \
  train.total_timesteps=1000000 \
  eval.episodes=50
```

The successful runs update `runs/latest_idm.txt`, `runs/latest_ppo.txt`, and
`runs/latest_sac.txt`. The smoke test uses `runs/latest_smoke.txt`, so it cannot replace the
latest report-quality PPO pointer.

MetaDrive owns one process-global simulation engine. During training, callback evaluation
runs in a separate subprocess even when PPO, SAC, or the smoke test uses one training
environment. Do not configure more than one MetaDrive environment with `DummyVecEnv`; use
`subproc` or set `n_envs: 1`.

Evaluate a trained best model:

```bash
python -m scripts.evaluate \
  --run-dir runs/<ppo-run> \
  --model best \
  --episodes 50 \
  --prefix best_unseen
```

If an interrupted short run did not create `best_model.zip`, the evaluation command gives
a clear error and can be rerun with `--model final`. Video recording falls back from best
to final automatically with a warning:

```bash
python -m scripts.record_video \
  --run-dir runs/<ppo-run> \
  --model best \
  --steps 1000
```

The Colab helper `restore_run_from_drive("<run-directory>", "ppo")` restores an existing
PPO run from Drive and recreates its latest-run pointer. Use `"sac"` for a SAC run. This
allows best-model evaluation or video recording in a new runtime without retraining.

## Interpreting a weak PPO result

The smoke test is intentionally easier than the Phase-1 benchmark: it has no traffic, two
simple scenarios, and only checks wiring. It is not a performance baseline. Compare PPO
with IDM on the same unseen seeds instead.

Always evaluate `best_model.zip` before judging a completed run. The final PPO update can
be worse than an earlier checkpoint. New runs save `best_vecnormalize.pkl` beside the best
model so that checkpoint is evaluated with the matching observation statistics. Older
runs fall back to final normalization statistics with a warning.

To distinguish poor learning from poor generalization, evaluate the same model on its
training seed range under a separate prefix:

```bash
python -m scripts.evaluate \
  --run-dir runs/<ppo-run> \
  --model best \
  --episodes 50 \
  --prefix best_train_seeds \
  eval.start_seed=0 \
  eval.num_scenarios=50
```

High training-seed success with low unseen success indicates overfitting. Low success on
both ranges indicates an optimization or reward problem. The Phase-1 configs now enable
lane-centered progress, strengthen success and safety terminal rewards, reduce the pure
speed incentive, and use a larger PPO policy network.

Manual summary comparison is still supported:

```bash
python -m scripts.compare_runs \
  --summaries \
    runs/<ppo-run>/eval/final_unseen_summary.json \
    runs/<sac-run>/eval/final_unseen_summary.json \
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
│   └── tensorboard/                 # when enabled
├── models/
│   ├── final_model.zip
│   ├── best_model.zip               # when EvalCallback runs
│   ├── vecnormalize.pkl
│   ├── best_vecnormalize.pkl         # statistics captured with best_model.zip
│   └── replay_buffer.pkl             # SAC
├── checkpoints/
├── eval/
│   ├── final_unseen_episodes.csv
│   ├── final_unseen_summary.json
│   ├── best_unseen_episodes.csv      # after explicit best evaluation
│   └── best_unseen_summary.json
├── plots/
│   ├── training_returns.png
│   ├── eval_route_completion.png
│   └── eval_outcome_rates.png
└── videos/
    └── <algorithm>_<best-or-final>_topdown.mp4
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

Per-episode CSV files include the episode number, scenario seed, return, length, success,
collision, off-road, timeout, cumulative cost, route completion, mean speed, and serialized
final MetaDrive `info` data.

Summary JSON files include:

- episode count;
- mean and standard deviation of return;
- mean episode length;
- success, collision, off-road, and timeout rates;
- mean cost;
- mean route completion;
- mean speed when MetaDrive exposes it.

## Report

`reports/main.tex` is the concise portfolio report. It compiles before results exist and
uses explicit TODO fields until the final values are copied from
`runs/phase1_comparison.csv`. `reports/surrogate_notes.tex` holds exact commands,
hyperparameters, runtime metadata, failed attempts, implementation notes, and deferred
ideas.

Build from the repository root:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error reports/main.tex
```
