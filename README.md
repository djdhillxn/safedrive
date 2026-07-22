# SafeRL-Drive: Reproducible Control and Procedural Generalization in MetaDrive

SafeRL-Drive is a focused student reinforcement-learning project for closed-loop driving
in MetaDrive. Phase 1 asks a deliberately bounded question: can PPO and SAC reliably learn
lane following and route completion on one traffic-free straight
road, under the same observations, reward, and evaluation metrics? The learned
agents are compared with MetaDrive's rule-based IDM controller.

Phase 2 advances the credible continuous-control learner, SAC, into an equal-budget
direct-versus-curriculum experiment on unseen three-block procedural roads. The project
uses vector observations and records success, collision, off-road, timeout,
route-completion, return, cost, speed, and episode-length metrics. Every experiment keeps
its resolved config, detailed logs, metadata, and artifacts in one run directory.

## Project structure

```text
safedrive/
├── configs/
│   ├── ppo_mvp.yaml
│   ├── sac_mvp.yaml
│   ├── sac_phase2_direct.yaml
│   ├── sac_phase2_curriculum.yaml
│   └── smoke_test.yaml
├── notebooks/
│   ├── colab_smoke_test.ipynb
│   ├── phase1_colab_driver.ipynb
│   └── phase2_colab_driver.ipynb
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
│   ├── train_curriculum.py
│   ├── evaluate.py
│   ├── evaluate_baseline.py
│   ├── sync_drive_runs.py
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

## Scope and end objective

The core claim is intentionally modest: **standard PPO and SAC can learn reproducible
lane-following controls in this pipeline**. Phase 1 is complete only when
each learned agent's frozen best checkpoint reaches at least 80% success and 90% mean
route completion, with at most 10% collision, off-road, and timeout outcomes on 20
untouched test episodes. It does not claim robust autonomous driving, procedural-map generalization, or
algorithmic superiority from a single seed. PPO uses MetaDrive's documented beginner
3-by-3 steering/throttle action grid. SAC remains continuous, with steering limited to
`[-0.1, 0.1]` for basic lane centering and full throttle/brake. Because these action
interfaces differ, Phase 1 demonstrates two working learning controls and does not support
a PPO-versus-SAC ranking.

Run this limited matrix in order:

1. **Wiring smoke test** — verify installation, training, evaluation, plots, and artifacts.
2. **Native-policy checks** — verify IDM determinism and task feasibility.
3. **Learning controls** — train PPO and SAC for at most 100,000 steps on one
   straight road with no traffic. Validation runs every 25,000 steps; training stops after
   saving a checkpoint that also stays at or below 10% collision, off-road, and timeout outcomes.
4. **Held-out evaluation** — compare IDM, PPO, and SAC on 20 untouched test seeds.
5. **Videos and report** — record one rollout per learned agent, generate the comparison,
   and fill the report placeholders.

Difficulty increases only after those deliverables exist: first test continuous PPO and
unrestricted SAC steering on the same straight road, then use a fixed curved road, a small
set of traffic-free procedural roads, and finally light traffic. A failure at one
level is diagnosed at that level; it is not hidden by adding timesteps or more scenarios.
Validation begins at seed 1000 and held-out testing at seed 4000. The road topology and
no-traffic setting stay fixed while seeded straight-block parameters vary. This is a
narrow reproducibility test, not evidence of broad procedural generalization. Seeds
3000--3019 were consumed by the local diagnosis and are not reused for final reporting.
Validation seeds are sent only to the simulator subprocess. They do not reset Python,
NumPy, or PyTorch in the training process, so changing the validation episode count cannot
silently change later PPO updates.

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

### Pulling Drive runs onto a Mac

With Google Drive for desktop running, pull persistent artifacts into this repository's
root `runs/` folder with:

```bash
python -m scripts.sync_drive_runs
```

On macOS the script auto-detects a single
`~/Library/CloudStorage/GoogleDrive-*/My Drive/SafeDrive` folder. The location can also be
provided explicitly:

```bash
python -m scripts.sync_drive_runs \
  --drive-project "/Users/dhillo/Library/CloudStorage/GoogleDrive-dhillondheeraj84@gmail.com/My Drive/SafeDrive"
```

The operation is one-way from Drive to the repository. It merges completed and failed run
artifacts, skips experiments still marked `running`, preserves local-only files, copies
top-level comparison artifacts, and rebuilds the `latest_*.txt` pointers. The Mac default
is analysis-only: it keeps resolved configs, metadata, evaluation CSV/JSON, logs,
TensorBoard traces, plots, and videos, but does not even traverse Drive's `models/` or
`checkpoints/` directories. Model archives, checkpoint archives, and replay-buffer pickle
files therefore remain in Drive.

Use `--dry-run` to preview a sync. To remove training artifacts downloaded by an older
version of the script without changing Google Drive, run:

```bash
python -m scripts.sync_drive_runs --prune-local-training-artifacts
```

Section 4.1 of the Colab notebook passes `--include-training-artifacts` because the Colab
checkout needs models and checkpoints for evaluation, video recording, and possible run
continuation. That full mode should normally not be used on the Mac.

Notebook run order:

1. Define paths and experiment constants.
2. Check the runtime, GPU, and PyTorch CUDA status.
3. Mount Drive and approve Google's authentication prompt.
4. Clone or fast-forward pull the public GitHub repository.
5. Remove legacy Gym and install the repository with `pip install -e .`.
6. Run the smoke test.
7. Run the expert task-sanity and deterministic IDM reproducibility gates.
8. Train the Phase-1 PPO and SAC controls, enforce their gates, and copy them to Drive.
9. Inspect each qualifying validation summary.
10. Run the IDM, best PPO, and best SAC held-out tests and copy them to Drive.
11. Record one video for each learned agent.
12. Build and display the Phase-1 comparison.
13. Compile the report if `latexmk` is available, then perform the final artifact sync.

The experiment sections are independent. In a later Colab session, rerun the setup
sections and then continue from the required experiment. `PHASE1_TIMESTEPS` is 100,000;
the success-first callback may stop earlier after saving a qualifying checkpoint.
The smaller `notebooks/colab_smoke_test.ipynb` remains available as a quick VS Code,
Colab, GitHub, Drive, GPU, and headless-MetaDrive connection check.

## Phase-2 procedural generalization experiment

Use [`notebooks/phase2_colab_driver.ipynb`](notebooks/phase2_colab_driver.ipynb) for the
complete workflow. It reuses the Phase-1 GitHub, Colab, and Drive setup, closes Phase 1
with an exact held-out IDM run and corrected videos, and then runs one focused study:

- **Direct SAC:** train from scratch on 100 three-block procedural maps.
- **Curriculum SAC:** train on `C`, then `SC`, then three-block procedural maps while
  preserving actor, critic, optimizer, and replay-buffer state.
- **Shared target:** full continuous control, no training traffic, 500,000 maximum steps,
  25 fixed validation scenarios, and 100 untouched test scenarios.
- **Ablation:** the training sequence is the intended difference; architecture, reward,
  action interface, final task, test split, and total budget match.

Every evaluation summary stores two compatibility hashes. The task hash covers map,
traffic, horizon, termination, reward, and evaluation split. The strict hash also covers
the policy-facing action interface, SAC architecture, normalization, stopping target, and
maximum training budget. Phase-2 comparison requires the strict hash to match and refuses
to produce a plot or report macros otherwise. Phase 1 permits only a
descriptive task-outcome matrix because its controller interfaces intentionally differ.

The Phase-2 notebook order is:

1. Initialize paths, inspect the GPU, mount Drive, clone/pull, restore artifacts, and
   install the repository.
2. Run the lightweight wiring test.
3. Rerun IDM on Phase-1 test seeds 4000--4019, generate the fingerprint-checked Phase-1
   table, and record corrected PPO/SAC videos.
4. Evaluate IDM and ExpertPolicy on the Phase-2 validation task to confirm feasibility.
5. Train and test seed-0 direct SAC.
6. Train seed-0 curriculum SAC in three resumable stage cells. Each stage is copied to
   Drive before the next begins.
7. Compare the two pilots. Confirmation runs are justified only if either policy reaches
   50% success or 80% route completion.
8. Conditionally run seeds 1 and 2 for both conditions.
9. Evaluate Phase-2 IDM and ExpertPolicy on the exact test task, run the learned policies'
   zero-shot light-traffic stress tests, create videos, and generate the final comparison.
10. Compile both LaTeX reports and sync all final artifacts to Drive.

The direct command is:

```bash
python -m scripts.train \
  --config configs/sac_phase2_direct.yaml \
  --seed 0
```

The curriculum can run end-to-end:

```bash
python -m scripts.train_curriculum \
  --config configs/sac_phase2_curriculum.yaml \
  --seed 0
```

For safer Colab persistence, pause and sync at stage boundaries:

```bash
python -m scripts.train_curriculum \
  --config configs/sac_phase2_curriculum.yaml \
  --seed 0 \
  --stop-after-stage curve

python -m scripts.train_curriculum \
  --config configs/sac_phase2_curriculum.yaml \
  --run-dir runs/<curriculum-run> \
  --seed 0 \
  --stop-after-stage straight_curve

python -m scripts.train_curriculum \
  --config configs/sac_phase2_curriculum.yaml \
  --run-dir runs/<curriculum-run> \
  --seed 0
```

The large intermediate replay buffer is necessary for faithful curriculum continuation
and is retained in Drive. The default Mac sync excludes the entire `models/` directory,
so it is not downloaded for analysis.

After explicit held-out evaluations, compare one or more complete seed pairs:

```bash
python -m scripts.compare_runs --phase2 --seeds 0
python -m scripts.compare_runs --phase2 --seeds 0 1 2
```

Outputs include `phase2_seed_results.csv`, `phase2_comparison.csv/json/png`, combined
training curves, optional light-traffic tables and plots, videos, and generated LaTeX
macros. The comparison command fails closed if even one selected learned run has a
different task, action interface, reward, or test split.

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

Run the Phase-1 controls. These are the report runs; duplicating the same
configuration would spend compute without answering a new question:

```bash
python -m scripts.train --config configs/ppo_mvp.yaml --run-name ppo_phase1_control

python -m scripts.train --config configs/sac_mvp.yaml --run-name sac_phase1_control
```

The notebook enforces 80% success, 90% route completion, and maximum 10% collision,
off-road, and timeout gates. If either control fails,
stop and inspect its validation history and `training_diagnostics.json`; do not increase
task difficulty. A passing run has already selected and frozen its validation winner
without inspecting the held-out test split.

Dotlist overrides remain available:

```bash
python -m scripts.train --config configs/ppo_mvp.yaml \
  logging.console_level=DEBUG
```

The successful runs update `runs/latest_idm.txt`, `runs/latest_ppo.txt`, and
`runs/latest_sac.txt`. The smoke test uses `runs/latest_smoke.txt`, so it cannot replace the
latest report-quality PPO pointer.

MetaDrive owns one process-global simulation engine. During training, validation
runs in a separate subprocess even when PPO, SAC, or the smoke test uses one training
environment. Do not configure more than one MetaDrive environment with `DummyVecEnv`; use
`subproc` or set `n_envs: 1`.

The PPO configuration keeps its MLP policy on CPU and collects experience in four
MetaDrive subprocesses. Each update receives 4,096 samples and uses batches of 256. Its
3-by-3 discrete action grid follows MetaDrive's beginner PPO example and removes
destructive full-lock continuous exploration from the Phase-1 proof. More workers are not
automatically better: keep `n_envs` at or below the runtime's logical CPU count.

SAC keeps `device: auto`; Colab can use CUDA while a machine without CUDA falls back to
CPU. The bounded run uses a 100,000-transition buffer, begins updates after 5,000 steps,
and fixes the entropy coefficient at 0.05. The automatic control collapsed from about
0.22 to near zero during diagnosis, but fixed entropy alone still failed with unrestricted
steering. SAC's Phase-1 steering limit was the decisive task-boundary change. PPO instead
uses the documented discrete control abstraction. Replay-buffer
serialization is disabled because the file is large and is unnecessary for
evaluation, plots, or videos. The selected device is printed and saved in
`run_metadata.json`.

MetaDrive's vector observation is already bounded in `[0, 1]`, so the configs do not add
`VecNormalize`. Phase 1 deliberately returns to MetaDrive's documented reference reward:
dense longitudinal progress and speed, plus terminal success/failure values. The previous
custom penalties created a profitable stall policy under a long horizon. Safety remains
visible through success, collision, off-road, timeout, and cost metrics. The 500-step
horizon is treated as a Gymnasium truncation so value learning can bootstrap through the
artificial time limit.

After both learned configurations are frozen, run the IDM test and evaluate each trained
best model on the same held-out split:

```bash
python -m scripts.evaluate_baseline \
  --config configs/ppo_mvp.yaml \
  --split test \
  --episodes 20 \
  --prefix idm_test

python -m scripts.evaluate \
  --run-dir runs/<ppo-run> \
  --model best \
  --split test \
  --episodes 20 \
  --prefix best_test
```

If an interrupted short run did not create `best_model.zip`, the evaluation command gives
a clear error and can be rerun with `--model final`. Video recording falls back from best
to final automatically with a warning:

```bash
python -m scripts.record_video \
  --run-dir runs/<ppo-run> \
  --model best \
  --seed 4007 \
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

High training-seed success with low validation success indicates overfitting. Low success
on both ranges indicates an optimization, reward, termination, or task-difficulty problem.
Do not respond by jumping directly to more maps, traffic, or reward terms. First reproduce
the straight-road control; then change one difficulty dimension at a time.

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
│   └── replay_buffer.pkl             # SAC only when explicitly enabled
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
