# SafeRL-Drive: MetaDrive MVP with PPO and SAC

Phase-1 pet/research project for autonomous-driving reinforcement learning in closed-loop simulation.

This repo trains and evaluates **PPO** and **SAC** agents in **MetaDrive** using **Stable-Baselines3**. The goal is not to claim “Tesla-style self-driving,” but to build a credible, reproducible RL autonomy benchmark with:

- continuous vehicle control in MetaDrive,
- PPO and SAC baselines,
- unseen-scenario evaluation,
- closed-loop driving metrics,
- top-down rollout videos,
- training/evaluation plots.

## Project structure

```text
saferl-drive-metadrive-mvp/
├── configs/
│   ├── ppo_mvp.yaml          # PPO baseline config
│   ├── sac_mvp.yaml          # SAC baseline config
│   └── smoke_test.yaml       # tiny installation/sanity test
├── notebooks/
│   └── colab_smoke_test.ipynb # VS Code/Colab/Drive connection test
├── saferl_drive/
│   ├── algorithms.py         # PPO/SAC construction
│   ├── config.py             # YAML loading and CLI overrides
│   ├── envs.py               # MetaDrive + SB3 VecEnv factories
│   ├── evaluation.py         # AV-specific closed-loop metrics
│   └── utils.py              # general helpers and plots
├── scripts/
│   ├── train.py              # train PPO/SAC
│   ├── evaluate.py           # evaluate final/best model
│   ├── record_video.py       # top-down rollout video
│   ├── plot_results.py       # generate plots
│   └── compare_runs.py       # compare PPO vs SAC summaries
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

Create a clean environment. Python 3.10 or 3.11 is usually the safest choice for simulation/RL packages.

```bash
conda create -n saferl-drive python=3.10 -y
conda activate saferl-drive

cd saferl-drive-metadrive-mvp
pip install -e .
```

The MetaDrive simulator is pinned to an upstream revision that supports Python 3.12, which is
used by current Colab runtimes. The latest PyPI release of `metadrive-simulator` still declares
Python `<3.12` support.

Quick sanity check:

```bash
python -m scripts.train --config configs/smoke_test.yaml
```

That should create a small run under `runs/` and verify that MetaDrive, Stable-Baselines3, training, evaluation, and plotting are wired correctly.

## VS Code and Google Colab

Use `notebooks/colab_smoke_test.ipynb` to verify the complete remote workflow before
starting a training run. The intended layout is:

```text
VS Code working tree --git push--> GitHub --clone/pull--> Colab /content/safedrive
                                                           |
                                                           +--> Google Drive/SafeDrive
                                                                (results and checkpoints)
```

GitHub is the source of truth for code. Google Drive is persistent artifact storage, not a
second Git working tree. Running the repository directly from mounted Drive is discouraged
because Drive-backed operations are slower and can fail when many small files are involved.

Before the first test:

1. Commit and push the local branch so Colab can clone the same revision.
2. Open `notebooks/colab_smoke_test.ipynb` in VS Code.
3. Choose **Select Kernel > Colab > New Colab Server**, sign in to Google, and select a GPU
   machine. **Auto Connect** is enough when a GPU is not required.
4. Run the notebook from top to bottom. Approve the separate Google Drive mount prompt.
5. If this repository is private, create a fine-grained GitHub token with read-only access to
   this repository, save it as `GITHUB_TOKEN` in Colab Secrets, and set `PRIVATE_REPO = True`
   in the notebook. Never paste the token into a code cell or commit it.

The notebook verifies the remote runtime, executes a small CUDA calculation, mounts and writes
to Drive, installs SafeDrive, and steps a headless MetaDrive environment. It intentionally does
not start RL training or consume more compute than needed for the connection test.

The workflow follows the [official Colab VS Code extension guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide).
Google notes that Colab VMs are temporary and recommends reducing reads and writes through a
[mounted Drive filesystem](https://research.google.com/colaboratory/faq.html). Also, a GPU does
not automatically make the current PPO configuration faster: Stable-Baselines3 recommends CPU
execution for [PPO with an MLP policy](https://stable-baselines3.readthedocs.io/en/v2.8.0/modules/ppo.html).
The GPU connection test is still useful now and becomes more relevant for larger networks or
image-based policies later.

## Train PPO

```bash
python -m scripts.train --config configs/ppo_mvp.yaml
```

Useful laptop-friendly override:

```bash
python -m scripts.train --config configs/ppo_mvp.yaml \
  train.total_timesteps=200000 \
  train.n_envs=2 \
  train.vec_env=dummy
```

Stronger run:

```bash
python -m scripts.train --config configs/ppo_mvp.yaml \
  train.total_timesteps=1000000 \
  metadrive.num_scenarios=100 \
  eval.episodes=100
```

## Train SAC

```bash
python -m scripts.train --config configs/sac_mvp.yaml
```

Useful shorter run:

```bash
python -m scripts.train --config configs/sac_mvp.yaml \
  train.total_timesteps=200000 \
  eval.episodes=30
```

## Evaluate a trained run

After training, each run directory will look something like:

```text
runs/20260426_123456_ppo_mvp_ppo_seed0/
```

Evaluate final model:

```bash
python -m scripts.evaluate \
  --run-dir runs/20260426_123456_ppo_mvp_ppo_seed0 \
  --model final \
  --episodes 50
```

Evaluate best SB3 EvalCallback model:

```bash
python -m scripts.evaluate \
  --run-dir runs/20260426_123456_ppo_mvp_ppo_seed0 \
  --model best \
  --episodes 50 \
  --prefix best_unseen
```

Outputs:

```text
run_dir/eval/*_episodes.csv
run_dir/eval/*_summary.json
run_dir/plots/eval_route_completion.png
run_dir/plots/eval_outcome_rates.png
```

## Record a rollout video

```bash
python -m scripts.record_video \
  --run-dir runs/20260426_123456_ppo_mvp_ppo_seed0 \
  --model best \
  --steps 1000
```

Output:

```text
run_dir/videos/ppo_best_topdown.mp4
```

## Plot results

```bash
python -m scripts.plot_results \
  --run-dir runs/20260426_123456_ppo_mvp_ppo_seed0
```

## Compare PPO and SAC

```bash
python -m scripts.compare_runs \
  --summaries \
    runs/<ppo-run>/eval/final_unseen_summary.json \
    runs/<sac-run>/eval/final_unseen_summary.json \
  --output runs/ppo_vs_sac_eval_summary.png
```

## Metrics collected

Per episode:

- return,
- episode length,
- success flag,
- collision flag,
- out-of-road flag,
- max-step/timeout flag,
- cumulative cost,
- route completion,
- mean speed when exposed by the simulator,
- final MetaDrive `info` dictionary.

Summary metrics:

- mean return,
- success rate,
- collision rate,
- out-of-road rate,
- timeout/max-step rate,
- mean cost,
- mean route completion.

## Config philosophy

Training scenarios and evaluation scenarios are intentionally separated by seed range.

Default training:

```yaml
metadrive:
  start_seed: 0
  num_scenarios: 50
```

Default evaluation:

```yaml
eval:
  start_seed: 1000
  num_scenarios: 50
```

This gives you a clean train/test split over procedurally generated roads.

## Phase-1 resume framing

Once you have real numbers, a strong resume entry could look like:

> **SafeRL-Drive: Reinforcement Learning for Autonomous Driving** — Built a closed-loop autonomous-driving simulation benchmark in MetaDrive, training PPO and SAC agents for continuous vehicle control across procedurally generated traffic scenarios.

> Evaluated policies on unseen road seeds using route completion, collision rate, out-of-road rate, cumulative safety cost, and success rate; generated rollout videos and PPO-vs-SAC comparison plots for reproducible analysis.

After results are available, replace generic claims with actual values:

> Improved unseen-route success rate from **A%** to **B%** after tuning traffic density, reward weights, and normalization; reduced collision rate by **X%** relative to baseline.

## Practical notes

- Start with `smoke_test.yaml` first.
- If `subproc` causes issues on macOS, override with `train.vec_env=dummy`.
- PPO benefits from multiple environments; SAC is configured with one environment by default to keep replay-buffer behavior simple.
- The default 500k timesteps are meant as an MVP, not a final benchmark. For stronger plots, run 1M+ timesteps and 100+ unseen evaluation episodes.
- Videos require `imageio-ffmpeg`; it is included in `requirements.txt`.

## Suggested Phase-2 extension

After Phase 1 works, the natural next step is a safety layer:

- TTC-style risk penalty,
- action shield,
- cost-constrained evaluation,
- ablation: PPO/SAC with vs without shield.
