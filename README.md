# SafeRL-Drive

SafeRL-Drive is a bounded reinforcement-learning study in MetaDrive. Its final question is:

> Can a geometry-competent SAC policy adapt to interactive procedural traffic while
> reducing collisions and retaining traffic-free road generalization?

The learned policy controls one ego vehicle. MetaDrive’s native traffic manager controls
all surrounding vehicles. Every final condition uses `num_agents: 1` and
`is_multi_agent: false`; surrounding vehicles are never trained.

This is not a complete autonomous-driving stack and does not claim real-world safety.

## Project story

Phase 1 established trustworthy infrastructure on an intentionally easy straight,
traffic-free task. IDM, PPO, and SAC each reached 100% success on 20 held-out episodes.
The result validated training, checkpoint selection, evaluation, metrics, rendering, and
Drive persistence; it was not a robustness benchmark.

Phase 2 tested procedural road geometry. With the same SAC architecture, action space,
reward, budget, and held-out task:

| Condition | Success | Collision | Off-road | Route completion |
|---|---:|---:|---:|---:|
| Direct SAC | 12.0% | 15.5% | 83.0% | 47.0% |
| Curriculum SAC | 87.0% | 0.0% | 9.5% | 95.1% |

These are means across completed seeds 0 and 1. Curriculum seed 2 failed its mandatory
curve gate and remains preserved as negative evidence.

At traffic density 0.05, without traffic training, frozen curriculum SAC achieved 69%
success and 21% collision. That is promising initialization, not trained traffic
competence. The final extension therefore warm-starts from the successful geometry
curriculum and changes one difficulty dimension: traffic exposure.

No new PPO traffic model, algorithm sweep, network-size experiment, or reward search is
part of the final plan.

## Final three-run plan

Only these long runs are planned:

1. `SAC-Traffic`, seed 0: collision penalties remain 5.
2. `SAC-Traffic-Safe`, seed 0: vehicle/object collision penalties become 10.
3. Seed-1 confirmation of the predeclared seed-0 winner.

Both pilots warm-start from the best completed Phase-2 curriculum seed-0 checkpoint.
The confirmation warm-starts from curriculum seed 1. Each adaptation uses:

- stage A: density 0.02, at most 100,000 new transitions;
- stage B: density 0.05, at most 200,000 new transitions;
- map 3, 200 training scenarios beginning at seed 40000;
- respawn traffic, randomized during training;
- fixed 25-episode validation beginning at seed 50000;
- full continuous control and default LidarState vector observation;
- `MlpPolicy` with `[256, 256]`;
- learning rate `1e-4`, batch 512, buffer 300,000, starts 5,000;
- `gamma=0.99`, `tau=0.005`, `ent_coef=0.05`;
- correct timeout handling and standard replay memory layout.

Stage A must achieve at least 70% success and 80% route completion, with at most 25%
collision and 15% off-road. A failed gate pauses the lineage and does not create another
pilot. Its success-first Stage-A checkpoint remains available for diagnosis, but the
lineage is labeled `failed_gate` and excluded from completed-lineage aggregates.

The seed-0 pilots are evaluated on 50 fixed scenarios at densities 0, 0.05, and 0.10.
The saved selection rule uses success, collision, off-road, route completion, and
traffic-free retention. Only the selected variant advances to seed 1. Seed 2 is not run.

## Warm start and replay behavior

`scripts.train_curriculum` resolves the source run from
`latest_sac_phase2_curriculum_seed{seed}.txt` unless `--source-run-dir` is given.
Normally it selects `best_model.zip`.

Before training, it:

1. hashes the source checkpoint;
2. records the source run fingerprint and summary;
3. constructs a new SAC model using the traffic configuration;
4. verifies exact observation and action spaces;
5. copies actor, critic, and target-network policy state;
6. leaves the new optimizer and traffic replay buffer fresh.

The historical traffic-free replay buffer is not imported, so it cannot dominate early
adaptation. Loading it requires the explicit `--load-source-replay-buffer` flag and is not
used by the canonical notebook.

After adaptation starts, the traffic model and its own replay buffer are saved at every
stage boundary. A resumed run restores that traffic state and requires the exact immutable
`resolved_config.yaml`.

## Repository structure

```text
configs/
  sac_traffic_curriculum.yaml       final shared config and reward variants
  sac_phase2_*.yaml                 historical geometry experiments
  ppo_mvp.yaml, sac_mvp.yaml        historical Phase-1 controls
notebooks/
  phase2_colab_driver.ipynb         single complete Colab driver
saferl_drive/
  algorithms.py                     SB3 model construction and validation
  config.py                         YAML, overrides, and fingerprints
  envs.py                           MetaDrive and vector-environment factories
  evaluation.py                     episode metrics and aggregation
  utils.py                          logging, artifacts, plots, and atomic writes
scripts/
  train.py                          Phase-1/direct training
  train_curriculum.py               geometry and traffic curricula
  evaluate.py                       learned-policy density matrices
  evaluate_baseline.py              IDM/Expert density matrices
  compare_runs.py                   Phase 1, Phase 2, and traffic comparisons
  record_video.py                   chase or diagnostic top-down video
  sync_drive_runs.py                Mac restore and Colab-to-Drive persistence
reports/
  main.tex, main.pdf                compact public report
  surrogate_notes.tex, .pdf         detailed internal record
tests/
```

## Installation

Use Python 3.10 or newer. MetaDrive is pinned to commit
`85e5dadc6c7436d324348f6e3d8f8e680c06b4db`.

```bash
python -m pip install -e .
python -m compileall saferl_drive scripts tests
pytest -q
```

The project uses Gymnasium. A Colab image that also contains obsolete `gym` can remove it:

```bash
python -m pip uninstall -y gym
```

## One canonical Colab notebook

Open `notebooks/phase2_colab_driver.ipynb` in VS Code with a Google Colab runtime. It has
exactly these sections:

0. Project direction and experiment policy
1. Constants and paths
2. Runtime, CPU, RAM, CUDA, and L4 inspection
3. Google Drive mount
4. Clone or fast-forward pull repository
5. Install repository and verify package versions
6. Restore artifacts from Drive
7. Compile, test, MetaDrive smoke test, and chase-camera smoke test
8. Summarize existing Phase-1 and Phase-2 results without retraining
9. Experiment 0: IDM and ExpertPolicy traffic solvability
10. Experiment 1: frozen pre-adaptation evaluation matrix
11. Experiment 2: seed-0 SAC-Traffic pilot
12. Experiment 3: seed-0 SAC-Traffic-Safe pilot
13. Pilot comparison and saved selection decision
14. Experiment 5: selected seed-1 confirmation
15. Final evaluation matrix
16. Third-person videos
17. Final plots and tables
18. Build main and surrogate reports
19. Final Drive synchronization

The notebook runs the repository through visible `!python -m ...` entry points. It does
not embed training or evaluation implementations. Completed runs are reused, paused runs
resume, failed gates remain terminal, and unknown or mismatched states fail closed.

The live clone is `/content/safedrive`. Persistent artifacts are stored under
`/content/drive/MyDrive/SafeDrive`. Do not run Git directly from mounted Drive.

## Experiment commands

The notebook computes:

```python
CPU_COUNT = os.cpu_count() or 2
N_ENVS = min(4, max(1, CPU_COUNT - 1))
```

Use `SubprocVecEnv` when `N_ENVS > 1`; one MetaDrive instance runs in each process.
`gradient_steps` is resolved to `N_ENVS`.

Run the native solvability gate:

```bash
python -m scripts.evaluate_baseline \
  --config configs/sac_traffic_curriculum.yaml \
  --policy expert \
  --split validation \
  --episodes 50 \
  --densities 0.05 0.10 \
  --prefix traffic_solvability_expert \
  validation.num_scenarios=50
```

Freeze the source matrix for each completed geometry seed:

```bash
python -m scripts.evaluate \
  --run-dir runs/<geometry-curriculum-run> \
  --model best \
  --split test \
  --episodes 100 \
  --densities 0.0 0.05 0.10 \
  --prefix traffic_before
```

Run the seed-0 pilots:

```bash
python -m scripts.train_curriculum \
  --config configs/sac_traffic_curriculum.yaml \
  --variant reference \
  --seed 0 \
  --n-envs 4 \
  --vec-env subproc \
  --progress

python -m scripts.train_curriculum \
  --config configs/sac_traffic_curriculum.yaml \
  --variant safety \
  --seed 0 \
  --n-envs 4 \
  --vec-env subproc \
  --progress
```

To resume a paused lineage, pass its saved run and reuse its resolved environment count:

```bash
python -m scripts.train_curriculum \
  --config configs/sac_traffic_curriculum.yaml \
  --variant reference \
  --seed 0 \
  --n-envs 4 \
  --vec-env subproc \
  --progress \
  --run-dir runs/<traffic-run>
```

Evaluate both pilots and save the decision:

```bash
python -m scripts.evaluate \
  --run-dir runs/<pilot> \
  --model best \
  --split validation \
  --episodes 50 \
  --densities 0.0 0.05 0.10 \
  --prefix traffic_pilot \
  validation.num_scenarios=50

python -m scripts.compare_runs --traffic-extension --select-pilots
```

Run seed 1 with the `selected_variant` stored in
`runs/traffic_extension_selection.json`, then evaluate the two source and two adapted
lineages on the final test matrix.

Run native controllers on that same matrix:

```bash
python -m scripts.evaluate_baseline \
  --config configs/sac_traffic_curriculum.yaml \
  --policy idm \
  --split test \
  --episodes 100 \
  --densities 0.0 0.05 0.10 \
  --prefix traffic_final_idm

python -m scripts.evaluate_baseline \
  --config configs/sac_traffic_curriculum.yaml \
  --policy expert \
  --split test \
  --episodes 100 \
  --densities 0.0 0.05 0.10 \
  --prefix traffic_final_expert
```

Generate final tables and the five public plots:

```bash
python -m scripts.compare_runs --traffic-extension
```

## Metrics and fingerprints

Per-episode CSVs retain raw non-exclusive MetaDrive flags and add one exclusive outcome:
`success`, `collision`, `out_of_road`, `timeout`, or `other`.

Summaries contain:

- success rate and Wilson 95% interval;
- aggregate collision, vehicle collision, object collision, and collision-free rates;
- off-road and timeout rates;
- return, episode length, cost, route completion, and speed;
- mean absolute steering and mean action change;
- episode count, training seed, density, source checkpoint, condition, and fingerprint.

The comparison reports density-0.05 success gain and collision change, traffic-free
success and route retention, density-0.10 degradation, and across-seed mean and standard
deviation. Fingerprints retain the training definition, while final compatibility permits
only the intended source/adaptation and collision-penalty differences and refuses
undeclared scenario, map, traffic, reward, horizon, or learned-action mismatches.

## Chase-camera video

Chase view is the public default:

```bash
python -m scripts.record_video \
  --run-dir runs/<adapted-run> \
  --model best \
  --view chase \
  --density 0.05 \
  --episodes-csv runs/<adapted-run>/eval/traffic_final_d005_episodes.csv \
  --scenario-rule first_success
```

The recorder creates a separate offscreen environment with MetaDrive’s moving main
camera. At the pinned commit, the recording-only environment must enable MetaDrive’s
image-service switch to retain an offscreen camera; it simultaneously sets
`agent_observation=LidarStateObservation`. The MLP therefore receives the same vector
observation used in training, while rendered RGB frames are only encoded to MP4.
The pinned macOS offscreen path also requires leaving MetaDrive's unused mouse flag
enabled so `MainCamera` does not call a window-only method on a graphics buffer; all
visible interface panels remain hidden.

The pinned API is `main_camera.perceive(to_float=False)`. Expected frames are
1280×720 RGB. Failure does not silently fall back to top-down; diagnostics include render
mode, sensors, version, commit, and frame shape.

Scenario selection is systematic: the lowest seed satisfying `first`, `first_success`,
or `first_failure`. Sidecar JSON records model, seed, density, traffic mode, outcome,
return, completion, collision, frames, FPS, camera settings, observation shape, and
fingerprint. `--view topdown` remains available for diagnostics.

## Progress, logging, and resources

Training uses one SB3 progress bar per curriculum stage. `--progress` and
`--no-progress` override the config. Evaluation uses tqdm only for episode loops. SB3
tabular verbosity remains zero.

Console output is concise. Detailed DEBUG logs record commands, resolved config, Git and
package versions, hardware, checkpoint lineage, stage transitions, validation metrics,
checkpoints, exceptions, and output paths. Startup records observation/action shapes,
lidar beams, and policy/actor/critic parameter counts.

`logs/resource_usage.csv` samples CPU, RAM, GPU utilization/memory when available, and
environment steps per second about every 60 seconds.

## Google Drive persistence and Mac analysis sync

Persist one long run from Colab:

```bash
python -m scripts.sync_drive_runs \
  --drive-project /content/drive/MyDrive/SafeDrive \
  --to-drive \
  --run-dir runs/<run>
```

The command atomically merges the run, updates and verifies its latest pointer, skips
TensorBoard and intermediate checkpoints, and verifies critical config, metadata,
lineage, model, replay, validation, log, and resource artifacts in Drive.

Persist final comparisons, videos, report sources, bibliography, generated TeX, and PDFs:

```bash
python -m scripts.sync_drive_runs \
  --drive-project /content/drive/MyDrive/SafeDrive \
  --project-artifacts-to-drive
```

Restore all training artifacts in a fresh Colab session:

```bash
python -m scripts.sync_drive_runs \
  --drive-project /content/drive/MyDrive/SafeDrive \
  --local-runs /content/safedrive/runs \
  --include-training-artifacts
```

On a Mac with Google Drive for desktop, this defaults to an analysis-only restore:

```bash
python -m scripts.sync_drive_runs
```

It skips models, checkpoints, replay buffers, and TensorBoard data while retaining logs,
metrics, plots, videos, and pointers.

## Reports

Build the compact two-column public report and detailed two-column surrogate:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error reports/main.tex
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error reports/surrogate_notes.tex
```

`reports/main.tex` shows pending traffic results until
`runs/traffic_extension_comparison.*` and `reports/generated_traffic_results.tex` exist.
No geometry metric is relabeled as a traffic-adaptation result.

## Final outputs

```text
runs/traffic_extension_seed_results.csv
runs/traffic_extension_comparison.csv
runs/traffic_extension_comparison.json
runs/traffic_extension_selection.json
runs/traffic_extension_outcomes.png
runs/traffic_extension_success_collision.png
runs/traffic_extension_route_completion.png
runs/traffic_extension_training_returns.png
runs/traffic_extension_retention.png
reports/generated_traffic_results.tex
reports/main.pdf
reports/surrogate_notes.pdf
```

Until the three Colab training runs finish, traffic adaptation remains an implemented,
predeclared experiment—not a claimed result.
