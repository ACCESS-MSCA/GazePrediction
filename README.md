
# GazePrediction
![Example GIF](imgs/access_gaze.gif)

GazePrediction: a compact codebase for gaze prediction in VR without eye tracking, using visual saliency (UNISAL) and head-mounted display (HMD) motion with temporal models (Conv-LSTM / TSMixer). It covers dataset preparation, training, evaluation, inference, CoreML export and an example iOS app.

This branch (`paper-extension`) contains the journal extension of our ISVC 2025 paper, *Behaviour-Aware Gaze Prediction in Virtual Reality Without Eye Tracking*: a multi-task model that forecasts gaze up to 666 ms ahead and recognises the viewing-task condition (free viewing, visual search, saliency judgement, tracking), and uses the predicted task to choose how far ahead to forecast.

## Overview
This repository provides:
- Integration of the UNISAL saliency model for saliency map extraction.
- Temporal gaze prediction models (Conv-LSTM, TSMixer), with an optional task-classification head.
- Training scripts and spherical-coordinate-aware losses.
- Dataset pipelines for EHTask and DSAV360, with participant-exclusive splits.
- Evaluation tools: horizon studies, task-conditioned horizon selection, task-entropy analysis, zero-shot transfer to DSAV360.
- Inference pipeline with visualization utilities and CLI.
- CoreML conversion utilities and an example iOS app for on-device validation.
- `revision/`: the experiments of the journal revision (cross-validation, multiple seeds, baselines, horizon-policy controls, leave-one-task-out), with the split definitions and summary results.

## Repo layout (key files & folders)
- `train.py` — training script and loss implementations (SphericalMSELoss, SphericalAngularLoss); `--use_task_output` and `--task_weight` enable the task head.
- `run_all.sh` — end-to-end pipeline: splits, horizon study, task-conditioned horizons, entropy analysis, zero-shot DSAV360.
- `core/`
  - `models.py` — temporal models (Conv_LSTM, Conv_tsmixer) and the task head.
  - `dataloader.py` — dataset classes and dataloader helpers.
  - `utils.py` — preprocessing, UNISAL/model loading, angular and visualization utilities.
  - `setup_logger.py` — logging helper.
- `pipelines/`
  - `create_EHT_dataset.py` — cut EHTask videos and gaze/head metadata into FOV clips.
  - `create_dsav360_dataset.py` — the same for DSAV360.
  - `prepare_eht_splits.py`, `prepare_dsav_splits.py` — participant-exclusive train/val/test splits.
  - `run_experiment_orchestrator.py` — zero-shot / retrain / fine-tune runs.
- `tools/`
  - `evaluation.py` — step-wise error and optimal-horizon study.
  - `uncertainty_aware_evaluation.py` — task-conditioned (adaptive) horizon evaluation.
  - `uncertainty_correlation.py` — correlation between task entropy and gaze error.
  - `inference.py`, `inference_viz.py` — inference on processed datasets or raw videos.
  - `convertcoreml.py` — convert trained PyTorch models to CoreML.
  - `visualize_*.py`, `inspect_*.py`, `check_dsav_mapping.py`, `test_dsav_mapping.py` — plots and dataset checks.
- `revision/` — journal-revision experiments (see below).
- `unisal/` — bundled UNISAL project with training and saliency tools.
- `models/` — UNISAL weights, exported CoreML models, and the behaviour-aware model of the journal extension (`models/behaviour_aware_lambda20/`, λ = 20, three seeds; see its README).
- `coreml-gazePred/` — Xcode project and example app for CoreML model validation.
- `imgs/` — demo GIFs and images used in README/visualization.

## Quickstart

1) Environment setup
- Recommended: conda env using UNISAL env file and install project dependencies:
```bash
conda env create -f unisal/environment.yml
conda activate unisal
pip install -r requirements.txt  # if provided
```

2) Prepare datasets
- Produce FOV clips from the raw EHTask videos and metadata, then the splits:
```bash
python pipelines/create_EHT_dataset.py -inv /path/to/raw_videos -int /path/to/txts -o /path/to/output
python pipelines/prepare_eht_splits.py --processed_dir /path/to/output --output_dir /path/to/splits
```
- DSAV360 (zero-shot evaluation only): `pipelines/create_dsav360_dataset.py --extracted_path ... --save_folder ...` and `pipelines/prepare_dsav_splits.py`.

3) Train
- Gaze-only model:
```bash
python train.py --past_timepoints 15 --future_timepoints 20 --select_time_model lstm --dataset_path /path/to/output
```
- Behaviour-aware model (task head): add `--use_task_output --task_weight 20`.
- Training options and hyperparameters are listed in `get_args()` in `train.py`.

4) Evaluate
- See `run_all.sh` for the full sequence (horizon study, adaptive horizon, entropy correlation, DSAV360 zero-shot). Set the dataset and checkpoint paths at the top of the script.

5) Inference
- Run model inference on processed datasets or raw videos:
```bash
python tools/inference.py --cut_dataset_path /path/to/processed --raw_videos_path "/path/to/raw/*.mp4" --trained_model_folder path/to/trained --model_base_name model_name --model_epoch_name_path model_epoch_5_val_loss_49.1553.pth
```

6) CoreML conversion
- Convert and validate models with `tools/convertcoreml.py`:
```bash
python tools/convertcoreml.py --trained_model_folder /path/to/trained --model_base_name model_name --model_epoch_name_path model_epoch_5_val_loss_49.1553.pth --sal_path /path/to/unisal.pth
```
- Use the coreml-gazePred Xcode example to run models on device.

## Journal revision experiments (`revision/`)
The revision re-trains and re-evaluates all models on participant-exclusive splits, with 3 seeds per configuration. Every table and figure is derived from one per-window output file per model.

- Splits: `revision/splits/ehtask_splits.json` and `revision/splits/dsav360_splits.json` (train/val/test recordings); 5-fold participant-level cross-validation folds in `revision/results/cv_folds.json` (generated by `make_cv_folds.py`).
- Pipeline:
  1. `build_sal_cache.py` — precompute the frozen-UNISAL 25×25 saliency maps once per frame.
  2. `train_cached.py` — train from the cache (`--task_weight` = λ, 0 for gaze-only; `--seed`; `--exclude_task` for leave-one-task-out; `--split_file` for CV folds). Batches of runs: `run_queue.sh queue_phaseN.txt`.
  3. `eval_unified.py` (`eval_runs.sh`, `eval_cv.sh`) — per-window outputs: step-wise great-circle error, HMD-centre error, task probabilities.
  4. Analyses: `analyze.py` (horizon policies, entropy, risk–coverage), `seeds_summary.py`, `baselines.py` (constant-velocity head extrapolation, ridge eye–head model, saliency × head prior), `speed_policy.py` (head-speed horizon policy), `lobo_ood.py` (leave-one-task-out), `dsav_checks.py`.
  5. `make_paper_assets.py`, `make_fig2.py` — tables (LaTeX) and figures of the revised paper.
- DSAV360: head and gaze are recorded for the first 30 s of each viewing (the videos are 34–60 s), so only clips 1–6 are used (`cached_data.DSAV_MAX_CLIP`; applied when windows are built and when stored per-window outputs are loaded).
- Summary results: `revision/results/*.md` / `*.json`. Per-window outputs (~1 GB), checkpoints and logs are not tracked.
- Data, cache and checkpoint locations are set as constants at the top of `cached_data.py`, `eval_unified.py` and `make_cv_folds.py`; adjust them to your setup.

## Key functions & utilities
- Data loader & dataset: `core/dataloader.py`.
- Models: Conv_tsmixer, Conv_LSTM (`core/models.py`).
- UNISAL & saliency utilities: `core/utils.py` (load_unisal), `unisal.run.train`.
- Model loader: `core/utils.py` (load_time_model) — loads model config and checkpoint.
- Angular helpers and visualization: compute_spherical_motion, CalAngularDist, draw_gaze, draw_fov_frame (`core/utils.py`).

## Tips & troubleshooting
- Set dataset env vars used by UNISAL: DHF1K_DATA_DIR, SALICON_DATA_DIR, HOLLYWOOD_DATA_DIR, UCFSPORTS_DATA_DIR, MIT1003_DATA_DIR, MIT300_DATA_DIR — see unisal/README.md.
- If a model fails to load, verify the checkpoint and JSON config file under trained_model_folder/model_base_name/.
- During CoreML conversion, verify that the model inputs (shapes, preprocessing) match the exported converter examples.

## Contributing
- Add models in `core/models.py` and update the loader logic in `core/utils.py`.
- Add dataset helpers or new dataset classes to `core/dataloader.py` or UNISAL’s dataset definitions.
- Add tests, examples or scripts to validate model changes and preprocessing.

## Citation
Petrou, C., Partaourides, H., Balomenos, A., Kopsinis, Y., Chatzis, S.: Gaze prediction in virtual reality without eye tracking using visual and head motion cues. In: ISVC 2025, LNCS, vol. 16396, pp. 399–410. Springer, Cham (2026). https://doi.org/10.1007/978-3-032-14492-8_31

## License
- UNISAL code included is Apache 2.0 (see unisal/LICENSE). Respect licensing for redistribution.

## Short reference
- Train: `train.py` (cached revision runs: `revision/train_cached.py`)
- Evaluate: `tools/evaluation.py`, `run_all.sh`, `revision/eval_unified.py`
- Inference: `tools/inference.py`
- Convert model: `tools/convertcoreml.py`
- Dataset tools: `pipelines/`
- Models: `core/models.py`
- Utilities: `core/utils.py`
