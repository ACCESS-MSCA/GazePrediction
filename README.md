
# GazePrediction
![Example GIF](imgs/access_gaze.gif)

GazePrediction: a compact codebase for video gaze prediction using saliency (UNISAL) + temporal models (TSMixer / Conv-LSTM), training, inference, CoreML export and an example iOS app.

## Overview
This repository provides:
- Integration of the UNISAL saliency model for saliency map extraction.
- Temporal gaze prediction models (Conv-LSTM, TSMixer).
- Training scripts and spherical-coordinate-aware losses.
- Inference pipeline with visualization utilities and CLI.
- CoreML conversion utilities and an example iOS app for on-device validation.

## Repo layout (key files & folders)
- train.py — training script and loss implementations (SphericalMSELoss, SphericalAngularLoss).
- inference.py — video processing and inference outputs with visualization functions.
- convertcoreml.py — convert trained PyTorch models to CoreML.
- create_fov_vid_dataset.py — generate FOV video clips and dataset metadata.
- dataloader.py — dataset classes and get_dataloader helper.
- models.py — temporal model implementations (Conv_tsmixer, Conv_LSTM).
- utils.py — preprocessing, loading/unloading helpers, angular & visualization utilities.
- unisal/ — bundled UNISAL project with training and saliency tools.
- coreml-gazePred/ — Xcode project and example app for CoreML model validation.
- imgs/ — demo GIFs and images used in README/visualization.

## Quickstart

1) Environment setup
- Recommended: conda env using UNISAL env file and install project dependencies:
```bash
conda env create -f unisal/environment.yml
conda activate unisal
pip install -r requirements.txt  # if provided
```

2) Prepare dataset
- Produce FOV datasets from raw video inputs:
```bash
python create_fov_vid_dataset.py -inv /path/to/raw_videos -int /path/to/txts -o /path/to/output
```
- See create_fov_vid_dataset.py for CLI options and expected formats.

3) Train
- Example training command:
```bash
python train.py --past_timepoints 15 --future_timepoints 10 --batch_size 16 --num_epochs 20
```
- Training options and hyperparameters are listed in get_args() in train.py.

4) Inference
- Run model inference on processed datasets or raw videos:
```bash
python inference.py --cut_dataset_path /path/to/processed --raw_videos_path "/path/to/raw/*.mp4" --trained_model_folder path/to/trained --model_base_name model_name --model_epoch_name_path model_epoch_5_val_loss_49.1553.pth
```

5) CoreML conversion
- Convert and validate models with convertcoreml.py:
```bash
python convertcoreml.py --trained_model_folder /path/to/trained --model_base_name model_name --model_epoch_name_path model_epoch_5_val_loss_49.1553.pth --sal_path /path/to/unisal.pth
```
- Use the coreml-gazePred Xcode example to run models on device.

## Key functions & utilities
- Data loader & dataset: dataloader.get_dataloader, GazePredDataset (dataloader.py).
- Models: Conv_tsmixer, Conv_LSTM (models.py).
- UNISAL & saliency utilities: utils.load_unisal, unisal.run.train.
- Model loader: utils.load_time_model — loads model config and checkpoint.
- Angular helpers and visualization: utils.compute_spherical_motion, utils.CalAngularDist, utils.draw_gaze, utils.draw_fov_frame.

## Tips & troubleshooting
- Set dataset env vars used by UNISAL: DHF1K_DATA_DIR, SALICON_DATA_DIR, HOLLYWOOD_DATA_DIR, UCFSPORTS_DATA_DIR, MIT1003_DATA_DIR, MIT300_DATA_DIR — see unisal/README.md.
- If a model fails to load, verify the checkpoint and JSON config file under trained_model_folder/model_base_name/.
- During CoreML conversion, verify that the model inputs (shapes, preprocessing) match the exported converter examples.

## Contributing
- Add models in models.py or new files and update utils.load_time_model loader logic.
- Add dataset helpers or new dataset classes to dataloader.py or UNISAL’s dataset definitions.
- Add tests, examples or scripts to validate model changes and preprocessing.

## License
- UNISAL code included is Apache 2.0 (see unisal/LICENSE). Respect licensing for redistribution.

## Short reference
- Train: train.py
- Inference: inference.py
- Convert model: convertcoreml.py
- Dataset tools: create_fov_vid_dataset.py
- Models: models.py
- Utilities: utils.py