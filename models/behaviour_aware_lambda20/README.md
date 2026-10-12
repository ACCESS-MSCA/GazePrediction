# Behaviour-aware gaze prediction model (λ = 20)

The model used in the journal extension, *Behaviour-Aware Gaze Prediction in Virtual Reality Without Eye Tracking*: a Conv-LSTM with a gaze head and a task head, trained with task-loss weight λ = 20 on EHTask. Three seeds were trained; the paper reports mean ± s.d. over the three, and single-seed analyses use seed 0.

| File | Content |
|---|---|
| `seedK/best.pth` | PyTorch state dict (24,862 parameters), checkpoint with the best validation objective |
| `seedK/conv_e2_dual20_sK.json` | Configuration and per-epoch training log |

**Training data.** EHTask, split from `revision/splits/ehtask_splits.json`: 4 held-out test participants; training and validation recordings come from the other 26 participants. Trained with `revision/train_cached.py --task_weight 20 --seed K`.

**Inputs** (15 past frames at 30 Hz):
- 25×25 saliency maps of the HMD viewport from the frozen UniSal model (`models/unisal.pth`), shape `(B, 15, 1, 1, 25, 25)`;
- 4 HMD motion features per frame: azimuth and elevation angular velocity (deg/s) and frame-to-frame angular displacement (deg), unnormalised, shape `(B, 15, 4)`.

**Outputs:**
- gaze offsets (azimuth, elevation, degrees) for the next 20 frames (666 ms), relative to the head direction at the first predicted frame, shape `(B, 20, 2)`;
- task logits, shape `(B, 4)`, in the order Free Viewing, Visual Search, Saliency judgement, Tracking.

**Horizon selection.** Horizons per predicted task, selected on validation data with a 10° tolerance: 13 / 15 / 10 / 20 steps for seeds 0 and 1, 13 / 16 / 10 / 20 for seed 2.

**Results** (mean ± s.d. over the three seeds):

| | 333 ms | 666 ms | Task accuracy |
|---|---|---|---|
| EHTask test, endpoint error | 8.48 ± 0.05° | 11.99 ± 0.04° | 68.6 ± 0.3% |
| DSAV360 zero-shot, endpoint error | 15.59 ± 0.06° | 23.81 ± 0.11° | – |

## Loading

```python
import json, torch
from core.models import Conv_LSTM

cfg = json.load(open("models/behaviour_aware_lambda20/seed0/conv_e2_dual20_s0.json"))
model = Conv_LSTM(num_time_features=cfg["num_time_features"], conv_output_embeddings=cfg["conv_output_embeddings"],
                  sequence_length=cfg["past_timepoints"], prediction_length=cfg["future_timepoints"],
                  prediction_features=cfg["prediction_features"], lstm_hidden_size=cfg["lstm_hidden_states"],
                  lstm_num_layers=cfg["lstm_layers"], use_task_output=True)
model.load_state_dict(torch.load("models/behaviour_aware_lambda20/seed0/best.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
    gaze, task_logits = model(saliency, motion)  # (B, 15, 1, 1, 25, 25), (B, 15, 4)
```

To evaluate a seed with the revision pipeline (GPU required; data paths are set at the top of `revision/cached_data.py`):

```bash
python revision/eval_unified.py --models lambda20_s0 --targets eht:test \
    --extra lambda20_s0=models/behaviour_aware_lambda20/seed0/best.pth
```
