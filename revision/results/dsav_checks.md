# E8 DSAV360 data checks

Processed subset: videos 1–10 (of 85), 88 participants, 684 recordings (train 478 / val 103 / test 103; only test is used, zero-shot). Stereo viewers among test recordings: 50.5%.

## Zero-shot error by viewing condition

| model | condition | N | step10 | step20 | HMD step10 | HMD step20 |
|---|---|---|---|---|---|---|
| e1_gaze_s0 | mono | 4393 | 17.75 | 22.35 | 20.14 | 25.08 |
| e1_gaze_s0 | stereo | 4296 | 16.99 | 21.89 | 19.71 | 25.25 |
| e2_dual20_s0 | mono | 4393 | 17.89 | 22.40 | 20.14 | 25.08 |
| e2_dual20_s0 | stereo | 4296 | 17.21 | 21.96 | 19.71 | 25.25 |
| e1_dual100_s0 | mono | 4393 | 17.61 | 22.09 | 20.14 | 25.08 |
| e1_dual100_s0 | stereo | 4296 | 16.91 | 21.60 | 19.71 | 25.25 |
| base_hmd_cv | mono | 4393 | 16.87 | 22.28 | 20.14 | 25.08 |
| base_hmd_cv | stereo | 4296 | 16.02 | 21.56 | 19.71 | 25.25 |

## Step-0 head–gaze offset at prediction time (great-circle, deg)

| dataset | N windows | mean | median | 90th pct |
|---|---|---|---|---|
| eht | 11432 | 11.27 | 10.17 | 19.09 |
| dsav | 8689 | 15.24 | 12.05 | 25.98 |