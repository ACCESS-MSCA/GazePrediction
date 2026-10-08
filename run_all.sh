#!/bin/bash

# ==============================================================================
# Script Name: run_all.sh
# Description: Master orchestration for EHTask Retraining & Task-Aware Study.
# ==============================================================================

# Exit on error
set -e

# Configuration
BASE_DIR=$(pwd)
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
RUN_LOG_DIR="${BASE_DIR}/logs/run_${TIMESTAMP}"
export RUN_LOG_DIR
mkdir -p "$RUN_LOG_DIR"

REPORT_FILE="${RUN_LOG_DIR}/summary_report.log"

# Dataset Paths
EHT_RAW_PATH="${BASE_DIR}/../eht_dataset"
EHT_PROCESSED_PATH="${BASE_DIR}/../processed_dataset/eht_dataset_prossesed_vid"
DSAV_PROCESSED_PATH="${BASE_DIR}/../processed_dataset/dsav360_processed_vid"
SPLIT_OUTPUT_PATH="${BASE_DIR}/../processed_dataset/splits"

# Hyperparameters
FUTURE_TIMEPOINTS=20
MODEL_TYPE="lstm"
SALIENCY_SAMPLING=1
TASK_WEIGHT=100.0
USE_TASK_OUTPUT=true # Set to true for dual-output, false for gaze-only
PYTHON_CMD="conda run --no-capture-output -n access_env python3"
STEP4_EVAL_HORIZON=20
FORCE_RECALC=true # Set to true to bypass cache and rerun inference (1.3h)

# ... (rest of config) ...

# --- PRETRAINED MODELS ---
# Best Dual-Task Model (Task Head + Gaze)
MODEL_DUAL="${BASE_DIR}/../experiments/eht_scratch_improved_head/dsav_scratch/conv_lstm_past_15_future_20_window_15_samp1_20260416_161628/model_epoch_10_val_loss_172.1438.pth"
# Best Gaze-Only Model (No Task Head)
MODEL_GAZE="${BASE_DIR}/../experiments/eht_scratch_no_task_head/dsav_scratch/conv_lstm_past_15_future_20_window_15_samp1_20260424_161043/model_epoch_10_val_loss_74.7576.pth"

# Initialize Logging
echo "Pipeline Execution Report - $(date)" > "$REPORT_FILE"
echo "Log Folder: $RUN_LOG_DIR" >> "$REPORT_FILE"
echo "Horizon: $FUTURE_TIMEPOINTS steps" >> "$REPORT_FILE"
echo "Dual Model: $MODEL_DUAL" >> "$REPORT_FILE"
echo "Gaze-Only Model: $MODEL_GAZE" >> "$REPORT_FILE"
echo "==========================================" >> "$REPORT_FILE"

log() {
    local level=$1
    local msg=$2
    local ts=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$ts] [$level] $msg"
    echo "[$ts] [$level] $msg" >> "$REPORT_FILE"
}

log "INFO" "Starting EHTask Task-Aware Training Pipeline..."

# ==============================================================================
# Step 1: Task-Balanced Splitting (EHTask)
# ==============================================================================
log "INFO" "Step 1: Generating Task-Balanced Splits for EHTask..."
if $PYTHON_CMD pipelines/prepare_eht_splits.py \
    --processed_dir "$EHT_PROCESSED_PATH" \
    --output_dir "$SPLIT_OUTPUT_PATH" >> "$RUN_LOG_DIR/step1_splits.log" 2>&1; then
    log "INFO" "Step 1: EHTask splits generated successfully."
else
    log "ERROR" "Step 1: Split generation failed."
    exit 1
fi

# ==============================================================================
# Step 2: Training (Skipped in Comparison Mode)
# ==============================================================================
log "INFO" "Step 2: Skipped (Using provided pretrained models for comparison)."

# ==============================================================================
# Step 3: Optimal Horizon Study (Comparison)
# ==============================================================================
log "INFO" "Step 3: Performing Comparative Horizon Study (Dual vs Gaze-Only)..."

FORCE_FLAG=""
if [ "$FORCE_RECALC" = true ]; then
    FORCE_FLAG="--force"
fi

# 1. Evaluate Dual-Task Model
log "INFO" "  - Evaluating Dual-Task Model..."
$PYTHON_CMD tools/evaluation.py --lstm_model_path "$MODEL_DUAL" --dataset_path "$EHT_PROCESSED_PATH" --study_mode $FORCE_FLAG >> "$RUN_LOG_DIR/step3_eval_dual.log" 2>&1
DUAL_ERR=$(grep "Overall Avg LSTM Error:" "$RUN_LOG_DIR/step3_eval_dual.log" | awk '{print $5}')
DUAL_ACC=$(grep "Task Classification Accuracy:" "$RUN_LOG_DIR/step3_eval_dual.log" | awk '{print $4}')
log "INFO" "    -> Dual-Task Overall Error: $DUAL_ERR deg"
log "INFO" "    -> Dual-Task Classification Accuracy: $DUAL_ACC"
log "INFO" "    -> Confusion Matrix (detailed):"
sed -n '/TASK CLASSIFICATION CONFUSION MATRIX:/,/Task Classification Accuracy:/p' "$RUN_LOG_DIR/step3_eval_dual.log" >> "$REPORT_FILE"

# 2. Evaluate Gaze-Only Model
log "INFO" "  - Evaluating Gaze-Only Model..."
$PYTHON_CMD tools/evaluation.py --lstm_model_path "$MODEL_GAZE" --dataset_path "$EHT_PROCESSED_PATH" --study_mode $FORCE_FLAG >> "$RUN_LOG_DIR/step3_eval_gaze.log" 2>&1
GAZE_ERR=$(grep "Overall Avg LSTM Error:" "$RUN_LOG_DIR/step3_eval_gaze.log" | awk '{print $5}')
log "INFO" "    -> Gaze-Only Overall Error: $GAZE_ERR deg"

# 3. Generate Comparative Graph
JSON_DUAL="$(dirname "$MODEL_DUAL")/optimal_horizon_study_eht.json"
JSON_GAZE="$(dirname "$MODEL_GAZE")/optimal_horizon_study_eht.json"

log "INFO" "  - Generating Comparative Figure..."
$PYTHON_CMD tools/visualize_horizon_study.py \
    --json "$JSON_DUAL" "$JSON_GAZE" \
    --labels "Dual-Task LSTM" "Gaze-Only LSTM" \
    --output "$RUN_LOG_DIR/comparative_horizon_eht.pdf"

# 4. Generate Confusion Matrix (Dual Model only)
CM_JSON_EHT="$(dirname "$MODEL_DUAL")/task_confusion_matrix_eht.json"
if [ -f "$CM_JSON_EHT" ]; then
    $PYTHON_CMD tools/visualize_horizon_study.py --json "$CM_JSON_EHT" --output "$RUN_LOG_DIR/task_confusion_matrix_eht.pdf" --mode cm
fi

# Hyperparameters for Uncertainty-Aware Study
UA_HORIZON_TRACK=20
UA_HORIZON_SEARCH=15
UA_HORIZON_FREE=13
UA_HORIZON_SALIENCY=10

# 5. Uncertainty-Aware Comparison Study (Dynamic Horizon vs Fixed)
log "INFO" "  - Performing Uncertainty-Aware Comparison Study (Tabular)..."
if $PYTHON_CMD tools/uncertainty_aware_evaluation.py \
    --dual_model_path "$MODEL_DUAL" \
    --gaze_only_model_path "$MODEL_GAZE" \
    --dataset_path "$EHT_PROCESSED_PATH" \
    --horizon_track $UA_HORIZON_TRACK \
    --horizon_search $UA_HORIZON_SEARCH \
    --horizon_free $UA_HORIZON_FREE \
    --horizon_saliency $UA_HORIZON_SALIENCY \
    $FORCE_FLAG >> "$RUN_LOG_DIR/step3_ua_eval_table.log" 2>&1; then
    UA_ERR=$(grep "MAE UA-Model (Dynamic)" "$RUN_LOG_DIR/step3_ua_eval_table.log" | awk -F': ' '{print $2}')
    log "INFO" "  - Uncertainty-Aware tabular results generated in step3_ua_eval_table.log"
    log "INFO" "    -> Uncertainty-Aware Overall Error: $UA_ERR deg"
else
    log "ERROR" "  - Uncertainty-Aware comparison failed."
fi

# 6. Uncertainty Correlation Analysis
log "INFO" "  - Performing Uncertainty Correlation Analysis (Step 20)..."
if $PYTHON_CMD tools/uncertainty_correlation.py \
    --model_path "$MODEL_DUAL" \
    --eht_path "$EHT_PROCESSED_PATH" \
    --horizon 20 $FORCE_FLAG >> "$RUN_LOG_DIR/step3_uncertainty_correlation.log" 2>&1; then
    log "INFO" "  - Uncertainty correlation results generated in step3_uncertainty_correlation.log"
else
    log "ERROR" "  - Uncertainty correlation analysis failed."
fi

# ==============================================================================
# Step 4: Cross-Dataset Validation (DSAV360 - Dual Only)
# ==============================================================================
log "INFO" "Step 4: Running Zero-Shot Evaluation on DSAV360..."

# 1. Standard Study Mode Evaluation
log "INFO" "  - Running standard Zero-Shot study..."
if $PYTHON_CMD pipelines/run_experiment_orchestrator.py \
    --mode zero_shot \
    --model_dir "${BASE_DIR}/../experiments/dsav_eval_dual_only" \
    --dataset_dir "$DSAV_PROCESSED_PATH" \
    --future_steps "$FUTURE_TIMEPOINTS" \
    --eval_horizon "$STEP4_EVAL_HORIZON" \
    --saliency_sampling "$SALIENCY_SAMPLING" \
    --pretrained_path "$MODEL_DUAL" \
    --use_task_output \
    --study_mode \
    $FORCE_FLAG >> "$RUN_LOG_DIR/step4_dsav_eval.log" 2>&1; then
    
    DSAV_ERR=$(grep "Overall Avg LSTM Error:" "$RUN_LOG_DIR/step4_dsav_eval.log" | awk '{print $5}')
    log "INFO" "    -> DSAV360 Zero-Shot Overall Error: $DSAV_ERR deg"

    # Generate Graph for Dual-Only
    STUDY_JSON_DSAV="$(dirname "$MODEL_DUAL")/optimal_horizon_study_dsav.json"
    $PYTHON_CMD tools/visualize_horizon_study.py \
        --json "$STUDY_JSON_DSAV" \
        --labels "Dual-Task LSTM" \
        --output "$RUN_LOG_DIR/horizon_study_dsav_dual_only.pdf" \
        --no_fixed_ylim
else
    log "ERROR" "Step 4: DSAV360 Standard Evaluation failed."
fi

# 2. Uncertainty-Aware & Task Distribution Analysis
log "INFO" "  - Running Uncertainty-Aware / Task Distribution analysis on DSAV360..."
if $PYTHON_CMD tools/uncertainty_aware_evaluation.py \
    --dual_model_path "$MODEL_DUAL" \
    --gaze_only_model_path "$MODEL_GAZE" \
    --dataset_path "$DSAV_PROCESSED_PATH" \
    --horizon_track $UA_HORIZON_TRACK \
    --horizon_search $UA_HORIZON_SEARCH \
    --horizon_free $UA_HORIZON_FREE \
    --horizon_saliency $UA_HORIZON_SALIENCY \
    $FORCE_FLAG >> "$RUN_LOG_DIR/step4_dsav_ua_task_analysis.log" 2>&1; then
    UA_ERR_DSAV=$(grep "MAE UA-Model (Dynamic)" "$RUN_LOG_DIR/step4_dsav_ua_task_analysis.log" | awk -F': ' '{print $2}')
    log "INFO" "  - DSAV360 UA analysis and task percentages generated in step4_dsav_ua_task_analysis.log"
    log "INFO" "    -> DSAV360 Uncertainty-Aware Overall Error: $UA_ERR_DSAV deg"
else
    log "ERROR" "  - DSAV360 UA analysis failed."
fi

# 3. Uncertainty Correlation Analysis
log "INFO" "  - Performing Uncertainty Correlation Analysis on DSAV360 (Step 20)..."
if $PYTHON_CMD tools/uncertainty_correlation.py \
    --model_path "$MODEL_DUAL" \
    --dsav_path "$DSAV_PROCESSED_PATH" \
    --horizon 20 $FORCE_FLAG >> "$RUN_LOG_DIR/step4_dsav_uncertainty_correlation.log" 2>&1; then
    log "INFO" "  - DSAV360 uncertainty correlation results generated in step4_dsav_uncertainty_correlation.log"
else
    log "ERROR" "  - DSAV360 uncertainty correlation analysis failed."
fi

# Final Summary
log "INFO" "Full Workflow Completed."
cat "$REPORT_FILE"
