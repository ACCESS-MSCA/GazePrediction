import os
import sys
import argparse
import subprocess

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.setup_logger import setup_logger

def run_command(command, logger):
    """Executes a shell command and logs output in real-time."""
    logger.info(f"Executing: {command}")
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr into stdout
            text=True,
            bufsize=1 # Line buffered
        )
        
        # Stream output line by line
        for line in process.stdout:
            clean_line = line.strip()
            if clean_line:
                logger.info(clean_line)
                # Also print to console so it reaches the redirected file in run_all.sh
                print(clean_line, flush=True)
                
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Command Failed with exit code {process.returncode}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to execute command: {str(e)}")
        return False

def orchestrate_experiments(mode, model_dir, dataset_dir, model_type="tsmixer", future_steps=10, pretrained_path=None, saliency_sampling=1, task_weight=0.5, study_mode=False, eval_horizon=None, use_task_output=False, force=False):
    """
    Orchestrates the Zero-Shot, Retraining, and Fine-Tuning workflows.
    """
    logger = setup_logger(f"orchestrator_{mode}")
    
    # Define common paths and flags
    task_flag = "--use_task_output" if use_task_output else ""
    base_train_cmd = f"python3 train.py --select_time_model {model_type} {task_flag} --future_timepoints {future_steps} --saliency_sampling {saliency_sampling} --task_weight {task_weight}"
    
    try:
        if mode == "zero_shot":
            logger.info(f"Starting Zero-Shot Evaluation (Future Steps: {future_steps})")
            
            # Support both "path1,path2" and single "path"
            if ',' in pretrained_path:
                lstm_path, tsmixer_path = pretrained_path.split(',')
                model_args = f"--lstm_model_path {lstm_path} --tsmixer_model_path {tsmixer_path}"
            else:
                model_args = f"--lstm_model_path {pretrained_path}"
            
            eval_cmd = (
                f"python3 tools/evaluation.py "
                f"{model_args} "
                f"--dataset_path {dataset_dir} "
                f"--future_timepoints {future_steps} "
                f"--saliency_sampling {saliency_sampling}"
            )
            
            if study_mode:
                eval_cmd += " --study_mode"
            
            if force:
                eval_cmd += " --force"
            
            if eval_horizon:
                eval_cmd += f" --eval_horizon {eval_horizon}"
            
            if run_command(eval_cmd, logger):
                logger.info("Zero-Shot Evaluation completed successfully.")
            else:
                logger.error("Zero-Shot Evaluation failed.")
                sys.exit(1)

        elif mode == "retrain":
            logger.info("Starting Retraining (Scratch) on DSAV360")
            
            # Construct training command
            # Using --dataset_path instead of --data_source based on train.py
            train_cmd = f"{base_train_cmd} --model_dir {os.path.join(model_dir, 'dsav_scratch')} --dataset_path {dataset_dir}"
            
            if run_command(train_cmd, logger):
                logger.info("Retraining completed successfully.")
            else:
                logger.error("Retraining failed.")
                sys.exit(1)

        elif mode == "finetune":
            logger.info("Starting Fine-Tuning (EHT Weights -> DSAV360 Data)")
            
            if not pretrained_path:
                logger.error("Pretrained model path is required for fine-tuning.")
                sys.exit(1)
            
            # Construct fine-tuning command
            # Using --previous_run_weights instead of --resume based on train.py
            ft_cmd = f"{base_train_cmd} --model_dir {os.path.join(model_dir, 'dsav_finetune')} --dataset_path {dataset_dir} --previous_run_weights {pretrained_path}"
            
            if run_command(ft_cmd, logger):
                logger.info("Fine-Tuning completed successfully.")
            else:
                logger.error("Fine-Tuning failed.")
                sys.exit(1)
        
        else:
            logger.error(f"Unknown mode: {mode}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate Gaze Prediction Experiments")
    parser.add_argument("--mode", type=str, choices=["zero_shot", "retrain", "finetune"], required=True)
    parser.add_argument("--model_dir", type=str, required=True, help="Directory for model outputs")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Root of processed dataset")
    parser.add_argument("--model_type", type=str, default="tsmixer", choices=["tsmixer", "lstm"])
    parser.add_argument("--future_steps", type=int, default=10)
    parser.add_argument("--pretrained_path", type=str, help="Path to existing EHT checkpoint")
    parser.add_argument("--saliency_sampling", type=int, default=1, help="Saliency sampling rate")
    parser.add_argument("--task_weight", type=float, default=0.5, help="Weight for the task classification loss")
    parser.add_argument("--study_mode", action="store_true", help="Enable Optimal Horizon Study mode")
    parser.add_argument("--eval_horizon", type=int, help="Truncated horizon for evaluation error calculation")
    parser.add_argument("--use_task_output", action="store_true", help="Enable task classification head")
    parser.add_argument("--force", action="store_true", help="Bypass cache and force recalculation")
    
    args = parser.parse_args()
    orchestrate_experiments(args.mode, args.model_dir, args.dataset_dir, args.model_type, args.future_steps, args.pretrained_path, args.saliency_sampling, args.task_weight, args.study_mode, args.eval_horizon, args.use_task_output, args.force)
