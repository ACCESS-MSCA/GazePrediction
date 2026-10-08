import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import seaborn as sns

def plot_confusion_matrix(json_path, output_path):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    matrix = np.array(data["matrix"])
    labels = data["labels"]
    
    # Normalize by row (true labels)
    matrix_norm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    matrix_norm = np.nan_to_num(matrix_norm) # Handle division by zero

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.4)  # Scale up seaborn defaults
    sns.heatmap(matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 14, "weight": "bold"})
    
    plt.title('Task Classification Confusion Matrix (Normalized)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=16, labelpad=10)
    plt.ylabel('True Label', fontsize=16, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    print(f"Confusion matrix graph saved to: {output_path}")

def plot_horizon_study(json_paths, output_path, model_labels=None, fixed_ylim=True):
    all_data = []
    for jp in json_paths:
        if not os.path.exists(jp):
            print(f"Error: {jp} not found.")
            return
        with open(jp, 'r') as f:
            all_data.append(json.load(f))

    # Enforce specific display order
    requested_order = ["Saliency", "Free Viewing", "Visual Search", "Track"] #"Overall",
    # Use the first JSON to determine which tasks are available
    tasks = [t for t in requested_order if t in all_data[0]]
    num_tasks = len(tasks)
    
    # Determine grid layout
    if num_tasks == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes_flat = [axes]
    else:
        # Default to 2x2 for EHT-style 4-task data
        rows = (num_tasks + 1) // 2
        cols = 2 if num_tasks > 1 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes_flat = axes.flatten() if num_tasks > 1 else [axes]

    # Time steps in ms (30 FPS -> ~33ms per step)
    horizon_steps = len(all_data[0][tasks[0]]['lstm_err'])
    time_ms = np.arange(horizon_steps) * 33.3

    # Define colors for multiple models
    model_colors = ['r', 'g', 'b', 'm', 'c']

    for i, task_name in enumerate(tasks):
        ax = axes_flat[i]
        
        # Plot HMD-center baseline once (using first dataset)
        ax.plot(time_ms, all_data[0][task_name]['hmd_center_err'], 'k--', label='HMD-center', linewidth=2)
        
        # Plot each model
        for j, data in enumerate(all_data):
            label = model_labels[j] if model_labels else f"Model {j+1}"
            color = model_colors[j % len(model_colors)]
            ax.plot(time_ms, data[task_name]['lstm_err'], color + '-o', label=label, markersize=5, linewidth=2.5)
        
        ax.set_title(f"Task: {task_name}", fontsize=16, fontweight='bold')
        ax.set_xlabel("Horizon (ms)", fontsize=14)
        ax.set_ylabel("Angular Error (Degrees)", fontsize=14)
        if fixed_ylim:
            ax.set_ylim(0, 30) # Fix range for consistent comparison
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Comparative graph saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", nargs='+', required=True, help="List of JSON results files to compare")
    parser.add_argument("--labels", nargs='+', help="Optional labels for the models")
    parser.add_argument("--output", help="Output image path")
    parser.add_argument("--mode", choices=["horizon", "cm"], default="horizon", 
                        help="Choose between 'horizon' study or 'cm' (confusion matrix)")
    parser.add_argument("--no_fixed_ylim", action="store_true", help="Disable the 0-30 degree Y-axis limit")
    args = parser.parse_args()
    
    if args.mode == "horizon":
        output = args.output if args.output else "comparative_horizon_study.pdf"
        plot_horizon_study(args.json, output, args.labels, not args.no_fixed_ylim)
    else:
        # Confusion matrix still takes single JSON for now
        output = args.output if args.output else "confusion_matrix.pdf"
        plot_confusion_matrix(args.json[0], output)
