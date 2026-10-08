import os
import random
import json
import argparse
import sys
from glob import glob
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.setup_logger import setup_logger

def create_splits(processed_dir, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Generates consistent Train/Val/Test splits for the DSAV360 dataset.
    """
    logger = setup_logger("prepare_dsav_splits")
    
    try:
        logger.info(f"Starting split generation for directory: {processed_dir}")
        
        if not os.path.exists(processed_dir):
            logger.error(f"Processed dataset directory not found: {processed_dir}")
            sys.exit(1)

        # Find all user/video folders
        # Structure: processed_dataset/dsav360_processed_vid/Video_X/User_Y_Task_Z
        search_pattern = os.path.join(processed_dir, "Video_*", "User_*")
        all_folders = glob(search_pattern)
        
        if not all_folders:
            logger.error("No processed video folders found. Check the search pattern or preprocessing step.")
            sys.exit(1)

        logger.info(f"Found {len(all_folders)} sequences.")

        # Ensure determinism
        random.seed(seed)
        random.shuffle(all_folders)

        # Calculate split indices
        total = len(all_folders)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        train_set = all_folders[:train_end]
        val_set = all_folders[train_end:val_end]
        test_set = all_folders[val_end:]

        logger.info(f"Split sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        # Save splits
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        splits = {
            "train": [os.path.relpath(p, processed_dir) for p in train_set],
            "val": [os.path.relpath(p, processed_dir) for p in val_set],
            "test": [os.path.relpath(p, processed_dir) for p in test_set]
        }
        
        split_file_path = os.path.join(output_dir, "dsav360_splits.json")
        
        with open(split_file_path, 'w') as f:
            json.dump(splits, f, indent=4)
            
        logger.info(f"Splits successfully saved to {split_file_path}")
        
    except Exception as e:
        logger.error(f"An error occurred during split generation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Train/Val/Test splits for DSAV360")
    parser.add_argument("--processed_dir", type=str, required=True, help="Path to processed DSAV360 data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save split files")
    
    args = parser.parse_args()
    create_splits(args.processed_dir, args.output_dir)
