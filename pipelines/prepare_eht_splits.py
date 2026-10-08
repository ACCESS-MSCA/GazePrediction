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

def create_eht_splits(processed_dir, output_dir, test_user_ratio=0.15, seed=42):
    """
    Creates User-Exclusive and Task-Balanced splits for EHTask.
    """
    logger = setup_logger("prepare_eht_splits")
    
    try:
        logger.info(f"Starting EHT split generation: {processed_dir}")
        
        # Pattern: Video_X/User_Y_Task_Z/clip_N.mp4
        all_videos = glob(os.path.join(processed_dir, "Video_*", "User_*_Task_*", "*.mp4"))
        
        if not all_videos:
            logger.error("No processed EHT videos found.")
            sys.exit(1)

        # Map users to their videos and tasks
        user_to_data = {}
        for vp in all_videos:
            # e.g. .../Video_1/User_5_Task_2/clip_1.mp4
            parts = vp.split(os.sep)
            folder_name = parts[-2] # User_5_Task_2
            user_id = folder_name.split('_')[1]
            task_id = folder_name.split('_')[3]
            
            if user_id not in user_to_data:
                user_to_data[user_id] = {'paths': [], 'tasks': set()}
            
            # Save the folder path (parent of mp4)
            folder_path = os.path.dirname(vp)
            rel_path = os.path.relpath(folder_path, processed_dir)
            
            if rel_path not in user_to_data[user_id]['paths']:
                user_to_data[user_id]['paths'].append(rel_path)
                user_to_data[user_id]['tasks'].add(task_id)

        all_users = list(user_to_data.keys())
        random.seed(seed)
        random.shuffle(all_users)

        # 1. Pick Test Users (Exclusive)
        num_test_users = max(1, int(len(all_users) * test_user_ratio))
        test_users = all_users[:num_test_users]
        remaining_users = all_users[num_test_users:]

        # 2. Check Task Coverage in Test Set
        test_tasks = set()
        for u in test_users:
            test_tasks.update(user_to_data[u]['tasks'])
        
        logger.info(f"Initial test users: {test_users}. Tasks covered: {test_tasks}")

        # If any task is missing, pull a user from remaining that has it
        all_possible_tasks = {'1', '2', '3', '4'}
        missing_tasks = all_possible_tasks - test_tasks
        
        for mt in missing_tasks:
            for i, ru in enumerate(remaining_users):
                if mt in user_to_data[ru]['tasks']:
                    test_users.append(remaining_users.pop(i))
                    logger.info(f"Added user {ru} to test set to cover task {mt}")
                    break

        # 3. Finalize Paths
        test_paths = []
        for u in test_users:
            test_paths.extend(user_to_data[u]['paths'])
            
        train_val_paths = []
        for u in remaining_users:
            train_val_paths.extend(user_to_data[u]['paths'])
            
        random.shuffle(train_val_paths)
        split_idx = int(0.85 * len(train_val_paths))
        train_paths = train_val_paths[:split_idx]
        val_paths = train_val_paths[split_idx:]

        # Save
        os.makedirs(output_dir, exist_ok=True)
        splits = {"train": train_paths, "val": val_paths, "test": test_paths}
        
        with open(os.path.join(output_dir, "ehtask_splits.json"), 'w') as f:
            json.dump(splits, f, indent=4)
            
        logger.info(f"EHT Splits Saved. Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
        logger.info(f"Test Set Tasks: {all_possible_tasks} (Confirmed)")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    create_eht_splits(args.processed_dir, args.output_dir)
