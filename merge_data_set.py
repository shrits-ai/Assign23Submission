import os
import json
import glob
import random
import argparse
from tqdm import tqdm

# Default configuration (can be overridden by command-line arguments)
DEFAULT_INPUT_DIR = "./stl10_vlm_dataset_simple_parallel/json_parts" # Adjusted default
DEFAULT_OUTPUT_DIR = "./stl10_vlm_dataset_final" # Adjusted default
DEFAULT_SPLIT_RATIO = 0.9 # 90% for training, 10% for validation

def merge_and_split_datasets(input_dir, output_dir, split_ratio):
    """
    Merges partial JSON datasets, shuffles, splits, and saves final dataset.

    Args:
        input_dir (str): Directory containing the partial JSON files (e.g., 'json_parts').
        output_dir (str): Directory where final train.json, val.json, and metadata.json will be saved.
        split_ratio (float): Ratio of data to use for the training set (e.g., 0.9 for 90%).
    """
    # Find all partial JSON files
    json_pattern = os.path.join(input_dir, "dataset_part_*.json")
    partial_files = glob.glob(json_pattern)

    if not partial_files:
        print(f"Error: No partial dataset files found matching '{json_pattern}'")
        return

    print(f"Found {len(partial_files)} partial dataset files.")

    # Load all data into a single list
    all_data = []
    print("Loading data from partial files...")
    for file_path in tqdm(partial_files, desc="Loading JSON parts"):
        try:
            with open(file_path, 'r') as f:
                data_part = json.load(f)
                # Basic check if loaded data is a list
                if isinstance(data_part, list):
                    all_data.extend(data_part)
                else:
                    print(f"Warning: Expected a list in {file_path}, found {type(data_part)}. Skipping.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}. Skipping.")
        except Exception as e:
            print(f"Warning: Error reading {file_path}: {e}. Skipping.")

    if not all_data:
        print("Error: No data loaded from partial files. Final dataset cannot be created.")
        return

    print(f"Loaded a total of {len(all_data)} data entries.")

    # Shuffle the combined data
    print("Shuffling data...")
    random.shuffle(all_data)

    # Split into train/val
    split_idx = int(len(all_data) * split_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    print(f"Splitting data: {len(train_data)} train, {len(val_data)} val")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save train and val JSON files
    train_output_path = os.path.join(output_dir, "train.json")
    val_output_path = os.path.join(output_dir, "val.json")

    print(f"Saving {train_output_path}...")
    with open(train_output_path, 'w') as f:
        json.dump(train_data, f, indent=2)

    print(f"Saving {val_output_path}...")
    with open(val_output_path, 'w') as f:
        json.dump(val_data, f, indent=2)

    # --- Determine classes based on content if possible, or use defaults ---
    # This part assumes you know the classes for the dataset being merged
    # For robustness, you could try inferring from saved data or pass as arg
    if "stl10" in input_dir.lower():
         classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
         dataset_name = "STL10-VLM-Merged"
    elif "cifar10" in input_dir.lower():
         classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
         dataset_name = "CIFAR10-VLM-Merged"
    else: # Default fallback
         classes = ["unknown"] * 10
         dataset_name = "Unknown-VLM-Merged"
    # ---

    # Create and save metadata
    metadata = {
        "dataset_name": dataset_name,
        "num_images": len(all_data),
        "num_train": len(train_data),
        "num_val": len(val_data),
        "classes": classes
    }
    metadata_output_path = os.path.join(output_dir, "metadata.json")
    print(f"Saving {metadata_output_path}...")
    with open(metadata_output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nDataset merging complete!")
    print(f"Final dataset saved in: {output_dir}")
    print(f"Remember to ensure the 'images' directory (likely in ../images relative to {input_dir}) is correctly placed relative to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge partial JSON dataset files into final train/val splits.")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing the partial JSON files (default: {DEFAULT_INPUT_DIR}).")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save the final train.json and val.json (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--split_ratio", type=float, default=DEFAULT_SPLIT_RATIO,
                        help=f"Fraction of data for the training set (default: {DEFAULT_SPLIT_RATIO}).")

    args = parser.parse_args()

    merge_and_split_datasets(args.input_dir, args.output_dir, args.split_ratio)