import os
import json
import argparse
from PIL import Image
import re
from tqdm import tqdm

def check_dataset(dataset_dir, verbose=False):
    """
    Check if the class label is correctly reflected in the first response for each image.
    
    Args:
        dataset_dir: Directory containing the dataset (with train.json and val.json)
        verbose: Whether to print details for each image
    
    Returns:
        Dictionary with statistics about correct/incorrect classifications
    """
    results = {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "errors_by_class": {},
        "incorrect_examples": []
    }
    
    # Load the dataset files
    dataset_files = []
    for filename in ["train.json", "val.json"]:
        filepath = os.path.join(dataset_dir, filename)
        if os.path.exists(filepath):
            dataset_files.append((filename, filepath))
    
    if not dataset_files:
        print(f"Error: No dataset files found in {dataset_dir}")
        return results
    
    # Process each dataset file
    for file_name, file_path in dataset_files:
        print(f"Checking {file_name}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            continue
        
        # Process each image in the dataset
        for item in tqdm(data, desc=f"Analyzing {file_name}"):
            results["total"] += 1
            
            # Extract image path and class from filename
            image_path = item["image"]
            # Extract class from image path (e.g., "images/cat_123.png" -> "cat")
            class_name = os.path.basename(image_path).split('_')[0]
            
            # Get the first conversation pair
            if not item["conversations"] or len(item["conversations"]) < 2:
                print(f"Warning: No conversations found for {image_path}")
                continue
                
            first_prompt = item["conversations"][0]["value"]
            first_response = item["conversations"][1]["value"]
            
            # Check if the class name is mentioned in the first response
            class_mentioned = class_name.lower() in first_response.lower()
            
            # Check for negations near the class name
            negation_patterns = [
                r"not\s+a\s+" + class_name,
                r"isn't\s+a\s+" + class_name,
                r"isn't\s+really\s+a\s+" + class_name,
                r"doesn't\s+appear\s+to\s+be\s+a\s+" + class_name,
                r"doesn't\s+look\s+like\s+a\s+" + class_name
            ]
            
            has_negation = any(re.search(pattern, first_response.lower()) for pattern in negation_patterns)
            
            # Determine if the response correctly identifies the class
            is_correct = class_mentioned and not has_negation
            
            # Update statistics
            if is_correct:
                results["correct"] += 1
            else:
                results["incorrect"] += 1
                
                # Track errors by class
                if class_name not in results["errors_by_class"]:
                    results["errors_by_class"][class_name] = 0
                results["errors_by_class"][class_name] += 1
                
                # Save example of incorrect classification
                if len(results["incorrect_examples"]) < 10:  # Limit to 10 examples
                    results["incorrect_examples"].append({
                        "image": image_path,
                        "class": class_name,
                        "prompt": first_prompt,
                        "response": first_response
                    })
            
            # Print details if verbose
            if verbose:
                status = "✓" if is_correct else "✗"
                print(f"{status} {image_path} (class: {class_name})")
                print(f"  Prompt: {first_prompt}")
                print(f"  Response: {first_response[:100]}...")
                print()
    
    return results

def print_results(results):
    """Print the results in a readable format."""
    print("\n" + "="*50)
    print("DATASET QUALITY CHECK RESULTS")
    print("="*50)
    
    total = results["total"]
    correct = results["correct"]
    incorrect = results["incorrect"]
    
    if total == 0:
        print("No images were processed.")
        return
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"Total images analyzed: {total}")
    print(f"Correctly labeled: {correct} ({accuracy:.2f}%)")
    print(f"Incorrectly labeled: {incorrect} ({100-accuracy:.2f}%)")
    
    if incorrect > 0:
        print("\nErrors by class:")
        for class_name, count in sorted(results["errors_by_class"].items(), key=lambda x: x[1], reverse=True):
            class_total = sum(1 for item in results["incorrect_examples"] if item["class"] == class_name)
            error_rate = (count / class_total) * 100 if class_total > 0 else 0
            print(f"  {class_name}: {count} errors")
        
        print("\nExamples of incorrect classifications:")
        for i, example in enumerate(results["incorrect_examples"], 1):
            print(f"\n{i}. Image: {example['image']} (class: {example['class']})")
            print(f"   Prompt: {example['prompt']}")
            print(f"   Response: {example['response'][:150]}...")
    
    print("\nRecommendation:")
    if accuracy < 90:
        print("The dataset quality is concerning. Consider regenerating the dataset with improved prompts.")
    elif accuracy < 95:
        print("The dataset quality is acceptable but could be improved.")
    else:
        print("The dataset quality is good.")

def main():
    parser = argparse.ArgumentParser(description="Check the quality of the generated dataset")
    parser.add_argument("--dataset_dir", type=str, default="cifar10_vlm_dataset",
                        help="Directory containing the dataset")
    parser.add_argument("--verbose", action="store_true",
                        help="Print details for each image")
    
    args = parser.parse_args()
    
    # Check if the dataset directory exists
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory {args.dataset_dir} does not exist")
        return
    
    # Run the check
    results = check_dataset(args.dataset_dir, args.verbose)
    
    # Print the results
    print_results(results)

if __name__ == "__main__":
    main()
