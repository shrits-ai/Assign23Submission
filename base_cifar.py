import os
import tempfile
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
# import matplotlib.pyplot as plt # Not needed for generation
from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np
import json
import random
from tqdm import tqdm
import argparse # <-- Import argparse

# Create a temporary directory in the user's home directory
temp_dir = os.path.join(os.path.expanduser("~"), "temp")
os.makedirs(temp_dir, exist_ok=True)
os.environ["TMPDIR"] = temp_dir
os.environ["TEMP"] = temp_dir
os.environ["TMP"] = temp_dir

# Set the cache directory for Hugging Face to a location with sufficient space
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.path.expanduser("~"), "hf_cache")
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

def download_cifar10():
    """Download CIFAR-10 dataset and return the trainset and classes."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Download and load CIFAR-10
    full_trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Class names for CIFAR-10
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return full_trainset, classes

def load_vlm_model(gpu_id):
    """Load SmolVLM 2 model onto a specific GPU."""
    model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Loading model onto {device}...")

    processor = AutoProcessor.from_pretrained(model_path)
    # Load directly to the target device if possible, or use .to()
    try:
         model = AutoModelForImageTextToText.from_pretrained(
             model_path,
             torch_dtype=torch.float16,
             device_map={'': device} # Try direct loading
         )
    except Exception:
         print("Direct device_map failed, loading to CPU then moving.")
         model = AutoModelForImageTextToText.from_pretrained(
             model_path,
             torch_dtype=torch.float16,
         ).to(device) # Fallback to .to()

    print(f"Model loaded successfully on {device}.")
    return model, processor, device

# --- generate_conversation_data including modified ask_question ---
def generate_conversation_data(model, processor, image, class_name, device):
    """
    Generate conversational data with robust fallback and CONDITIONAL quality filtering.
    Low quality turns are skipped based on question type.
    """
    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    # Define categories clearly
    CAT_IDENTIFICATION = "class_identification"
    CAT_DESCRIPTION = "detailed_description"
    CAT_REASONING = "complex_reasoning"

    conversation_templates = {
        CAT_IDENTIFICATION: {
            "with_label": [
                f"Is this a {class_name}? Describe its features.", # Still asks for description
                f"This image shows a {class_name}. Can you describe its characteristics?", # Still asks for description
                f"What are the distinctive features of this {class_name}?",
                f"How would you recognize a {class_name} from its appearance?",
                f"Identify this {class_name} and list its main traits.",
                f"What details confirm this is a {class_name}?"
                 # Example of a purely ID question (add if needed):
                 # f"What specific type of {class_name} is this, if identifiable?"
            ],
        },
        CAT_DESCRIPTION: {
            "with_label": [
                f"What does this {class_name} look like?",
                f"Describe the appearance of this {class_name}.",
                f"What are the main colors and shapes of this {class_name}?",
                f"How is this {class_name} positioned in the image?",
                f"What textures or patterns do you observe on this {class_name}?",
                f"Give a detailed visual description of this {class_name}."
            ],
        },
        CAT_REASONING: {
            "with_label": [
                f"What is the typical function or purpose of a {class_name}?",
                f"How do people interact with a {class_name}?",
                f"What makes this {class_name} stand out compared to similar objects?",
                f"In what environments would you usually find a {class_name}?",
                f"What interesting facts do you know about {class_name}s?",
                f"Why is a {class_name} considered important in its context?"
            ],
        }
    }

    # --- Build prompt pool ---
    overall_pool = []
    for category, prompts_dict in conversation_templates.items():
        if "with_label" in prompts_dict: # Check if key exists
             overall_pool.extend([(category, q) for q in prompts_dict["with_label"]])

    conversation = []
    # Anchor question is descriptive by nature
    default_first_question = (
        f"Based on the CIFAR‑10 label, this image should depict a {class_name}. "
        f"Could you confirm this and describe its features?"
    )
    ANCHOR_CAT = CAT_DESCRIPTION # Treat anchor as descriptive for filtering

    # --- Helper functions defined inside generate_conversation_data ---
    def is_uncertain_or_mismatch(answer_text, true_label):
        lower = answer_text.lower()
        dynamic_keyword = f"it is not {true_label}".lower()
        uncertain_keywords = [
            "not sure", "unclear", "can't tell", "cannot tell",
            "i'm sorry", "i cannot", "i can't", dynamic_keyword
        ]
        if any(kw in lower for kw in uncertain_keywords):
            return True
        mentioned_other = False
        mentioned_true = true_label.lower() in lower
        for c in cifar10_classes:
            if c != true_label and c.lower() in lower:
                 mentioned_other = True
                 break
        if mentioned_other and not mentioned_true:
             return True
        return False

    def fallback_response(label):
        return f"Confirmed: The image shows a {label}. Key features include characteristics typical of a {label}."

    # --- Inner function ask_question modified for CONDITIONAL quality filtering ---
    def ask_question(question_text, question_category, use_fallback=True, skip_low_quality=True): # Add question_category
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question_text}]}]
        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device, dtype=torch.float16)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=300
                )

            raw_answer = processor.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
            if raw_answer.startswith("Assistant:"):
                 raw_answer = raw_answer.replace("Assistant:", "").strip()

            # --- Fallback Logic ---
            if use_fallback and is_uncertain_or_mismatch(raw_answer, class_name):
                print(f"Info: Fallback triggered for anchor question.")
                return fallback_response(class_name)

            # --- Conditional Quality Filter ---
            answer_lower = raw_answer.lower()
            refusal_phrases = [
                "i cannot", "i'm sorry", "unable to", "can't provide", "no information",
                "cannot answer", "don't have access", "not possible", "cannot see",
                "not clear enough", "image is too blurry", "i am not able", "as an ai",
                "i do not have"
                ]

            # Determine required length based on category
            # Assume identification might be short, others need description
            if question_category in [CAT_DESCRIPTION, CAT_REASONING]:
                 min_answer_length = 20 # Require longer answers for these
            else: # Assume CAT_IDENTIFICATION or unknown category
                 min_answer_length = 5 # Allow shorter answers, e.g., just the class name + confirmation

            # Check for refusal OR insufficient length based on category
            is_low_quality = any(phrase in answer_lower for phrase in refusal_phrases) or \
                             len(raw_answer) < min_answer_length

            if is_low_quality:
                print(f"Warning: Low quality answer detected for category '{question_category}'. MinLen={min_answer_length}. Answer: '{raw_answer}'")
                if skip_low_quality:
                    return None # Signal to skip
                else:
                    return "[SKIPPED_LOW_QUALITY]" # Placeholder
            else:
                return raw_answer # Return good answer

        except Exception as e:
            print(f"Error during model generation for '{question_text}': {e}")
            if use_fallback:
                return fallback_response(class_name)
            elif skip_low_quality:
                 return None
            else:
                 return "[ERROR_DURING_GENERATION]"
    # --- End ask_question ---

    # 1) Ask the first (anchor) question - treat as descriptive
    first_answer = ask_question(default_first_question, question_category=ANCHOR_CAT, use_fallback=True, skip_low_quality=True)
    if first_answer is not None:
         conversation.append({"human": default_first_question, "assistant": first_answer})
    else:
         print(f"Skipped anchor question turn due to low quality/error for class {class_name}.")

    if not overall_pool:
         print(f"Warning: Overall prompt pool is empty for class {class_name}. Cannot select questions.")
         return conversation

    # --- Select and ask additional questions ---
    num_additional_questions = 5
    questions_to_ask = []
    # ... [rest of question selection logic remains the same] ...
    selected_categories = set()
    available_categories = list(conversation_templates.keys())
    random.shuffle(available_categories)

    for category in available_categories:
        if len(questions_to_ask) >= num_additional_questions: break
        prompts_in_category = conversation_templates.get(category, [])
        if isinstance(prompts_in_category, dict):
             prompts_in_category = prompts_in_category.get("with_label", [])
        if prompts_in_category:
            q = random.choice(prompts_in_category)
            if q != default_first_question:
                questions_to_ask.append((category, q))
                selected_categories.add(category)

    num_needed = num_additional_questions - len(questions_to_ask)
    if num_needed > 0:
         already_selected_prompts = set(q for _, q in questions_to_ask)
         extra_candidates = [entry for entry in overall_pool if entry[1] not in already_selected_prompts and entry[1] != default_first_question]
         random.shuffle(extra_candidates)
         questions_to_ask.extend(extra_candidates[:num_needed])


    print(f"Selected {len(questions_to_ask)} additional questions.")
    for category, question in questions_to_ask:
        # Pass the category to ask_question
        answer = ask_question(question, question_category=category, use_fallback=False, skip_low_quality=True) # Pass category here
        if answer is not None:
            conversation.append({
                "human": question,
                "assistant": answer
            })
        else:
            print(f"Skipped low-quality turn for question: '{question}'")

    # ... [rest of the function, final check, return conversation] ...
    if len(conversation) <= 1 and (not conversation or conversation[0].get("assistant", "").startswith("Confirmed:")) :
         print(f"Warning: Conversation for class {class_name} has only anchor/fallback or is empty after filtering.")
    return conversation
# --- End generate_conversation_data ---


def save_partial_dataset(output_dir, data_list, process_id):
    """Saves the generated data list for a specific process."""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"dataset_part_{process_id}.json")
    try:
        with open(filename, 'w') as f:
            json.dump(data_list, f, indent=2)
        print(f"Saved partial dataset for process {process_id} to {filename} ({len(data_list)} items)")
    except Exception as e:
        print(f"Error saving partial dataset {filename}: {e}")


def main(args):
    # Download CIFAR-10 dataset
    print("Downloading CIFAR-10 dataset...")
    trainset, classes = download_cifar10()
    num_total_images = len(trainset)
    print(f"Total training images: {num_total_images}")

    # Determine indices for this process
    start_index = args.start_index
    end_index = args.end_index # User provides exclusive end index

    if end_index > num_total_images:
        print(f"Warning: End index {args.end_index} > total images {num_total_images}. Adjusting end index.")
        end_index = num_total_images
    if start_index >= end_index:
        print(f"Start index {start_index} is >= end index {end_index}. No images to process.")
        return

    process_range = range(start_index, end_index)
    num_images_to_process = len(process_range)
    process_id = f"gpu{args.gpu_id}_{start_index}-{end_index-1}"

    print(f"Process {process_id} (GPU: {args.gpu_id}) processing indices {start_index} to {end_index-1} ({num_images_to_process} images)")

    # Setup output directories
    output_dir = args.output_dir
    image_output_dir = os.path.join(output_dir, "images")
    json_output_dir = os.path.join(output_dir, "json_parts")
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)

    # Load the VLM model onto the specified GPU
    print(f"Loading SmolVLM 2 model for process {process_id}...")
    model, processor, device = load_vlm_model(args.gpu_id)
    print(f"Model loaded successfully for process {process_id} on {device}!")

    process_data = []
    pbar = tqdm(process_range, desc=f"GPU {args.gpu_id} Processing", unit="image")

    for i in pbar:
        try:
            image_tensor, label = trainset[i]
            class_name = classes[label]

            image_np = image_tensor.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)

            pbar.set_description(f"GPU {args.gpu_id} Proc {class_name} ({i})")

            # Generate conversation data
            conversation = generate_conversation_data(model, processor, pil_image, class_name, device)

            # Check if conversation is valid before saving
            if conversation and len(conversation) > 0:
                # Save image
                image_filename_base = f"{class_name}_{i}.png"
                image_save_path = os.path.join(image_output_dir, image_filename_base)
                pil_image.save(image_save_path)
                image_relative_path = os.path.join("images", image_filename_base)

                # Format and check turns again before final add
                final_conversation_turns = []
                valid_turns_count = 0
                for turn in conversation:
                    human_val = turn.get("human")
                    assistant_val = turn.get("assistant")
                    # Ensure both parts exist and assistant isn't a placeholder/error
                    if human_val and assistant_val and \
                       not assistant_val.startswith("[SKIPPED") and \
                       not assistant_val.startswith("[ERROR"):
                           final_conversation_turns.append({"from": "human", "value": human_val})
                           final_conversation_turns.append({"from": "assistant", "value": assistant_val})
                           valid_turns_count += 1 # Count valid assistant turns
                    # else: # Optionally log if a turn from the list was invalid somehow
                    #    print(f"Debug: Invalid turn structure or placeholder found for idx {i}. Turn: {turn}")


                # Add to dataset only if there's at least one valid generated assistant response
                if valid_turns_count > 0:
                    process_data.append({
                        "image": image_relative_path,
                        "conversations": final_conversation_turns, # Save cleaned/formatted turns
                        "original_index": i
                    })
                else:
                    pbar.write(f"Skipping save for image {i} (class: {class_name}) due to no valid assistant responses after final check.")
            else:
                 pbar.write(f"Skipping save for image {i} (class: {class_name}) because conversation generation failed or resulted in empty list.")

        except Exception as e:
            pbar.write(f"Critical Error processing image index {i}: {str(e)}")


        # Save progress periodically
        if (i + 1) % args.save_interval == 0:
             current_step_in_batch = i - start_index + 1
             if current_step_in_batch > 0 and process_data: # Check if there's actually data for this process
                 # Find data generated since last save (or from start) for checkpoint
                 # This simple way saves all current data; more complex logic could save only new data
                 save_partial_dataset(json_output_dir, process_data, f"{process_id}_checkpoint_{i+1}")
                 # Optionally clear process_data after checkpoint saving if memory is a concern
                 # print(f"Checkpoint saved for process {process_id} at index {i+1}.")


    # Final save for any remaining data
    if process_data: # Check if there's data left
        # More robust check: filter out data already saved in last checkpoint if checkpointing logic doesn't clear list
        save_partial_dataset(json_output_dir, process_data, f"{process_id}_final")
    print(f"Process {process_id} finished generating data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CIFAR-10 conversation dataset in parallel with conditional quality filtering.")
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID to use (e.g., 0, 1, 2, 3).")
    parser.add_argument("--start_index", type=int, required=True, help="Starting index (inclusive) of the dataset slice to process.")
    parser.add_argument("--end_index", type=int, required=True, help="Ending index (exclusive) of the dataset slice to process.")
    parser.add_argument("--output_dir", type=str, default="cifar10_vlm_dataset_parallel_filtered", help="Directory to save images and partial JSON files.")
    parser.add_argument("--save_interval", type=int, default=200, help="Save partial data every N images processed.")

    args = parser.parse_args()
    main(args)