#!/usr/bin/env python
import os
import json
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Import the SigLIPProjectionModel, extract_phi_embeddings, and training config.
from train_projection import SigLIPProjectionModel, extract_phi_embeddings, CONFIG as TRAIN_CONFIG

# Import the AutoTokenizer from transformers.
from transformers import AutoTokenizer

def process_sample(image_path, prompt, model, tokenizer, device):
    """
    Process a single image and prompt to compute cosine similarity and L2 norms.

    Args:
        image_path (str): Path to the image file.
        prompt (str): Text prompt (should include the correct class label).
        model: The SigLIPProjectionModel instance.
        tokenizer: The tokenizer for the Phi‑3 model.
        device (str): Device to run the computations.

    Returns:
        dict: Dictionary containing 'cosine_similarity', 'image_norm', 'text_norm'.
    """
    # Preprocess the image using the appropriate transform.
    # Here we resize and normalize the image so that it matches the expected input size.
    from dataloader import CONFIG as DATALOADER_CONFIG  # import expected normalization config
    transform = transforms.Compose([
        transforms.Resize((DATALOADER_CONFIG["IMAGE_SIZE"], DATALOADER_CONFIG["IMAGE_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATALOADER_CONFIG["NORMALIZE_MEAN"],
                             std=DATALOADER_CONFIG["NORMALIZE_STD"])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Obtain the projected image embedding.
        image_embedding = model(image_tensor)  # shape: [1, embedding_dim]

    # Tokenize the prompt for the Phi‑3 model.
    encoded = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    # Retrieve text embeddings using the extract_phi_embeddings helper.
    text_embedding = extract_phi_embeddings(model.phi, encoded["input_ids"], encoded["attention_mask"])

    # Normalize both image and text embeddings.
    normalized_image_embedding = F.normalize(image_embedding, p=2, dim=-1)
    normalized_text_embedding = F.normalize(text_embedding, p=2, dim=-1)

    # Compute cosine similarity between the normalized embeddings.
    cosine_sim = F.cosine_similarity(normalized_image_embedding, normalized_text_embedding, dim=-1).item()
    # Compute L2 norms (magnitudes) for image and text embeddings (from the original embeddings).
    image_norm = torch.norm(image_embedding, p=2).item()
    text_norm = torch.norm(text_embedding, p=2).item()

    return {
        "cosine_similarity": cosine_sim,
        "image_norm": image_norm,
        "text_norm": text_norm
    }

def main(args):
    # Choose the device.
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the SigLIPProjectionModel onto the device.
    model = SigLIPProjectionModel(
        siglip_model_name=TRAIN_CONFIG["SIGLIP_MODEL"],
        phi_model_name=TRAIN_CONFIG["PHI_MODEL"],
        projection_dim=TRAIN_CONFIG["PROJECTION_DIM"],
        num_layers=TRAIN_CONFIG["PROJECTION_LAYERS"]
    ).to(device)
    model.eval()

    # Load the AutoTokenizer for the Phi‑3 model.
    tokenizer = AutoTokenizer.from_pretrained(TRAIN_CONFIG["PHI_MODEL"])

    # Load the JSON file containing the dataset metadata.
    json_path = os.path.join(args.data_dir, f"{args.split}.json")
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    if len(data) == 0:
        print("No data found in JSON file.")
        return

    # Load checkpoint (if available)
    checkpoint_path = "siglip_phi3_projection/best_model.pt"  # change this as needed
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.projection.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint weights from {checkpoint_path}")
    else:
        print("Checkpoint not found; using freshly initialized weights.")

    # Process up to the requested number of images.
    num_samples = min(args.num_images, len(data))
    total_cosine = 0.0
    total_image_norm = 0.0
    total_text_norm = 0.0

    print(f"Processing {num_samples} samples from '{args.split}' split in '{args.data_dir}'...")
    for i in tqdm(range(num_samples), desc="Processing samples", unit="sample"):
        sample = data[i]
        # The image path is stored as a relative path (e.g., "images/cat_12.png").
        image_rel_path = sample["image"]
        image_path = os.path.join(args.data_dir, image_rel_path)

        # Extract the class name from the filename.
        # Expected file format: "images/<class_name>_<index>.png"
        base_name = os.path.basename(image_rel_path)
        class_name = base_name.split("_")[0]

        # Build a prompt using the class label.
        prompt = f"This is a {class_name}."

        metrics = process_sample(image_path, prompt, model, tokenizer, device)
        total_cosine += metrics["cosine_similarity"]
        total_image_norm += metrics["image_norm"]
        total_text_norm += metrics["text_norm"]

    avg_cosine = total_cosine / num_samples
    avg_image_norm = total_image_norm / num_samples
    avg_text_norm = total_text_norm / num_samples

    print("\nResults:")
    print(f"Processed {num_samples} samples from the dataset.")
    print(f"Average Cosine Similarity (normalized): {avg_cosine:.4f}")
    print(f"Average Image Embedding Norm: {avg_image_norm:.4f}")
    print(f"Average Text Embedding Norm: {avg_text_norm:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute average cosine similarity and L2 norms between image and text embeddings "
                    "from a dataset created with base.py"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="cifar10_vlm_dataset",
        help="Directory containing the dataset (with train.json/val.json)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to use (train or val)"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1000,
        help="Number of images to process"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda' or 'cpu'). If not set, it will use 'cuda' if available."
    )
    args = parser.parse_args()
    main(args)