import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoProcessor

# Configuration
CONFIG = {
    # Dataset parameters
    "IMAGE_SIZE": 384,  # SigLIP input size
    "MAX_LENGTH": 512,  # Maximum sequence length for text
    "NORMALIZE_MEAN": [0.4467, 0.4398, 0.4066],  # STL-10 normalization mean
    "NORMALIZE_STD": [0.2241, 0.2215, 0.2239],   # STL-10 normalization std
    
    # Dataloader parameters
    "BATCH_SIZE": 8,
    "NUM_WORKERS": 4,
    "PIN_MEMORY": True,
    
    # Text processing
    "USE_FIRST_RESPONSE_ONLY": True,  # Whether to use only the first assistant response
}

class SigLIPPhi3Dataset(Dataset):
    """Dataset for training a projection layer between SigLIP and Phi-3."""
    
    def __init__(self, data_dir, json_file, phi_processor_name, max_length=CONFIG["MAX_LENGTH"]):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing the dataset
            json_file (str): Path to the JSON file with conversation data
            phi_processor_name (str): Name of the Phi-3 processor to use
            max_length (int): Maximum sequence length for text
        """
        self.data_dir = data_dir
        self.phi_processor = AutoProcessor.from_pretrained(phi_processor_name)
        self.max_length = max_length
        
        # Load the dataset
        with open(os.path.join(data_dir, json_file), 'r') as f:
            self.data = json.load(f)
            
        # Image transformation for SigLIP
        self.image_transform = transforms.Compose([
            transforms.Resize((CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=CONFIG["NORMALIZE_MEAN"], std=CONFIG["NORMALIZE_STD"])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and transform image
        image_path = os.path.join(self.data_dir, item["image"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)
        
        # Extract text from conversations
        # We'll use the assistant's responses as targets
        text = ""
        for i, conv in enumerate(item["conversations"]):
            if conv["from"] == "assistant":
                text += conv["value"] + " "
                # Only use the first assistant response if configured
                if CONFIG["USE_FIRST_RESPONSE_ONLY"] and i > 1:
                    break
        
        # Tokenize text for Phi-3
        text_encoding = self.phi_processor(
            text, 
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        input_ids = text_encoding.input_ids.squeeze(0)
        attention_mask = text_encoding.attention_mask.squeeze(0)
        
        return {
            "image": image_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text": text  # Keep original text for debugging
        }

def create_dataloaders(data_dir, phi_processor_name, batch_size=CONFIG["BATCH_SIZE"], num_workers=CONFIG["NUM_WORKERS"]):
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir (str): Directory containing the dataset
        phi_processor_name (str): Name of the Phi-3 processor
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = SigLIPPhi3Dataset(
        data_dir=data_dir,
        json_file="train.json",
        phi_processor_name=phi_processor_name
    )
    
    val_dataset = SigLIPPhi3Dataset(
        data_dir=data_dir,
        json_file="val.json",
        phi_processor_name=phi_processor_name
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=CONFIG["PIN_MEMORY"]
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=CONFIG["PIN_MEMORY"]
    )
    
    return train_dataloader, val_dataloader

if __name__ == "__main__":
    # Test the dataloader
    data_dir = "stl10_vlm_dataset_final"
    phi_processor_name = "microsoft/phi-3-mini-4k-instruct"
    
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        phi_processor_name=phi_processor_name,
        batch_size=2
    )
    
    # Print sample batch
    for batch in train_loader:
        print(f"Image shape: {batch['image'].shape}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Sample text: {batch['text'][0][:100]}...")
        break 