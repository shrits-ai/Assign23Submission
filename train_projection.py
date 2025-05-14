import os
HF_CACHE_PATH = "/opt/dlami/nvme/huggingface_cache"
os.makedirs(HF_CACHE_PATH, exist_ok=True)
os.environ['HF_HOME'] = HF_CACHE_PATH
os.environ['HF_DATASETS_CACHE'] = os.path.join(HF_CACHE_PATH, 'datasets')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(HF_CACHE_PATH, 'transformers')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import argparse
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor
import wandb
import numpy as np
from dataloader import create_dataloaders
import torch.nn.functional as F

# Configuration
CONFIG = {
    # Model parameters
    "SIGLIP_MODEL": "google/siglip-so400m-patch14-384",
    "PHI_MODEL": "microsoft/phi-3-mini-4k-instruct",
    "PROJECTION_DIM": 4096,
    "PROJECTION_LAYERS": 2,  # Number of layers in projection MLP
    "ACTIVATION": "gelu",    # Activation function in projection MLP
    
    # Training parameters
    "BATCH_SIZE": 16,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 0.01,
    "EPOCHS": 10,
    "SAVE_EVERY": 5,
    
    # Optimizer parameters
    "OPTIMIZER": "adamw",
    "SCHEDULER": "cosine",
    "LR_MIN_FACTOR": 0.01,   # Minimum LR will be LR * this factor
    
    # Loss function
    "LOSS_FUNCTION": "mse",
    
    # Mixed precision
    "USE_MIXED_PRECISION": True,
    
    # Data parameters
    "DATA_DIR": "stl10_vlm_dataset_final",
    "NUM_WORKERS": 4,
    
    # Output parameters
    "OUTPUT_DIR": "siglip_phi3_projection",
    "USE_WANDB": False,
}

class SigLIPProjectionModel(nn.Module):
    """Model that projects SigLIP embeddings to Phi-3 embedding space."""
    
    def __init__(self, siglip_model_name, phi_model_name, projection_dim=None, num_layers=CONFIG["PROJECTION_LAYERS"]):
        """
        Initialize the projection model.
        
        Args:
            siglip_model_name (str): Name of the SigLIP model
            phi_model_name (str): Name of the Phi-3 model
            projection_dim (int, optional): Dimension of the projection layer
            num_layers (int): Number of layers in projection MLP
        """
        super().__init__()
        
        # Load SigLIP model
        self.siglip = AutoModel.from_pretrained(siglip_model_name)
        
        # Load Phi-3 model
        self.phi = AutoModelForCausalLM.from_pretrained(
            phi_model_name, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Freeze both models
        for param in self.siglip.parameters():
            param.requires_grad = False
        
        for param in self.phi.parameters():
            param.requires_grad = False
        
        # Get embedding dimensions
        # SigLIP uses 'vision_config.hidden_size' instead of 'hidden_size'
        if hasattr(self.siglip.config, 'vision_config'):
            siglip_dim = self.siglip.config.vision_config.hidden_size
        elif hasattr(self.siglip.config, 'projection_dim'):
            siglip_dim = self.siglip.config.projection_dim
        elif hasattr(self.siglip.config, 'embed_dim'):
            siglip_dim = self.siglip.config.embed_dim
        else:
            # Fallback to a common dimension for SigLIP models
            siglip_dim = 1024
            print(f"Warning: Could not determine SigLIP embedding dimension, using default: {siglip_dim}")
            print(f"Available config attributes: {dir(self.siglip.config)}")
        
        phi_dim = self.phi.config.hidden_size
        
        print(f"SigLIP embedding dimension: {siglip_dim}")
        print(f"Phi-3 embedding dimension: {phi_dim}")
        
        # Create projection layer
        if projection_dim is None:
            projection_dim = phi_dim
        
        # Choose activation function
        if CONFIG["ACTIVATION"].lower() == "gelu":
            activation = nn.GELU()
        elif CONFIG["ACTIVATION"].lower() == "relu":
            activation = nn.ReLU()
        elif CONFIG["ACTIVATION"].lower() == "silu":
            activation = nn.SiLU()
        else:
            activation = nn.GELU()  # Default to GELU
        
        # Build projection MLP
        if num_layers == 1:
            self.projection = nn.Linear(siglip_dim, phi_dim)
        else:
            layers = []
            # First layer
            layers.append(nn.Linear(siglip_dim, projection_dim))
            layers.append(activation)
            
            # Middle layers (if any)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(projection_dim, projection_dim))
                layers.append(activation)
            
            # Last layer
            layers.append(nn.Linear(projection_dim, phi_dim))
            
            self.projection = nn.Sequential(*layers)
        
        # Initialize weights
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, images):
        """
        Forward pass through the model.
        
        Args:
            images (torch.Tensor): Batch of images
            
        Returns:
            torch.Tensor: Projected embeddings
        """
        # Get SigLIP embeddings
        with torch.no_grad():
            # For SigLIP vision-only embedding, we need to use the vision_model directly
            vision_outputs = self.siglip.vision_model(pixel_values=images)
            siglip_embeddings = vision_outputs.pooler_output  # Use pooled output
        
        # Project to Phi-3 embedding space
        projected_embeddings = self.projection(siglip_embeddings)
        
        return projected_embeddings

def compute_loss(projected_embeddings, target_embeddings, loss_fn):
    """
    Compute the loss between projected embeddings and target embeddings.
    
    Args:
        projected_embeddings (torch.Tensor): Projected SigLIP embeddings
        target_embeddings (torch.Tensor): Target Phi-3 embeddings
        loss_fn: Loss function
        
    Returns:
        torch.Tensor: Loss value
    """
    return loss_fn(projected_embeddings, target_embeddings)

def extract_phi_embeddings(model, input_ids, attention_mask):
    """
    Extract embeddings from Phi-3 model.
    
    Args:
        model: Phi-3 model
        input_ids (torch.Tensor): Input token IDs
        attention_mask (torch.Tensor): Attention mask
        
    Returns:
        torch.Tensor: Phi-3 embeddings
    """
    with torch.no_grad():
        # Get the embedding layer
        embeddings = model.get_input_embeddings()(input_ids)
        
        # Average the embeddings across the sequence dimension (weighted by attention mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled_embeddings = (embeddings * mask).sum(dim=1) / mask.sum(dim=1)
        
    return pooled_embeddings

def train_epoch(model, train_loader, optimizer, scaler, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_cosine = 0.0  # Initialize total_cosine
    num_batches = 0     # Initialize num_batches

    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        # Move data to device
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=CONFIG["USE_MIXED_PRECISION"]):
            # Get projected embeddings
            projected_embeddings = model(images)
            
            # Get target Phi-3 embeddings
            target_embeddings = extract_phi_embeddings(
                model.phi, 
                input_ids, 
                attention_mask
            )
            
            # Normalize embeddings for cosine similarity
            projected_norm = torch.nn.functional.normalize(projected_embeddings, p=2, dim=-1)
            target_norm = torch.nn.functional.normalize(target_embeddings, p=2, dim=-1)
            
            # Compute loss
            loss = compute_loss(projected_norm, target_norm, loss_fn)
            
        # Compute cosine similarity for logging.
        # F.cosine_similarity returns a value per sample; take mean for the batch.
        batch_cosine = F.cosine_similarity(projected_norm, target_norm, dim=-1).mean().item()
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update accumulators
        total_loss += loss.item()
        total_cosine += batch_cosine
        num_batches += 1
        
        progress_bar.set_postfix({"loss": loss.item(), "cosine": f"{batch_cosine:.4f}"})
        
    avg_loss = total_loss / len(train_loader)
    avg_cosine = total_cosine / num_batches if num_batches > 0 else 0
    return avg_loss, avg_cosine

def validate(model, val_loader, loss_fn, device):
    """Validate the model and log average loss and cosine similarity."""
    model.eval()
    total_loss = 0
    total_cosine = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to device.
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Obtain embeddings.
            projected_embeddings = model(images)
            target_embeddings = extract_phi_embeddings(model.phi, input_ids, attention_mask)
            
            # Normalize both embeddings.
            projected_norm = torch.nn.functional.normalize(projected_embeddings, p=2, dim=-1)
            target_norm = torch.nn.functional.normalize(target_embeddings, p=2, dim=-1)
            
            # Compute validation loss on normalized embeddings.
            loss = loss_fn(projected_norm, target_norm)
            total_loss += loss.item()
            
            # Compute the cosine similarity for the batch.
            batch_cosine = F.cosine_similarity(projected_norm, target_norm, dim=-1).mean().item()
            total_cosine += batch_cosine
            num_batches += 1
    
    avg_loss = total_loss / len(val_loader)
    avg_cosine = total_cosine / num_batches if num_batches > 0 else 0
    print(f"Validation Loss: {avg_loss:.4f}, Avg Cosine Similarity: {avg_cosine:.4f}")
    return avg_loss, avg_cosine


def main(args):
    # Print configuration
    print("Running with configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        phi_processor_name=args.phi_model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Created dataloaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    # Create model
    model = SigLIPProjectionModel(
        siglip_model_name=args.siglip_model_name,
        phi_model_name=args.phi_model_name,
        projection_dim=args.projection_dim,
        num_layers=args.projection_layers
    ).to(device)
    
    print(f"Created model with projection dimension: {args.projection_dim}, layers: {args.projection_layers}")
    
    # Set up optimizer
    if args.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            model.projection.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.projection.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.AdamW(
            model.projection.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    
    # Set up learning rate scheduler
    if args.scheduler.lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.learning_rate * args.lr_min_factor
        )
    elif args.scheduler.lower() == "linear":
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args.lr_min_factor,
            total_iters=args.epochs
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.learning_rate * args.lr_min_factor
        )
    
    # Set up loss function
    if args.loss_function.lower() == "mse":
        loss_fn = nn.MSELoss()
    elif args.loss_function.lower() == "l1":
        loss_fn = nn.L1Loss()
    elif args.loss_function.lower() == "smoothl1":
        loss_fn = nn.SmoothL1Loss()
    else:
        loss_fn = nn.MSELoss()
    
    print(f"Using {args.loss_function} loss function")
    
    # Set up gradient scaler for mixed precision
    scaler = GradScaler(enabled=args.use_mixed_precision)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop with logging of cosine similarity.
    best_val_loss = float('inf')
    
    print(f"Starting training for {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train: now returning both loss and average cosine similarity.
        train_loss, train_cosine = train_epoch(model, train_loader, optimizer, scaler, loss_fn, device)
        
        # Validate with logging of cosine similarity.
        val_loss, val_cosine = validate(model, val_loader, loss_fn, device)
        
        # Step learning rate scheduler.
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log metrics.
        print(f"Train Loss: {train_loss:.4f}, Train Cosine: {train_cosine:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Cosine: {val_cosine:.4f}, LR: {current_lr:.6f}")
        if args.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "train_cosine_similarity": train_cosine,
                "val_loss": val_loss,
                "val_cosine_similarity": val_cosine,
                "learning_rate": current_lr
            })
        
        # Save best model checkpoint.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.projection.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, os.path.join(args.output_dir, "best_model.pt"))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save checkpoint every save_every epochs.
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.projection.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
    
    # Save final model.
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.projection.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss
    }, os.path.join(args.output_dir, "final_model.pt"))
    
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SigLIP to Phi-3 projection layer")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="stl10_vlm_dataset_final",
        help="Directory containing the dataset (with train.json/val.json)"
    )
    parser.add_argument(
        "--siglip_model_name",
        type=str,
        default="google/siglip-so400m-patch14-384",
        help="Name of the SigLIP model"
    )
    parser.add_argument(
        "--phi_model_name",
        type=str,
        default="microsoft/phi-3-mini-4k-instruct",
        help="Name of the Phi-3 model"
    )
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=4096,
        help="Projection dimension for the intermediate layer"
    )
    parser.add_argument(
        "--projection_layers",
        type=int,
        default=2,
        help="Number of layers in the projection MLP"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--lr_min_factor",
        type=float,
        default=0.01,
        help="Minimum learning rate factor"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="Optimizer to use"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        help="Learning rate scheduler to use"
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        default="mse",
        help="Loss function to use"
    )
    parser.add_argument(
        "--use_mixed_precision",
        action="store_true",
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights and Biases for logging"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="siglip_phi3_projection",
        help="Output directory for model checkpoints and logs"
    )

    args = parser.parse_args()
    main(args)
