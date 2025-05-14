import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
import numpy as np
from dataloader import create_dataloaders

# Import both the model class and CONFIG from train_projection
from train_projection import SigLIPProjectionModel, CONFIG as PROJECTION_CONFIG

# Configuration
CONFIG = {
    # Model parameters
    "SIGLIP_MODEL": "google/siglip-so400m-patch14-384",
    "PHI_MODEL": "microsoft/phi-3-mini-4k-instruct",
    "PROJECTION_MODEL_PATH": "siglip_phi3_projection/best_model.pt",
    
    # Use the same projection parameters from train_projection.py
    "PROJECTION_DIM": PROJECTION_CONFIG["PROJECTION_DIM"],
    "PROJECTION_LAYERS": PROJECTION_CONFIG["PROJECTION_LAYERS"],
    
    # QLoRA parameters
    "LORA_R": 16,
    "LORA_ALPHA": 32,
    "LORA_DROPOUT": 0.05,
    "LORA_TARGET_MODULES": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # Quantization parameters
    "QUANTIZATION_BITS": 4,  # 4-bit quantization
    "QUANT_TYPE": "nf4",     # NF4 data type
    
    # Training parameters
    "BATCH_SIZE": 8,
    "LEARNING_RATE": 2e-5,
    "WEIGHT_DECAY": 0.01,
    "EPOCHS": 5,
    "SAVE_EVERY": 1,
    "GRADIENT_ACCUMULATION_STEPS": 4,
    
    # Optimizer parameters
    "OPTIMIZER": "adamw",
    "SCHEDULER": "cosine",
    "LR_MIN_FACTOR": 0.1,   # Minimum LR will be LR * this factor
    "WARMUP_RATIO": 0.03,   # Percentage of steps for warmup
    
    # Mixed precision
    "USE_MIXED_PRECISION": True,
    "COMPUTE_DTYPE": torch.bfloat16, # Define compute dtype centrally (bfloat16 recommended for Ampere+)
    
    # Data parameters
    "DATA_DIR": "stl10_vlm_dataset_final",
    "NUM_WORKERS": 16,
    
    # Output parameters
    "OUTPUT_DIR": "phi3_qlora_adapter",
    "USE_WANDB": False,
}

class MultimodalPhiModel(nn.Module):
    """Model that combines SigLIP projection with Phi-3 for multimodal capabilities."""

    def __init__(self, siglip_model_name, phi_model_name, projection_model_path,
                 quantization_config=None, projection_dim=None, projection_layers=None,
                 compute_dtype=torch.bfloat16): # Added compute_dtype argument
        """
        Initialize the multimodal model.
        
        Args:
            siglip_model_name (str): Name of the SigLIP model
            phi_model_name (str): Name of the Phi-3 model
            projection_model_path (str): Path to the trained projection model
            quantization_config: Configuration for quantization
            projection_dim: Dimension of the projection layer
            projection_layers: Number of layers in projection MLP
            compute_dtype: Data type for computation (e.g., torch.bfloat16)
        """
        super().__init__()

        self.compute_dtype = compute_dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load processor first to check dimensions if needed
        print(f"Loading processor for {phi_model_name}...")
        # Use trust_remote_code=True if required by the model, common for newer models like Phi-3
        self.processor = AutoProcessor.from_pretrained(phi_model_name, trust_remote_code=True)

        # --- Start Fix for Pad Token ---
        # Access pad_token directly on the processor object
        if not hasattr(self.processor, 'pad_token') or self.processor.pad_token is None:
             print("Processor does not have pad_token set.")
             # Check if eos_token exists and is suitable
             if hasattr(self.processor, 'eos_token') and self.processor.eos_token:
                 print(f"Setting pad_token to eos_token: '{self.processor.eos_token}'")
                 self.processor.pad_token = self.processor.eos_token
             else:
                 # Fallback: Add a generic pad token if EOS is also missing (unlikely but safe)
                 print("Warning: EOS token also missing. Adding a default <|pad|> token.")
                 self.processor.add_special_tokens({'pad_token': '<|pad|>'})
        else:
            print(f"Processor pad_token already set: '{self.processor.pad_token}'")
        # --- End Fix for Pad Token ---


        # Load SigLIP projection model
        self.projection_model = SigLIPProjectionModel(
            siglip_model_name=siglip_model_name,
            phi_model_name=phi_model_name,
            projection_dim=projection_dim,
            num_layers=projection_layers
        )
        
        # Load trained projection weights
        if projection_model_path and os.path.exists(projection_model_path):
            print(f"Loading projection model from {projection_model_path}")
            checkpoint = torch.load(projection_model_path, map_location="cpu")
            self.projection_model.projection.load_state_dict(checkpoint["model_state_dict"])
        else:
            print(f"Warning: Projection model path {projection_model_path} not found. Using untrained projection.")
        # Move the projection model to the correct device and data type
        print(f"Casting projection model to {self.compute_dtype} on device {self.device}")
        self.projection_model.to(self.device, dtype=self.compute_dtype)
        
        # Freeze the projection model
        for param in self.projection_model.parameters():
            param.requires_grad = False
        
        # Load Phi-3 model with quantization for QLoRA
        # Use trust_remote_code=True here as well
        model_load_kwargs = {
            "torch_dtype": self.compute_dtype,
            "device_map": "auto", # Handles device placement for large models
            "trust_remote_code": True
        }
        if quantization_config:
            print("Loading Phi-3 model with quantization...")
            model_load_kwargs["quantization_config"] = quantization_config
            self.phi = AutoModelForCausalLM.from_pretrained(
                phi_model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            # --- Set Model's Pad Token ID ---
            # Ensure the model's config uses the same pad token ID as the processor
            # Access pad_token_id directly on the processor
            if hasattr(self.processor, 'pad_token_id') and self.processor.pad_token_id is not None:
                print(f"Setting model's pad_token_id to: {self.processor.pad_token_id}")
                self.phi.config.pad_token_id = self.processor.pad_token_id
            else:
                print("Warning: Could not determine processor's pad_token_id. Model config might be inconsistent.")
            # ---

            # Prepare model for k-bit training
            self.phi = prepare_model_for_kbit_training(self.phi)
        else:
            self.phi = AutoModelForCausalLM.from_pretrained(
                phi_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        
        # Load processor for tokenization
        self.processor = AutoProcessor.from_pretrained(phi_model_name)
    
    def forward(self, images, input_ids, attention_mask, labels=None):
        """
        Forward pass through the model.
        
        Args:
            images (torch.Tensor): Batch of images
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            labels (torch.Tensor, optional): Labels for computing loss
            
        Returns:
            torch.Tensor: Loss or logits
        """
        # Ensure images are on the same device and dtype as the projection model expects
        images = images.to(self.projection_model.siglip.device, dtype=self.compute_dtype)

        # Get projected image embeddings from the frozen projection model
        # No need for torch.no_grad() as the model is frozen and in eval mode
        image_embeddings = self.projection_model(images) # Shape: [batch_size, proj_dim]

        # Ensure projected embeddings match Phi's expected embedding dimension
        phi_embedding_dim = self.phi.get_input_embeddings().weight.shape[-1]
        if image_embeddings.shape[-1] != phi_embedding_dim:
             raise ValueError(f"Projected image embedding dimension ({image_embeddings.shape[-1]}) "
                              f"does not match Phi-3 embedding dimension ({phi_embedding_dim}). "
                              "Check your SigLIPProjectionModel's output dimension (projection_dim).")

        # Get the token embeddings from Phi model
        # Ensure input_ids are on the correct device (handled by device_map="auto" usually)
        # input_ids = input_ids.to(self.phi.device) # Usually not needed with device_map
        input_embeddings = self.phi.get_input_embeddings()(input_ids) # Shape: [batch_size, seq_len, phi_emb_dim]

        # Prepare image embeddings for concatenation: ensure same device and dtype as input_embeddings
        image_embeddings = image_embeddings.to(input_embeddings.device, dtype=input_embeddings.dtype)

        # Add the sequence length dimension: [batch_size, 1, phi_emb_dim]
        image_embeddings_unsqueezed = image_embeddings.unsqueeze(1)

        # Concatenate image embedding at the beginning
        # Input Embeddings: [ImageEmb, TokenEmb_1, TokenEmb_2, ...]
        final_embeddings = torch.cat(
            [image_embeddings_unsqueezed, input_embeddings[:, 1:, :]], # Exclude the original first token embedding
            dim=1
        )

        # Adjust attention mask: The image embedding corresponds to the first position.
        # The original attention mask should be kept as is if it correctly masks padding.
        # The image token should be attended to (mask=1).
        # Make sure attention mask is on the correct device
        # attention_mask = attention_mask.to(self.phi.device) # Usually not needed with device_map

        # Ensure labels are also on the correct device if provided
        if labels is not None:
            # labels = labels.to(self.phi.device) # Usually not needed with device_map
            pass # device_map should handle label device placement relative to logits

        # Forward pass through Phi model using the combined embeddings
        outputs = self.phi(
            inputs_embeds=final_embeddings,
            attention_mask=attention_mask, # Use original mask, assuming image replaces a non-padded token
            labels=labels,
            return_dict=True
        )
        
        return outputs

# --- *** MODIFIED train_epoch *** ---
def train_epoch(model, train_loader, optimizer, lr_scheduler, scaler, device, gradient_accumulation_steps=1, grad_clip_norm=1.0, compute_dtype=torch.bfloat16):
    """Train for one epoch with corrected loss accumulation and scheduler stepping."""
    model.train() # Ensure model is in training mode (for dropout, etc.)
    total_loss = 0.0
    num_samples = 0 # Initialize num_samples

    progress_bar = tqdm(train_loader, desc="Training")
    optimizer.zero_grad() # Clear gradients at the start of the epoch

    for step, batch in enumerate(progress_bar):
        # Move data to device
        images = batch["image"] # Keep images on CPU initially, move inside forward
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Create labels for causal language modeling
        labels = input_ids.clone()
        labels[:, 0] = -100  # Don't compute loss for the image token position
        # Ensure labels are on the same device as the final logits will be (handled by device_map)
        # labels = labels.to(device) # Can move here or rely on device_map

        batch_size = images.size(0) # Get batch size for averaging

        # Forward pass with mixed precision
        # Use device.type for autocast device
        with autocast(device_type=device.type, dtype=compute_dtype, enabled=scaler.is_enabled()):
             outputs = model(
                 images=images,
                 input_ids=input_ids,
                 attention_mask=attention_mask,
                 labels=labels
             )
             loss = outputs.loss # Loss is computed internally by the model

        # Check if loss is valid before accumulation and backward pass
        if loss is not None and not torch.isnan(loss):
            loss = loss / gradient_accumulation_steps # Normalize loss for accumulation
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Accumulate total loss correctly weighted by batch size
            # Use loss.item() for accumulation to free graph memory
            total_loss += loss.item() * gradient_accumulation_steps * batch_size
            num_samples += batch_size # Increment sample count

            progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps}) # Show effective loss for this step batch
        else:
             print(f"Warning: Step {step}: Received None or NaN loss. Skipping gradient update for this batch.")
             # If loss is invalid, we might need to skip the optimizer step for this accumulation cycle
             # Easiest is to just not call backward and proceed. Optimizer step check will handle it.


        # --- Optimizer Step Block ---
        # Check if it's time to update weights
        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.phi.parameters()), grad_clip_norm)
            # Optimizer step (will check for inf/NaNs)
            scaler.step(optimizer)
            # Update the scaler for next iteration
            scaler.update()
            # Step the learning rate scheduler
            lr_scheduler.step() # <<<--- SCHEDULER STEP MOVED HERE
            # Zero gradients for the next accumulation cycle
            optimizer.zero_grad()

    # Calculate average loss per sample for the epoch
    avg_epoch_loss = total_loss / num_samples if num_samples > 0 else 0.0
    print(f"\nEpoch Train Summary: total_loss={total_loss:.4f}, num_samples={num_samples}")
    return avg_epoch_loss
# --- *** END MODIFIED train_epoch *** ---

def validate(model, val_loader, device, compute_dtype=torch.bfloat16):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to device
            images = batch["image"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Create labels for causal language modeling
            labels = input_ids.clone()
            labels[:, 0] = -100  # Don't compute loss for the image token position
            labels = labels.to(device) # Move labels initially
            
            # Forward pass
            with autocast(device_type=device.type, dtype=compute_dtype, enabled=CONFIG["USE_MIXED_PRECISION"]): # Use config flag directly
                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            total_loss += outputs.loss.item() * images.size(0) # Accumulate loss weighted by batch size
            num_samples += images.size(0)
    
    # Return average loss per sample for the validation set
    avg_val_loss = total_loss / num_samples if num_samples > 0 else 0.0
    return avg_val_loss

def main(args):
    # Print configuration
    print("Running with configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Determine compute dtype from config
    compute_dtype = CONFIG["COMPUTE_DTYPE"]
    print(f"Using compute dtype: {compute_dtype}")
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project="phi3-qlora-multimodal",
            config=vars(args)
        )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        phi_processor_name=args.phi_model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Created dataloaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    # Set up quantization config for QLoRA
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype, # Use defined compute_dtype
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_use_double_quant=True
    )
    
    # Create model
    model = MultimodalPhiModel(
        siglip_model_name=args.siglip_model_name,
        phi_model_name=args.phi_model_name,
        projection_model_path=args.projection_model_path,
        quantization_config=quantization_config,
        projection_dim=args.projection_dim,
        projection_layers=args.projection_layers,
        compute_dtype=compute_dtype # Pass compute_dtype
    )
    
    # Apply LoRA to the model
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model.phi = get_peft_model(model.phi, lora_config)
    model.phi.print_trainable_parameters()
    
    # Set up optimizer
    # Set up optimizer - only optimize trainable LoRA parameters
    # Filter parameters that require gradients
    trainable_params = filter(lambda p: p.requires_grad, model.phi.parameters())
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Set up learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    if args.scheduler.lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps), # Ensure T_max is at least 1
            eta_min=args.learning_rate * args.lr_min_factor
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_update_steps - warmup_steps),
            eta_min=args.learning_rate * args.lr_min_factor
        )
    
    # Set up gradient scaler for mixed precision
    scaler = GradScaler(enabled=args.use_mixed_precision)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"Starting training for {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            scheduler,
            scaler, 
            device, 
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            compute_dtype=compute_dtype
        )
        
        # Validate
        val_loss = validate(model, val_loader, device, compute_dtype=compute_dtype)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr'] # Get current LR after scheduler step
        
        # Log metrics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        if args.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.phi.save_pretrained(os.path.join(args.output_dir, "best_model"))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            model.phi.save_pretrained(os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}"))
    
    # Save final model
    model.phi.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Phi-3 with SigLIP projection")

    # Get CONFIG from CONFIG dict
    compute_dtype = CONFIG["COMPUTE_DTYPE"]
    # Model arguments
    parser.add_argument("--siglip_model_name", type=str, default=CONFIG["SIGLIP_MODEL"],
                        help="Name of the SigLIP model")
    parser.add_argument("--phi_model_name", type=str, default=CONFIG["PHI_MODEL"],
                        help="Name of the Phi-3 model")
    parser.add_argument("--projection_model_path", type=str, default=CONFIG["PROJECTION_MODEL_PATH"],
                        help="Path to the trained projection model")
    
    # QLoRA arguments
    parser.add_argument("--lora_r", type=int, default=CONFIG["LORA_R"],
                        help="Rank of the LoRA update matrices")
    parser.add_argument("--lora_alpha", type=int, default=CONFIG["LORA_ALPHA"],
                        help="Scaling factor for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=CONFIG["LORA_DROPOUT"],
                        help="Dropout probability for LoRA layers")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=CONFIG["LORA_TARGET_MODULES"],
                        help="List of module names to apply LoRA to")
    parser.add_argument("--quant_type", type=str, default=CONFIG["QUANT_TYPE"],
                        help="Quantization type (nf4 or fp4)")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default=CONFIG["DATA_DIR"],
                        help="Directory containing the dataset")
    parser.add_argument("--batch_size", type=int, default=CONFIG["BATCH_SIZE"],
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=CONFIG["NUM_WORKERS"],
                        help="Number of workers for data loading")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=CONFIG["EPOCHS"],
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=CONFIG["LEARNING_RATE"],
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=CONFIG["WEIGHT_DECAY"],
                        help="Weight decay")
    parser.add_argument("--save_every", type=int, default=CONFIG["SAVE_EVERY"],
                        help="Save checkpoint every N epochs")
    parser.add_argument("--optimizer", type=str, default=CONFIG["OPTIMIZER"],
                        help="Optimizer to use (adamw, adam)")
    parser.add_argument("--scheduler", type=str, default=CONFIG["SCHEDULER"],
                        help="Learning rate scheduler (cosine, linear)")
    parser.add_argument("--lr_min_factor", type=float, default=CONFIG["LR_MIN_FACTOR"],
                        help="Minimum learning rate factor")
    parser.add_argument("--warmup_ratio", type=float, default=CONFIG["WARMUP_RATIO"],
                        help="Percentage of steps for warmup")
    parser.add_argument("--use_mixed_precision", action="store_true", default=CONFIG["USE_MIXED_PRECISION"],
                        help="Whether to use mixed precision training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=CONFIG["GRADIENT_ACCUMULATION_STEPS"],
                        help="Number of steps to accumulate gradients")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default=CONFIG["OUTPUT_DIR"],
                        help="Directory to save models")
    parser.add_argument("--use_wandb", action="store_true", default=CONFIG["USE_WANDB"],
                        help="Whether to use Weights & Biases for logging")
    
    # Projection arguments
    parser.add_argument("--projection_dim", type=int, default=CONFIG["PROJECTION_DIM"],
                        help="Dimension of the projection layer")
    parser.add_argument("--projection_layers", type=int, default=CONFIG["PROJECTION_LAYERS"],
                        help="Number of layers in projection MLP")
    
    args = parser.parse_args()
    main(args) 
