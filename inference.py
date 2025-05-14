#!/usr/bin/env python
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import argparse
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from train_projection import SigLIPProjectionModel, CONFIG as PROJECTION_CONFIG
from train_qlora import CONFIG as QLORA_CONFIG

def patch_dynamic_cache():
    try:
        from transformers.models.phi3.modeling_phi3 import DynamicCache
        if not hasattr(DynamicCache, "get_max_length"):
            print("Monkey patching DynamicCache: aliasing get_max_length to get_seq_length")
            DynamicCache.get_max_length = DynamicCache.get_seq_length
    except Exception as e:
        print("Warning: Failed to patch DynamicCache:", e)

def load_models_for_inference(phi_model_name, siglip_model_name, projection_checkpoint_path, adapter_checkpoint_path, projection_dim, projection_layers, quant_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = QLORA_CONFIG["COMPUTE_DTYPE"]

    print(f"Loading base Phi-3 model: {phi_model_name} with quantization...")
    phi_model = AutoModelForCausalLM.from_pretrained(
        phi_model_name,
        #quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    phi_tokenizer = AutoTokenizer.from_pretrained(phi_model_name, trust_remote_code=True)
    if phi_tokenizer.pad_token is None:
        print("Setting pad_token to eos_token for tokenizer.")
        phi_tokenizer.pad_token = phi_tokenizer.eos_token
        phi_model.config.pad_token_id = phi_tokenizer.pad_token_id

    print(f"Loading QLoRA adapter from: {adapter_checkpoint_path}...")
    #model_with_adapter = PeftModel.from_pretrained(phi_model, adapter_checkpoint_path)
    #model_with_adapter.eval()
    print("Phi-3 model with QLoRA adapter loaded.")

    print(f"Loading SigLIP projection model from: {projection_checkpoint_path}...")
    projection_model = SigLIPProjectionModel(
        siglip_model_name=siglip_model_name,
        phi_model_name=phi_model_name,
        projection_dim=projection_dim,
        num_layers=projection_layers
    )
    proj_checkpoint = torch.load(projection_checkpoint_path, map_location="cpu")
    projection_model.projection.load_state_dict(proj_checkpoint["model_state_dict"])
    projection_model.to(device, dtype=compute_dtype).eval()
    print("SigLIP projection model loaded.")

    return model_with_adapter, projection_model, phi_tokenizer, device, compute_dtype

def process_image(image_path, transform=None):
    image = Image.open(image_path).convert("RGB")
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616])
        ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def generate_text_from_image(phi_model_with_adapter, projection_model, image_tensor, phi_tokenizer, device, compute_dtype, max_length=200, prompt="Describe what you see in this image."):
    patch_dynamic_cache()
    with torch.no_grad():
        image_embeddings = projection_model(image_tensor.to(device, dtype=compute_dtype))
    
    special_token = "<image>"
    if special_token not in phi_tokenizer.get_vocab():
        print(f"Adding special token '{special_token}' to tokenizer vocabulary.")
        phi_tokenizer.add_tokens([special_token])
        phi_model_with_adapter.resize_token_embeddings(len(phi_tokenizer))
    special_token_id = phi_tokenizer.convert_tokens_to_ids(special_token)
    print(f"Special token '{special_token}' has ID: {special_token_id}")
    
    full_prompt = f"{special_token} {prompt}"
    tokenized = phi_tokenizer(full_prompt, return_tensors="pt").to(device)
    print("Decoded full prompt:", phi_tokenizer.decode(tokenized.input_ids[0], skip_special_tokens=False))
    print("Tokenized input IDs:", tokenized.input_ids)
    
    embedding_layer = phi_model_with_adapter.get_input_embeddings()
    original_weight = embedding_layer.weight.data[special_token_id].clone()
    injected_embedding = image_embeddings.squeeze()
    embedding_layer.weight.data[special_token_id] = injected_embedding
    print("Injected special token embedding (norm):", torch.norm(injected_embedding).item())
    print("Injected special token embedding (first 5 values):", injected_embedding[:5].tolist())
    
    outputs = phi_model_with_adapter.generate(
        input_ids=tokenized.input_ids,
        attention_mask=tokenized.attention_mask,
        max_length=tokenized.input_ids.shape[1] + max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=phi_tokenizer.pad_token_id,
        eos_token_id=phi_tokenizer.eos_token_id
    )
    print("Generated output token IDs:", outputs[0])
    embedding_layer.weight.data[special_token_id] = original_weight
    generated_text = phi_tokenizer.decode(outputs[0][tokenized.input_ids.shape[1]:], skip_special_tokens=True)
    print("Decoded generated text (raw):", phi_tokenizer.decode(outputs[0], skip_special_tokens=False))
    return generated_text

def main(args):
    compute_dtype = QLORA_CONFIG["COMPUTE_DTYPE"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=QLORA_CONFIG["QUANT_TYPE"],
        bnb_4bit_use_double_quant=True
    )
    
    phi_model_with_adapter, projection_model, phi_tokenizer, device, compute_dtype = load_models_for_inference(
        phi_model_name=args.phi_model_name,
        siglip_model_name=args.siglip_model_name,
        projection_checkpoint_path=args.projection_model_path,
        adapter_checkpoint_path=args.adapter_checkpoint_path,
        projection_dim=args.projection_dim,
        projection_layers=args.projection_layers,
        quant_config=quantization_config
    )
    
    image_tensor = process_image(args.image_path)
    
    generated_text = generate_text_from_image(
        phi_model_with_adapter,
        projection_model,
        image_tensor,
        phi_tokenizer,
        device,
        compute_dtype,
        max_length=args.max_length,
        prompt=args.prompt
    )
    
    print(f"Generated text for image {args.image_path}:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with SigLIP to Phi-3 projection model and QLoRA adapter")
    parser.add_argument("--siglip_model_name", type=str, default="google/siglip-so400m-patch14-384", help="Name of the SigLIP model")
    parser.add_argument("--phi_model_name", type=str, default="microsoft/phi-3-mini-4k-instruct", help="Name of the Phi-3 model")
    parser.add_argument("--projection_model_path", type=str, required=True, help="Path to the trained projection model checkpoint (.pt file from train_projection.py)")
    parser.add_argument("--adapter_checkpoint_path", type=str, required=True, help="Path to the trained QLoRA adapter directory (e.g., ./phi3_qlora_adapter/final_model/)")
    parser.add_argument("--projection_dim", type=int, default=PROJECTION_CONFIG["PROJECTION_DIM"], help="Projection dimension")
    parser.add_argument("--projection_layers", type=int, default=PROJECTION_CONFIG["PROJECTION_LAYERS"], help="Number of layers in projection MLP")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length for generated text")
    parser.add_argument("--prompt", type=str, default="What is the object in the image? Describe it in detail.", help="Prompt to guide image description generation")
    args = parser.parse_args()
    main(args)
