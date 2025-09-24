import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- settings ---
base_model_name = "mistralai/Mistral-7B-v0.1"   # your base model
lora_model_path = "./lora_checkpoint"           # path where you trained LoRA
output_dir = "./mistral-7b-merged"              # where merged model will be saved

# --- load base model + tokenizer ---
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# --- load LoRA and merge ---
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, lora_model_path)

print("Merging LoRA weights into base model...")
model = model.merge_and_unload()  # merges and drops adapter

# --- save final merged model ---
print(f"Saving merged model to {output_dir}...")
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

print("✅ Done! Merged model saved.")
