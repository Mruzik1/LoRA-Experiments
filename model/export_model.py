from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch


peft_model_path = "../llm_checkpoint"
config = PeftConfig.from_pretrained(peft_model_path)

base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map="cpu",
    torch_dtype=torch.float32,
)
model = PeftModel.from_pretrained(base_model, peft_model_path).to(base_model.device)

merged = model.merge_and_unload()
out_folder = "../llama_merged"
merged.save_pretrained(out_folder)