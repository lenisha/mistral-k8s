import torch,os
from transformers import AutoModelForCausalLM, AutoTokenizer
# Load the model and tokenizer
print(f"starting to load model")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", attn_implementation="eager", revision="26bca36bde8333b5d7f72e9ed20ccda6a618af24")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", revision="26bca36bde8333b5d7f72e9ed20ccda6a618af24")
# save the model

save_dir = os.getenv("MODEL_PATH", "./models/mistral-7B-v0.1")
print(f"Saving model to path: {save_dir}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"Model saved to {save_dir}")
exit(0)