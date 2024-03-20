from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from safetensors.torch import load
import os

def preprocess_data(data, tokenizer):
    input_ids = tokenizer.encode(data, return_tensors='pt')
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask

model_path = input("Enter the path to your optimized model: ")
tokenizer = AutoTokenizer.from_pretrained(model_path)

config = AutoConfig.from_pretrained(model_path)

# Correctly reading the .safetensors file in binary mode and loading it
model_state_dict_path = os.path.join(model_path, "model.safetensors")
with open(model_state_dict_path, "rb") as f:
    data = f.read()

model_state_dict = load(data)

# Print the keys of the loaded state dictionary for debugging
print("Loaded state dictionary keys:")
for key in model_state_dict.keys():
    print(key)

# Create a new model instance with the loaded config
model = AutoModelForCausalLM.from_config(config)

# Print the keys of the model's state dictionary for debugging
print("Model state dictionary keys:")
for key in model.state_dict().keys():
    print(key)

# Load the state dictionary into the model
# Adjust keys to remove 'model.' prefix
adjusted_model_state_dict = {key.replace('model.', ''): value for key, value in model_state_dict.items()}

# Check if the size of lm_head.weight matches the expected shape
vocab_size = config.vocab_size
hidden_size = config.hidden_size
expected_lm_head_size = vocab_size * hidden_size
actual_lm_head_size = adjusted_model_state_dict['lm_head.weight'].numel()

if actual_lm_head_size != expected_lm_head_size:
    print(f"Warning: The size of 'lm_head.weight' in the loaded state dictionary ({actual_lm_head_size}) does not match the expected size ({expected_lm_head_size}).")
    print("Creating a new tensor with the expected shape and copying the values.")

    loaded_lm_head_weight = adjusted_model_state_dict['lm_head.weight']
    new_lm_head_weight = torch.zeros(vocab_size, hidden_size, dtype=loaded_lm_head_weight.dtype, device=loaded_lm_head_weight.device)

    # Copy the values from the loaded tensor to the new tensor
    num_elements_to_copy = min(actual_lm_head_size, expected_lm_head_size)
    new_lm_head_weight.view(-1)[:num_elements_to_copy] = loaded_lm_head_weight.view(-1)[:num_elements_to_copy]

    adjusted_model_state_dict['lm_head.weight'] = new_lm_head_weight

else:
    adjusted_model_state_dict['lm_head.weight'] = adjusted_model_state_dict['lm_head.weight'].view(vocab_size, hidden_size)

# Load the adjusted state dictionary into the model
model.load_state_dict(adjusted_model_state_dict, strict=False)

# Move the model to the appropriate device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Example usage
input_text = input("Enter the text to process: ")
input_ids, attention_mask = preprocess_data(input_text, tokenizer)

with torch.no_grad():
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:", generated_text)
