import torch
import torch.nn as nn
from transformers import LlamaConfig
from quantization_utils import quantize_tensor, pack_quantized_tensor, unpack_quantized_tensor
from llama_model import LlamaModel, BitLinear
from tqdm import tqdm
import os
import time

def quantize_activations(x, b=8):
    Q_b = 2 ** (b - 1)
    return torch.clamp(x, -Q_b, Q_b)

def optimize_matrix_multiplications(model):
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            def optimized_forward(input):
                quantized_weight = module.ternarize_weights_groupwise()
                output = nn.functional.linear(input, quantized_weight, module.bias)
                return output.clamp(-128, 127).to(torch.int8)

            module.forward = optimized_forward

    return model

def validate_quantization(original_tensor, quantized_tensor, eps=1e-5):
    reconstructed_tensor = quantized_tensor.float() * original_tensor.abs().mean()
    max_error = torch.max(torch.abs(original_tensor - reconstructed_tensor))
    mean_error = torch.mean(torch.abs(original_tensor - reconstructed_tensor))
    print(f"Max Error: {max_error.item()}, Mean Error: {mean_error.item()}")
    assert max_error < eps, "Quantization error exceeds tolerance"

def calculate_model_dimensions(num_params):
    min_hidden_size = 256
    max_hidden_size = 2048
    min_num_layers = 2
    max_num_layers = 48
    target_ratio = 0.99

    def params_from_dimensions(hidden_size, num_layers):
        config = LlamaConfig(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            num_attention_heads=max(1, hidden_size // 64),
            num_hidden_layers=num_layers,
            vocab_size=32000,
        )
        model = LlamaModel(config)
        return sum(p.numel() for p in model.parameters())

    best_config = None
    best_num_params = 0

    for num_layers in range(max_num_layers, min_num_layers - 1, -1):
        lower_hidden_size = min_hidden_size
        upper_hidden_size = max_hidden_size

        while lower_hidden_size <= upper_hidden_size:
            hidden_size = (lower_hidden_size + upper_hidden_size) // 2
            hidden_size = (hidden_size // 64) * 64  # Ensure hidden_size is divisible by 64
            total_params = params_from_dimensions(hidden_size, num_layers)

            if total_params < num_params:
                lower_hidden_size = hidden_size + 64
            else:
                upper_hidden_size = hidden_size - 64

            if abs(total_params - num_params) / num_params < (1 - target_ratio):
                if total_params > best_num_params:
                    best_config = LlamaConfig(
                        hidden_size=hidden_size,
                        intermediate_size=4 * hidden_size,
                        num_attention_heads=max(1, hidden_size // 64),
                        num_hidden_layers=num_layers,
                        vocab_size=32000,
                    )
                    best_num_params = total_params

        if best_config is not None:
            break

    return best_config

import re

def parse_num_params(input_str):
    input_str = input_str.upper()
    if 'M' in input_str:
        num_params = float(input_str.replace('M', '')) * 1_000_000
    elif 'B' in input_str:
        num_params = float(input_str.replace('B', '')) * 1_000_000_000
    else:
        num_params = int(input_str)
    return int(num_params)

# User input for the desired number of parameters
input_str = input("Enter the desired number of parameters (e.g., 300M, 1.5B): ")
num_params = parse_num_params(input_str)
assert 300_000_000 <= num_params <= 200_000_000_000, "Number of parameters must be between 300M and 200B"

# Calculate the model dimensions based on the desired number of parameters
config = calculate_model_dimensions(num_params)
model = LlamaModel(config)

# Move the model to the appropriate device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Optimize matrix multiplications
print("Optimizing matrix multiplications...")
model = optimize_matrix_multiplications(model)

print(f"Hidden size: {config.hidden_size}")
print(f"Number of attention heads: {config.num_attention_heads}")

print("Saving ternarized and quantized model...")
save_dir = f"llama_{num_params}_ternary_quantized_optimized"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)

print("Ternarized and quantized model saved at:", save_dir)

# Example usage and testing
tensor = torch.randn(10, 20)
quantized_tensor = quantize_tensor(tensor)
packed_data = pack_quantized_tensor(quantized_tensor)
unpacked_tensor = unpack_quantized_tensor(packed_data, quantized_tensor.shape)

validate_quantization(tensor, unpacked_tensor)

# Performance testing
large_tensor = torch.randn(1000, 1000)
quantized_large_tensor = quantize_tensor(large_tensor)

start_time = time.time()
packed_large_data = pack_quantized_tensor(quantized_large_tensor)
end_time = time.time()
packing_time = end_time - start_time

start_time = time.time()
unpacked_large_tensor = unpack_quantized_tensor(packed_large_data, quantized_large_tensor.shape)
end_time = time.time()
unpacking_time = end_time - start_time

print(f"Packing time: {packing_time:.4f} seconds")
print(f"Unpacking time: {unpacking_time:.4f} seconds")
