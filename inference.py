import torch
import torch.nn as nn
from transformers import LlamaConfig, AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.activations import ACT2FN
from llama_model import LlamaModel
from quantization_utils import activation_norm_quant, gemm_lowbit_kernel_mps
from safetensors.torch import safe_open, load
import time
import psutil
import json
import os

ACT2FN["llamamlp"] = lambda x: x * torch.sigmoid(x)

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.weight_scale = None

    def forward(self, x):
        w = self.weight  # a 1.58-bit weight tensor with shape [d, k]
        w_scale = self.weight_scale  # a full-precision weight scale tensor with shape [1]
        x_quant, x_scale = activation_norm_quant(x)
        y = gemm_lowbit_kernel_mps(x_quant, w) / w_scale / x_scale
        return y


def load_quantized_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = LlamaConfig.from_pretrained(model_path)

    # Correctly reading the .safetensors file in binary mode
    model_state_dict_path = os.path.join(model_path, "model.safetensors")
    with open(model_state_dict_path, "rb") as f:
        model_state_dict_bytes = f.read()

    # Load the state dictionary from the bytes
    model_state_dict = load(model_state_dict_bytes)

    # Remove the "model." prefix from the state dictionary keys
    model_state_dict = {k.replace("model.", ""): v for k, v in model_state_dict.items()}

    # Create a new model instance with the loaded config
    model = LlamaModel(config)

    # Load the state dictionary into the model
    model.load_state_dict(model_state_dict)

    # Quantize the model weights offline to 1.58 bits
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            module.weight_scale = module.weight.abs().mean().clamp_(min=1e-5)
            module.weight = ((module.weight * module.weight_scale).round().clamp_(-1, 1) / module.weight_scale).to(torch.int8)

    # Move the model to the appropriate device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Set the pad_token_id and eos_token_id in the tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.pad_token_id

    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.lm_head.weight.device)

    # Generate cos and sin values
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    position_embeddings = model.embed_positions(position_ids)
    cos = position_embeddings[:, :, 0::2].cos()
    sin = position_embeddings[:, :, 1::2].sin()

    generation_config = GenerationConfig(
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
    )

    start_time = time.time()
    output_ids = model.generate(input_ids, generation_config=generation_config, cos=cos, sin=sin)
    end_time = time.time()

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    generation_time = end_time - start_time
    num_tokens = len(output_ids[0])
    tokens_per_second = num_tokens / generation_time

    print(f"Generated {num_tokens} tokens in {generation_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")

    return generated_text

def evaluate_metrics(model, tokenizer, prompts, max_length=100):
    perplexities = []
    runtimes = []
    memory_usages = []

    generation_config = GenerationConfig(max_length=max_length, pad_token_id=tokenizer.pad_token_id)

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.lm_head.weight.device)
        attention_mask = torch.ones_like(input_ids)

        # Generate cos and sin values
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = model.embed_positions(position_ids)
        cos = position_embeddings[:, :, 0::2].cos()
        sin = position_embeddings[:, :, 1::2].sin()

        start_time = time.time()
        output_ids = model.generate(input_ids, attention_mask=attention_mask, generation_config=generation_config, cos=cos, sin=sin)
        end_time = time.time()

        runtime = end_time - start_time
        memory_usage = psutil.Process().memory_info().rss / 1024 ** 2  # Convert bytes to MB

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        perplexity = calculate_perplexity(model, tokenizer, generated_text)

        perplexities.append(perplexity)
        runtimes.append(runtime)
        memory_usages.append(memory_usage)

    avg_perplexity = sum(perplexities) / len(perplexities)
    avg_runtime = sum(runtimes) / len(runtimes)
    avg_memory_usage = sum(memory_usages) / len(memory_usages)

    print("Evaluation Metrics:")
    print(f"Average Perplexity: {avg_perplexity:.2f}")
    print(f"Average Runtime: {avg_runtime:.2f} seconds")
    print(f"Average Memory Usage: {avg_memory_usage:.2f} MB")

def calculate_perplexity(model, tokenizer, text):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.lm_head.weight.device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=torch.ones_like(input_ids))
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

model_path = input("Enter the path to your optimized model: ")

model, tokenizer = load_quantized_model(model_path)

prompts = [
    "The quick brown fox",
    "Artificial intelligence is",
    "In a shocking turn of events,",
]

evaluate_metrics(model, tokenizer, prompts)

while True:
    prompt = input("Enter a prompt (or 'quit' to exit): ")
    if prompt.lower() == "quit":
        break

    generated_text = generate_text(model, tokenizer, prompt)
    print("Generated text:", generated_text)
