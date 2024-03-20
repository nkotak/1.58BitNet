import os
import json
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaConfig
from safetensors.torch import load_file
from llama_model import LlamaModel
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def download_dataset(dataset_path):
    if dataset_path.startswith("https://huggingface.co/datasets/"):
        dataset_path = dataset_path.split("/")[-1]
    dataset = load_dataset(dataset_path)
    return dataset

def preprocess_dataset(file_path, file_format):
    """Preprocesses the dataset based on its format."""
    data = []
    if file_format == "txt":
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                data.append(line.strip())
    elif file_format == "json":
        with open(file_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)
            for item in json_data:
                data.append(item["text"])
    elif file_format == "jsonl":
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                data.append(json.loads(line)["text"])
    return data

def loss(model, inputs, targets, lengths):
    # Move lengths to the same device as inputs
    inputs = inputs.to(device)
    lengths = lengths.to(device)
    targets = targets.to(device)

    # Run model on inputs
    attention_mask = torch.arange(inputs.shape[1], device=device)[None, :] < lengths[:, None]
    attention_mask = attention_mask.unsqueeze(1).repeat(1, inputs.shape[1], 1)
    attention_mask = attention_mask.to(inputs.device)

    # Generate cos and sin embeddings
    seq_length = inputs.shape[1]
    position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs.device)
    cos = torch.zeros(seq_length, model.config.hidden_size // model.config.num_attention_heads, device=inputs.device)
    sin = torch.zeros(seq_length, model.config.hidden_size // model.config.num_attention_heads, device=inputs.device)
    div_term = torch.exp(torch.arange(0, model.config.hidden_size // model.config.num_attention_heads, 2, device=inputs.device) * (-torch.log(torch.tensor(10000.0)) / (model.config.hidden_size // model.config.num_attention_heads)))
    cos[:, 0::2] = torch.cos(position_ids[:, None] * div_term)
    sin[:, 1::2] = torch.sin(position_ids[:, None] * div_term)

    logits = model(inputs, attention_mask=attention_mask, cos=cos, sin=sin)
    logits = logits.view(-1, logits.size(-1))

    #print("Logits shape:", logits.shape)
    #print("Targets shape:", targets.shape)
    #print("Logits values:", logits)
    #print("Targets values:", targets)

    # Reshape logits to match the shape of targets
    logits = logits.view(targets.shape[0], targets.shape[1], -1)

     # Compute the loss
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    targets = targets.contiguous().view(-1)
    loss = loss_fn(logits.view(-1, logits.size(-1)), targets)

    toks = logits.size(0) * logits.size(1)
    return loss, torch.tensor(toks, device=device)

def iterate_batches(dset, tokenizer, batch_size, train=False, max_length=4096):
    # Shuffle indices
    while True:
        indices = np.arange(len(dset))
        if train:
            indices = np.random.permutation(indices)

        # Collect batches from dataset
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Encode batch
            batch = [tokenizer.encode(dset[indices[i + j]]) for j in range(batch_size)]
            lengths = [len(x) for x in batch]

            # Check if any sequence is longer than max_length tokens
            if max(lengths) > max_length:
                print(
                    f"[WARNING] Some sequences are longer than {max_length} tokens. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to the max length
            batch_arr = np.zeros((batch_size, max(lengths)), np.int32)
            for j in range(batch_size):
                batch_arr[j, : lengths[j]] = batch[j]
            batch = torch.tensor(batch_arr).to(device)
            lengths = torch.tensor(lengths).to(device)

            yield batch[:, :-1].to(device), batch[:, 1:].to(device), lengths.to(device)

        if not train:
            break

def evaluate(model, dataset, loss_fn, tokenizer, batch_size, num_batches, max_length, device):
    model.eval()
    all_losses = []
    ntokens = 0

    with torch.no_grad():
        for it, batch in zip(
            range(num_batches),
            iterate_batches(dataset, tokenizer, batch_size, max_length=max_length),
        ):
            batch = tuple(t.to(device) for t in batch)
            inputs, targets, lengths = batch
            attention_mask = torch.arange(inputs.shape[1], device=device)[None, :] < lengths[:, None]
            attention_mask = attention_mask.to(device)
            cos = torch.zeros(inputs.shape[1], model.config.hidden_size // model.config.num_attention_heads, device=device)
            sin = torch.zeros(inputs.shape[1], model.config.hidden_size // model.config.num_attention_heads, device=device)
            losses, toks = loss_fn(model, inputs, targets, lengths)
            all_losses.append((losses * toks).item())
            ntokens += toks.item()

    return np.sum(all_losses) / ntokens

def train(model, tokenizer, dataset, batch_size, num_epochs, learning_rate, iters, val_batches, steps_per_report, steps_per_eval, max_length, grad_accum_steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print("Learning rate:", optimizer.param_groups[0]['lr'])
    trainable_params = sum(v.numel() for _, v in model.named_parameters() if v.requires_grad) / 10**6
    total_params = sum(v.numel() for _, v in model.named_parameters()) / 10**6
    print(f"Total parameters: {total_params:.3f}M")
    print(f"Trainable parameters: {trainable_params:.3f}M")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    step = 0
    while step < iters:
        for batch_idx, batch in enumerate(tqdm(iterate_batches(dataset, tokenizer, batch_size, train=True, max_length=max_length), desc=f"Iteration {step // steps_per_eval + 1}")):
            batch = tuple(t.to(device) for t in batch)
            loss_value, ntoks = loss(model, *batch)
            loss_value = loss_value / grad_accum_steps
            loss_value.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % steps_per_report == 0:
                print(f"Step {step}: Loss = {loss_value.item():.4f}")

            if step % steps_per_eval == 0:
                cos = torch.zeros(max_length, model.config.hidden_size // model.config.num_attention_heads, device=device)
                sin = torch.zeros(max_length, model.config.hidden_size // model.config.num_attention_heads, device=device)
                val_loss = evaluate(model, dataset, loss, tokenizer, batch_size, val_batches, max_length, device)
                print(f"Validation Loss at Step {step}: {val_loss:.4f}")
                model.train()

            step += 1
            if step >= iters:
                break

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning script.")
    parser.add_argument("--dataset", type=str, help="Path to the dataset file.")
    parser.add_argument("--model_path", type=str, help="Path to the pre-trained model.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_model", help="Output directory to save the fine-tuned model.")
    parser.add_argument("--iters", type=int, default=1000, help="Steps to train for.")
    parser.add_argument("--val_batches", type=int, default=25, help="Number of validation batches, -1 uses the entire validation set.")
    parser.add_argument("--steps_per_report", type=int, default=10, help="Number of training steps between loss reporting.")
    parser.add_argument("--steps_per_eval", type=int, default=20, help="Number of training steps between validations.")
    parser.add_argument("--max_length", type=int, default=8192, help="Maximum sequence length for input tokens.")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Number of steps for gradient accumulation.")

    args = parser.parse_args()

    file_format = args.dataset.split(".")[-1]
    dataset = preprocess_dataset(args.dataset, file_format)

    model_path = input("Please enter the path to the pre-trained model you want to fine-tune: ")

    # Load the model configuration from the pre-trained model directory
    config = LlamaConfig.from_pretrained(model_path)
    print(f"Loaded hidden size: {config.hidden_size}")
    print(f"Loaded number of attention heads: {config.num_attention_heads}")

    # Create a new model instance with the loaded configuration
    model = LlamaModel(config)

    # Move the model to the appropriate device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = train(model, tokenizer, dataset, args.batch_size, args.num_epochs, args.learning_rate, args.iters, args.val_batches, args.steps_per_report, args.steps_per_eval, args.max_length, args.grad_accum_steps)
    model.save_pretrained(args.output_dir)
