import math
import torch
import torch.nn as nn
from transformers import LlamaConfig, AutoTokenizer
from quantization_utils import quantize_tensor, pack_quantized_tensor, unpack_quantized_tensor
from safetensors.torch import save_file, load_file
import os
import json
from tqdm import tqdm
import numpy as np
import time
from custom_gradient_checkpointing import custom_checkpoint

class QuantizedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.eps = 1e-5

    def forward(self, input):
        quantized_weight = quantize_tensor(self.weight, self.eps)
        return nn.functional.embedding(input, quantized_weight)

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, num_groups=1):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.num_groups = num_groups
        self.eps = 1e-5
        self.quantized_weight = None

    def ternarize_weights_groupwise(self):
        if self.quantized_weight is None:
            group_size = self.weight.shape[0] // self.num_groups
            self.quantized_weight = torch.zeros_like(self.weight, dtype=torch.int8)

            for g in range(self.num_groups):
                start_idx = g * group_size
                end_idx = (g + 1) * group_size
                weight_group = self.weight[start_idx:end_idx]
                self.quantized_weight[start_idx:end_idx] = quantize_tensor(weight_group, self.eps)

        return self.quantized_weight

    def forward(self, input):
        quantized_weight = self.ternarize_weights_groupwise()
        dequantized_weight = quantized_weight.float() * self.weight.abs().mean()
        output = nn.functional.linear(input, dequantized_weight, self.bias)
        return output

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q: (batch_size, seq_length, hidden_size)
    # k: (batch_size, seq_length, hidden_size)
    # cos: (seq_length, head_dim)
    # sin: (seq_length, head_dim)

    batch_size, seq_length, hidden_size = q.shape
    head_dim = cos.shape[-1]

    q = q.view(batch_size, seq_length, -1, head_dim)
    k = k.view(batch_size, seq_length, -1, head_dim)

    cos = cos.unsqueeze(1).unsqueeze(0)
    sin = sin.unsqueeze(1).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.view(batch_size, seq_length, hidden_size), k_embed.view(batch_size, seq_length, hidden_size)

class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = BitLinear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = BitLinear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = BitLinear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = BitLinear(config.hidden_size, config.hidden_size, bias=False)
        self.pretraining_tp = config.pretraining_tp

    def forward(self, hidden_states, attention_mask, cos, sin):
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_probs, value_states)
        attention_output = self.o_proj(attention_output)

        return attention_output

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = BitLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = BitLinear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = BitLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.pretraining_tp = config.pretraining_tp

    def forward(self, hidden_states):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [
                    F.linear(hidden_states, gate_proj_slices[i])
                    for i in range(self.pretraining_tp)
                ],
                dim=-1,
            )
            up_proj = torch.cat(
                [
                    F.linear(hidden_states, up_proj_slices[i])
                    for i in range(self.pretraining_tp)
                ],
                dim=-1,
            )

            intermediate_states = (gate_proj * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            gate_proj = self.gate_proj(hidden_states)
            up_proj = self.up_proj(hidden_states)
            hidden_gelu = gate_proj * up_proj
            down_proj = self.down_proj(hidden_gelu)

        return down_proj

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=getattr(config, "layer_norm_eps", 1e-5))
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=getattr(config, "layer_norm_eps", 1e-5))
        self.mlp = LlamaMLP(config)

    def forward(self, hidden_states, attention_mask, cos, sin):
        residual = hidden_states

        def attn_forward(hidden_states, attention_mask, cos, sin):
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(hidden_states, attention_mask, cos, sin)
            return hidden_states

        hidden_states = custom_checkpoint(attn_forward, hidden_states, attention_mask, cos, sin)

        if hidden_states.shape != residual.shape:
            hidden_states = hidden_states.view(residual.shape)

        hidden_states = residual + hidden_states

        residual = hidden_states

        def mlp_forward(hidden_states):
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            return hidden_states

        hidden_states = custom_checkpoint(mlp_forward, hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = QuantizedEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=getattr(config, "layer_norm_eps", 1e-5))

        # Add missing bias terms
        for layer in self.layers:
            layer.input_layernorm.bias = nn.Parameter(torch.zeros(config.hidden_size))
            layer.post_attention_layernorm.bias = nn.Parameter(torch.zeros(config.hidden_size))

        # Add lm_head
        self.lm_head = BitLinear(config.hidden_size, config.vocab_size, bias=False)

        print("Embedding layer weight mean:", self.embed_tokens.weight.mean())
        print("Embedding layer weight std:", self.embed_tokens.weight.std())

        for i, layer in enumerate(self.layers):
            print(f"Layer {i} input layernorm weight mean:", layer.input_layernorm.weight.mean())
            print(f"Layer {i} input layernorm bias mean:", layer.input_layernorm.bias.mean())
            print(f"Layer {i} self-attention query projection weight mean:", layer.self_attn.q_proj.weight.mean())
            print(f"Layer {i} self-attention key projection weight mean:", layer.self_attn.k_proj.weight.mean())
            print(f"Layer {i} self-attention value projection weight mean:", layer.self_attn.v_proj.weight.mean())
            print(f"Layer {i} self-attention output projection weight mean:", layer.self_attn.o_proj.weight.mean())
            print(f"Layer {i} post-attention layernorm weight mean:", layer.post_attention_layernorm.weight.mean())
            print(f"Layer {i} post-attention layernorm bias mean:", layer.post_attention_layernorm.bias.mean())
            print(f"Layer {i} MLP gate projection weight mean:", layer.mlp.gate_proj.weight.mean())
            print(f"Layer {i} MLP down projection weight mean:", layer.mlp.down_proj.weight.mean())
            print(f"Layer {i} MLP up projection weight mean:", layer.mlp.up_proj.weight.mean())

        print("Output layer norm weight mean:", self.norm.weight.mean())
        print("Output layer norm bias mean:", self.norm.bias.mean())
        print("Language model head weight mean:", self.lm_head.weight.mean())

        # Move the model to the appropriate device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.to(device)

    @classmethod
    def load_pretrained(cls, model_path):
        # Load the model configuration
        config = LlamaConfig.from_pretrained(model_path)

        # Create a new model instance with the loaded configuration
        model = cls(config)

        # Load the state dict from the model.safetensors file
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))

        # Remove the 'model.' prefix from the state dict keys
        adjusted_state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}

        # Load the adjusted state dict into the model
        model.load_state_dict(adjusted_state_dict)

        return model

    def forward(self, input_ids, attention_mask, cos, sin):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, cos, sin)
        hidden_states = self.norm(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits

    def save_pretrained(self, save_directory):
        # Update the model configuration with the quantized model's parameters
        self.config.hidden_size = self.embed_tokens.embedding_dim
        self.config.num_attention_heads = self.config.hidden_size // self.layers[0].self_attn.head_dim
        self.config.num_hidden_layers = len(self.layers)
        self.config.intermediate_size = self.layers[0].mlp.gate_proj.out_features
        self.config.max_position_embeddings = self.embed_tokens.num_embeddings
        self.config.vocab_size = self.embed_tokens.num_embeddings

        if hasattr(self.config, "num_key_value_heads"):
            self.config.num_key_value_heads = self.config.num_attention_heads

        self.config.hidden_act = self.layers[0].mlp.__class__.__name__.lower()
        self.config.initializer_range = self.embed_tokens.weight.data.std().item()
        self.config.rms_norm_eps = self.layers[0].input_layernorm.variance_epsilon
        self.config.use_cache = True
        self.config.tie_word_embeddings = False
        self.config.model_type = self.__class__.__name__.lower()

        if hasattr(self.layers[0].self_attn, "attention_dropout"):
            self.config.attention_dropout = self.layers[0].self_attn.attention_dropout
        else:
            self.config.attention_dropout = 0.0

        # Find the hidden_dropout value from the model's layers
        hidden_dropout = None
        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, nn.Dropout):
                    hidden_dropout = module.p
                    break
            if hidden_dropout is not None:
                break

        if hidden_dropout is None:
            hidden_dropout = 0.0  # Set a default value if no dropout module is found

        self.config.hidden_dropout = hidden_dropout
        self.config.attention_bias = False

        if hasattr(self.config, "pretraining_tp"):
            self.config.pretraining_tp = self.config.pretraining_tp

        self.config.bos_token_id = self.config.bos_token_id if hasattr(self.config, "bos_token_id") else None
        self.config.eos_token_id = self.config.eos_token_id if hasattr(self.config, "eos_token_id") else None
        self.config.torch_dtype = str(self.embed_tokens.weight.dtype).split(".")[-1]
        self.config.transformers_version = "4.39.0.dev0"

        # Save the updated model configuration
        self.config.save_pretrained(save_directory)

        # Load the pre-trained LLaMA tokenizer
        tokenizer = AutoTokenizer.from_pretrained("DeepInfra/Llama-2-70b-chat-tokenizer")
        # Save the tokenizer files to the save directory
        tokenizer.save_pretrained(save_directory)

        # Copy additional tokenizer files if available
        additional_files = ["tokenizer.model", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]
        for file_name in additional_files:
            src_path = os.path.join("DeepInfra/Llama-2-70b-chat-tokenizer", file_name)
            dst_path = os.path.join(save_directory, file_name)
            if os.path.isfile(src_path):
                shutil.copyfile(src_path, dst_path)


        state_dict = self.state_dict()
        quantized_state_dict = {}
        total_size = 0
        weight_map = {}

        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                print(f"Before quantization: {name} - Size: {param.numel() * param.element_size()} bytes")

                # Use 'model.' prefix for compatibility
                adjusted_name = f'model.{name}'

                if "embed_tokens" in name:
                    eps = 1e-5  # Default eps value for embedding layer
                else:
                    eps = 1e-5  # Default eps value for other parameters
                quantized_param = quantize_tensor(param)
                packed_param = pack_quantized_tensor(quantized_param)
                quantized_state_dict[adjusted_name] = packed_param
                quantized_size = packed_param.numel() * packed_param.element_size()
                total_size += quantized_size
                print(f"After quantization: {adjusted_name} - Size: {quantized_size} bytes")
                weight_map["model." + name] = "model.safetensors"

        # Quantize activations
        # for name, module in self.named_modules():
        #    if isinstance(module, BitLinear):
        #        module.quantization_bits = quantization_bits

        index_data = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }

        with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
            json.dump(index_data, f, indent=2)

        save_file(quantized_state_dict, os.path.join(save_directory, "model.safetensors"))

    def save_sharded_safetensors(self, output_path, shard_size=9*1024*1024*1024):
        state_dict = self.state_dict()
        num_shards = math.ceil(sum(v.numel() * v.element_size() for v in state_dict.values()) / shard_size)

        os.makedirs(output_path, exist_ok=True)

        shard_id = 1
        shard_state_dict = {}
        shard_size_bytes = 0

        for key, value in state_dict.items():
            shard_state_dict[key] = value
            shard_size_bytes += value.numel() * value.element_size()

            if shard_size_bytes >= shard_size:
                shard_file = os.path.join(output_path, f"model-{shard_id:05d}-of-{num_shards:05d}.safetensors")
                save_file(shard_state_dict, shard_file)
                print(f"Saved shard {shard_id} at: {shard_file}")

                shard_id += 1
                shard_state_dict = {}
                shard_size_bytes = 0

        if shard_state_dict:
            shard_file = os.path.join(output_path, f"model-{shard_id:05d}-of-{num_shards:05d}.safetensors")
            save_file(shard_state_dict, shard_file)
            print(f"Saved shard {shard_id} at: {shard_file}")

    def create_additional_files(self, save_directory, model_path, state_dicts, num_shards):
        # Create model.safetensors.index.json
        weight_map = {}
        total_size = 0
        for shard_id, state_dict in enumerate(state_dicts, start=1):
            shard_file = f"model-{shard_id:05d}-of-{num_shards:05d}.safetensors"
            for key in state_dict.keys():
                weight_map[key] = shard_file
            total_size += sum(v.numel() * v.element_size() for v in state_dict.values())

        index_data = {
            "metadata": {
                "total_size": total_size
            },
            "weight_map": weight_map
        }
        with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
            json.dump(index_data, f, indent=4)

        # Create generation_config.json
        generation_config = {
            "max_length": 4096,
            "min_length": 0,
            "do_sample": False,
            "early_stopping": False,
            "num_beams": 1,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "num_return_sequences": 1,
            "attention_mask_column": 0,
        }
        with open(os.path.join(save_directory, "generation_config.json"), "w") as f:
            json.dump(generation_config, f, indent=4)
