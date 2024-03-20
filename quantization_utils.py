import torch
import numpy as np

def quantize_tensor(x: torch.Tensor, eps: float = 1e-5):
    gamma = x.abs().mean()
    quantized_x = torch.clamp(torch.round(x / (gamma + eps)), -1, 1).to(torch.int8)
    return quantized_x

def pack_quantized_tensor(quantized_tensor: torch.Tensor):
    padded_length = (quantized_tensor.numel() + 4) // 5 * 5
    padded_tensor = torch.full((padded_length,), -1, dtype=torch.int8)
    padded_tensor[:quantized_tensor.numel()] = quantized_tensor.reshape(-1)
    reshaped_tensor = padded_tensor.view(-1, 5)

    unsigned_tensor = reshaped_tensor + 1
    packed_data = torch.zeros(reshaped_tensor.size(0), dtype=torch.uint8)
    shifts = torch.arange(0, 10, 2, dtype=torch.uint8)
    for i in range(5):
        packed_data |= unsigned_tensor[:, i] << shifts[i]

    return packed_data

def unpack_quantized_tensor(packed_data: torch.Tensor, original_shape):
    unpacked_data = torch.zeros(packed_data.size(0), 5, dtype=torch.int8)
    shifts = torch.arange(0, 10, 2, dtype=torch.uint8)
    for i in range(5):
        unpacked_data[:, i] = (packed_data >> shifts[i]) & 3

    ternary_tensor = unpacked_data - 1
    original_numel = np.prod(original_shape)
    return ternary_tensor.view(-1)[:original_numel].view(original_shape)
