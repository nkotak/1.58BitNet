import torch
import numpy as np

def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

def activation_norm_quant(x):
    x = RMSNorm(x)
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127)
    return y, scale

def act_quant_8bit(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127)
    return y

def act_quant_4bit(x):
    scale = 7.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-8, 7)
    return y

def gemm_lowbit_kernel_mps(x, w):
    # Ensure the tensors are on the MPS device
    x = x.to(device='mps')
    w = w.to(device='mps')

    # Extract the dimensions
    batch_size, input_size = x.shape
    output_size, _ = w.shape

    # Create a custom Metal kernel for low-bit matrix multiplication
    kernel = '''
        #include <metal_stdlib>
        using namespace metal;

        kernel void gemm_lowbit(device char* x_ptr   [[buffer(0)]],
                                device char* w_ptr   [[buffer(1)]],
                                device float* out_ptr [[buffer(2)]],
                                uint batch_size      [[buffer(3)]],
                                uint input_size      [[buffer(4)]],
                                uint output_size     [[buffer(5)]]) {
            uint idx = threadgroup_position_in_grid.x;
            if (idx >= batch_size * output_size) {
                return;
            }

            uint row = idx / output_size;
            uint col = idx % output_size;

            int32_t acc = 0;
            for (uint i = 0; i < input_size; ++i) {
                int8_t x_val = x_ptr[row * input_size + i];
                int8_t w_val = w_ptr[col * input_size + i];
                acc += static_cast<int32_t>(x_val) * static_cast<int32_t>(w_val);
            }

            out_ptr[idx] = static_cast<float>(acc);
        }
    '''

    # Compile the Metal kernel
    device = torch.device('mps')
    kernel_func = torch.jit.CompilationUnit().create_kernel(kernel, 'gemm_lowbit')

    # Allocate output tensor on the MPS device
    out = torch.empty(batch_size, output_size, dtype=torch.float32, device=device)

    # Convert input tensors to int8
    x_int8 = x.to(dtype=torch.int8)
    w_int8 = w.to(dtype=torch.int8)

    # Launch the Metal kernel
    kernel_func(
        x_int8,
        w_int8,
        out,
        torch.tensor([batch_size], dtype=torch.int32, device=device),
        torch.tensor([input_size], dtype=torch.int32, device=device),
        torch.tensor([output_size], dtype=torch.int32, device=device),
        threads=batch_size * output_size,
    )

    return out

def quantize_tensor(x: torch.Tensor, eps: float = 1e-5):
    gamma = x.abs().mean()
    quantized_x = torch.clamp(torch.round(x / (gamma + eps)), -1, 1).to(torch.int8)
    return quantized_x

def quantize_tensor_1_58bit(x: torch.Tensor, eps: float = 1e-5):
    gamma = x.abs().mean()
    quantized_x = torch.clamp(torch.round(x / (gamma + eps)), -1, 1).to(torch.int8)
    return quantized_x

def kv_cache_quant(x):
    scale = 15.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-16, 15)
    return y

#def pack_quantized_tensor(quantized_tensor: torch.Tensor):
#    padded_length = (quantized_tensor.numel() + 4) // 5 * 5
#    padded_tensor = torch.full((padded_length,), -1, dtype=torch.int8)
#    padded_tensor[:quantized_tensor.numel()] = quantized_tensor.reshape(-1)
#    reshaped_tensor = padded_tensor.view(-1, 5)
#
#    unsigned_tensor = reshaped_tensor + 1
#    packed_data = torch.zeros(reshaped_tensor.size(0), dtype=torch.uint8)
#    shifts = torch.arange(0, 10, 2, dtype=torch.uint8)
#    for i in range(5):
#        packed_data |= unsigned_tensor[:, i] << shifts[i]
#
#    return packed_data

#def unpack_quantized_tensor(packed_data: torch.Tensor, original_shape):
#    unpacked_data = torch.zeros(packed_data.size(0), 5, dtype=torch.int8)
#    shifts = torch.arange(0, 10, 2, dtype=torch.uint8)
#    for i in range(5):
#        unpacked_data[:, i] = (packed_data >> shifts[i]) & 3
#
#    ternary_tensor = unpacked_data - 1
#    original_numel = np.prod(original_shape)
#    return ternary_tensor.view(-1)[:original_numel].view(original_shape)
