import torch
import torch.nn as nn
import torch.autograd as autograd

class CustomCheckpoint(autograd.Function):
    @staticmethod
    def forward(ctx, run_function, *args):
        ctx.run_function = run_function
        with torch.no_grad():
            outputs = run_function(*args)
        ctx.save_for_backward(*args)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = ctx.saved_tensors
        with torch.enable_grad():
            outputs = ctx.run_function(*inputs)
            gradients = autograd.grad(outputs, inputs, grad_outputs, allow_unused=True)
        return (None,) + gradients

def custom_checkpoint(function, *args):
    return CustomCheckpoint.apply(function, *args)
