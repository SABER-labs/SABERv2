import math
from utils.config import config
import numpy as np
import torch
import torch.distributed as dist

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

def get_adam_warmup_cosine_schedule(train_iters_per_epoch):
    start_lr = config.trainer.start_lr
    final_lr = config.trainer.final_lr
    learning_rate = config.trainer.learning_rate
    warmup_epochs = config.trainer.warmup_epochs
    max_epochs = config.trainer.max_epochs

    warmup_lr_schedule = np.linspace(
        start_lr, learning_rate, int(train_iters_per_epoch * warmup_epochs)
    )

    iters = np.arange(int(train_iters_per_epoch * (max_epochs - warmup_epochs)))

    cosine_lr_schedule = np.array([
            final_lr + 0.5 * (learning_rate - final_lr) *
            (1 + math.cos(math.pi * t / (int(train_iters_per_epoch * (max_epochs - warmup_epochs)))))
            for t in iters
        ])

    return np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

def length_to_mask(length, stride=1, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    length = (length / stride).ceil_().to(dtype=length.dtype, device=length.device)
    max_len = math.ceil(max_len / stride) if max_len else length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask