import math
from utils.config import config
import numpy as np
import torch
import torch.distributed as dist

class GatherLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]

        dist.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, async_op=False)

        idx_from = dist.get_rank() * ctx.batch_size
        idx_to = (dist.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]

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

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, _ = x.shape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()