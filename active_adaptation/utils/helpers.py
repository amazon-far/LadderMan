import torch
from functools import wraps


def batchify(func, broadcast=True):
    """Flatten the leading (batch) dims of every argument, call `func`, and restore them."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        batch_shapes = [arg.shape[:-1] for arg in args]
        if broadcast:
            batch_shape = torch.broadcast_shapes(*batch_shapes)
        else:
            batch_shape = set(batch_shapes)
            if len(batch_shape) != 1:
                raise ValueError()
            batch_shape = batch_shape.pop()
        args = [
            arg.expand(*batch_shape, arg.shape[-1]).reshape(-1, arg.shape[-1])
            for arg in args
        ]
        ret = func(*args, **kwargs)
        return ret.reshape(*batch_shape, *ret.shape[1:])
    return wrapped
