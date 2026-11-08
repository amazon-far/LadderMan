import pickle
import types

import torch


class _MissingClass:
    """Placeholder for classes referenced by a checkpoint that no longer exist in this code base."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # e.g. config dataclasses of older algorithm versions pickled inside `cfg`
            if module.startswith("active_adaptation"):
                return type(name, (_MissingClass,), {"__module__": module})
            raise


_compat_pickle = types.ModuleType("_compat_pickle")
_compat_pickle.__dict__.update(pickle.__dict__)
_compat_pickle.Unpickler = _CompatUnpickler


def load_checkpoint(path: str, map_location="cpu") -> dict:
    """``torch.load`` that also works for checkpoints saved by older versions of this code."""
    return torch.load(path, map_location=map_location, weights_only=False, pickle_module=_compat_pickle)
