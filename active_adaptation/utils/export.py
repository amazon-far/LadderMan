import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase as ModBase


def _prepare_onnx_ir_for_version_convert(model, *, max_rounds: int = 64) -> None:
    """Clear leftover ONNX local functions before PyTorch's convert_version pass.

    PyTorch calls ``onnxscript.version_converter.convert_version``, whose pipeline runs
    ``InlinePass`` then immediately errors if ``model.functions`` is non-empty.
    Unused function definitions are only removed *after* version conversion, so models
    that retain orphan local functions hit:

        ValueError: The model contains functions. The version conversion pass does not support functions.

    Running ``RemoveUnusedFunctionsPass`` (with repeated inlining) beforehand matches
    the intent of the upstream pipeline and avoids that failure.
    """
    from onnxscript.ir.passes.common import InlinePass, RemoveUnusedFunctionsPass

    for _ in range(max_rounds):
        if not model.functions:
            return
        InlinePass()(model)
        RemoveUnusedFunctionsPass()(model)


@torch.inference_mode()
def export_onnx(module: ModBase, td: TensorDictBase, path: str, meta=None):
    if not path.endswith(".onnx"):
        raise ValueError(f"Export path must end with .onnx, got {path}.")

    td = td.cpu().select(*module.in_keys, strict=True)
    module = module.cpu()

    # Patch convert_version for the duration of dynamo_export so orphaned ONNX functions
    # are dropped before the strict ``model.functions`` check (see _prepare_onnx_ir_for_version_convert).
    from torch.onnx._internal._lazy_import import onnxscript_apis

    _orig_convert_version = onnxscript_apis.convert_version

    def _convert_version(*args, **kwargs):
        if args:
            _prepare_onnx_ir_for_version_convert(args[0])
        return _orig_convert_version(*args, **kwargs)

    try:
        onnxscript_apis.convert_version = _convert_version
        onnx_program = torch.onnx.dynamo_export(module, **td.to_dict())
    finally:
        onnxscript_apis.convert_version = _orig_convert_version

    onnx_program.save(path)
    print(f"Exported ONNX model to {path}.")

    import json

    meta_path = path.replace(".onnx", ".json")
    if meta is None:
        meta = {}
    meta["in_keys"] = module.in_keys
    meta["out_keys"] = module.out_keys
    meta["in_shapes"] = ([td[k].shape for k in module.in_keys],)

    json.dump(meta, open(meta_path, "w"), indent=4)
    print(f"Exported metadata to {meta_path}.")

    import onnxruntime as ort

    ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    onnx_input = tuple(td[k] for k in module.in_keys)
    onnxruntime_input = {
        k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)
    }

    ort_output = ort_session.run(None, onnxruntime_input)
    assert len(ort_output) == len(module.out_keys)
