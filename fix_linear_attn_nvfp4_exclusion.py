#!/usr/bin/env python3
"""Fix NVFP4 quantization exclusion for Qwen3.5 Mamba-hybrid models.

Patches modelopt.py to exclude layers that should remain BF16, and
patches qwen3_5.py to handle any remaining size mismatches gracefully.
"""
import sys, os, glob


def remove_pyc():
    for pattern in [
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/__pycache__/modelopt*.pyc",
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/__pycache__/qwen3_5*.pyc",
    ]:
        for f in glob.glob(pattern):
            os.remove(f)
            print(f"Removed: {f}")


def patch_modelopt():
    target = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py"
    with open(target) as f:
        content = f.read()

    if "PATCH_V6" in content:
        print("modelopt.py: already patched v6")
        remove_pyc()
        return

    # Remove all previous patch versions
    for marker in ["PATCH_V5", "PATCH_V4", "QWEN35_LINEAR_ATTN_PATCH_V3", "QWEN35_LINEAR_ATTN_PATCH_V2"]:
        while marker in content:
            idx = content.index(marker)
            line_start = content.rfind("\n", 0, idx) + 1
            guard = "        if len(self.exclude_modules) == 0:\n            return False"
            guard_idx = content.index(guard, idx)
            content = content[:line_start] + content[guard_idx:]

    old = "        if len(self.exclude_modules) == 0:\n            return False"
    # Exclude: linear_attn (Mamba), mtp, shared_expert_gate, mlp.gate (MoE router),
    # visual (vision encoder is BF16 not quantized)
    # These are all stored as BF16 in Qwen3.5 NVFP4 checkpoints
    new = """\
        # PATCH_V6: Exclude BF16 layers in Qwen3.5 NVFP4 checkpoints.
        # The model's ignore list has these but HF-to-vLLM name mapping
        # fails to translate the patterns correctly.
        _bf16_markers = ["linear_attn", "shared_expert_gate", ".mlp.gate", "visual."]
        for _m in _bf16_markers:
            if _m in prefix:
                return True
        if prefix.startswith("mtp.") or prefix.startswith("visual."):
            return True

        if len(self.exclude_modules) == 0:
            return False"""
    if old not in content:
        print("ERROR: cannot find target in modelopt.py")
        sys.exit(1)
    content = content.replace(old, new, 1)
    with open(target, "w") as f:
        f.write(content)
    print("modelopt.py: patched v6 (linear_attn + gate + shared_expert_gate + mtp + visual)")
    remove_pyc()


def patch_qwen3_5():
    target = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py"
    with open(target) as f:
        content = f.read()

    if "LOAD_PATCH_V2" in content:
        print("qwen3_5.py: already patched v2")
        remove_pyc()
        return

    # Remove old v1 patch if present
    old_v1 = (
        '                    # LOAD_PATCH_V1: handle BF16/FP4 size mismatch for linear_attn\n'
        '                    if param.size() != loaded_weight.size() and "linear_attn" in name:\n'
        '                        import torch\n'
        '                        new_data = torch.empty(loaded_weight.size(), dtype=loaded_weight.dtype, device=param.device)\n'
        '                        new_data.copy_(loaded_weight)\n'
        '                        param.data = new_data\n'
        '                        loaded_params.add(name)\n'
        '                        continue\n'
    )
    if old_v1 in content:
        content = content.replace(old_v1, '')

    old = (
        '                    param = params_dict[name]\n'
        '                    weight_loader = getattr(\n'
        '                        param, "weight_loader", default_weight_loader\n'
        '                    )\n'
        '                    weight_loader(param, loaded_weight)'
    )
    new = (
        '                    param = params_dict[name]\n'
        '                    # LOAD_PATCH_V2: handle size mismatch for unquantized layers\n'
        '                    if param.size() != loaded_weight.size():\n'
        '                        import logging as _logging\n'
        '                        _log = _logging.getLogger(__name__)\n'
        '                        _log.warning(\n'
        '                            f"Size mismatch for {name}: param={param.size()} "\n'
        '                            f"loaded={loaded_weight.size()}, "\n'
        '                            f"re-materializing as unquantized"\n'
        '                        )\n'
        '                        import torch\n'
        '                        param.data = loaded_weight.to(\n'
        '                            dtype=param.dtype, device=param.device\n'
        '                        )\n'
        '                        loaded_params.add(name)\n'
        '                        continue\n'
        '                    weight_loader = getattr(\n'
        '                        param, "weight_loader", default_weight_loader\n'
        '                    )\n'
        '                    weight_loader(param, loaded_weight)'
    )

    if old not in content:
        print("ERROR: cannot find target in qwen3_5.py")
        sys.exit(1)
    content = content.replace(old, new, 1)
    with open(target, "w") as f:
        f.write(content)
    print("qwen3_5.py: patched load_weights v2")
    remove_pyc()


def patch_compressed_tensors():
    """Patch compressed_tensors.py to exclude visual layers from quantization."""
    target = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
    with open(target) as f:
        content = f.read()

    if "CT_VISUAL_EXCLUDE_V1" in content:
        print("compressed_tensors.py: already patched visual exclude v1")
        for f in glob.glob(
            "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/__pycache__/compressed_tensors*.pyc"
        ):
            os.remove(f)
            print(f"Removed: {f}")
        return

    # Find the should_ignore_layer call and add visual exclusion before it
    # Handle both debug-patched and original versions
    for old_pattern in [
        "        # IGNORE_DEBUG_V1\n",
        "        if should_ignore_layer(\n            layer_name, ignore=self.ignore, fused_mapping=self.packed_modules_mapping\n        ):\n            return None",
    ]:
        if old_pattern in content:
            break
    else:
        print("ERROR: cannot find target in compressed_tensors.py")
        sys.exit(1)

    # Replace with clean version that excludes visual/mtp/linear_attn
    if "IGNORE_DEBUG_V1" in content:
        # Remove entire debug block first
        start = content.index("        # IGNORE_DEBUG_V1")
        end = content.index("        if _should_ignore:\n            return None") + len("        if _should_ignore:\n            return None")
        content = content[:start] + "        # PLACEHOLDER_FOR_CT_PATCH\n" + content[end:]
        old_pattern = "        # PLACEHOLDER_FOR_CT_PATCH\n"

    new = (
        '        # CT_VISUAL_EXCLUDE_V1: Skip quantization for BF16 vision/linear_attn layers\n'
        '        _bf16_markers = ["visual.", "linear_attn", "shared_expert_gate", ".mlp.gate"]\n'
        '        if layer_name and any(_m in layer_name for _m in _bf16_markers):\n'
        '            return None\n'
        '        if layer_name and (layer_name.startswith("mtp.") or layer_name.startswith("visual.")):\n'
        '            return None\n'
        '        if should_ignore_layer(\n'
        '            layer_name, ignore=self.ignore, fused_mapping=self.packed_modules_mapping\n'
        '        ):\n'
        '            return None'
    )

    content = content.replace(old_pattern, new, 1)
    with open(target, "w") as f:
        f.write(content)
    print("compressed_tensors.py: patched visual exclude v1")
    for f in glob.glob(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/__pycache__/compressed_tensors*.pyc"
    ):
        os.remove(f)
        print(f"Removed: {f}")


def patch_qwen3_vl():
    """Patch qwen3_vl.py vision load_weights to handle missing/mismatched params."""
    target = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_vl.py"
    with open(target) as f:
        content = f.read()

    if "VISION_LOAD_PATCH_V1" in content:
        print("qwen3_vl.py: already patched v1")
        for f in glob.glob(
            "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/__pycache__/qwen3_vl*.pyc"
        ):
            os.remove(f)
            print(f"Removed: {f}")
        return

    # Patch the vision load_weights else branch to handle KeyError
    old = (
        '            else:\n'
        '                param = params_dict[name]\n'
        '                weight_loader = getattr(param, "weight_loader", default_weight_loader)\n'
        '                weight_loader(param, loaded_weight)'
    )
    new = (
        '            else:\n'
        '                # VISION_LOAD_PATCH_V1: handle missing/mismatched vision params\n'
        '                if name not in params_dict:\n'
        '                    continue\n'
        '                param = params_dict[name]\n'
        '                if param.size() != loaded_weight.size():\n'
        '                    import logging as _logging\n'
        '                    _log = _logging.getLogger(__name__)\n'
        '                    _log.warning(\n'
        '                        f"Vision size mismatch for {name}: param={param.size()} "\n'
        '                        f"loaded={loaded_weight.size()}, re-materializing"\n'
        '                    )\n'
        '                    import torch\n'
        '                    param.data = loaded_weight.to(dtype=param.dtype, device=param.device)\n'
        '                    loaded_params.add(name)\n'
        '                    continue\n'
        '                weight_loader = getattr(param, "weight_loader", default_weight_loader)\n'
        '                weight_loader(param, loaded_weight)'
    )

    if old not in content:
        print("ERROR: cannot find target in qwen3_vl.py")
        sys.exit(1)
    content = content.replace(old, new, 1)
    with open(target, "w") as f:
        f.write(content)
    print("qwen3_vl.py: patched vision load_weights v1")
    for f in glob.glob(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/__pycache__/qwen3_vl*.pyc"
    ):
        os.remove(f)
        print(f"Removed: {f}")


if __name__ == "__main__":
    patch_modelopt()
    patch_qwen3_5()
    patch_qwen3_vl()
    patch_compressed_tensors()
