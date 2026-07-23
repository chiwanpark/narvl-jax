from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from narvl.models.mobileclip2 import (
    MOBILECLIP2_IMAGE_CONFIGS,
    MobileCLIP2ImageEncoder,
    MobileCLIP2TextEncoder,
)


MOBILECLIP2_PRETRAINED_REPOS: Mapping[str, str] = {name: f"timm/{name}-OpenCLIP" for name in MOBILECLIP2_IMAGE_CONFIGS}


@dataclass(frozen=True)
class _TensorMapping:
    target: str
    transform: Literal["identity", "weight", "squeeze"] = "identity"


type _LoaderScope = Literal["image", "text"]
type _SourceScope = Literal["global", "image", "text"]


def _is_norm_weight(name: str) -> bool:
    parent = name.rsplit(".", 2)[-2]
    return parent in {"bn", "identity", "ln_1", "ln_2", "ln_final", "norm"}


def _map_text_tensor(name: str) -> _TensorMapping | None:
    direct = {
        "text.token_embedding.weight": "text.token_embedding.embedding",
        "text.positional_embedding": "text.positional_embedding",
        "text.text_projection": "text.projection.kernel",
        "text.ln_final.weight": "text.final_norm.scale",
        "text.ln_final.bias": "text.final_norm.bias",
    }
    if name in direct:
        return _TensorMapping(direct[name])
    prefix = "text.transformer.resblocks."
    if not name.startswith(prefix):
        return None
    target = "text.blocks." + name.removeprefix(prefix)
    target = target.replace(".attn.in_proj_", ".attention.qkv.")
    target = target.replace(".attn.out_proj.", ".attention.proj.")
    target = target.replace(".ln_1.", ".norm_1.")
    target = target.replace(".ln_2.", ".norm_2.")
    target = target.replace(".mlp.c_fc.", ".mlp.fc1.")
    target = target.replace(".mlp.c_proj.", ".mlp.fc2.")
    if target.endswith(".weight"):
        target = target.removesuffix(".weight") + (".scale" if _is_norm_weight(name) else ".kernel")
        return _TensorMapping(target, "identity" if _is_norm_weight(name) else "weight")
    return _TensorMapping(target)


def _map_fastvit_tensor(name: str) -> _TensorMapping | None:
    prefix = "visual.trunk."
    if not name.startswith(prefix):
        return None
    relative = name.removeprefix(prefix)
    if not relative.startswith(("stem.", "stages.", "final_conv.", "head.fc.")):
        return None
    target = relative.replace("head.fc.", "projection.")
    target = target.replace(".downsample.proj.0.", ".downsample.spatial.")
    target = target.replace(".downsample.proj.1.", ".downsample.pointwise.")
    target = target.replace(".pos_emb.pos_enc.", ".pos_emb.conv.")
    target = target.replace(".mlp.conv.conv.", ".mlp.spatial.conv.")
    target = target.replace(".mlp.conv.bn.", ".mlp.spatial.norm.")
    target = target.replace(".token_mixer.qkv.", ".attention.qkv.")
    target = target.replace(".token_mixer.proj.", ".attention.proj.")
    target = target.replace(".conv_kxk.", ".branches.")
    target = target.replace(".conv_scale.", ".scale_branch.")
    target = target.replace(".bn.", ".norm.")
    target = f"vision.{target}"
    if target.endswith(".gamma"):
        return _TensorMapping(target.removesuffix(".gamma"), "squeeze")
    if target.endswith(".running_mean"):
        return _TensorMapping(target.removesuffix(".running_mean") + ".mean")
    if target.endswith(".running_var"):
        return _TensorMapping(target.removesuffix(".running_var") + ".var")
    if target.endswith(".weight"):
        norm_weight = _is_norm_weight(name)
        target = target.removesuffix(".weight") + (".scale" if norm_weight else ".kernel")
        return _TensorMapping(target, "identity" if norm_weight else "weight")
    return _TensorMapping(target)


def _map_openclip_tensor(name: str) -> _TensorMapping | None:
    if name.endswith(".num_batches_tracked"):
        return None
    if name.startswith("text."):
        return _map_text_tensor(name)
    return _map_fastvit_tensor(name)


def _channel_norm_target(target: str) -> str:
    parts = target.split(".")
    if "blocks" not in parts:
        return target
    block_index = parts.index("blocks")
    norm_index = block_index + 2
    if len(parts) > norm_index and parts[norm_index] == "norm":
        parts.insert(norm_index + 1, "norm")
    return ".".join(parts)


def _converted_shape(shape: tuple[int, ...], transform: str) -> tuple[int, ...]:
    if transform == "identity":
        return shape
    if transform == "squeeze":
        if len(shape) < 2 or shape[-2:] != (1, 1):
            raise ValueError(f"Cannot squeeze a layer scale with shape {shape}")
        return shape[:-2]
    if len(shape) == 2:
        return shape[::-1]
    if len(shape) == 4:
        return shape[2], shape[3], shape[1], shape[0]
    raise ValueError(f"Cannot convert a PyTorch weight with shape {shape}")


def _convert_tensor(tensor: np.ndarray | jax.Array, transform: str) -> np.ndarray | jax.Array:
    if transform == "identity":
        return tensor
    if transform == "squeeze":
        return np.asarray(tensor).reshape(tensor.shape[:-2])
    if tensor.ndim == 2:
        return tensor.T
    if tensor.ndim == 4:
        return tensor.transpose(2, 3, 1, 0)
    raise ValueError(f"Cannot convert a PyTorch weight with shape {tensor.shape}")


def _target_variables(model: nnx.Module) -> dict[str, nnx.Variable[jax.Array]]:
    return {
        ".".join(str(part) for part in path): variable
        for path, variable in nnx.to_flat_state(nnx.state(model))
        if isinstance(variable, (nnx.Param, nnx.BatchStat))
    }


def _resolve_target(
    mapping: _TensorMapping,
    targets: Mapping[str, nnx.Variable[jax.Array]],
) -> _TensorMapping | None:
    if mapping.target in targets:
        return mapping
    target = _channel_norm_target(mapping.target)
    return _TensorMapping(target, mapping.transform) if target in targets else None


def _source_scope(name: str) -> _SourceScope | None:
    if name == "logit_scale":
        return "global"
    if name.startswith("visual."):
        return "image"
    if name.startswith("text."):
        return "text"
    return None


def _scoped_mapping(mapping: _TensorMapping, scope: _LoaderScope) -> _TensorMapping | None:
    prefix = "vision." if scope == "image" else "text."
    if not mapping.target.startswith(prefix):
        return None
    return _TensorMapping(mapping.target.removeprefix(prefix), mapping.transform)


def _plan_assignments(
    targets: Mapping[str, nnx.Variable[jax.Array]],
    keys: Iterable[str],
    get_shape: Callable[[str], tuple[int, ...]],
    scope: _LoaderScope,
) -> tuple[dict[str, tuple[str, _TensorMapping]], list[str], list[str]]:
    assignments: dict[str, tuple[str, _TensorMapping]] = {}
    unexpected: list[str] = []
    shape_errors: list[str] = []
    for source in keys:
        source_scope = _source_scope(source)
        if source_scope is not None and source_scope != scope:
            continue
        if source.endswith(".num_batches_tracked"):
            continue
        source_mapping = _map_openclip_tensor(source)
        scoped_mapping = _scoped_mapping(source_mapping, scope) if source_mapping is not None else None
        mapping = _resolve_target(scoped_mapping, targets) if scoped_mapping is not None else None
        if mapping is None:
            unexpected.append(source)
            continue
        if mapping.target in assignments:
            previous = assignments[mapping.target][0]
            raise ValueError(f"Both {previous!r} and {source!r} map to {mapping.target!r}")
        source_shape = tuple(get_shape(source))
        converted_shape = _converted_shape(source_shape, mapping.transform)
        target_shape = tuple(targets[mapping.target].shape)
        if converted_shape != target_shape:
            shape_errors.append(f"{source}: checkpoint {source_shape} -> model {target_shape}")
            continue
        assignments[mapping.target] = source, mapping
    return assignments, unexpected, shape_errors


def _validate_assignments(
    targets: Mapping[str, nnx.Variable[jax.Array]],
    assignments: Mapping[str, tuple[str, _TensorMapping]],
    unexpected: list[str],
    shape_errors: list[str],
    *,
    strict: bool,
) -> None:
    missing = sorted(set(targets) - set(assignments))
    if not shape_errors and (not strict or (not missing and not unexpected)):
        return
    details = []
    if shape_errors:
        details.append("shape mismatches: " + "; ".join(shape_errors[:10]))
    if strict and missing:
        details.append("missing tensors: " + ", ".join(missing[:10]))
    if strict and unexpected:
        details.append("unexpected tensors: " + ", ".join(sorted(unexpected)[:10]))
    raise ValueError("Invalid MobileCLIP2 checkpoint (" + "; ".join(details) + ")")


def _load_tensors(
    model: nnx.Module,
    keys: Iterable[str],
    get_shape: Callable[[str], tuple[int, ...]],
    get_tensor: Callable[[str], np.ndarray | jax.Array],
    *,
    scope: _LoaderScope,
    strict: bool,
) -> None:
    targets = _target_variables(model)
    assignments, unexpected, shape_errors = _plan_assignments(targets, keys, get_shape, scope)
    _validate_assignments(targets, assignments, unexpected, shape_errors, strict=strict)
    for target, (source, mapping) in assignments.items():
        variable = targets[target]
        tensor = _convert_tensor(get_tensor(source), mapping.transform)
        variable[...] = jnp.asarray(tensor, dtype=variable[...].dtype)


def _load_state_dict(
    model: nnx.Module,
    state_dict: Mapping[str, np.ndarray | jax.Array],
    *,
    scope: _LoaderScope,
    strict: bool,
) -> None:
    _load_tensors(
        model,
        state_dict,
        lambda name: tuple(state_dict[name].shape),
        state_dict.__getitem__,
        scope=scope,
        strict=strict,
    )


def _load_weights(
    model: nnx.Module,
    path: str | Path,
    *,
    scope: _LoaderScope,
    strict: bool,
) -> None:
    with safe_open(str(path), framework="numpy") as tensors:
        _load_tensors(
            model,
            tensors.keys(),
            lambda name: tuple(tensors.get_slice(name).get_shape()),
            tensors.get_tensor,
            scope=scope,
            strict=strict,
        )


def _download_weights(
    repo_id: str,
    *,
    filename: str,
    revision: str | None,
    cache_dir: str | Path | None,
    token: bool | str | None,
    local_files_only: bool,
) -> str:
    return hf_hub_download(
        repo_id,
        filename,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )


def load_mobileclip2_image_encoder_state_dict(
    encoder: MobileCLIP2ImageEncoder,
    state_dict: Mapping[str, np.ndarray | jax.Array],
    *,
    strict: bool = True,
) -> MobileCLIP2ImageEncoder:
    _load_state_dict(encoder, state_dict, scope="image", strict=strict)
    return encoder


def load_mobileclip2_text_encoder_state_dict(
    encoder: MobileCLIP2TextEncoder,
    state_dict: Mapping[str, np.ndarray | jax.Array],
    *,
    strict: bool = True,
) -> MobileCLIP2TextEncoder:
    _load_state_dict(encoder, state_dict, scope="text", strict=strict)
    return encoder


def load_mobileclip2_image_encoder_weights(
    encoder: MobileCLIP2ImageEncoder,
    path: str | Path,
    *,
    strict: bool = True,
) -> MobileCLIP2ImageEncoder:
    _load_weights(encoder, path, scope="image", strict=strict)
    return encoder


def load_mobileclip2_text_encoder_weights(
    encoder: MobileCLIP2TextEncoder,
    path: str | Path,
    *,
    strict: bool = True,
) -> MobileCLIP2TextEncoder:
    _load_weights(encoder, path, scope="text", strict=strict)
    return encoder


def load_mobileclip2_image_encoder_from_hf(
    encoder: MobileCLIP2ImageEncoder,
    repo_id: str,
    *,
    filename: str = "open_clip_model.safetensors",
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: bool | str | None = None,
    local_files_only: bool = False,
    strict: bool = True,
) -> MobileCLIP2ImageEncoder:
    path = _download_weights(
        repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    return load_mobileclip2_image_encoder_weights(encoder, path, strict=strict)


def load_mobileclip2_text_encoder_from_hf(
    encoder: MobileCLIP2TextEncoder,
    repo_id: str,
    *,
    filename: str = "open_clip_model.safetensors",
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: bool | str | None = None,
    local_files_only: bool = False,
    strict: bool = True,
) -> MobileCLIP2TextEncoder:
    path = _download_weights(
        repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    return load_mobileclip2_text_encoder_weights(encoder, path, strict=strict)
