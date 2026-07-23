from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

from narvl.layers import FeedForward, MultiHeadAttention
from narvl.models.fastvit import FastViTConfig, FastViTEncoder


@dataclass(frozen=True)
class TextConfig:
    output_dim: int
    width: int
    heads: int
    layers: int = 12
    context_length: int = 77
    vocab_size: int = 49_408
    mlp_ratio: float = 4.0
    causal: bool = False
    drop_rate: float = 0.0


type MobileCLIP2ImageConfig = FastViTConfig
type MobileCLIP2TextConfig = TextConfig


def _fastvit_config(
    output_dim: int,
    layers: tuple[int, ...],
    embed_dims: tuple[int, ...],
    mlp_ratios: tuple[float, ...],
    token_mixers: tuple[Literal["repmixer", "attention"], ...],
    *,
    se_downsamples: tuple[bool, ...],
    pos_embs: tuple[bool, ...],
    norm_layer: Literal["batch_norm", "layer_norm"] = "batch_norm",
    stem_use_scale_branch: bool = True,
) -> MobileCLIP2ImageConfig:
    return FastViTConfig(
        image_size=256,
        output_dim=output_dim,
        layers=layers,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        token_mixers=token_mixers,
        downsamples=(False,) + (True,) * (len(layers) - 1),
        se_downsamples=se_downsamples,
        pos_embs=pos_embs,
        norm_layer=norm_layer,
        stem_use_scale_branch=stem_use_scale_branch,
    )


def _mobileclip2_configs() -> tuple[dict[str, MobileCLIP2ImageConfig], dict[str, MobileCLIP2TextConfig]]:
    s0_vision = _fastvit_config(
        512,
        (2, 6, 10, 2),
        (64, 128, 256, 512),
        (3, 3, 3, 3),
        ("repmixer", "repmixer", "repmixer", "attention"),
        se_downsamples=(False, False, True, True),
        pos_embs=(False, False, False, True),
    )
    s2_vision = _fastvit_config(
        512,
        (4, 12, 24, 4),
        (80, 160, 320, 640),
        (3, 3, 3, 3),
        ("repmixer", "repmixer", "repmixer", "attention"),
        se_downsamples=(False, False, True, True),
        pos_embs=(False, False, False, True),
    )
    s3_vision = _fastvit_config(
        768,
        (2, 12, 24, 4, 2),
        (96, 192, 384, 768, 1536),
        (4, 4, 4, 4, 4),
        ("repmixer", "repmixer", "repmixer", "attention", "attention"),
        se_downsamples=(False, False, False, False, False),
        pos_embs=(False, False, False, True, True),
        norm_layer="layer_norm",
        stem_use_scale_branch=False,
    )
    s4_vision = _fastvit_config(
        768,
        (2, 12, 24, 4, 4),
        (128, 256, 512, 1024, 2048),
        (4, 4, 4, 4, 4),
        ("repmixer", "repmixer", "repmixer", "attention", "attention"),
        se_downsamples=(False, False, False, False, False),
        pos_embs=(False, False, False, True, True),
        norm_layer="layer_norm",
        stem_use_scale_branch=False,
    )

    def text(output_dim: int) -> MobileCLIP2TextConfig:
        heads = 8 if output_dim == 512 else 12
        return TextConfig(output_dim=output_dim, width=output_dim, heads=heads)

    image_configs: dict[str, MobileCLIP2ImageConfig] = {
        "MobileCLIP2-S0": s0_vision,
        "MobileCLIP2-S2": s2_vision,
        "MobileCLIP2-S3": s3_vision,
        "MobileCLIP2-S4": s4_vision,
    }
    text_configs = {
        "MobileCLIP2-S0": text(512),
        "MobileCLIP2-S2": text(512),
        "MobileCLIP2-S3": text(768),
        "MobileCLIP2-S4": text(768),
    }
    return image_configs, text_configs


_IMAGE_CONFIGS, _TEXT_CONFIGS = _mobileclip2_configs()
MOBILECLIP2_IMAGE_CONFIGS: Mapping[str, MobileCLIP2ImageConfig] = MappingProxyType(_IMAGE_CONFIGS)
MOBILECLIP2_TEXT_CONFIGS: Mapping[str, MobileCLIP2TextConfig] = MappingProxyType(_TEXT_CONFIGS)


def _get_mobileclip2_config[T](name: str, configs: Mapping[str, T]) -> T:
    normalized = name.lower().replace("_", "-")
    if not normalized.startswith("mobileclip2-"):
        normalized = f"mobileclip2-{normalized}"
    for model_name, config in configs.items():
        if model_name.lower() == normalized:
            return config
    options = ", ".join(configs)
    raise ValueError(f"Unknown MobileCLIP2 variant {name!r}. Available variants: {options}")


def get_mobileclip2_image_config(name: str) -> MobileCLIP2ImageConfig:
    return _get_mobileclip2_config(name, MOBILECLIP2_IMAGE_CONFIGS)


def get_mobileclip2_text_config(name: str) -> MobileCLIP2TextConfig:
    return _get_mobileclip2_config(name, MOBILECLIP2_TEXT_CONFIGS)


class _TextTransformerBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        mlp_ratio: float,
        drop_rate: float,
        dtype: DTypeLike | None,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ) -> None:
        self.norm_1 = nnx.LayerNorm(
            dim,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.attention = MultiHeadAttention(
            dim,
            heads,
            qkv_bias=True,
            attn_drop=drop_rate,
            proj_drop=drop_rate,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.norm_2 = nnx.LayerNorm(
            dim,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = FeedForward(
            dim,
            int(dim * mlp_ratio),
            hidden_drop=drop_rate,
            output_drop=drop_rate,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        mask: jax.Array | None = None,
        deterministic: bool,
    ) -> jax.Array:
        x += self.attention(self.norm_1(x), mask=mask, deterministic=deterministic)
        return x + self.mlp(self.norm_2(x), deterministic=deterministic)


class TextTransformer(nnx.Module):
    def __init__(
        self,
        config: MobileCLIP2TextConfig,
        *,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.token_embedding = nnx.Embed(
            config.vocab_size,
            config.width,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(0.02),
            rngs=rngs,
        )
        self.positional_embedding = nnx.Param(
            jax.nn.initializers.normal(0.01)(rngs.params(), (config.context_length, config.width), param_dtype)
        )
        self.blocks = nnx.List(
            [
                _TextTransformerBlock(
                    config.width,
                    config.heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_rate=config.drop_rate,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                )
                for _ in range(config.layers)
            ]
        )
        self.final_norm = nnx.LayerNorm(
            config.width,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.projection = nnx.Linear(
            config.width,
            config.output_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.width**-0.5),
            rngs=rngs,
        )

    def __call__(
        self,
        tokens: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        deterministic: bool = True,
        return_all_tokens: bool = False,
    ) -> jax.Array:
        if tokens.ndim != 2:
            raise ValueError(f"Text tokens must have shape [batch, length], got {tokens.shape}")
        length = tokens.shape[1]
        if length > self.config.context_length:
            raise ValueError(f"Text length {length} exceeds context length {self.config.context_length}")
        x = self.token_embedding(tokens) + self.positional_embedding[:length]
        mask = None
        if self.config.causal:
            mask = jnp.tril(jnp.ones((length, length), dtype=jnp.bool_))
        if attention_mask is not None:
            if attention_mask.shape != tokens.shape:
                raise ValueError(f"Attention mask must have shape {tokens.shape}, got {attention_mask.shape}")
            key_mask = attention_mask.astype(jnp.bool_)[:, None, :]
            mask = key_mask if mask is None else mask[None, :, :] & key_mask
        for block in self.blocks:
            x = block(x, mask=mask, deterministic=deterministic)
        x = self.final_norm(x)
        if return_all_tokens:
            return x
        pooled = x[jnp.arange(tokens.shape[0]), jnp.argmax(tokens, axis=-1)]
        return self.projection(pooled)


type MobileCLIP2ImageEncoder = FastViTEncoder
type MobileCLIP2TextEncoder = TextTransformer


def create_mobileclip2_image_encoder(
    name: str = "MobileCLIP2-S0",
    *,
    pretrained: bool | str = False,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: bool | str | None = None,
    local_files_only: bool = False,
    dtype: DTypeLike | None = None,
    param_dtype: DTypeLike = jnp.float32,
    rngs: nnx.Rngs,
) -> MobileCLIP2ImageEncoder:
    config = get_mobileclip2_image_config(name)
    encoder = FastViTEncoder(
        config,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    if pretrained:
        from narvl.models.io.mobileclip2 import (
            MOBILECLIP2_PRETRAINED_REPOS,
            load_mobileclip2_image_encoder_from_hf,
        )

        canonical_name = next(
            model_name for model_name, preset in MOBILECLIP2_IMAGE_CONFIGS.items() if preset is config
        )
        repo_id = pretrained if isinstance(pretrained, str) else MOBILECLIP2_PRETRAINED_REPOS[canonical_name]
        load_mobileclip2_image_encoder_from_hf(
            encoder,
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    return encoder


def create_mobileclip2_text_encoder(
    name: str = "MobileCLIP2-S0",
    *,
    pretrained: bool | str = False,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: bool | str | None = None,
    local_files_only: bool = False,
    dtype: DTypeLike | None = None,
    param_dtype: DTypeLike = jnp.float32,
    rngs: nnx.Rngs,
) -> MobileCLIP2TextEncoder:
    config = get_mobileclip2_text_config(name)
    encoder = TextTransformer(config, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
    if pretrained:
        from narvl.models.io.mobileclip2 import (
            MOBILECLIP2_PRETRAINED_REPOS,
            load_mobileclip2_text_encoder_from_hf,
        )

        canonical_name = next(model_name for model_name, preset in MOBILECLIP2_TEXT_CONFIGS.items() if preset is config)
        repo_id = pretrained if isinstance(pretrained, str) else MOBILECLIP2_PRETRAINED_REPOS[canonical_name]
        load_mobileclip2_text_encoder_from_hf(
            encoder,
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    return encoder
