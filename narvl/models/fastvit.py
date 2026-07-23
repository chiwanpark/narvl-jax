from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

from narvl.layers import MultiHeadAttention


@dataclass(frozen=True)
class FastViTConfig:
    image_size: int
    output_dim: int
    layers: tuple[int, ...]
    embed_dims: tuple[int, ...]
    mlp_ratios: tuple[float, ...]
    token_mixers: tuple[Literal["repmixer", "attention"], ...]
    downsamples: tuple[bool, ...]
    se_downsamples: tuple[bool, ...]
    pos_embs: tuple[bool, ...]
    norm_layer: Literal["batch_norm", "layer_norm"] = "batch_norm"
    stem_use_scale_branch: bool = True
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    layer_scale_init_value: float = 1e-5
    cls_ratio: float = 2.0

    def __post_init__(self) -> None:
        lengths = {
            len(self.layers),
            len(self.embed_dims),
            len(self.mlp_ratios),
            len(self.token_mixers),
            len(self.downsamples),
            len(self.se_downsamples),
            len(self.pos_embs),
        }
        if lengths != {len(self.layers)}:
            raise ValueError("All FastViT stage settings must have the same length")


def _make_divisible(value: float, divisor: int) -> int:
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


class _ConvNormAct(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        *,
        strides: int = 1,
        padding: str | tuple[tuple[int, int], tuple[int, int]] | None = None,
        groups: int = 1,
        use_bias: bool = False,
        use_norm: bool = True,
        use_act: bool = True,
        dtype: DTypeLike | None,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ) -> None:
        if padding is None:
            pad = kernel_size // 2
            padding = ((pad, pad), (pad, pad))
        self.conv = nnx.Conv(
            in_features,
            out_features,
            kernel_size=(kernel_size, kernel_size),
            strides=(strides, strides),
            padding=padding,
            feature_group_count=groups,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.norm = (
            nnx.BatchNorm(
                out_features,
                momentum=0.9,
                epsilon=1e-5,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            if use_norm
            else None
        )
        self.use_act = use_act

    def __call__(self, x: jax.Array, *, deterministic: bool) -> jax.Array:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x, use_running_average=deterministic)
        return jax.nn.gelu(x, approximate=False) if self.use_act else x


class _SqueezeExcite(nnx.Module):
    def __init__(
        self,
        channels: int,
        *,
        ratio: float,
        divisor: int,
        dtype: DTypeLike | None,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ) -> None:
        reduced_channels = _make_divisible(channels * ratio, divisor)
        self.fc1 = nnx.Conv(
            channels,
            reduced_channels,
            kernel_size=(1, 1),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.fc2 = nnx.Conv(
            reduced_channels,
            channels,
            kernel_size=(1, 1),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        scale = jnp.mean(x, axis=(1, 2), keepdims=True)
        scale = jax.nn.relu(self.fc1(scale))
        return x * jax.nn.sigmoid(self.fc2(scale))


class _MobileOneBlock(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        *,
        strides: int = 1,
        groups: int = 1,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
        dtype: DTypeLike | None,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ) -> None:
        self.identity = (
            nnx.BatchNorm(
                in_features,
                momentum=0.9,
                epsilon=1e-5,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            if in_features == out_features and strides == 1
            else None
        )
        self.branches = nnx.List(
            [
                _ConvNormAct(
                    in_features,
                    out_features,
                    kernel_size,
                    strides=strides,
                    groups=groups,
                    use_act=False,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                )
                for _ in range(num_conv_branches)
            ]
        )
        self.scale_branch = (
            _ConvNormAct(
                in_features,
                out_features,
                1,
                strides=strides,
                groups=groups,
                use_act=False,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            if kernel_size > 1 and use_scale_branch
            else None
        )
        self.se = (
            _SqueezeExcite(
                out_features,
                ratio=1 / 16,
                divisor=1,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            if use_se
            else None
        )
        self.use_act = use_act

    def __call__(self, x: jax.Array, *, deterministic: bool) -> jax.Array:
        outputs = [branch(x, deterministic=deterministic) for branch in self.branches]
        if self.scale_branch is not None:
            outputs.append(self.scale_branch(x, deterministic=deterministic))
        if self.identity is not None:
            outputs.append(self.identity(x, use_running_average=deterministic))
        if not outputs:
            raise RuntimeError("MobileOneBlock has no active branch")
        x = sum(outputs[1:], start=outputs[0])
        if self.se is not None:
            x = self.se(x)
        return jax.nn.gelu(x, approximate=False) if self.use_act else x


class _ReparamLargeKernelConv(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        kernel_size: int,
        strides: int,
        use_se: bool,
        use_act: bool,
        dtype: DTypeLike | None,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ) -> None:
        self.large_conv = _ConvNormAct(
            in_features,
            out_features,
            kernel_size,
            strides=strides,
            groups=in_features,
            use_act=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.small_conv = _ConvNormAct(
            in_features,
            out_features,
            3,
            strides=strides,
            groups=in_features,
            use_act=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.se = (
            _SqueezeExcite(
                out_features,
                ratio=0.25,
                divisor=8,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            if use_se
            else None
        )
        self.use_act = use_act

    def __call__(self, x: jax.Array, *, deterministic: bool) -> jax.Array:
        x = self.large_conv(x, deterministic=deterministic) + self.small_conv(x, deterministic=deterministic)
        if self.se is not None:
            x = self.se(x)
        return jax.nn.gelu(x, approximate=False) if self.use_act else x


class _PatchEmbed(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_se: bool,
        dtype: DTypeLike | None,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ) -> None:
        self.spatial = _ReparamLargeKernelConv(
            in_features,
            out_features,
            kernel_size=7,
            strides=2,
            use_se=use_se,
            use_act=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.pointwise = _MobileOneBlock(
            out_features,
            out_features,
            1,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, *, deterministic: bool) -> jax.Array:
        x = self.spatial(x, deterministic=deterministic)
        return self.pointwise(x, deterministic=deterministic)


class _DropPath(nnx.Module):
    def __init__(self, rate: float, ndim: int, *, rngs: nnx.Rngs) -> None:
        self.dropout = nnx.Dropout(rate, broadcast_dims=tuple(range(1, ndim)), rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool) -> jax.Array:
        return self.dropout(x, deterministic=deterministic)


class _ConvMlp(nnx.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        drop_rate: float,
        dtype: DTypeLike | None,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ) -> None:
        self.spatial = _ConvNormAct(
            dim,
            dim,
            7,
            groups=dim,
            use_act=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.fc1 = nnx.Conv(
            dim,
            hidden_dim,
            kernel_size=(1, 1),
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            rngs=rngs,
        )
        self.fc2 = nnx.Conv(
            hidden_dim,
            dim,
            kernel_size=(1, 1),
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(drop_rate, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool) -> jax.Array:
        x = self.spatial(x, deterministic=deterministic)
        x = jax.nn.gelu(self.fc1(x), approximate=False)
        x = self.dropout(x, deterministic=deterministic)
        x = self.fc2(x)
        return self.dropout(x, deterministic=deterministic)


class _RepMixer(nnx.Module):
    def __init__(
        self,
        dim: int,
        *,
        layer_scale_init_value: float,
        dtype: DTypeLike | None,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ) -> None:
        self.norm = _MobileOneBlock(
            dim,
            dim,
            3,
            groups=dim,
            use_act=False,
            use_scale_branch=False,
            num_conv_branches=0,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mixer = _MobileOneBlock(
            dim,
            dim,
            3,
            groups=dim,
            use_act=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layer_scale = nnx.Param(jnp.full((dim,), layer_scale_init_value, dtype=param_dtype))

    def __call__(self, x: jax.Array, *, deterministic: bool) -> jax.Array:
        mixed = self.mixer(x, deterministic=deterministic) - self.norm(x, deterministic=deterministic)
        return x + mixed * self.layer_scale[None, None, None, :]


class _RepMixerBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        *,
        mlp_ratio: float,
        drop_rate: float,
        drop_path_rate: float,
        layer_scale_init_value: float,
        dtype: DTypeLike | None,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ) -> None:
        self.token_mixer = _RepMixer(
            dim,
            layer_scale_init_value=layer_scale_init_value,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = _ConvMlp(
            dim,
            int(dim * mlp_ratio),
            drop_rate=drop_rate,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layer_scale = nnx.Param(jnp.full((dim,), layer_scale_init_value, dtype=param_dtype))
        self.drop_path = _DropPath(drop_path_rate, 4, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool) -> jax.Array:
        x = self.token_mixer(x, deterministic=deterministic)
        residual = self.mlp(x, deterministic=deterministic) * self.layer_scale[None, None, None, :]
        return x + self.drop_path(residual, deterministic=deterministic)


class _ChannelNorm(nnx.Module):
    def __init__(
        self,
        dim: int,
        *,
        dtype: DTypeLike | None,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ) -> None:
        self.norm = nnx.LayerNorm(
            dim,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, *, deterministic: bool = True) -> jax.Array:
        del deterministic
        return self.norm(x)


class _AttentionBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        *,
        mlp_ratio: float,
        norm_layer: Literal["batch_norm", "layer_norm"],
        drop_rate: float,
        drop_path_rate: float,
        layer_scale_init_value: float,
        dtype: DTypeLike | None,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ) -> None:
        self.norm = (
            nnx.BatchNorm(
                dim,
                momentum=0.9,
                epsilon=1e-5,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            if norm_layer == "batch_norm"
            else _ChannelNorm(dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        )
        self.attention = MultiHeadAttention(
            dim,
            dim // 32,
            qkv_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = _ConvMlp(
            dim,
            int(dim * mlp_ratio),
            drop_rate=drop_rate,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layer_scale_1 = nnx.Param(jnp.full((dim,), layer_scale_init_value, dtype=param_dtype))
        self.layer_scale_2 = nnx.Param(jnp.full((dim,), layer_scale_init_value, dtype=param_dtype))
        self.drop_path_1 = _DropPath(drop_path_rate, 4, rngs=rngs)
        self.drop_path_2 = _DropPath(drop_path_rate, 4, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool) -> jax.Array:
        if isinstance(self.norm, nnx.BatchNorm):
            normalized = self.norm(x, use_running_average=deterministic)
        else:
            normalized = self.norm(x)
        batch, height, width, channels = normalized.shape
        attended = self.attention(
            normalized.reshape(batch, height * width, channels),
            deterministic=deterministic,
        ).reshape(batch, height, width, channels)
        attended *= self.layer_scale_1[None, None, None, :]
        x = x + self.drop_path_1(attended, deterministic=deterministic)
        residual = self.mlp(x, deterministic=deterministic) * self.layer_scale_2[None, None, None, :]
        return x + self.drop_path_2(residual, deterministic=deterministic)


class _ConditionalPositionalEncoding(nnx.Module):
    def __init__(
        self,
        dim: int,
        *,
        dtype: DTypeLike | None,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ) -> None:
        self.conv = nnx.Conv(
            dim,
            dim,
            kernel_size=(7, 7),
            padding="SAME",
            feature_group_count=dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.conv(x)


class _FastViTStage(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        depth: int,
        token_mixer: Literal["repmixer", "attention"],
        mlp_ratio: float,
        downsample: bool,
        se_downsample: bool,
        pos_emb: bool,
        norm_layer: Literal["batch_norm", "layer_norm"],
        drop_rate: float,
        drop_path_rates: tuple[float, ...],
        layer_scale_init_value: float,
        dtype: DTypeLike | None,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ) -> None:
        self.downsample = (
            _PatchEmbed(
                in_dim,
                out_dim,
                use_se=se_downsample,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            if downsample
            else None
        )
        self.pos_emb = (
            _ConditionalPositionalEncoding(out_dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
            if pos_emb
            else None
        )
        block_type = _RepMixerBlock if token_mixer == "repmixer" else _AttentionBlock
        blocks: list[nnx.Module] = []
        for index in range(depth):
            common = {
                "mlp_ratio": mlp_ratio,
                "drop_rate": drop_rate,
                "drop_path_rate": drop_path_rates[index],
                "layer_scale_init_value": layer_scale_init_value,
                "dtype": dtype,
                "param_dtype": param_dtype,
                "rngs": rngs,
            }
            if block_type is _AttentionBlock:
                blocks.append(_AttentionBlock(out_dim, norm_layer=norm_layer, **common))
            else:
                blocks.append(_RepMixerBlock(out_dim, **common))
        self.blocks = nnx.List(blocks)

    def __call__(self, x: jax.Array, *, deterministic: bool) -> jax.Array:
        if self.downsample is not None:
            x = self.downsample(x, deterministic=deterministic)
        if self.pos_emb is not None:
            x = self.pos_emb(x)
        for block in self.blocks:
            x = block(x, deterministic=deterministic)
        return x


class FastViTEncoder(nnx.Module):
    def __init__(
        self,
        config: FastViTConfig,
        *,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        first_dim = config.embed_dims[0]
        self.stem = nnx.List(
            [
                _MobileOneBlock(
                    3,
                    first_dim,
                    3,
                    strides=2,
                    use_scale_branch=config.stem_use_scale_branch,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                ),
                _MobileOneBlock(
                    first_dim,
                    first_dim,
                    3,
                    strides=2,
                    groups=first_dim,
                    use_scale_branch=config.stem_use_scale_branch,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                ),
                _MobileOneBlock(
                    first_dim,
                    first_dim,
                    1,
                    use_scale_branch=config.stem_use_scale_branch,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                ),
            ]
        )
        total_blocks = sum(config.layers)
        denominator = max(total_blocks - 1, 1)
        rates = tuple(config.drop_path_rate * index / denominator for index in range(total_blocks))
        offset = 0
        previous_dim = first_dim
        stages: list[_FastViTStage] = []
        for index, depth in enumerate(config.layers):
            out_dim = config.embed_dims[index]
            downsample = config.downsamples[index] or previous_dim != out_dim
            stage_rates = rates[offset : offset + depth]
            stages.append(
                _FastViTStage(
                    previous_dim,
                    out_dim,
                    depth=depth,
                    token_mixer=config.token_mixers[index],
                    mlp_ratio=config.mlp_ratios[index],
                    downsample=downsample,
                    se_downsample=config.se_downsamples[index],
                    pos_emb=config.pos_embs[index],
                    norm_layer=config.norm_layer,
                    drop_rate=config.drop_rate,
                    drop_path_rates=stage_rates,
                    layer_scale_init_value=config.layer_scale_init_value,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                )
            )
            previous_dim = out_dim
            offset += depth
        self.stages = nnx.List(stages)
        final_dim = int(config.embed_dims[-1] * config.cls_ratio)
        self.final_conv = _MobileOneBlock(
            config.embed_dims[-1],
            final_dim,
            3,
            groups=config.embed_dims[-1],
            use_se=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.projection = nnx.Linear(
            final_dim,
            config.output_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(final_dim**-0.5),
            rngs=rngs,
        )

    def _features(self, images: jax.Array, *, deterministic: bool) -> jax.Array:
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Images must have shape [batch, height, width, 3], got {images.shape}")
        x = images
        for block in self.stem:
            x = block(x, deterministic=deterministic)
        for stage in self.stages:
            x = stage(x, deterministic=deterministic)
        return self.final_conv(x, deterministic=deterministic)

    def encode_tokens(self, images: jax.Array, *, deterministic: bool = True) -> jax.Array:
        x = self._features(images, deterministic=deterministic)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return self.projection(x)

    def __call__(self, images: jax.Array, *, deterministic: bool = True) -> jax.Array:
        x = self._features(images, deterministic=deterministic)
        return self.projection(jnp.mean(x, axis=(1, 2)))
