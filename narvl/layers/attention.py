import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import Initializer
from jax.typing import DTypeLike


_DEFAULT_KERNEL_INIT = jax.nn.initializers.lecun_normal()


class MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        head_dim: int | None = None,
        fused_qkv: bool = True,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        softmax_dtype: DTypeLike | None = None,
        qkv_kernel_init: Initializer = _DEFAULT_KERNEL_INIT,
        query_kernel_init: Initializer = _DEFAULT_KERNEL_INIT,
        key_kernel_init: Initializer = _DEFAULT_KERNEL_INIT,
        value_kernel_init: Initializer = _DEFAULT_KERNEL_INIT,
        proj_kernel_init: Initializer = _DEFAULT_KERNEL_INIT,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {dim}")
        if heads <= 0:
            raise ValueError(f"Head count must be positive, got {heads}")
        if head_dim is None:
            if dim % heads:
                raise ValueError(f"Embedding dimension {dim} must be divisible by {heads} heads")
            head_dim = dim // heads
        if head_dim <= 0:
            raise ValueError(f"Head dimension must be positive, got {head_dim}")

        inner_dim = heads * head_dim
        if fused_qkv and inner_dim != dim:
            raise ValueError("Fused QKV projections require heads * head_dim to equal the embedding dimension")

        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = inner_dim
        self.scale = head_dim**-0.5 if scale is None else scale
        self.softmax_dtype = softmax_dtype
        self.qkv = (
            nnx.Linear(
                dim,
                inner_dim * 3,
                use_bias=qkv_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=qkv_kernel_init,
                rngs=rngs,
            )
            if fused_qkv
            else None
        )
        self.query = (
            None
            if fused_qkv
            else nnx.Linear(
                dim,
                inner_dim,
                use_bias=qkv_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=query_kernel_init,
                rngs=rngs,
            )
        )
        self.key = (
            None
            if fused_qkv
            else nnx.Linear(
                dim,
                inner_dim,
                use_bias=qkv_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=key_kernel_init,
                rngs=rngs,
            )
        )
        self.value = (
            None
            if fused_qkv
            else nnx.Linear(
                dim,
                inner_dim,
                use_bias=qkv_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=value_kernel_init,
                rngs=rngs,
            )
        )
        self.proj = nnx.Linear(
            inner_dim,
            dim,
            use_bias=proj_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=proj_kernel_init,
            rngs=rngs,
        )
        self.attn_drop = nnx.Dropout(attn_drop, rngs=rngs)
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)

    @staticmethod
    def _expand_attention_tensor(x: jax.Array, name: str) -> jax.Array:
        if x.ndim == 2:
            return x[None, None, :, :]
        if x.ndim == 3:
            return x[:, None, :, :]
        if x.ndim == 4:
            return x
        raise ValueError(f"{name} must have 2, 3, or 4 dimensions, got shape {x.shape}")

    def __call__(
        self,
        x: jax.Array,
        *,
        context: jax.Array | None = None,
        mask: jax.Array | None = None,
        attention_bias: jax.Array | None = None,
        deterministic: bool,
    ) -> jax.Array:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"Attention input must have shape [batch, length, {self.dim}], got {x.shape}")

        batch, query_length, _ = x.shape
        if self.qkv is not None:
            if context is not None:
                raise ValueError("Fused QKV projections do not support a separate attention context")
            qkv = self.qkv(x).reshape(batch, query_length, 3, self.heads, self.head_dim)
            qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
            query, key, value = qkv[0], qkv[1], qkv[2]
            key_length = query_length
        else:
            context = x if context is None else context
            if context.ndim != 3 or context.shape[0] != batch or context.shape[-1] != self.dim:
                raise ValueError(
                    f"Attention context must have shape [batch, length, {self.dim}] with batch {batch}, "
                    f"got {context.shape}"
                )
            if self.query is None or self.key is None or self.value is None:
                raise RuntimeError("Separate attention projections are incomplete")
            key_length = context.shape[1]
            query = self.query(x).reshape(batch, query_length, self.heads, self.head_dim)
            key = self.key(context).reshape(batch, key_length, self.heads, self.head_dim)
            value = self.value(context).reshape(batch, key_length, self.heads, self.head_dim)
            query = jnp.transpose(query, (0, 2, 1, 3))
            key = jnp.transpose(key, (0, 2, 1, 3))
            value = jnp.transpose(value, (0, 2, 1, 3))

        scores = jnp.matmul(query * self.scale, jnp.swapaxes(key, -1, -2))
        if attention_bias is not None:
            bias = self._expand_attention_tensor(attention_bias, "Attention bias").astype(scores.dtype)
            scores += bias
        if mask is not None:
            mask = self._expand_attention_tensor(mask, "Attention mask").astype(jnp.bool_)
            scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)

        softmax_scores = scores.astype(self.softmax_dtype) if self.softmax_dtype is not None else scores
        weights = jax.nn.softmax(softmax_scores, axis=-1).astype(value.dtype)
        weights = self.attn_drop(weights, deterministic=deterministic)
        x = jnp.matmul(weights, value)
        x = jnp.transpose(x, (0, 2, 1, 3)).reshape(batch, query_length, self.inner_dim)
        x = self.proj(x)
        return self.proj_drop(x, deterministic=deterministic)
