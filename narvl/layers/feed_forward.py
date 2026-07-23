from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import Initializer
from jax.typing import DTypeLike


_DEFAULT_KERNEL_INIT = jax.nn.initializers.lecun_normal()


def _gelu(x: jax.Array) -> jax.Array:
    return jax.nn.gelu(x, approximate=False)


class FeedForward(nnx.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        activation: Callable[[jax.Array], jax.Array] = _gelu,
        gated: bool = False,
        use_bias: bool = True,
        hidden_drop: float = 0.0,
        output_drop: float = 0.0,
        input_kernel_init: Initializer = _DEFAULT_KERNEL_INIT,
        gate_kernel_init: Initializer = _DEFAULT_KERNEL_INIT,
        output_kernel_init: Initializer = _DEFAULT_KERNEL_INIT,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError(f"Feed-forward dimensions must be positive, got {dim} and {hidden_dim}")

        self.activation = activation
        self.fc1 = nnx.Linear(
            dim,
            hidden_dim,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=input_kernel_init,
            rngs=rngs,
        )
        self.gate = (
            nnx.Linear(
                dim,
                hidden_dim,
                use_bias=use_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=gate_kernel_init,
                rngs=rngs,
            )
            if gated
            else None
        )
        self.fc2 = nnx.Linear(
            hidden_dim,
            dim,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=output_kernel_init,
            rngs=rngs,
        )
        self.hidden_dropout = nnx.Dropout(hidden_drop, rngs=rngs)
        self.output_dropout = nnx.Dropout(output_drop, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool) -> jax.Array:
        hidden = self.activation(self.fc1(x))
        if self.gate is not None:
            hidden *= self.gate(x)
        hidden = self.hidden_dropout(hidden, deterministic=deterministic)
        hidden = self.fc2(hidden)
        return self.output_dropout(hidden, deterministic=deterministic)
