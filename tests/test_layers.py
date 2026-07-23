import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from narvl.layers import FeedForward, MultiHeadAttention


def _state_paths(module: nnx.Module) -> set[tuple[object, ...]]:
    return {tuple(path) for path, _ in nnx.to_flat_state(nnx.state(module))}


def _gated_gelu(x: jax.Array) -> jax.Array:
    return jax.nn.gelu(x, approximate=True)


def test_fused_multi_head_attention() -> None:
    attention = MultiHeadAttention(8, 2, qkv_bias=True, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (2, 4, 8))

    output = attention(x, deterministic=True)

    assert output.shape == x.shape
    paths = _state_paths(attention)
    assert ("qkv", "kernel") in paths
    assert ("qkv", "bias") in paths
    assert ("proj", "kernel") in paths
    assert not any(path[0] in {"query", "key", "value"} for path in paths)
    with pytest.raises(ValueError, match="separate attention context"):
        attention(x, context=x, deterministic=True)


def test_t5_style_multi_head_attention() -> None:
    attention = MultiHeadAttention(
        5,
        2,
        head_dim=3,
        fused_qkv=False,
        qkv_bias=False,
        proj_bias=False,
        scale=1.0,
        softmax_dtype=jnp.float32,
        rngs=nnx.Rngs(2),
    )
    x = jax.random.normal(jax.random.key(3), (2, 3, 5))
    context = jax.random.normal(jax.random.key(4), (2, 4, 5))
    mask = jnp.array(
        [
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    attention_bias = jax.random.normal(jax.random.key(5), (1, 2, 3, 4))

    output = attention(
        x,
        context=context,
        mask=mask,
        attention_bias=attention_bias,
        deterministic=True,
    )

    query_projection = attention.query
    key_projection = attention.key
    value_projection = attention.value
    assert query_projection is not None
    assert key_projection is not None
    assert value_projection is not None
    query = jnp.transpose(query_projection(x).reshape(2, 3, 2, 3), (0, 2, 1, 3))
    key = jnp.transpose(key_projection(context).reshape(2, 4, 2, 3), (0, 2, 1, 3))
    value = jnp.transpose(value_projection(context).reshape(2, 4, 2, 3), (0, 2, 1, 3))
    scores = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) + attention_bias
    scores = jnp.where(mask[None, None, :, :], scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(scores.dtype)
    expected = jnp.matmul(weights, value)
    expected = jnp.transpose(expected, (0, 2, 1, 3)).reshape(2, 3, 6)
    expected = attention.proj(expected)

    assert output.shape == (2, 3, 5)
    assert jnp.allclose(output, expected)
    paths = _state_paths(attention)
    assert ("query", "kernel") in paths
    assert ("key", "kernel") in paths
    assert ("value", "kernel") in paths
    assert not any(path[-1] == "bias" for path in paths)


def test_multi_head_attention_validation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(5, 2, rngs=nnx.Rngs(6))
    with pytest.raises(ValueError, match=r"heads \* head_dim"):
        MultiHeadAttention(5, 2, head_dim=3, rngs=nnx.Rngs(7))

    attention = MultiHeadAttention(4, 2, rngs=nnx.Rngs(8))
    with pytest.raises(ValueError, match="Attention input"):
        attention(jnp.ones((2, 4)), deterministic=True)
    with pytest.raises(ValueError, match="2, 3, or 4 dimensions"):
        attention(jnp.ones((2, 3, 4)), mask=jnp.ones((2, 1, 3, 3, 1)), deterministic=True)


def test_standard_feed_forward() -> None:
    feed_forward = FeedForward(4, 6, rngs=nnx.Rngs(9))
    x = jax.random.normal(jax.random.key(10), (2, 3, 4))

    output = feed_forward(x, deterministic=True)
    expected = feed_forward.fc2(jax.nn.gelu(feed_forward.fc1(x), approximate=False))

    assert feed_forward.gate is None
    assert jnp.allclose(output, expected)
    paths = _state_paths(feed_forward)
    assert ("fc1", "kernel") in paths
    assert ("fc2", "kernel") in paths
    assert ("fc1", "bias") in paths
    assert ("fc2", "bias") in paths


def test_t5_style_gated_feed_forward() -> None:
    feed_forward = FeedForward(
        5,
        7,
        activation=_gated_gelu,
        gated=True,
        use_bias=False,
        rngs=nnx.Rngs(11),
    )
    x = jax.random.normal(jax.random.key(12), (2, 3, 5))

    output = feed_forward(x, deterministic=True)

    gate = feed_forward.gate
    assert gate is not None
    expected = feed_forward.fc2(_gated_gelu(feed_forward.fc1(x)) * gate(x))
    assert jnp.allclose(output, expected)
    paths = _state_paths(feed_forward)
    assert ("gate", "kernel") in paths
    assert not any(path[-1] == "bias" for path in paths)
