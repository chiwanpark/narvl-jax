from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from safetensors.numpy import save_file

import narvl.models as model_api
from narvl.models import (
    MobileCLIP2ImageConfig,
    MobileCLIP2ImageEncoder,
    MobileCLIP2TextConfig,
    MobileCLIP2TextEncoder,
)
from narvl.models.fastvit import FastViTConfig, FastViTEncoder
from narvl.models.mobileclip2 import (
    MOBILECLIP2_IMAGE_CONFIGS,
    MOBILECLIP2_TEXT_CONFIGS,
    TextConfig,
    TextTransformer,
    create_mobileclip2_image_encoder,
    create_mobileclip2_text_encoder,
    get_mobileclip2_image_config,
    get_mobileclip2_text_config,
)
from narvl.models.io import (
    load_mobileclip2_image_encoder_state_dict,
    load_mobileclip2_image_encoder_weights,
    load_mobileclip2_text_encoder_state_dict,
    load_mobileclip2_text_encoder_weights,
)
from narvl.models.io.mobileclip2 import MOBILECLIP2_PRETRAINED_REPOS


def _text_config() -> TextConfig:
    return TextConfig(
        output_dim=8,
        width=8,
        heads=2,
        layers=2,
        context_length=8,
        vocab_size=32,
    )


def _state_value(model: nnx.Module, path: tuple[str | int, ...]) -> jax.Array:
    variables = dict(nnx.to_flat_state(nnx.state(model)))
    variable = variables[path]
    assert isinstance(variable, (nnx.Param, nnx.BatchStat))
    return variable[...]


def _fastvit_config() -> FastViTConfig:
    return FastViTConfig(
        image_size=32,
        output_dim=8,
        layers=(1, 1),
        embed_dims=(8, 16),
        mlp_ratios=(2, 2),
        token_mixers=("repmixer", "repmixer"),
        downsamples=(False, True),
        se_downsamples=(False, True),
        pos_embs=(False, True),
    )


def test_mobileclip2_type_aliases() -> None:
    assert MobileCLIP2ImageConfig.__value__ is FastViTConfig
    assert MobileCLIP2ImageEncoder.__value__ is FastViTEncoder
    assert MobileCLIP2TextConfig.__value__ is TextConfig
    assert MobileCLIP2TextEncoder.__value__ is TextTransformer
    assert {
        "FastViTConfig",
        "FastViTEncoder",
        "MOBILECLIP2_IMAGE_CONFIGS",
        "MOBILECLIP2_PRETRAINED_REPOS",
        "MOBILECLIP2_TEXT_CONFIGS",
        "TextConfig",
        "load_mobileclip2_image_encoder_from_hf",
        "load_mobileclip2_image_encoder_state_dict",
        "load_mobileclip2_image_encoder_weights",
        "load_mobileclip2_text_encoder_from_hf",
        "load_mobileclip2_text_encoder_state_dict",
        "load_mobileclip2_text_encoder_weights",
        "TextTransformer",
    }.isdisjoint(model_api.__all__)


def test_mobileclip2_presets() -> None:
    variants = {
        "MobileCLIP2-S0",
        "MobileCLIP2-S2",
        "MobileCLIP2-S3",
        "MobileCLIP2-S4",
    }
    assert set(MOBILECLIP2_IMAGE_CONFIGS) == variants
    assert set(MOBILECLIP2_TEXT_CONFIGS) == variants
    assert set(MOBILECLIP2_PRETRAINED_REPOS) == variants
    assert get_mobileclip2_image_config("s0") is MOBILECLIP2_IMAGE_CONFIGS["MobileCLIP2-S0"]
    assert get_mobileclip2_text_config("mobileclip2_s4") is MOBILECLIP2_TEXT_CONFIGS["MobileCLIP2-S4"]
    assert not MOBILECLIP2_TEXT_CONFIGS["MobileCLIP2-S4"].causal

    with pytest.raises(ValueError, match="Unknown MobileCLIP2 variant"):
        get_mobileclip2_image_config("B")
    with pytest.raises(ValueError, match="Unknown MobileCLIP2 variant"):
        get_mobileclip2_text_config("L-14")


def test_mobileclip2_standalone_encoder_factories() -> None:
    image_encoder = create_mobileclip2_image_encoder("S0", param_dtype=jnp.float16, rngs=nnx.Rngs(0))
    assert isinstance(image_encoder, FastViTEncoder)
    assert image_encoder.config.output_dim == 512
    del image_encoder

    text_encoder = create_mobileclip2_text_encoder("S0", param_dtype=jnp.float16, rngs=nnx.Rngs(1))
    assert isinstance(text_encoder, TextTransformer)
    assert text_encoder.config.output_dim == 512


def test_fastvit_image_encoder() -> None:
    encoder = FastViTEncoder(_fastvit_config(), rngs=nnx.Rngs(2))
    images = jnp.ones((2, 32, 32, 3))

    features = encoder(images)
    tokens = encoder.encode_tokens(images)

    assert features.shape == (2, 8)
    assert tokens.shape == (2, 16, 8)
    assert jnp.allclose(jnp.mean(tokens, axis=1), features, atol=1e-5)
    with pytest.raises(ValueError, match="Images must have shape"):
        encoder(jnp.ones((2, 32, 32, 1)))


def test_text_encoder_attention_mask_and_all_tokens() -> None:
    encoder = TextTransformer(_text_config(), rngs=nnx.Rngs(3))
    tokens = jnp.array([[1, 2, 31, 0], [1, 4, 5, 31]])
    attention_mask = jnp.array([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=jnp.bool_)

    features = encoder(tokens, attention_mask=attention_mask)
    all_tokens = encoder(tokens, attention_mask=attention_mask, return_all_tokens=True)

    assert features.shape == (2, 8)
    assert all_tokens.shape == (2, 4, 8)


def test_image_encoder_openclip_state_dict_loading() -> None:
    encoder = FastViTEncoder(_fastvit_config(), rngs=nnx.Rngs(6))
    conv = np.arange(8 * 3 * 3 * 3, dtype=np.float32).reshape(8, 3, 3, 3)
    layer_scale = np.arange(8, dtype=np.float32).reshape(8, 1, 1)
    state_dict = {
        "logit_scale": np.array(3.0, dtype=np.float32),
        "text.token_embedding.weight": np.zeros((4, 4), dtype=np.float32),
        "visual.trunk.stem.0.conv_kxk.0.conv.weight": conv,
        "visual.trunk.stem.0.conv_kxk.0.bn.running_mean": np.arange(8, dtype=np.float32),
        "visual.trunk.stages.0.blocks.0.layer_scale.gamma": layer_scale,
    }

    result = load_mobileclip2_image_encoder_state_dict(encoder, state_dict, strict=False)

    assert result is encoder
    assert jnp.array_equal(
        _state_value(encoder, ("stem", 0, "branches", 0, "conv", "kernel")),
        conv.transpose(2, 3, 1, 0),
    )
    assert jnp.array_equal(
        _state_value(encoder, ("stem", 0, "branches", 0, "norm", "mean")),
        jnp.arange(8),
    )
    assert jnp.array_equal(_state_value(encoder, ("stages", 0, "blocks", 0, "layer_scale")), jnp.arange(8))


def test_text_encoder_openclip_state_dict_loading() -> None:
    encoder = TextTransformer(_text_config(), rngs=nnx.Rngs(7))
    qkv = np.arange(24 * 8, dtype=np.float32).reshape(24, 8)
    fc1 = np.arange(32 * 8, dtype=np.float32).reshape(32, 8)
    fc2 = np.arange(8 * 32, dtype=np.float32).reshape(8, 32)
    state_dict = {
        "logit_scale": np.array(3.0, dtype=np.float32),
        "text.transformer.resblocks.0.attn.in_proj_weight": qkv,
        "text.transformer.resblocks.0.mlp.c_fc.weight": fc1,
        "text.transformer.resblocks.0.mlp.c_proj.weight": fc2,
        "visual.trunk.stem.0.conv_kxk.0.conv.weight": np.zeros((1, 1), dtype=np.float32),
    }

    result = load_mobileclip2_text_encoder_state_dict(encoder, state_dict, strict=False)

    assert result is encoder
    assert jnp.array_equal(_state_value(encoder, ("blocks", 0, "attention", "qkv", "kernel")), qkv.T)
    assert jnp.array_equal(_state_value(encoder, ("blocks", 0, "mlp", "fc1", "kernel")), fc1.T)
    assert jnp.array_equal(_state_value(encoder, ("blocks", 0, "mlp", "fc2", "kernel")), fc2.T)


def test_layer_norm_fastvit_loading() -> None:
    config = FastViTConfig(
        image_size=32,
        output_dim=8,
        layers=(1,),
        embed_dims=(32,),
        mlp_ratios=(2,),
        token_mixers=("attention",),
        downsamples=(False,),
        se_downsamples=(False,),
        pos_embs=(True,),
        norm_layer="layer_norm",
        stem_use_scale_branch=False,
    )
    encoder = FastViTEncoder(config, rngs=nnx.Rngs(8))

    load_mobileclip2_image_encoder_state_dict(
        encoder,
        {"visual.trunk.stages.0.blocks.0.norm.weight": np.arange(32, dtype=np.float32)},
        strict=False,
    )

    assert jnp.array_equal(
        _state_value(encoder, ("stages", 0, "blocks", 0, "norm", "norm", "scale")),
        jnp.arange(32),
    )


def test_encoder_safetensors_loading(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    conv = np.arange(8 * 3 * 3 * 3, dtype=np.float32).reshape(8, 3, 3, 3)
    embedding = np.arange(32 * 8, dtype=np.float32).reshape(32, 8)
    save_file(
        {
            "logit_scale": np.array(4.0, dtype=np.float32),
            "visual.trunk.stem.0.conv_kxk.0.conv.weight": conv,
            "text.token_embedding.weight": embedding,
        },
        path,
    )
    image_encoder = FastViTEncoder(_fastvit_config(), rngs=nnx.Rngs(11))
    text_encoder = TextTransformer(_text_config(), rngs=nnx.Rngs(12))

    load_mobileclip2_image_encoder_weights(image_encoder, path, strict=False)
    load_mobileclip2_text_encoder_weights(text_encoder, path, strict=False)

    assert jnp.array_equal(
        _state_value(image_encoder, ("stem", 0, "branches", 0, "conv", "kernel")),
        conv.transpose(2, 3, 1, 0),
    )
    assert jnp.array_equal(_state_value(text_encoder, ("token_embedding", "embedding")), embedding)
    with pytest.raises(ValueError, match="missing tensors"):
        load_mobileclip2_image_encoder_weights(image_encoder, path)
    with pytest.raises(ValueError, match="missing tensors"):
        load_mobileclip2_text_encoder_weights(text_encoder, path)


def test_loader_validation() -> None:
    encoder = TextTransformer(_text_config(), rngs=nnx.Rngs(13))
    assert MOBILECLIP2_PRETRAINED_REPOS["MobileCLIP2-S0"] == "timm/MobileCLIP2-S0-OpenCLIP"

    with pytest.raises(ValueError, match="shape mismatches"):
        load_mobileclip2_text_encoder_state_dict(
            encoder,
            {"text.token_embedding.weight": np.zeros((4, 4), dtype=np.float32)},
            strict=False,
        )


def test_fastvit_config_validation() -> None:
    with pytest.raises(ValueError, match="same length"):
        FastViTConfig(
            image_size=32,
            output_dim=8,
            layers=(1,),
            embed_dims=(8, 16),
            mlp_ratios=(2,),
            token_mixers=("repmixer",),
            downsamples=(False,),
            se_downsamples=(False,),
            pos_embs=(False,),
        )
