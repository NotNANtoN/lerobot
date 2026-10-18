"""Unit tests for Cosmos 3 Edge LoRA adaptation and feature extraction components."""

import tempfile
from pathlib import Path

import pytest
import safetensors
import torch
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer

from lerobot.policies.vam.cosmos3_lora import (
    DUAL_PATHWAY_TARGET_MODULES,
    GEN_TARGET_MODULES,
    convert_legacy_cosmos3_lora_to_peft,
    cosmos3_lora_config,
    inject_cosmos3_lora,
    load_cosmos3_lora,
    load_cosmos3_peft_adapter,
    read_cosmos3_adapter_hyperparameters,
    save_cosmos3_lora,
)
from lerobot.policies.vam.cosmos_lora import LoRALinear


def build_tiny_real_cosmos3_transformer(
    num_layers: int = 3,
    hidden_size: int = 64,
    head_dim: int = 16,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    dtype: str = "float32",
) -> Cosmos3OmniTransformer:
    """Instantiate a real native diffusers Cosmos3OmniTransformer with compact dimensions for CPU tests."""
    return Cosmos3OmniTransformer(
        head_dim=head_dim,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        latent_channel=48,
        latent_patch_size=2,
        patch_latent_dim=192,
        vocab_size=1000,
        dtype=dtype,
    )


def test_inject_cosmos3_lora_dual_pathway() -> None:
    transformer = build_tiny_real_cosmos3_transformer(num_layers=3)
    wrapped = inject_cosmos3_lora(transformer, rank=4, alpha=8.0)

    # 14 dual-pathway modules (8 attn + 6 mlp) per layer * 3 layers = 42 modules
    assert len(wrapped) == 42

    for i in range(3):
        layer = transformer.layers[i]
        # Understanding pathway MUST have LoRALinear
        assert isinstance(layer.self_attn.to_q, LoRALinear)
        assert isinstance(layer.self_attn.to_k, LoRALinear)
        assert isinstance(layer.self_attn.to_v, LoRALinear)
        assert isinstance(layer.self_attn.to_out, LoRALinear)
        assert isinstance(layer.mlp.gate_proj, LoRALinear)
        assert isinstance(layer.mlp.up_proj, LoRALinear)
        assert isinstance(layer.mlp.down_proj, LoRALinear)

        # Generation pathway MUST also have LoRALinear
        assert isinstance(layer.self_attn.add_q_proj, LoRALinear)
        assert isinstance(layer.self_attn.add_k_proj, LoRALinear)
        assert isinstance(layer.self_attn.add_v_proj, LoRALinear)
        assert isinstance(layer.self_attn.to_add_out, LoRALinear)
        assert isinstance(layer.mlp_moe_gen.gate_proj, LoRALinear)
        assert isinstance(layer.mlp_moe_gen.up_proj, LoRALinear)
        assert isinstance(layer.mlp_moe_gen.down_proj, LoRALinear)

    trainable_params = dict(transformer.named_parameters())
    trainable_names = [name for name, p in trainable_params.items() if p.requires_grad]

    assert len(trainable_names) == 42 * 2  # lora_A and lora_B for each wrapped linear
    for name in trainable_names:
        assert any(target in name for target in DUAL_PATHWAY_TARGET_MODULES)


def test_zero_lora_initialization_parity() -> None:
    """Zero-LoRA test: with B initialized to 0, adapter output is identically zero and model matches base."""
    transformer = build_tiny_real_cosmos3_transformer(num_layers=2)

    und = torch.randn(5, 64)
    gen = torch.randn(8, 64)
    rotary = (torch.ones(5, 16), torch.zeros(5, 16), torch.ones(8, 16), torch.zeros(8, 16))

    # Base forward
    with torch.no_grad():
        u_base, g_base = transformer.layers[0](und, gen, rotary)

    # Inject LoRA
    inject_cosmos3_lora(transformer, rank=4, alpha=8.0)

    # Adapted forward with zero-initialized B
    with torch.no_grad():
        u_lora, g_lora = transformer.layers[0](und, gen, rotary)

    # Output with zero-LoRA must be identical to base
    assert torch.allclose(u_base, u_lora, atol=1e-7)
    assert torch.allclose(g_base, g_lora, atol=1e-7)


def test_save_and_load_cosmos3_lora_with_metadata() -> None:
    transformer1 = build_tiny_real_cosmos3_transformer(num_layers=2)
    inject_cosmos3_lora(transformer1, rank=4, alpha=16.0)

    # Modify lora weights
    with torch.no_grad():
        for p in transformer1.parameters():
            if p.requires_grad:
                p.fill_(0.42)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "lora.safetensors"
        save_cosmos3_lora(
            transformer1,
            save_path,
            rank=4,
            alpha=16.0,
            block_indices=[0, 1],
            target_modules=DUAL_PATHWAY_TARGET_MODULES,
        )

        # Check safetensors metadata header
        with safetensors.safe_open(str(save_path), framework="pt") as f:
            metadata = f.metadata()
            assert metadata is not None
            assert metadata["rank"] == "4"
            assert metadata["alpha"] == "16.0"

        # Create fresh transformer and load
        transformer2 = build_tiny_real_cosmos3_transformer(num_layers=2)
        load_cosmos3_lora(transformer2, save_path)

        # Check values match 0.42
        for _name, p in transformer2.named_parameters():
            if p.requires_grad:
                assert torch.allclose(p, torch.full_like(p, 0.42))


def _layer_outputs(transformer: Cosmos3OmniTransformer) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    und, gen = torch.randn(5, 64), torch.randn(8, 64)
    rotary = (torch.ones(5, 16), torch.zeros(5, 16), torch.ones(8, 16), torch.zeros(8, 16))
    with torch.no_grad():
        return transformer.layers[1](*transformer.layers[0](und, gen, rotary), rotary)


def test_peft_config_targets_match_legacy_injection() -> None:
    peft = pytest.importorskip("peft")
    legacy = build_tiny_real_cosmos3_transformer(num_layers=3)
    wrapped = set(inject_cosmos3_lora(legacy, rank=4, alpha=8.0, block_indices=[0, 2]))
    model = peft.get_peft_model(
        build_tiny_real_cosmos3_transformer(num_layers=3),
        cosmos3_lora_config(rank=4, alpha=8.0, block_indices=[0, 2]),
    )
    peft_wrapped = {
        name.removeprefix("base_model.model.").removesuffix(".lora_A")
        for name, _ in model.named_modules()
        if name.endswith(".lora_A")
    }
    assert peft_wrapped == wrapped
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == sum(
        p.numel() for p in legacy.parameters() if p.requires_grad
    )


def test_legacy_lora_converts_to_peft_with_identical_outputs(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    torch.manual_seed(0)
    base_state = build_tiny_real_cosmos3_transformer(num_layers=2).state_dict()
    legacy = build_tiny_real_cosmos3_transformer(num_layers=2)
    legacy.load_state_dict(base_state)
    inject_cosmos3_lora(legacy, rank=4, alpha=16.0, target_modules=GEN_TARGET_MODULES)
    with torch.no_grad():
        for p in legacy.parameters():
            if p.requires_grad:
                p.normal_(std=0.05)  # non-zero B so the adapter actually changes the output
    save_cosmos3_lora(legacy, tmp_path / "legacy.safetensors", rank=4, alpha=16.0)
    expected = _layer_outputs(legacy)

    adapter = convert_legacy_cosmos3_lora_to_peft(tmp_path / "legacy.safetensors", tmp_path / "peft")
    assert read_cosmos3_adapter_hyperparameters(adapter) == (4, 16.0)

    for merge in (False, True):
        fresh = build_tiny_real_cosmos3_transformer(num_layers=2)
        fresh.load_state_dict(base_state)
        base_out = _layer_outputs(fresh)
        adapted = load_cosmos3_peft_adapter(fresh, adapter, merge=merge)
        got = _layer_outputs(adapted)
        assert not torch.allclose(base_out[1], expected[1])
        for g, e in zip(got, expected, strict=True):
            torch.testing.assert_close(g, e, atol=1e-5, rtol=1e-5)
        assert not any(p.requires_grad for p in adapted.parameters())


def test_load_cosmos3_lora_missing_file_error() -> None:
    transformer = build_tiny_real_cosmos3_transformer(num_layers=2)
    non_existent = Path("/tmp/non_existent_lora_checkpoint_12345.safetensors")
    with pytest.raises(FileNotFoundError):
        load_cosmos3_lora(transformer, non_existent)


def test_load_cosmos3_lora_strict_missing_keys_check() -> None:
    """Strict scope check: loading a partial state dict into full scope must fail."""
    transformer1 = build_tiny_real_cosmos3_transformer(num_layers=4)
    # Inject only layer 0
    inject_cosmos3_lora(transformer1, rank=4, alpha=8.0, block_indices=[0])

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "partial_lora.safetensors"
        save_cosmos3_lora(transformer1, save_path, rank=4, alpha=8.0, block_indices=[0])

        # Try to load into transformer configured for layers 0..3 (missing layers 1..3 in checkpoint)
        transformer2 = build_tiny_real_cosmos3_transformer(num_layers=4)
        inject_cosmos3_lora(transformer2, rank=4, alpha=8.0, block_indices=[0, 1, 2, 3])

        with pytest.raises(RuntimeError, match="Missing.*required LoRA keys"):
            load_cosmos3_lora(transformer2, save_path, block_indices=[0, 1, 2, 3])
