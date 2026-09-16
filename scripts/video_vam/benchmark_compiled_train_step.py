# ruff: noqa: E402
#!/usr/bin/env python3
import gc
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.policies.vam.smol_expert import (
    ACTION_DIM,
    ACTION_HORIZON,
    SmolExpertActionDecoder,
    SmolVLANormalizer,
    sample_time_beta,
)

torch.set_float32_matmul_precision("high")


def get_vram_mb():
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def create_synthetic_batch(batch_size=8, num_tokens=600, channels=2048, device="cuda"):
    torch.manual_seed(42)
    state = torch.randn(batch_size, ACTION_DIM, device=device, dtype=torch.float32)
    action = torch.randn(batch_size, ACTION_HORIZON, ACTION_DIM, device=device, dtype=torch.float32)
    context = torch.randn(batch_size, num_tokens, channels, device=device, dtype=torch.bfloat16)
    action_is_pad = torch.zeros(batch_size, ACTION_HORIZON, device=device, dtype=torch.bool)
    action_is_pad[:, 24:] = True
    return {
        "state": state,
        "action": action,
        "context": context,
        "action_is_pad": action_is_pad,
    }


def init_model(device="cuda"):
    normalizer = SmolVLANormalizer(
        state_mean=torch.zeros(ACTION_DIM),
        state_std=torch.ones(ACTION_DIM),
        action_mean=torch.zeros(ACTION_DIM),
        action_std=torch.ones(ACTION_DIM),
    )
    decoder = SmolExpertActionDecoder.from_pretrained(
        "lerobot/smolvla_base",
        normalizer=normalizer,
        device=device,
        num_steps=10,
        input_channels=2048,
    )
    return decoder


# Graph-break free flow matching loss
def clean_flow_matching_loss(self, state, action, context, action_is_pad=None, generator=None):
    batch_size = action.shape[0]
    action_normalized = self.normalizer.normalize_action(action)
    action_normalized = self._pad_action_or_state(action_normalized, self.max_action_dim)
    epsilon = torch.randn(
        action_normalized.shape,
        device=action.device,
        dtype=action_normalized.dtype,
        generator=generator,
    )
    t = sample_time_beta(batch_size, action.device, alpha=1.5, beta=1.0, scale=0.999, offset=0.001)
    t = t.to(device=action.device, dtype=torch.float32)
    x_t = (1.0 - t[:, None, None]) * action_normalized + t[:, None, None] * epsilon
    target = epsilon - action_normalized
    prediction = self._vector_field(
        self._prepare_prefix(state, context),
        x_t,
        t,
    )
    squared_error = (prediction.float() - target.float()).square()
    if action_is_pad is None:
        return squared_error.mean()
    valid = (~action_is_pad).to(dtype=squared_error.dtype).unsqueeze(-1)
    valid_count = torch.clamp(valid.sum() * self.max_action_dim, min=1.0)
    return (squared_error * valid).sum() / valid_count


def benchmark_mode(mode_name, make_step_fn, steps=30, warmup=10, batch=None):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    print(f"Testing Mode: {mode_name}")
    decoder = init_model()
    # Patch method
    decoder.clean_flow_loss = clean_flow_matching_loss.__get__(decoder, SmolExpertActionDecoder)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=1e-2)
    step_fn = make_step_fn(decoder, optimizer)

    fixed_seeds = [1000 + i for i in range(warmup + steps)]

    t0 = time.perf_counter()
    for i in range(warmup):
        g = torch.Generator(device="cuda").manual_seed(fixed_seeds[i])
        loss = step_fn(batch, g)
        torch.cuda.synchronize()
    warmup_time = time.perf_counter() - t0
    print(f"  Warmup ({warmup} steps): {warmup_time:.2f}s")

    torch.cuda.synchronize()
    latencies = []
    losses = []
    for i in range(warmup, warmup + steps):
        g = torch.Generator(device="cuda").manual_seed(fixed_seeds[i])
        t_start = time.perf_counter()
        loss = step_fn(batch, g)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000.0)
        losses.append(float(loss.item()))

    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    p50_ms = sorted(latencies)[len(latencies) // 2]
    peak_vram = get_vram_mb()

    print(f"  Avg Latency:   {avg_ms:.2f} ms/step (min: {min_ms:.2f} ms, p50: {p50_ms:.2f} ms)")
    print(f"  Peak VRAM:     {peak_vram:.1f} MB")
    print(f"  Loss:          {losses[0]:.4f} -> {losses[-1]:.4f}")

    del decoder, optimizer, step_fn
    torch.cuda.empty_cache()
    return {
        "mode": mode_name,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "warmup_s": warmup_time,
        "peak_vram_mb": peak_vram,
        "losses": losses,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device = torch.device("cuda")
    print("PyTorch Version:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))

    batch = create_synthetic_batch(batch_size=8, num_tokens=600, channels=2048, device=device)

    # 1. Eager Baseline
    def make_eager(decoder, optimizer):
        def eager_step(b, generator):
            optimizer.zero_grad(set_to_none=True)
            loss = decoder.clean_flow_loss(
                state=b["state"],
                action=b["action"],
                context=b["context"],
                action_is_pad=b["action_is_pad"],
                generator=generator,
            )
            loss.backward()
            optimizer.step()
            return loss

        return eager_step

    res_eager = benchmark_mode("1. Eager (TF32 enabled)", make_eager, batch=batch)

    # 2. Compile clean_flow_loss (mode=default)
    def make_compiled_default(decoder, optimizer):
        compiled_loss = torch.compile(decoder.clean_flow_loss, mode="default")

        def step_fn(b, generator):
            optimizer.zero_grad(set_to_none=True)
            loss = compiled_loss(
                state=b["state"],
                action=b["action"],
                context=b["context"],
                action_is_pad=b["action_is_pad"],
                generator=generator,
            )
            loss.backward()
            optimizer.step()
            return loss

        return step_fn

    res_compiled_def = benchmark_mode(
        "2. Compiled (mode=default, graph-break free)", make_compiled_default, batch=batch
    )

    # 3. Compile clean_flow_loss (mode=reduce-overhead)
    def make_compiled_reduce(decoder, optimizer):
        compiled_loss = torch.compile(decoder.clean_flow_loss, mode="reduce-overhead")

        def step_fn(b, generator):
            optimizer.zero_grad(set_to_none=True)
            loss = compiled_loss(
                state=b["state"],
                action=b["action"],
                context=b["context"],
                action_is_pad=b["action_is_pad"],
                generator=generator,
            )
            loss.backward()
            optimizer.step()
            return loss

        return step_fn

    res_compiled_reduce = benchmark_mode(
        "3. Compiled (mode=reduce-overhead, CUDA graphs)", make_compiled_reduce, batch=batch
    )

    print("\n" + "=" * 85)
    print("BENCHMARK RESULTS: Offline SmolExpert Train Step (Batch Size 8, Context 600x2048)")
    print("=" * 85)
    for r in [res_eager, res_compiled_def, res_compiled_reduce]:
        if r is not None:
            name = r["mode"]
            avg_ms = r["avg_ms"]
            min_ms = r["min_ms"]
            warm_s = r["warmup_s"]
            vram = r["peak_vram_mb"]
            print(
                f"{name:<45} | Avg: {avg_ms:6.2f} ms | Min: {min_ms:6.2f} ms | Warmup: {warm_s:5.2f} s | VRAM: {vram:6.1f} MB"
            )


if __name__ == "__main__":
    main()
