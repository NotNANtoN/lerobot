#!/usr/bin/env python3
"""Ablation Study: Euler ODE Step Count vs. Trajectory RMSE and Compiled Latency.

Evaluates the best unaugmented Cosmos 3 Edge policy across Euler step counts
N in {10, 8, 6, 5, 4, 3, 2, 1} on both Eval-1 (V1 held-out eps 32-39) and
Eval-2 (OOD V2 held-out eps 90-99), measuring compiled action denoising latency
and generating publication-ready Pareto plots.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.policies.vam.smol_expert import (  # noqa: E402
    SmolExpertActionDecoder,
    SmolVLANormalizer,
)
from scripts.video_vam.train_smolexpert import (  # noqa: E402
    UnifiedFeatureCacheDataset,
    evaluate_validation,
)


def run_ablation():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    run_dir = REPO_ROOT / "outputs/train/v1-cosmos3-edge-lora-online-unaugmented-smolexpert"
    best_weights = run_dir / "best.safetensors"
    normalizer_path = run_dir / "normalizer.safetensors"
    eval1_manifest = REPO_ROOT / "outputs/features/cosmos3-edge-lora/val/manifest.json"
    eval2_manifest = (
        REPO_ROOT / "outputs/features/cosmos3-online-aug-eval-fix-20260914/v1-adapter-eval2/manifest.json"
    )

    out_dir = REPO_ROOT / "outputs/experiments/euler_step_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Normalizer & SmolExpert Decoder...")
    st = load_file(str(normalizer_path))
    normalizer = SmolVLANormalizer(
        state_mean=st["state_mean"],
        state_std=st["state_std"],
        action_mean=st["action_mean"],
        action_std=st["action_std"],
    )
    decoder = SmolExpertActionDecoder.from_training_checkpoint(
        best_weights,
        normalizer=normalizer,
        device=device,
        input_channels=2048,
    )
    decoder.eval()

    print("Loading Eval-1 and Eval-2 datasets...")
    eval1_ds = UnifiedFeatureCacheDataset(str(eval1_manifest))
    eval2_ds = UnifiedFeatureCacheDataset(str(eval2_manifest))
    print(f"Loaded Eval-1 ({len(eval1_ds)} samples) and Eval-2 ({len(eval2_ds)} samples).")

    # Fixed dummy input for latency benchmarking
    bench_state = torch.zeros(1, 6, device=device)
    bench_context = torch.zeros(1, 600, 2048, dtype=torch.bfloat16, device=device)
    bench_noise = torch.randn(1, 30, 32, device=device)

    step_counts = [10, 8, 6, 5, 4, 3, 2, 1]
    results = []

    for n in step_counts:
        print("\n========================================================")
        print(f"Evaluating Euler Step Count N = {n}")
        print("========================================================")

        # 1. Dual Evaluation
        print("Running Eval-1 (In-Distribution V1 Held-Out eps 32-39)...")
        eval1_res = evaluate_validation(decoder, eval1_ds, device=device, batch_size=8, num_steps=n, seed=42)
        print(
            f"  Eval-1 RMSE: {eval1_res['val_rmse']:.3f} | Arm: {eval1_res['arm_rmse_deg']:.3f}° | "
            f"Gripper: {eval1_res['gripper_rmse']:.3f} | H1: {eval1_res['val_h1']:.3f} | First-5: {eval1_res['val_first5']:.3f}"
        )

        print("Running Eval-2 (Out-of-Distribution V2 Held-Out eps 90-99)...")
        eval2_res = evaluate_validation(
            decoder, eval2_ds, device=device, batch_size=8, num_steps=n, seed=1042
        )
        print(
            f"  Eval-2 RMSE: {eval2_res['val_rmse']:.3f} | Arm: {eval2_res['arm_rmse_deg']:.3f}° | "
            f"Gripper: {eval2_res['gripper_rmse']:.3f} | H1: {eval2_res['val_h1']:.3f} | First-5: {eval2_res['val_first5']:.3f}"
        )

        # 2. Benchmarking Compiled Denoising Latency (batch_size=1, production CUDA graph / compiled runner)
        print("Benchmarking compiled action denoising latency (50 iterations)...")
        # Warmup and graph capture
        for _ in range(5):
            decoder.sample_actions(
                bench_state, bench_context, noise=bench_noise, num_steps=n, use_cuda_graph=True
            )
        torch.cuda.synchronize()

        timed_iters = 50
        t0 = time.perf_counter()
        for _ in range(timed_iters):
            decoder.sample_actions(
                bench_state, bench_context, noise=bench_noise, num_steps=n, use_cuda_graph=True
            )
        torch.cuda.synchronize()
        total_dt = (time.perf_counter() - t0) / timed_iters * 1000  # ms per chunk
        hz = 1000.0 / total_dt

        # Pure denoise loop latency (excluding prefix KV cache preparation)
        prefix = decoder.prepare_prefix_kv(bench_state, bench_context)
        for _ in range(5):
            decoder.sample_actions(
                bench_state,
                bench_context,
                noise=bench_noise,
                prefix_kv_cache=prefix,
                num_steps=n,
                use_cuda_graph=True,
            )
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(timed_iters):
            decoder.sample_actions(
                bench_state,
                bench_context,
                noise=bench_noise,
                prefix_kv_cache=prefix,
                num_steps=n,
                use_cuda_graph=True,
            )
        torch.cuda.synchronize()
        denoise_only_dt = (time.perf_counter() - t0) / timed_iters * 1000  # ms per chunk

        print(
            f"  Latency: End-to-End = {total_dt:.2f} ms ({hz:.1f} Hz) | Pure Denoise = {denoise_only_dt:.2f} ms"
        )

        record = {
            "euler_steps": n,
            "eval1_rmse": float(eval1_res["val_rmse"]),
            "eval1_arm_deg": float(eval1_res["arm_rmse_deg"]),
            "eval1_gripper": float(eval1_res["gripper_rmse"]),
            "eval1_h1": float(eval1_res["val_h1"]),
            "eval1_first5": float(eval1_res["val_first5"]),
            "eval2_rmse": float(eval2_res["val_rmse"]),
            "eval2_arm_deg": float(eval2_res["arm_rmse_deg"]),
            "eval2_gripper": float(eval2_res["gripper_rmse"]),
            "eval2_h1": float(eval2_res["val_h1"]),
            "eval2_first5": float(eval2_res["val_first5"]),
            "latency_e2e_ms": total_dt,
            "rate_e2e_hz": hz,
            "latency_denoise_only_ms": denoise_only_dt,
        }
        results.append(record)

    # Save structured JSON
    json_path = out_dir / "euler_ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation metrics saved to: {json_path}")

    # Generate Publication-Quality Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    steps = [r["euler_steps"] for r in results]
    eval1_rmse = [r["eval1_rmse"] for r in results]
    eval2_rmse = [r["eval2_rmse"] for r in results]

    denoise_ms = [r["latency_denoise_only_ms"] for r in results]
    hz_list = [r["rate_e2e_hz"] for r in results]

    # Subplot 1: RMSE vs. Euler Steps
    ax1 = axes[0]
    ax1.plot(steps, eval1_rmse, "o-", color="#1f77b4", linewidth=2.5, label="Eval-1 (In-Dist V1)")
    ax1.plot(steps, eval2_rmse, "s--", color="#d62728", linewidth=2.0, label="Eval-2 (OOD V2)")
    ax1.set_xlabel("Euler ODE Denoising Steps (N)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Trajectory RMSE", fontsize=11, fontweight="bold")
    ax1.set_title("Trajectory Accuracy vs. Euler Steps", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True, facecolor="white", edgecolor="none")
    ax1.set_xticks(steps)
    ax1.invert_xaxis()

    # Subplot 2: Latency vs. Euler Steps
    ax2 = axes[1]
    ax2.plot(steps, denoise_ms, "^-", color="#2ca02c", linewidth=2.5, label="Pure Denoise Latency (ms)")
    ax2.set_xlabel("Euler ODE Denoising Steps (N)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Denoising Latency (ms)", fontsize=11, fontweight="bold", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(steps)
    ax2.invert_xaxis()

    ax2_twin = ax2.twinx()
    ax2_twin.plot(steps, hz_list, "d-.", color="#ff7f0e", linewidth=2.0, label="Total Action Head Rate (Hz)")
    ax2_twin.set_ylabel("Inference Frequency (Hz)", fontsize=11, fontweight="bold", color="#ff7f0e")
    ax2_twin.tick_params(axis="y", labelcolor="#ff7f0e")
    ax2_twin.axhline(10.0, color="gray", linestyle=":", linewidth=1.5, label="10 Hz Control Budget")
    ax2.set_title("Action Head Speed vs. Euler Steps", fontsize=12, fontweight="bold")

    # Subplot 3: Pareto Frontier (RMSE vs. Denoise Latency)
    ax3 = axes[2]
    scatter = ax3.scatter(denoise_ms, eval1_rmse, c=steps, cmap="viridis_r", s=120, zorder=5)
    for i, txt in enumerate(steps):
        ax3.annotate(
            f"N={txt}",
            (denoise_ms[i], eval1_rmse[i]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
            fontweight="bold",
        )
    ax3.set_xlabel("Pure Denoise Latency (ms)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Eval-1 Trajectory RMSE", fontsize=11, fontweight="bold")
    ax3.set_title("Pareto Frontier: Accuracy vs. Denoise Latency", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label("Euler Steps (N)", fontsize=10)

    plt.tight_layout()
    plot_path = out_dir / "euler_step_ablation_pareto.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Publication-quality Pareto plot saved to: {plot_path}")


if __name__ == "__main__":
    run_ablation()
