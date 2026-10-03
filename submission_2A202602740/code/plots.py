"""plots.py — Vẽ biểu đồ kết quả thí nghiệm và so sánh theo nhóm.

Mỗi thí nghiệm một ảnh figures/<exp_id>.png gồm ít nhất 3 ô:
  (1) train_loss và val_loss theo epoch (cùng trục)
  (2) val_acc và val_macro_f1 theo epoch
  (3) grad_norm theo epoch (đo TRƯỚC khi clip)
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có 3 ô hoàn chỉnh."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    cfg = result["cfg"]
    hist = result["history"]
    sum_res = result["summary"]

    epochs = hist["epoch"]
    if not epochs:
        print(f"Cảnh báo: Thí nghiệm {cfg['exp_id']} không có dữ liệu lịch sử để vẽ.")
        return

    best_ep = sum_res.get("best_epoch", 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Ô 1: Train Loss & Val Loss
    axes[0].plot(epochs, hist["train_loss"], label="Train Loss (eval mode)", color="#1f77b4", lw=2)
    axes[0].plot(epochs, hist["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    axes[0].axvline(best_ep, color="gray", linestyle="--", alpha=0.7, label=f"Best epoch ({best_ep})")
    axes[0].set_title("Loss theo Epoch", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Ô 2: Val Accuracy & Macro-F1
    axes[1].plot(epochs, hist["val_acc"], label="Val Accuracy", color="#2ca02c", lw=2)
    axes[1].plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="#d62728", lw=2)
    axes[1].axvline(best_ep, color="gray", linestyle="--", alpha=0.7, label=f"Best epoch ({best_ep})")
    axes[1].set_title(f"Val Acc & Macro-F1 (Best F1: {sum_res['val_macro_f1']:.4f})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Điểm số")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Ô 3: Gradient Norm (Pre-clip)
    axes[2].plot(epochs, hist["grad_norm"], label="Gradient Norm (pre-clip)", color="#9467bd", lw=2)
    axes[2].set_title("Chuẩn Gradient Toàn cục", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("L2 Norm")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Tiêu đề chung
    opt_info = f"{cfg.get('optimizer')} (lr={cfg.get('lr')})"
    fig_title = f"[{cfg.get('exp_id')}] {cfg.get('description', '')} | {opt_info} | Batch {cfg.get('batch')} | Init {cfg.get('init')}"
    fig.suptitle(fig_title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu biểu đồ run -> {p}")


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số của nhiều thí nghiệm trên cùng một trục để so sánh nhóm."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for res in results:
        cfg = res["cfg"]
        hist = res["history"]
        exp_id = cfg.get("exp_id", "unknown")
        if metric in hist and hist[metric]:
            ax.plot(hist["epoch"], hist[metric], lw=2, label=f"{exp_id} ({cfg.get('optimizer')}, lr={cfg.get('lr')})")

    ax.set_title(title or f"So sánh {metric} giữa các thí nghiệm", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu biểu đồ so sánh -> {p}")
