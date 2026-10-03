"""optimizer.py — Quản lý bộ tối ưu hoá, scheduler và cắt gradient (clipping).

File này gom việc chọn bộ tối ưu và cắt gradient để `train.py` gọn và mọi thí nghiệm công bằng.

Công thức cần hiểu (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import torch
import torch.optim as optim

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer tương ứng."""
    name_clean = name.lower().strip()
    if name_clean not in OPTIMIZERS:
        raise ValueError(f"Bộ tối ưu '{name}' không nằm trong danh sách hỗ trợ: {OPTIMIZERS}")

    if name_clean == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif name_clean == "sgd_momentum":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name_clean == "adam":
        return optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name_clean == "adamw":
        return optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        raise ValueError(f"Không xác định được bộ tối ưu: {name_clean}")


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """Bộ lập lịch tốc độ học (tuỳ chọn)."""
    if name is None or not name:
        return None
    name_clean = name.lower().strip()
    if name_clean == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    return None


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Khi max_norm là None hoặc <= 0: tính chuẩn toàn cục mà không cắt.
    Khi dùng mixed precision FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    """
    params_list = list(params)
    if max_norm is None or max_norm <= 0:
        total_norm = torch.nn.utils.clip_grad_norm_(params_list, max_norm=float("inf"))
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(params_list, max_norm=float(max_norm))
    return float(total_norm)
