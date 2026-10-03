"""train.py — Vòng lặp huấn luyện, đánh giá và xuất kết quả dự đoán.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base).
DEFAULT_CFG = dict(
    exp_id="base-s1",
    group="baseline",
    description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # Giá trị khởi điểm hợp lý cho SGD+momentum
    weight_decay=0.0,
    momentum=0.9,
    batch=512,
    epochs=20,
    hidden=(256, 128),
    dropout=0.0,
    init="he",
    clip_norm=None,            # None = không clip; hoặc số dương (ví dụ 1.0)
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cố định cho random, numpy, torch, torch.cuda."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    model.eval()
    preds: list[torch.Tensor] = []
    N = len(X)
    for i in range(0, N, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds.append(logits.argmax(dim=1))
    return torch.cat(preds, dim=0)


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y.
    """
    loss_clean = loss_name.lower().strip()
    if loss_clean == "ce":
        return F.cross_entropy(logits, y)
    elif loss_clean == "mse":
        y_oh = F.one_hot(y, num_classes=7).float()
        return F.mse_loss(logits, y_oh)
    else:
        raise ValueError(f"Hàm mất mát không hợp lệ: {loss_name}")


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    N = len(X)
    cm = np.zeros((7, 7), dtype=np.int64)

    for i in range(0, N, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)

        if loss_name.lower() == "ce":
            batch_loss = F.cross_entropy(logits, yb, reduction="sum").item()
        else:
            y_oh = F.one_hot(yb, num_classes=7).float()
            # Nhân với 7 để đưa về tổng squared error trên batch cho reduction="sum"
            batch_loss = F.mse_loss(logits, y_oh, reduction="sum").item() / 7.0

        total_loss += batch_loss
        pred = logits.argmax(dim=1)
        total_correct += int((pred == yb).sum().item())
        np.add.at(cm, (yb.cpu().numpy(), pred.cpu().numpy()), 1)

    acc = float(total_correct / N)
    macro_f1 = macro_f1_from_confusion(cm)
    return {
        "loss": float(total_loss / N),
        "acc": acc,
        "macro_f1": macro_f1,
        "confusion": cm,
    }


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    set_seed(cfg["seed"])
    device = data["X_tr"].device

    # 1. Khởi tạo model và kiểm tra số tham số
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = str(cfg.get("init", "he"))
    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)

    expected_p = EXPECTED_PARAMS.get(hidden)
    if expected_p is not None:
        actual_p = count_params(model)
        assert actual_p == expected_p, f"Số tham số không khớp: kỳ vọng {expected_p}, thực tế {actual_p}"

    # 2. Khởi tạo optimizer
    optimizer = build_optimizer(
        name=cfg.get("optimizer", "sgd_momentum"),
        params=model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        momentum=float(cfg.get("momentum", 0.9)),
    )

    # 3. Chuẩn bị Mixed Precision
    precision = str(cfg.get("precision", "fp32")).lower()
    use_cuda = (device.type == "cuda")
    scaler = None
    if precision == "fp16" and use_cuda:
        scaler = torch.amp.GradScaler("cuda")

    # 4. Đo loss bước 0 trên val (trước cập nhật đầu tiên, kỳ vọng ≈ ln 7 ≈ 1.946)
    step0_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
    step0_loss = step0_res["loss"]

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = 1
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    best_state = None
    diverged = False

    generator = torch.Generator(device=device).manual_seed(cfg["seed"])
    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch", 512))
    clip_norm = cfg.get("clip_norm")

    # Đặt lại bộ đếm bộ nhớ GPU
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    total_train_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        grad_norms_epoch = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=batch_size,
                                      generator=generator, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            use_amp = (precision in ("fp16", "bf16")) and use_cuda
            amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                print(f"Cảnh báo: Thí nghiệm {cfg['exp_id']} bị phân kỳ (NaN/Inf) tại epoch {epoch}!")
                break

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            grad_norms_epoch.append(gn)

        if use_cuda:
            torch.cuda.synchronize(device)
        epoch_time = time.time() - t0

        if diverged:
            break

        # Cuối epoch: Đánh giá ở chế độ eval()
        # Train loss đo trên 50.000 mẫu cố định để nhanh và đúng chuẩn
        tr_subset_X = data["X_tr"][:50000]
        tr_subset_y = data["y_tr"][:50000]
        tr_res = evaluate(model, tr_subset_X, tr_subset_y, loss_name=cfg["loss"])
        val_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])

        avg_gn = float(np.mean(grad_norms_epoch)) if grad_norms_epoch else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["val_acc"].append(val_res["acc"])
        history["val_macro_f1"].append(val_res["macro_f1"])
        history["grad_norm"].append(avg_gn)
        history["epoch_time_s"].append(epoch_time)

        # Lưu lại epoch có val_loss thấp nhất
        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_epoch = epoch
            best_val_acc = val_res["acc"]
            best_val_macro_f1 = val_res["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_train_time = time.time() - total_train_start
    epochs_run = len(history["epoch"])
    time_per_epoch = total_train_time / max(epochs_run, 1)

    peak_mem_MB = 0.0
    if use_cuda:
        peak_mem_MB = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    final_train_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else float("nan")

    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "val_acc": float(best_val_acc),
        "val_macro_f1": float(best_val_macro_f1),
        "time_per_epoch_s": float(time_per_epoch),
        "peak_mem_MB": float(peak_mem_MB),
        "diverged": bool(diverged),
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề row_id,pred."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"row_id": row_id.astype(int), "pred": preds.astype(int)})
    df.to_csv(path, index=False)
    print(f"Đã lưu file dự đoán ({len(df)} dòng) -> {path}")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dự đoán eval từ best_state của mô hình và ghi file predictions."""
    device = data["X_tr"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    model = MLP(hidden=hidden, dropout=dropout).to(device)

    if result.get("best_state") is not None:
        model.load_state_dict(result["best_state"])
    model.eval()

    preds = predict(model, data["X_eval"])
    preds_np = preds.cpu().numpy()
    write_predictions(data["eval_row_id"], preds_np, pred_path)
