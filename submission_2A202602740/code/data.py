"""data.py — Xử lý và chuẩn bị dữ liệu Forest CoverType.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.
Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    p = Path(processed_dir)
    train_path = p / "train.npz"
    eval_path = p / "eval.npz"

    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu tại {processed_dir}. "
            "Hãy chạy 'python scripts/split_data.py' trước!"
        )

    train_data = np.load(train_path)
    eval_data = np.load(eval_path)

    X_train_full = train_data["X"].astype(np.float32)
    y_train_full = train_data["y"].astype(np.int64)
    X_eval = eval_data["X"].astype(np.float32)
    y_eval = eval_data["y"].astype(np.int64)
    eval_row_id = eval_data["row_id"].astype(np.int64)

    assert X_train_full.shape == (464809, 54), f"Kích thước X_train không đúng: {X_train_full.shape}"
    assert y_train_full.shape == (464809,), f"Kích thước y_train không đúng: {y_train_full.shape}"
    assert X_eval.shape == (116203, 54), f"Kích thước X_eval không đúng: {X_eval.shape}"
    assert y_eval.shape == (116203,), f"Kích thước y_eval không đúng: {y_eval.shape}"
    assert eval_row_id.shape == (116203,), f"Kích thước eval_row_id không đúng: {eval_row_id.shape}"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=val_fraction,
        stratify=y,
        random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Lý do: Không được tính trên val/eval để ngăn ngừa rò rỉ thông tin (data leakage).
    """
    numeric_data = X_tr[:, :N_NUMERIC]
    mean = numeric_data.mean(axis=0)
    std = numeric_data.std(axis=0)
    # Tránh chia cho 0 nếu một thuộc tính có phương sai bằng 0
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X_scaled = X.copy()
    X_scaled[:, :N_NUMERIC] = (X_scaled[:, :N_NUMERIC] - mean) / std
    return X_scaled


def prepare_data(device: str | torch.device, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và mảng numpy: eval_row_id
    """
    if isinstance(device, str):
        device = torch.device(device)

    # 1. Nạp và tách validation
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction, seed)

    # 2. Chuẩn hoá 10 cột liên tục bằng mean/std của phần train còn lại
    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    # 3. Chuyển thành tensor trên device
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.int64, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.int64, device=device)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device)
    y_eval_t = torch.tensor(y_eval, dtype=torch.int64, device=device)

    # 4. Tính toán thống kê kiểm tra
    counts_tr = np.bincount(y_tr, minlength=7)
    majority_class = int(counts_tr.argmax())
    acc_majority = float((y_val == majority_class).mean())

    print(f"--- Đã chuẩn bị dữ liệu thành công trên {device} ---")
    print(f"Train : X {tuple(X_tr_t.shape)}, y {tuple(y_tr_t.shape)}")
    print(f"Val   : X {tuple(X_val_t.shape)}, y {tuple(y_val_t.shape)}")
    print(f"Eval  : X {tuple(X_eval_t.shape)}, y {tuple(y_eval_t.shape)}")
    print(f"Đoán lớp đa số (lớp {majority_class}) trên Val cho Accuracy = {acc_majority:.4f}")

    return {
        "X_tr": X_tr_t,
        "y_tr": y_tr_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_eval": X_eval_t,
        "y_eval": y_eval_t,
        "eval_row_id": eval_row_id,
        "majority_class": majority_class,
        "acc_majority": acc_majority,
    }


def iterate_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int,
                    generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp mini-batch (xb, yb), thay thế DataLoader để đạt tốc độ tối đa."""
    N = len(X)
    if shuffle:
        perm = torch.randperm(N, generator=generator, device=X.device)
    else:
        perm = torch.arange(N, device=X.device)

    for i in range(0, N, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
