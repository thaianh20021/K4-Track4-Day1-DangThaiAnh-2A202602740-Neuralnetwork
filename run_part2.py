"""run_part2.py — Chạy và kiểm thử Part 2: Huấn luyện Baseline & Đo độ nhiễu Seed."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path("submission_2A202602740/code").resolve()))

from data import prepare_data
from train import DEFAULT_CFG, run_experiment
from plots import plot_run
from results_table import save_result

device = "cuda"
print(f"=== BẮT ĐẦU PART 2: BASELINE & ĐỘ NHIỄU SEED TRÊN {device.upper()} ===")

# 1. Nạp dữ liệu
data = prepare_data(device=device, val_fraction=0.2, seed=42, processed_dir="data/processed")

# 2. Cấu hình Baseline chuẩn
cfg_base = dict(DEFAULT_CFG)
cfg_base["lr"] = 0.05  # Tốc độ học tối ưu cho SGD + Momentum 0.9

results = []
seeds = [1, 2, 3]

for s in seeds:
    cfg = dict(cfg_base)
    cfg["seed"] = s
    cfg["exp_id"] = f"base-s{s}"
    cfg["description"] = f"Baseline M-base (Seed {s})"
    print(f"\n>>> Đang huấn luyện Baseline Seed {s} ({cfg['exp_id']})...")
    res = run_experiment(cfg, data)
    results.append(res)
    
    # Lưu JSON và vẽ ảnh
    save_result(res, "submission_2A202602740/results")
    plot_run(res, f"submission_2A202602740/figures/{cfg['exp_id']}.png")
    
    sum_res = res["summary"]
    print(f"  Best Val Loss: {sum_res['best_val_loss']:.4f} (Epoch {sum_res['best_epoch']})")
    print(f"  Val Acc: {sum_res['val_acc']:.4f} | Val Macro-F1: {sum_res['val_macro_f1']:.4f}")
    print(f"  Thời gian/epoch: {sum_res['time_per_epoch_s']:.2f}s | Bộ nhớ GPU: {sum_res['peak_mem_MB']:.1f} MB")

# 3. Tính độ nhiễu giữa các seed
f1_scores = [r["summary"]["val_macro_f1"] for r in results]
acc_scores = [r["summary"]["val_acc"] for r in results]

mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores, ddof=1) if len(f1_scores) > 1 else 0.0
threshold_2sigma = 2 * std_f1

print("\n" + "="*60)
print("KẾT QUẢ ĐO ĐỘ NHIỄU SEED (BASELINE):")
print(f"  Val Macro-F1 các seed: {[round(x, 4) for x in f1_scores]}")
print(f"  Trung bình (Mean)    : {mean_f1:.4f}")
print(f"  Độ lệch chuẩn (Std σ): {std_f1:.5f}")
print(f"  NGƯỠNG NHIỄU (2σ)    : {threshold_2sigma:.5f}")
print(f"  Val Accuracy trung bình: {np.mean(acc_scores):.4f}")
print("="*60)
print(">>> PART 2 HOÀN TẤT THÀNH CÔNG! <<<")
