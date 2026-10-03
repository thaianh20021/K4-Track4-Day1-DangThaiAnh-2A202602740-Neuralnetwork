"""run_part3.py — Chạy và kiểm thử Part 3: Thực hiện 7 nhóm thí nghiệm thử thách."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path("submission_2A202602740/code").resolve()))

from data import prepare_data
from train import DEFAULT_CFG, run_experiment
from plots import plot_run, plot_compare
from results_table import save_result

device = "cuda"
print(f"=== BẮT ĐẦU PART 3: 7 CHỦ ĐỀ THÍ NGHIỆM TRÊN {device.upper()} ===")

data = prepare_data(device=device, val_fraction=0.2, seed=42, processed_dir="data/processed")

# Base config chuẩn để kế thừa (mỗi thí nghiệm chỉ đổi đúng 1 yếu tố)
base_cfg = dict(DEFAULT_CFG)
base_cfg["lr"] = 0.05
base_cfg["seed"] = 1

experiments = [
    # 1. Hàm mất mát (Loss)
    dict(base_cfg, exp_id="loss-mse", group="loss", loss="mse",
         description="Dùng MSE Loss thay vì Cross-Entropy"),

    # 2. Bộ tối ưu hoá (Optimizer)
    dict(base_cfg, exp_id="opt-sgd", group="optimizer", optimizer="sgd", momentum=0.0,
         description="SGD thuần (không momentum) lr=0.05"),
    dict(base_cfg, exp_id="opt-adam-lr1e-3", group="optimizer", optimizer="adam", lr=0.001,
         description="Adam lr=0.001"),
    dict(base_cfg, exp_id="opt-adam-lr3e-4", group="optimizer", optimizer="adam", lr=0.0003,
         description="Adam lr=0.0003"),
    dict(base_cfg, exp_id="opt-adamw-lr1e-3", group="optimizer", optimizer="adamw", lr=0.001, weight_decay=0.01,
         description="AdamW lr=0.001 weight_decay=0.01"),

    # 3. Hyper-parameters (Hparam)
    dict(base_cfg, exp_id="hparam-batch128", group="hparam", batch=128,
         description="Batch size nhỏ = 128 (nhiều bước cập nhật hơn)"),
    dict(base_cfg, exp_id="hparam-batch2048", group="hparam", batch=2048,
         description="Batch size lớn = 2048 (ít bước cập nhật hơn)"),
    dict(base_cfg, exp_id="hparam-wide", group="hparam", hidden=(512, 256),
         description="Kiến trúc M-wide (54->512->256->7)"),
    dict(base_cfg, exp_id="hparam-deep", group="hparam", hidden=(256, 128, 64),
         description="Kiến trúc M-deep (54->256->128->64->7)"),

    # 4. Dropout
    dict(base_cfg, exp_id="drop-0.2", group="dropout", dropout=0.2,
         description="Dropout q=0.2 sau ReLU các lớp ẩn"),
    dict(base_cfg, exp_id="drop-0.5", group="dropout", dropout=0.5,
         description="Dropout q=0.5 sau ReLU các lớp ẩn"),

    # 5. Gradient Clipping
    dict(base_cfg, exp_id="clip-1.0", group="clipping", clip_norm=1.0,
         description="Cắt Gradient norm với ngưỡng c=1.0"),
    dict(base_cfg, exp_id="clip-highlr-noclip", group="clipping", lr=0.8, clip_norm=None,
         description="Phản chứng: lr cực cao (0.8) không clip"),
    dict(base_cfg, exp_id="clip-highlr-clip1.0", group="clipping", lr=0.8, clip_norm=1.0,
         description="Phản chứng: lr cực cao (0.8) có clip c=1.0"),

    # 6. Mixed Precision (AMP)
    dict(base_cfg, exp_id="amp-fp16", group="amp", precision="fp16",
         description="Huấn luyện độ chính xác hỗn hợp FP16 với GradScaler"),

    # 7. Khởi tạo tham số (Init)
    dict(base_cfg, exp_id="init-zeros", group="init", init="zeros",
         description="Khởi tạo trọng số bằng 0 (nơ-ron đối xứng)"),
    dict(base_cfg, exp_id="init-xavier", group="init", init="xavier",
         description="Khởi tạo Xavier Normal"),
]

all_results = {}
for i, cfg in enumerate(experiments, start=1):
    print(f"\n[{i}/{len(experiments)}] Huấn luyện thí nghiệm: {cfg['exp_id']} (Nhóm: {cfg['group']})...")
    res = run_experiment(cfg, data)
    all_results[cfg["exp_id"]] = res
    save_result(res, "submission_2A202602740/results")
    plot_run(res, f"submission_2A202602740/figures/{cfg['exp_id']}.png")
    
    s = res["summary"]
    status = "PHÂN KỲ (DIVERGED)" if s["diverged"] else "HỘI TỤ"
    print(f"  Trạng thái: {status} | Best Val Loss: {s['best_val_loss']:.4f} (Epoch {s['best_epoch']})")
    print(f"  Val Acc: {s['val_acc']:.4f} | Val Macro-F1: {s['val_macro_f1']:.4f}")

# Vẽ các ảnh so sánh chồng theo nhóm
print("\n>>> Đang vẽ các biểu đồ so sánh nhóm...")
# 1. So sánh Optimizer
opt_res = [all_results[k] for k in ["opt-sgd", "opt-adam-lr1e-3", "opt-adam-lr3e-4", "opt-adamw-lr1e-3"]]
plot_compare(opt_res, "val_macro_f1", "submission_2A202602740/figures/compare_optimizer.png",
             "So sánh Val Macro-F1 giữa các Bộ tối ưu")

# 2. So sánh Dropout
drop_res = [all_results[k] for k in ["drop-0.2", "drop-0.5"]]
plot_compare(drop_res, "val_loss", "submission_2A202602740/figures/compare_dropout.png",
             "So sánh Val Loss khi dùng Dropout")

# 3. So sánh Clipping ở lr cao
clip_res = [all_results[k] for k in ["clip-highlr-noclip", "clip-highlr-clip1.0"]]
plot_compare(clip_res, "grad_norm", "submission_2A202602740/figures/compare_clipping.png",
             "So sánh Chuẩn Gradient khi lr cao có và không Clip")

# 4. So sánh Kiến trúc / Batch
hparam_res = [all_results[k] for k in ["hparam-batch128", "hparam-batch2048", "hparam-wide", "hparam-deep"]]
plot_compare(hparam_res, "val_macro_f1", "submission_2A202602740/figures/compare_hparam.png",
             "So sánh Val Macro-F1 giữa các cấu hình Hyper-parameter")

print("\n>>> PART 3 HOÀN TẤT THÀNH CÔNG! ĐÃ CHẠY ĐỦ 7 CHỦ ĐỀ THÍ NGHIỆM! <<<")
