"""run_part4.py — Chạy và kiểm thử Part 4: Đánh giá Eval, Xuất file Excel & Chấm điểm chính thức."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path("submission_2A202602740/code").resolve()))

from data import prepare_data
from train import final_eval, run_experiment, DEFAULT_CFG
from results_table import load_results, to_row, write_xlsx

device = "cuda"
print(f"=== BẮT ĐẦU PART 4: ĐÁNH GIÁ TRÊN TẬP EVAL & XUẤT BẢNG EXCEL ===")

# 1. Nạp dữ liệu
data = prepare_data(device=device, val_fraction=0.2, seed=42, processed_dir="data/processed")

# 2. Tìm cấu hình có Val Macro-F1 cao nhất từ các file JSON kết quả
results = load_results("submission_2A202602740/results")
assert len(results) > 0, "Chưa có kết quả thí nghiệm trong results/. Hãy chạy run_part2 và run_part3 trước!"

# Lọc bỏ các lần chạy phân kỳ
valid_results = [r for r in results if not r.get("summary", {}).get("diverged", False)]
best_run = max(valid_results, key=lambda r: r["summary"]["val_macro_f1"])
best_cfg = best_run["cfg"]
best_id = best_cfg["exp_id"]
print(f"\n>>> CẤU HÌNH TỐI ƯU NHẤT TRÊN VAL: '{best_id}' (Val Macro-F1 = {best_run['summary']['val_macro_f1']:.4f})")
print(f"    Chi tiết: Optimizer={best_cfg.get('optimizer')}, lr={best_cfg.get('lr')}, Hidden={best_cfg.get('hidden')}")

# 3. Huấn luyện lại model tốt nhất (để lấy best_state trong RAM) và dự đoán trên Eval
print(f"\n>>> Đang sinh dự đoán trên toàn bộ tập Eval (116.203 mẫu) cho cấu hình {best_id}...")
best_trained = run_experiment(best_cfg, data)
pred_path = "submission_2A202602740/predictions_eval.csv"
final_eval(best_cfg, best_trained, data, pred_path)

# 4. Chấm điểm chính thức bằng scripts/evaluate.py
eval_json_path = "submission_2A202602740/eval_result.json"
cmd = [
    sys.executable, "-X", "utf8", "scripts/evaluate.py",
    "--pred", pred_path,
    "--out", eval_json_path
]
print(f"\n>>> Đang chạy script chấm điểm chính thức: {' '.join(cmd)}...")
proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
print(proc.stdout)
if proc.returncode != 0:
    print("LỖI CHẤM ĐIỂM:")
    print(proc.stderr)
    sys.exit(1)

with open(eval_json_path, "r", encoding="utf-8") as f:
    eval_result = json.load(f)

# 5. Đánh giá Baseline trên eval để ghi vào bảng và so sánh cải thiện
print("\n>>> Đang chạy đánh giá Baseline (base-s1) trên eval để đo mức độ cải thiện...")
cfg_base = dict(DEFAULT_CFG)
cfg_base["lr"] = 0.05
cfg_base["seed"] = 1
base_trained = run_experiment(cfg_base, data)
base_pred_path = "submission_2A202602740/pred_base_temp.csv"
final_eval(cfg_base, base_trained, data, base_pred_path)
cmd_base = [
    sys.executable, "-X", "utf8", "scripts/evaluate.py",
    "--pred", base_pred_path,
    "--out", "submission_2A202602740/eval_base_temp.json"
]
subprocess.run(cmd_base, check=True)
with open("submission_2A202602740/eval_base_temp.json", "r", encoding="utf-8") as f:
    base_eval_result = json.load(f)
Path(base_pred_path).unlink(missing_ok=True)
Path("submission_2A202602740/eval_base_temp.json").unlink(missing_ok=True)

# 6. Xuất bảng Excel
print("\n>>> Đang xuất bảng tổng hợp experiments.xlsx từ mẫu...")
all_runs = load_results("submission_2A202602740/results")
rows = []
for r in all_runs:
    r_id = r["cfg"]["exp_id"]
    if r_id == best_id:
        row = to_row(r, eval_scores=eval_result, notes="Cấu hình tối ưu nộp bài")
    elif r_id == "base-s1":
        row = to_row(r, eval_scores=base_eval_result, notes="Baseline tham chiếu")
    else:
        row = to_row(r)
    rows.append(row)

write_xlsx(
    rows=rows,
    template_path="templates/experiment_table_template.xlsx",
    out_path="submission_2A202602740/experiments.xlsx"
)

print("\n" + "="*60)
print("TỔNG KẾT ĐÁNH GIÁ EVAL TRÊN TẬP 116.203 MẪU:")
print(f"  Baseline (base-s1)  -> Accuracy: {base_eval_result['accuracy']:.4f} | Macro-F1: {base_eval_result['macro_f1']:.4f}")
print(f"  Cấu hình cuối ({best_id}) -> Accuracy: {eval_result['accuracy']:.4f} | Macro-F1: {eval_result['macro_f1']:.4f}")
diff_f1 = eval_result['macro_f1'] - base_eval_result['macro_f1']
print(f"  Mức cải thiện so với Baseline (ΔF1): {diff_f1:+.4f}")
print("="*60)
print(">>> PART 4 HOÀN TẤT THÀNH CÔNG! <<<")
