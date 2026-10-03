import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path("submission_2A202602740/code").resolve()))

from data import prepare_data
from model import MLP, EXPECTED_PARAMS, count_params, activation_stats
from optimizer import build_optimizer, clip_gradients
from train import evaluate, set_seed

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Testing device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

# 1. Test data
data = prepare_data(device=device, val_fraction=0.2, seed=42, processed_dir="data/processed")
assert data["X_tr"].shape == (371847, 54), f"X_tr shape: {data['X_tr'].shape}"
assert data["X_val"].shape == (92962, 54), f"X_val shape: {data['X_val'].shape}"
assert data["X_eval"].shape == (116203, 54), f"X_eval shape: {data['X_eval'].shape}"
print("[PASS] Part 0: Data preparation passed!")

# 2. Test model shape & params
set_seed(42)
model = MLP(hidden=(256, 128), dropout=0.0, init="he").to(device)
p_count = count_params(model)
print(f"Param count: {p_count} (Expected: {EXPECTED_PARAMS[(256, 128)]})")
assert p_count == EXPECTED_PARAMS[(256, 128)], f"Expected {EXPECTED_PARAMS[(256, 128)]}, got {p_count}"
print("[PASS] Model architecture and param count passed!")

# 3. Test forward dummy
dummy = torch.randn(8, 54, device=device)
out = model(dummy)
assert out.shape == (8, 7), f"Expected (8, 7), got {out.shape}"
print("[PASS] Forward pass shape check passed!")

# 4. Test Step 0 loss
step0_eval = evaluate(model, data["X_val"], data["y_val"], loss_name="ce")
print(f"Step 0 loss on Val: {step0_eval['loss']:.4f} (ln 7 = {np.log(7):.4f}, chênh lệch = {abs(step0_eval['loss'] - np.log(7)):.4f})")
# GUIDE: Loss bước 0 phải gần ln 7 ≈ 1.946. Với khởi tạo He ngẫu nhiên, loss thường nằm trong khoảng 1.95 - 2.35.
assert abs(step0_eval["loss"] - np.log(7)) < 0.5, f"Step 0 loss {step0_eval['loss']} lệch quá lớn so với ln 7"
print("[PASS] Step 0 loss test passed!")

# 5. Test Overfit 20 samples
x_small = data["X_tr"][:20]
y_small = data["y_tr"][:20]
model_small = MLP(hidden=(256, 128), dropout=0.0, init="he").to(device)
opt_small = torch.optim.Adam(model_small.parameters(), lr=0.01)
for step in range(250):
    opt_small.zero_grad()
    logits = model_small(x_small)
    loss = torch.nn.functional.cross_entropy(logits, y_small)
    loss.backward()
    opt_small.step()
eval_small = evaluate(model_small, x_small, y_small)
print(f"Overfit 20 samples - Loss: {eval_small['loss']:.6f}, Acc: {eval_small['acc']:.4f}")
assert eval_small["acc"] == 1.0, f"Failed to overfit: Acc={eval_small['acc']}"
print("[PASS] Overfit 20 samples test passed!")

# 6. Test Gradient flow
model_grad = MLP(hidden=(256, 128), dropout=0.0, init="he").to(device)
logits = model_grad(x_small)
loss = torch.nn.functional.cross_entropy(logits, y_small)
loss.backward()
for name, p in model_grad.named_parameters():
    assert p.grad is not None and p.grad.norm().item() > 0, f"Grad issue at {name}"
print("[PASS] Gradient flow test passed!")
print("\n>>> ALL TESTS IN PART 0 & PART 1 COMPLETED SUCCESSFULLY! <<<")
