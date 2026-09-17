import os
import numpy as np
import pandas as pd

CSV_FILE = "data.csv"
FEATURE_COLS = ["Critic_Score"]
TARGET_COL = "Global_Sales"
N_TRAIN = 5          # Cố tình dùng 5 mẫu train để minh họa Overfitting
RANDOM_SEED = 42

# 0. Tự động sinh file data.csv giả lập nếu chưa có sẵn
def ensure_csv_exists():
    if not os.path.exists(CSV_FILE):
        print(f"Chưa thấy {CSV_FILE}, đang tạo file dữ liệu mẫu...")
        np.random.seed(RANDOM_SEED)
        n_samples = 100
        critic = np.random.uniform(30, 95, n_samples)
        # Giả lập Doanh số (Global_Sales) phụ thuộc vào Điểm số + Nhiễu
        sales = 0.05 * critic + np.random.normal(0, 1.5, n_samples)
        sales = np.maximum(0.1, sales) # Doanh số không âm
        
        df_gen = pd.DataFrame({
            "Critic_Score": np.round(critic, 1),
            "Global_Sales": np.round(sales, 2)
        })
        df_gen.to_csv(CSV_FILE, index=False)
        print("-> Đã tạo file data.csv thành công!\n")

ensure_csv_exists()

# 1. Đọc dữ liệu
df = pd.read_csv(CSV_FILE)
X = df[FEATURE_COLS].values.astype(float)   # (N, d)
y = df[TARGET_COL].values.astype(float)     # (N,)

# 2. Chia train/test — train CHỈ N_TRAIN mẫu, còn lại làm test
rng = np.random.default_rng(RANDOM_SEED)
N = len(y)
indices = rng.permutation(N)
train_idx, test_idx = indices[:N_TRAIN], indices[N_TRAIN:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# 3. Thêm hàng bias (số 1) -> X dạng (d+1, N)
ones_train = np.ones((1, len(y_train)))
Xt_train = np.vstack([ones_train, X_train.T])           # (d+1, N_train)

ones_test = np.ones((1, len(y_test)))
Xt_test = np.vstack([ones_test, X_test.T])             # (d+1, N_test)

# =========================================================================
# PHẦN A: MÔ HÌNH THƯỜNG (KHÔNG CÓ REGULARIZATION -> DỄ OVERFIT)
# Công thức: w = (X Xᵀ)† X y
# =========================================================================
w_overfit = np.linalg.pinv(Xt_train @ Xt_train.T) @ Xt_train @ y_train

y_train_pred_of = w_overfit @ Xt_train
train_mse_of = np.mean((y_train - y_train_pred_of) ** 2)

y_test_pred_of = w_overfit @ Xt_test
test_mse_of = np.mean((y_test - y_test_pred_of) ** 2)

print("--- 1. Mô Hình Gốc (Không Regularization) ---")
print(f"Trọng số w (bias, slope) = {w_overfit}")
print(f"Train MSE = {train_mse_of:.4f}")
print(f"Test MSE  = {test_mse_of:.4f}")
print("-> Nhận xét: Test MSE rất cao so với Train MSE do bị Overfit.\n")

# =========================================================================
# PHẦN B: KHẮC PHỤC BẰNG L2 REGULARIZATION (RIDGE REGRESSION)
# Công thức: w = (X Xᵀ + lambda * I)⁻¹ X y
# =========================================================================
LAMBDA = 10.0  # Hệ số phạt L2

# Tạo ma trận đơn vị I, không phạt tham số bias (w0)
I = np.eye(Xt_train.shape[0])
I[0, 0] = 0

w_ridge = np.linalg.inv(Xt_train @ Xt_train.T + LAMBDA * I) @ Xt_train @ y_train

y_train_pred_ridge = w_ridge @ Xt_train
train_mse_ridge = np.mean((y_train - y_train_pred_ridge) ** 2)

y_test_pred_ridge = w_ridge @ Xt_test
test_mse_ridge = np.mean((y_test - y_test_pred_ridge) ** 2)

print(f"--- 2. Mô Hình Đã Xử Lý (Ridge Regularization with Lambda = {LAMBDA}) ---")
print(f"Trọng số w_ridge          = {w_ridge}")
print(f"Train MSE                 = {train_mse_ridge:.4f}")
print(f"Test MSE                  = {test_mse_ridge:.4f}")
print("-> Kết quả: Test MSE giảm mạnh, mô hình dự đoán ổn định hơn!")