import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

CSV_FILE = "data.csv"
FEATURE_COLS = ["Square_Feet"]
TARGET_COL = "House_Price"
N_TRAIN = 4          # Cố tình dùng 4 mẫu train để minh họa Overfitting
RANDOM_SEED = 42

# 1. Đọc dữ liệu
df = pd.read_csv(CSV_FILE)
X = df[FEATURE_COLS].values.astype(float)   # (N, d)
y = df[TARGET_COL].values.astype(float)     # (N,)

# 2. Chia tập Train / Test
rng = np.random.default_rng(RANDOM_SEED)
N = len(y)
indices = rng.permutation(N)
train_idx, test_idx = indices[:N_TRAIN], indices[N_TRAIN:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# 3. Chuẩn bị ma trận có hằng số Bias (cho mô hình gốc)
ones_train = np.ones((1, len(y_train)))
Xt_train = np.vstack([ones_train, X_train.T])           # (d+1, N_train)

ones_test = np.ones((1, len(y_test)))
Xt_test = np.vstack([ones_test, X_test.T])             # (d+1, N_test)

# =========================================================================
# PHẦN 1: MÔ HÌNH GỐC (KHÔNG REGULARIZATION -> BỊ OVERFIT)
# =========================================================================
w_overfit = np.linalg.pinv(Xt_train @ Xt_train.T) @ Xt_train @ y_train

y_train_pred_of = w_overfit @ Xt_train
train_mse_of = np.mean((y_train - y_train_pred_of) ** 2)

y_test_pred_of = w_overfit @ Xt_test
test_mse_of = np.mean((y_test - y_test_pred_of) ** 2)

print("--- 1. Mô Hình Dự Đoán Giá Nhà Gốc (Không Regularization) ---")
print(f"Trọng số w (Bias, Slope) = {w_overfit}")
print(f"Train MSE                = {train_mse_of:.4f}")
print(f"Test MSE                 = {test_mse_of:.4f}")
print("-> Nhận xét: Test MSE rất lớn do học thuộc lòng trên 4 căn nhà tập Train.\n")

# =========================================================================
# PHẦN 2: KHẮC PHỤC BẰNG L1 REGULARIZATION (LASSO REGRESSION)
# =========================================================================
ALPHA = 10.0  # Hệ số phạt L1

lasso_model = Lasso(alpha=ALPHA, random_state=RANDOM_SEED)
lasso_model.fit(X_train, y_train)

y_train_pred_l1 = lasso_model.predict(X_train)
y_test_pred_l1 = lasso_model.predict(X_test)

train_mse_l1 = np.mean((y_train - y_train_pred_l1) ** 2)
test_mse_l1 = np.mean((y_test - y_test_pred_l1) ** 2)

print(f"--- 2. Mô Hình Đã Xử Lý (L1 Regularization - Lasso, alpha = {ALPHA}) ---")
print(f"Hệ số góc (Slope)        = {lasso_model.coef_}")
print(f"Hệ số chặn (Bias)        = {lasso_model.intercept_:.4f}")
print(f"Train MSE                = {train_mse_l1:.4f}")
print(f"Test MSE                 = {test_mse_l1:.4f}")
print("-> Kết quả: Test MSE giảm rõ rệt, mô hình dự đoán giá nhà thực tế hơn!")