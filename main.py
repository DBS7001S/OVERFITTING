import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso

CSV_FILE = "data.csv"
FEATURE_COLS = ["Square_Feet", "Bedrooms", "House_Age"]
TARGET_COL = "Price"
N_TRAIN = 4          # Cố tình dùng ít mẫu train để tạo hiện tượng Overfitting
RANDOM_SEED = 42

# 1. Đọc dữ liệu
df = pd.read_csv(CSV_FILE)
X = df[FEATURE_COLS].values.astype(float)
y = df[TARGET_COL].values.astype(float)

# 2. Chia tập Train / Test
rng = np.random.default_rng(RANDOM_SEED)
indices = rng.permutation(len(y))
train_idx, test_idx = indices[:N_TRAIN], indices[N_TRAIN:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# 3. Chuẩn hóa dữ liệu (Feature Scaling) - Rất quan trọng đối với L1/L2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================================
# PHẦN A: MÔ HÌNH HỒI QUY THƯỜNG (OLS - KHÔNG REGULARIZATION)
# =========================================================================
ols_model = LinearRegression()
ols_model.fit(X_train_scaled, y_train)

y_train_ols = ols_model.predict(X_train_scaled)
y_test_ols = ols_model.predict(X_test_scaled)

train_mse_ols = np.mean((y_train - y_train_ols) ** 2)
test_mse_ols = np.mean((y_test - y_test_ols) ** 2)

print("--- 1. Mô Hình Gốc (Không Regularization) ---")
print(f"Hệ số góc (Coefficients) = {ols_model.coef_}")
print(f"Train MSE                 = {train_mse_ols:.4f}")
print(f"Test MSE                  = {test_mse_ols:.4f}")
print("-> Nhận xét: Test MSE rất cao do mô hình bị Overfit trên 4 mẫu train.\n")

# =========================================================================
# PHẦN B: MÔ HÌNH L1 REGULARIZATION (LASSO REGRESSION)
# =========================================================================
ALPHA = 5.0  # Hệ số phạt L1 (lambda)

lasso_model = Lasso(alpha=ALPHA, random_state=RANDOM_SEED)
lasso_model.fit(X_train_scaled, y_train)

y_train_lasso = lasso_model.predict(X_train_scaled)
y_test_lasso = lasso_model.predict(X_test_scaled)

train_mse_lasso = np.mean((y_train - y_train_lasso) ** 2)
test_mse_lasso = np.mean((y_test - y_test_lasso) ** 2)

print(f"--- 2. Mô Hình Dùng L1 Regularization (Lasso, alpha = {ALPHA}) ---")
print(f"Hệ số góc (Coefficients) = {lasso_model.coef_}")
print(f"Train MSE                 = {train_mse_lasso:.4f}")
print(f"Test MSE                  = {test_mse_lasso:.4f}")
print("-> Kết quả: L1 giúp kéo bớt trọng số thừa về 0, giảm Test MSE đáng kể!")