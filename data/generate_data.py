import numpy as np
import torch
from sklearn.model_selection import train_test_split

def generate_and_prepare_data(num_samples=30, test_size=0.4, noise_std=0.2, seed=42):
    """
    Tạo dữ liệu giả lập từ hàm y = sin(2 * pi * x) + noise
    Giới hạn số lượng mẫu nhỏ (30 mẫu) để gây ra hiện tượng Overfitting.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    X = np.linspace(0, 1, num_samples).reshape(-1, 1)
    y = np.sin(2 * np.pi * X) + np.random.normal(0, noise_std, X.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Chuyển đổi sang PyTorch Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    return (X_train, X_val, y_train, y_val), (X_train_t, y_train_t, X_val_t, y_val_t)
