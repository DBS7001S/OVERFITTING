import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_results(X_train, y_train, X_val, y_val, model_of, model_reg, train_loss_of, val_loss_of, train_loss_reg, val_loss_reg):
    X_test = np.linspace(0, 1, 200).reshape(-1, 1)
    X_test_t = torch.FloatTensor(X_test)
    y_true = np.sin(2 * np.pi * X_test)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Overfitted Model Predictions
    model_of.eval()
    with torch.no_grad():
        y_pred_of = model_of(X_test_t).numpy()

    axes[0, 0].scatter(X_train, y_train, color='blue', label='Train Data')
    axes[0, 0].scatter(X_val, y_val, color='red', label='Val Data')
    axes[0, 0].plot(X_test, y_true, 'g--', label='True Function')
    axes[0, 0].plot(X_test, y_pred_of, 'r-', label='Overfitted Model')
    axes[0, 0].set_title("1. Overfitted Model (Học thuộc lòng nhiễu)")
    axes[0, 0].legend()

    # 2. Overfit Loss Curve
    axes[0, 1].plot(train_loss_of, label='Train Loss')
    axes[0, 1].plot(val_loss_of, label='Validation Loss')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title("2. Learning Curve (Overfit: Val Loss tăng)")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].legend()

    # 3. Regularized Model Predictions
    model_reg.eval()
    with torch.no_grad():
        y_pred_reg = model_reg(X_test_t).numpy()

    axes[1, 0].scatter(X_train, y_train, color='blue', label='Train Data')
    axes[1, 0].scatter(X_val, y_val, color='red', label='Val Data')
    axes[1, 0].plot(X_test, y_true, 'g--', label='True Function')
    axes[1, 0].plot(X_test, y_pred_reg, 'm-', label='Regularized Model')
    axes[1, 0].set_title("3. Regularized Model (Đã mượt hóa)")
    axes[1, 0].legend()

    # 4. Regularized Loss Curve
    axes[1, 1].plot(train_loss_reg, label='Train Loss')
    axes[1, 1].plot(val_loss_reg, label='Validation Loss')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title("4. Learning Curve (Đã xử lý: Loss ổn định)")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()
