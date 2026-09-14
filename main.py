import torch
import torch.nn as nn
import torch.optim as optim

from data.generate_data import generate_and_prepare_data
from models.overfitting_net import OverfittedNet
from models.regularized_net import RegularizedNet
from utils.visualize import plot_results

def train_model(model, X_train_t, y_train_t, X_val_t, y_val_t, weight_decay=0.0, epochs=2500, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t)
            
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
    return train_losses, val_losses

def main():
    print("--- 1. Đang sinh dữ liệu (Data Generation) ---")
    (X_train, X_val, y_train, y_val), (X_train_t, y_train_t, X_val_t, y_val_t) = generate_and_prepare_data()
    
    print("--- 2. Huấn luyện mô hình Overfit (No Regularization) ---")
    model_of = OverfittedNet()
    t_loss_of, v_loss_of = train_model(model_of, X_train_t, y_train_t, X_val_t, y_val_t, weight_decay=0.0)
    
    print("--- 3. Huấn luyện mô hình đã xử lý (L2 Weight Decay + Dropout) ---")
    model_reg = RegularizedNet(dropout_rate=0.2)
    t_loss_reg, v_loss_reg = train_model(model_reg, X_train_t, y_train_t, X_val_t, y_val_t, weight_decay=0.01)
    
    print("--- 4. Trực quan hóa kết quả ---")
    plot_results(X_train, y_train, X_val, y_val, model_of, model_reg, t_loss_of, v_loss_of, t_loss_reg, v_loss_reg)

if __name__ == "__main__":
    main()
