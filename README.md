# Bài Tập Dự Đoán Giá Nhà với L1 Regularization (Lasso)

Dự án minh họa hiện tượng **Overfitting** khi huấn luyện mô hình Hồi quy tuyến tính trên dữ liệu diện tích nhà (`Square_Feet`) và giá nhà (`House_Price`) với kích thước tập train cực nhỏ ($N_{train} = 4$), sau đó khắc phục bằng **L1 Regularization (Lasso Regression)**.

## 1. Cấu trúc thư mục
```text
house-price-lab/
├── data.csv          # Dữ liệu diện tích & giá nhà
├── main.py            # Mã nguồn chạy mô hình
└── README.md          # Hướng dẫn dự án