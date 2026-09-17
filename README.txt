# Thực Nghiệm Overfitting Trong Hồi Quy Tuyến Tính (Linear Regression)

Dự án minh họa hiện tượng **Overfitting** khi huấn luyện mô hình Hồi quy tuyến tính bằng công thức đại số ma trận trên tập dữ liệu CSV có kích thước mẫu cực nhỏ ($N_{train} = 5$), đồng thời áp dụng **L2 Regularization (Ridge Regression)** để khắc phục.

## 1. Cấu trúc dự án
```text
linear-overfitting-lab/
├── data.csv              # Dữ liệu Doanh số game & Điểm đánh giá
├── main.py                # Mã nguồn chính thực thi toán ma trận
├── requirements.txt       # Danh sách thư viện
└── README.md              # Hướng dẫn chi tiết