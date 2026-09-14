# Bài Tập Thực Nghiệm Overfitting & Cách Khắc Phục

Dự án mô phỏng hiện tượng Overfitting trong Machine Learning/Deep Learning trên tập dữ liệu hàm sóng Sin có thêm nhiễu (Noise).

## 1. Cấu trúc dự án
```text
ml-overfitting-lab/
├── data/
│   └── generate_data.py
├── models/
│   ├── overfitting_net.py
│   └── regularized_net.py
├── utils/
│   └── visualize.py
├── main.py
├── requirements.txt
└── README.md
```

## 2. Hướng dẫn chạy
1. Cài đặt môi trường & thư viện:
   ```bash
   pip install -r requirements.txt
   ```
2. Chạy chương trình:
   ```bash
   python main.py
   ```
