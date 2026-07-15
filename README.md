# Deposit Forecasting MLOps Platform

Hệ thống MLOps toàn diện cho dự báo dòng tiền (Cashflow Forecasting), được xây dựng trên nền tảng **ClearML**. Dự án áp dụng thiết kế **Champion/Challenger**, đảm bảo tính ổn định, khả năng truy vết (Lineage) và tự động hóa hoàn toàn vòng đời của mô hình AI.

## 🚀 Tính năng nổi bật

* **Champion/Challenger Workflow**: Sử dụng Registry Singleton để quản lý mô hình Production một cách deterministic.
* **Two-Artifact Lineage**: Mọi Task đều xuất bản `summary` (kết quả) và `lineage` (nguồn gốc) để truy vết 100%.
* **Data-Centric**: Kiểm soát chất lượng qua Validate Data và KS-Drift Detection.
* **Auto-Retraining**: Tự động kích hoạt Training Pipeline khi phát hiện rò rỉ hiệu năng hoặc Drift dữ liệu.
* **Explainable AI (XAI)**: Tích hợp SHAP TreeExplainer để giải thích quyết định của mô hình.

## 🛠 Cấu trúc dự án

```text
├── business/           # Logic nghiệp vụ (Training, HPO, Eval, Alert...)
├── pipelines/          # Định nghĩa Training & Production Pipeline
├── tasks/              # Các Task đơn lẻ (Extract, Feature, Train...)
├── helpers/            # Tiện ích chung (ClearML utils, wait functions)
├── docs/               # Tài liệu kỹ thuật chi tiết
├── config.py           # Cấu hình hệ thống (Project names, Thresholds)
└── register_templates.py # Đăng ký Task Templates lên ClearML
```

## 🏁 Hướng dẫn nhanh (Quick Start)

Để triển khai hệ thống, vui lòng thực hiện theo trình tự sau:

1. **Cài đặt môi trường**:

    Cung cấp nội dung: GIT_REPO của Project trong file `.env.example`, sau đó chạy bash dưới đây:

    ```bash
    python -m venv .venv
    source .venv/bin/activate 
    pip install -r requirements.txt
    cp .env.example .env 
    ```

2. **Cấu hình dự án**:

    Chỉnh sửa file `config.py` để định nghĩa:

    * `PROJECT_TEMPLATE`: Tên dự án dùng để chứa các Task Templates.
    * `PROJECT_DATASET`: Tên dự án chuyên lưu trữ các version Dataset.
    * Các ngưỡng (Thresholds): `MAPE_THRESHOLD`, `DRIFT_RATIO_THRESHOLD`.
    * Cấu hình SMTP: cho chức năng tự động gửi email cảnh báo mô hình suy thoái.

3. **Thiết lập ClearML**: Xem [Hạ tầng & Agent](docs/02_infrastructure.md).

4. **Đăng ký Templates**:
    Trước khi chạy Pipeline lần đầu, bạn phải đăng ký các task templates lên ClearML Server, lệnh dưới đây sẽ tạo ra các Task ở trạng thái `Draft`. Pipeline Controller sẽ sử dụng IDs của các Task này để nhân bản (Clone) khi thực thi:

    ```bash
    python register_templates.py

    ```

    Tiếp theo cần:
    * Copy các template IDs được in từ lệnh trên gán cho các giá trị của các biến tương ứng có dạng `TEMPLATE_<TASK>_ID` trong file `config.py`.
    * Sau đó cần commit và push những thay đổi này của file `config.py` lên Git

    **Lưu ý bước này chỉ chạy 1 lần duy nhất.**

5. **Chạy Training Pipeline**:

    ```bash
    python -m pipelines.training_pipeline
    ```

    Tiêp theo cần:
    * Copy task ID của pipeline được in từ lệnh gán cho giá trị của biến `TRAINING_PIPELINE_ID` trong file `config.py`.
    * Sau đó cần commit và push thay đổi này của file `config.py` lên Git

6. **Chạy Production Pipeline**:

    ```bash
    python -m pipelines.production_pipeline
    ```

    **Lưu ý: bước auto retrain pipeline (nếu có) sẽ được clone từ training pipeline ở bước trên qua `TRAINING_PIPELINE_ID`.**

## 📚 Tài liệu chi tiết

Vui lòng đọc tài liệu theo thứ tự dưới đây để hiểu rõ hệ thống:

1. [Kiến trúc hệ thống](docs/01_architecture.md)
2. [Hạ tầng & Agent](docs/02_infrastructure.md)
3. [Quy trình Huấn luyện (Training Pipeline)](docs/03_training_pipeline.md)
4. [Quy trình Vận hành (Production Pipeline)](docs/04_production_pipeline.md)
5. [Giám sát & Drift](docs/05_monitoring_drift.md)
6. [Xóa toàn bộ tài nguyên của dự án](docs/06_delete_project.md)
