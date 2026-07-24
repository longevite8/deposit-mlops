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
    * Cấu hình nguồn dữ liệu PostgreSQL trong `.env`: `DB_*`, `SOURCE_PROJECT_NAME`, `SOURCE_FLOW_TYPE`, `SOURCE_APPROVAL_STATUS`, `SOURCE_FROM_DATE`, `SOURCE_TO_DATE`.
    * Các ngưỡng (Thresholds): `MAPE_THRESHOLD`, `DRIFT_RATIO_THRESHOLD`.
    * Cấu hình SMTP: cho chức năng tự động gửi email cảnh báo mô hình suy thoái.

    Bước `Extract Data` hiện đọc dữ liệu từ bảng cashflow thật theo pattern của `vc-mco-mlops`, lọc theo project/flow/status/date window, sau đó aggregate theo ngày thành schema `date`, `cashflow`.

3. **Thiết lập ClearML**: Xem [Hạ tầng & Agent](docs/02_infrastructure.md).

4. **Đăng ký Templates**:
    Trước khi chạy Pipeline lần đầu, bạn phải đăng ký các task templates lên ClearML Server, lệnh dưới đây sẽ tạo ra các Task ở trạng thái `Draft`. Pipeline Controller sẽ sử dụng IDs của các Task này để nhân bản (Clone) khi thực thi:

    ```bash
    python register_templates.py

    ```

    Tiếp theo cần:
    * Script sẽ tự động ghi các template IDs mới vào file `.env` theo dạng `TEMPLATE_<TASK>_ID=...`.
    * Không commit các ClearML IDs mới vào `config.py`; file `config.py` chỉ giữ fallback legacy IDs.

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

7. **Tự động chạy Production Pipeline hằng ngày bằng ClearML Scheduler**:

    Sau khi bước 6 chạy thành công, copy Task ID của Production Pipeline Controller
    từ ClearML UI và cấu hình:

    ```bash
    export PRODUCTION_PIPELINE_ID="<production-pipeline-controller-task-id>"
    python -m scripts.py.schedule_production_pipeline
    ```

    Mặc định lịch chạy là `19:00 UTC`, tương ứng `02:00 Asia/Ho_Chi_Minh`,
    trên queue `mco-services`, với `single_instance=True`.

## 📚 Tài liệu chi tiết

Vui lòng đọc tài liệu theo thứ tự dưới đây để hiểu rõ hệ thống:

1. [Kiến trúc hệ thống](docs/01_architecture.md)
2. [Hạ tầng & Agent](docs/02_infrastructure.md)
3. [Quy trình Huấn luyện (Training Pipeline)](docs/03_training_pipeline.md)
4. [Quy trình Vận hành (Production Pipeline)](docs/04_production_pipeline.md)
5. [ClearML Serving Deployment](docs/07_serving_deployment.md)
6. [Giám sát & Drift](docs/05_monitoring_drift.md)
7. [Xóa toàn bộ tài nguyên của dự án](docs/06_delete_project.md)
