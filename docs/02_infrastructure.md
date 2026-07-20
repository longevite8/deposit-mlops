# Hạ tầng & ClearML Agent

Tài liệu này hướng dẫn chi tiết về cách thiết lập, cấu hình và quản trị hệ thống Agent để thực thi các Pipeline tự động.

## 1. Kiến trúc hàng đợi (Queue Architecture)

Hệ thống hoạt động theo mô hình **Producer-Consumer**:

- **Producer**: Pipeline Controller phân rã các bước và đẩy Task vào Queue tương ứng.
- **Queue**: Bộ đệm lưu trữ các Task đang chờ xử lý.
- **Consumer (Agent)**: Lắng nghe Queue, tải mã nguồn từ Git, khởi tạo môi trường và thực thi.

## 2. Thiết lập Agent (Agent Setup)

### Bước 1: Cài đặt

```bash
pip install clearml-agent
```

### Bước 2: Cấu hình Credentials

Máy chạy `python -m pipelines.training_pipeline`, `python -m pipelines.production_pipeline`, `register_templates.py`, hoặc ClearML Agent đều cần ClearML client credentials.

Chạy lệnh sau và nhập thông tin từ ClearML Web UI (Profile -> Create new credentials):

```bash
clearml-init
```

Kết quả cấu hình sẽ được lưu tại `~/.clearml/clearml.conf`.

Với ClearML Agent chạy bằng `pip`, cần đảm bảo agent dùng pip mới và cài đúng CPU wheel index cho PyTorch. Thêm hoặc cập nhật block sau trong `~/.clearml/clearml.conf` trên máy chạy agent:

```ini
agent {
  package_manager {
    pip_version: [">=24.3"]
    extra_index_url: ["https://download.pytorch.org/whl/cpu"]
    force_repo_requirements_txt: true
  }
}
```

Nếu không cấu hình `pip_version`, agent có thể tạo virtualenv với pip cũ (`pip<22.3`), dẫn tới lỗi PyPI `Content-Type: application/vnd.pypi.simple.v1+json`. Nếu không cấu hình requirements/index, agent có thể tự suy luận sai wheel PyTorch thành `cu0`.

Nếu muốn cấu hình qua `.env`, set đủ các biến sau:

```bash
CLEARML_API_ACCESS_KEY="your-access-key"
CLEARML_API_SECRET_KEY="your-secret-key"
CLEARML_API_HOST="http://<clearml-server-host>:8008"
CLEARML_WEB_HOST="http://<clearml-server-host>:8080"
CLEARML_FILES_HOST="http://<clearml-server-host>:8081"
CLEARML_SERVER_URL="http://<clearml-server-host>:8080"
```

## 3. Quản lý Queues & Worker

Dự án này sử dụng 2 loại Queue chính để tối ưu hóa tài nguyên:

| Tên Queue | Mục đích | Loại tài nguyên |
| :--- | :--- | :--- |
| `services` | Chạy Pipeline Controller, Automation logic, Auto-Retraining trigger. | Yêu cầu RAM thấp, chạy đa luồng (multi-workers). |
| `cpu_queue` | Thực thi các Task tính toán: Feature Engineering, Training, Drift Detection. | Yêu cầu CPU và RAM cao. |

### Lệnh khởi chạy Agent bằng bash

```bash
# Agent dành cho quản lý (Services) - Cho phép chạy 2-4 workers song song
clearml-agent daemon --queue services --workers 2

# Agent dành cho tính toán (Compute)
export CLEARML_ALERT_SMTP_PASSWORD="xxxxxx" # để chức năng Alerting hoạt động
clearml-agent daemon --queue cpu_queue --name "compute-worker-1"
```

### Thực thi Agent với Docker (Recommended)

Để đảm bảo tính nhất quán (Consistency) giữa các môi trường, nên sử dụng Docker Agent. Nó sẽ tự động dựng Container cho mỗi Task:

```bash
export CLEARML_ALERT_SMTP_PASSWORD="xxxxxx" # để chức năng Alerting hoạt động
clearml-agent daemon --queue cpu_queue --docker python:3.12-slim --docker-args "-v /mnt/data:/data"
```

## 4. Quản lý tài nguyên & Bảo mật

- **Git Access**: Agent cần quyền truy cập Git để tải code. Cấu hình PAT (Personal Access Token) trong `clearml.conf`:

  ```ini
  [agent.git]
  git_user = "your-username"
  git_password = "your-github-token"
  ```

- **SMTP**: Đảm bảo biến môi trường `CLEARML_ALERT_SMTP_PASSWORD` được set trên máy chạy Agent để chức năng Alerting hoạt động.
- **Dọn dẹp bộ nhớ**: Agent lưu cache các gói thư viện và Dataset. Định kỳ cần dọn dẹp bộ nhớ:

  ```bash
  # Xóa cache ClearML
  rm -rf ~/.clearml/cache/*
  # Dọn dẹp Docker images không dùng
  docker image prune -f
  ```

## 5. Xử lý sự cố thường gặp (Troubleshooting)

- **Lỗi "Failed reloading Task: missing id"**: Thường do Race Condition khi Task được tạo nhưng database chưa kịp commit. Giải pháp: Đã tích hợp `retry logic` và `initial delay` trong mã nguồn.
- **Agent Offline**: Kiểm tra kết nối tới server bằng `curl http://<clearml-server-url>:8080`. Restart daemon nếu cần.
- **Wrong Python Version**: Nếu task yêu cầu phiên bản Python cụ thể, khởi chạy agent với:

  ```bash
  clearml-agent daemon --queue cpu_queue --python /usr/bin/python3.12
  ```

## 6. Tối ưu hóa hiệu suất

- **Worker Threads**: Chỉnh sửa `worker_threads` trong `clearml.conf` để cho phép một Agent xử lý nhiều Task nhỏ đồng thời (chủ yếu cho `services` queue).
- **Shallow Clone**: Thêm `--git-clone-depth 1` khi chạy daemon để tăng tốc độ tải mã nguồn đối với các Repository có lịch sử commit lớn.
