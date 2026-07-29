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

Chạy lệnh sau và nhập thông tin từ ClearML Web UI (Profile -> Create new credentials):

```bash
clearml-agent init
```

Kết quả cấu hình sẽ được lưu tại `~/clearml.conf`.

### Bước 2.5: Cấu hình Package Manager (Pip) trong `clearml.conf`

Để tránh lỗi khi cài đặt dependencies, cần điều chỉnh section `package_manager` trong file `~/clearml.conf`:

```hocon
agent {
    package_manager {
        # Loại package manager (pip, conda, poetry, uv)
        type: "pip"
        
        # QUAN TRỌNG: Phải để danh sách rỗng để tắt cờ --use-deprecated=legacy-resolver
        # (Cờ này không tồn tại trên Pip >= 26.x)
        pip_legacy_resolver: []
        
        # Để trống để dùng bản Pip mặc định trên hệ thống
        pip_version: ""
        
        # QUAN TRỌNG: Ép Agent dùng requirements.txt từ Repository
        # (Thay vì danh sách phụ thuộc tự động từ môi trường local)
        force_repo_requirements_txt: true
        
        # Tắt PyTorch resolver thông minh (để tránh detect CUDA version sai)
        pytorch_resolve: "none"
        
        # Chỉ định rõ index URL cho PyTorch và NVIDIA packages nếu máy Agent không có CUDA (chỉ CPU)
        extra_index_url: [
            "https://download.pytorch.org/whl/cpu"
        ]
        
        ## Chỉ định rõ index URL cho PyTorch và NVIDIA packages nếu máy Agent có CUDA (có GPU)
        # extra_index_url: [
        #     "https://download.pytorch.org/whl/cu124"
        # ]


        # Để trống để dùng Python mặc định
        python_binary: ""

        ...
    }
}
```

**Giải thích từng tham số:**

- **`pip_legacy_resolver: []`**: Bắt buộc. Nếu để có giá trị khác (ví dụ `[">=20.3,<24.3"]`), Agent sẽ chèn cờ `--use-deprecated=legacy-resolver` vào lệnh `pip install`. Cờ này được xóa từ Pip 26.x trở đi, gây lỗi `exit status 1`.
- **`force_repo_requirements_txt: true`**: Ép Agent dùng file `requirements.txt` chính xác từ Git Repository thay vì file temp ClearML tự tạo. Điều này tránh xung đột phiên bản thư viện.
- **`pytorch_resolve: "none"`**: Tắt logic auto-detect CUDA version của ClearML. Nếu để mặc định, Agent có thể detect sai version (ví dụ detect thành `cu0` thay vì `cu124`), dẫn đến tải bản Torch không tương thích.
- **`extra_index_url`**: Thêm URL PyPI index riêng để tìm các gói đặc biệt như `torch`, `nvidia-*` kernels.

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

# Agent dành cho tính toán (Compute) - Với Pip >= 26.x (Recommended)
clearml-agent daemon --queue cpu_queue --name "compute-worker-1"

# Agent dành cho tính toán (Compute) - Nếu gặp lỗi Pip, chạy lệnh này trước
export CLEARML_AGENT_PACKAGE_PIP_LEGACY_RESOLVER=""  # Tắt legacy resolver flag
clearml-agent daemon --queue cpu_queue --name "compute-worker-1"
```

**Lưu ý**: Trước khi khởi chạy Agent, hãy đảm bảo file `~/.clearml/clearml.conf` đã được cấu hình đúng (xem Bước 2.5 ở trên).

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

- **Lỗi `--use-deprecated=legacy-resolver` không được hỗ trợ (Pip >= 26.x)**:

  ```
  ERROR: Command '[..., 'pip', ..., '--use-deprecated=legacy-resolver']' returned non-zero exit status 1.
  ```

  **Nguyên nhân**: Tham số `pip_legacy_resolver` trong `clearml.conf` không để trống.
  **Khắc phục**: Sửa file `~/.clearml/clearml.conf`, tìm section `package_manager` và đảm bảo:

  ```hocon
  pip_legacy_resolver: []
  ```

  Sau đó restart Agent:

  ```bash
  pkill -f clearml-agent
  rm -rf ~/.clearml/venvs-builds
  clearml-agent daemon --queue cpu_queue
  ```

- **Lỗi "torch-2.13.0 is not a supported wheel on this platform"**:

  ```
  ERROR: torch-2.13.0-cp310-cp310m-linux_x86_64.whl is not a supported wheel on this platform.
  clearml_agent: ERROR: Could not download wheel name of "http://download.pytorch.org/whl/cu0/torch-..."
  ```

  **Nguyên nhân**: Agent detect sai CUDA version hoặc PyTorch resolver đang cố tự động chọn bản.
  **Khắc phục**: Cấu hình `clearml.conf`:

  ```hocon
  pytorch_resolve: "none"  # Tắt auto-resolve
  extra_index_url: [
      "https://download.pytorch.org/whl/cu124"  # Chỉ định đúng CUDA version
  ]
  ```

- **Lỗi "Parameter verbose should be of type int, got False"** (LightGBM >= 4.7):

  ```
  lightgbm.basic.LightGBMError: Parameter verbose should be of type int, got "False"
  ```

  **Nguyên nhân**: LightGBM 4.7+ yêu cầu `verbose` phải là `int`, không phải boolean.
  **Khắc phục**: Đã được sửa trong [business/models/lightgbm_model.py](../business/models/lightgbm_model.py). Đảm bảo file được cập nhật.

- **Agent Offline**: Kiểm tra kết nối tới server bằng `curl http://<clearml-server-url>:8080`. Restart daemon nếu cần.

- **Wrong Python Version**: Nếu task yêu cầu phiên bản Python cụ thể, khởi chạy agent với:

  ```bash
  clearml-agent daemon --queue cpu_queue --python /usr/bin/python3.10
  ```

- **Lỗi "ModuleNotFoundError: No module named pip"**:

  ```
  /home/user/.clearml/venvs-builds/.../bin/python: No module named pip
  ```

  **Nguyên nhân**: Venv do ClearML tạo không có `pip`. Thường xảy ra sau khi dùng `uv` hoặc seed-based venv.
  **Khắc phục**: Xóa cache venv và tạo lại:

  ```bash
  rm -rf ~/.clearml/venvs-builds
  rm -rf ~/.clearml/venvs-cache
  clearml-agent daemon --queue cpu_queue
  ```

## 6. Tối ưu hóa hiệu suất

- **Worker Threads**: Chỉnh sửa `worker_threads` trong `clearml.conf` để cho phép một Agent xử lý nhiều Task nhỏ đồng thời (chủ yếu cho `services` queue).
- **Shallow Clone**: Thêm `--git-clone-depth 1` khi chạy daemon để tăng tốc độ tải mã nguồn đối với các Repository có lịch sử commit lớn.
