# Giám sát và Drift Detection

## 1. Thuật toán Data Drift

Hệ thống sử dụng kiểm định **Kolmogorov-Smirnov (KS)** để so sánh hai phân phối dữ liệu cho từng đặc trưng.

- Nếu tỷ lệ các đặc trưng bị Drift (`drift_ratio`) > `DRIFT_RATIO_THRESHOLD`, trạng thái Drift sẽ được báo là `FAIL`.

## 2. Quality Gates

Hệ thống áp dụng các ngưỡng cứng (Static Thresholds) trong `config.py`:

- `MAPE_THRESHOLD`: Sai số phần trăm tuyệt đối trung bình tối đa cho phép.
- `R2_THRESHOLD`: Độ phù hợp mô hình tối thiểu.

## 3. SMTP Alerting

Giao thức gửi mail được thực hiện qua `business/alerting.py`.

- **Nội dung Alert**: Bao gồm bảng so sánh Metrics, lý do cảnh báo (Performance Drop hoặc Data Drift) và đường dẫn đến Task trên ClearML UI.
