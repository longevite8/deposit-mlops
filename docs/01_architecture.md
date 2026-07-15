# Kiến trúc hệ thống Deposit MLOps

Hệ thống được thiết kế dựa trên triết lý **Champion vs Challenger** và mô hình **Decoupled Architecture** (Tách biệt logic nghiệp vụ khỏi hạ tầng).

## 1. Champion Registry (Singleton Pattern)

Hệ thống sử dụng một Task duy nhất mang tên `Model Registry` để quản lý ID của mô hình đang hoạt động tại Production.

- **Champion**: Mô hình tốt nhất hiện tại, được gán tag `champion`.
- **Inference**: Bước dự báo luôn truy vấn Registry này để lấy đúng Model ID, đảm bảo tính nhất quán (Deterministic).

## 2. Two-Artifact Lineage Pattern

Mỗi Task trong Pipeline phải xuất bản hai loại Artifact:

- `{task_name}_summary`: Chứa kết quả tính toán, metadata, các chỉ số metrics chính.
- `{task_name}_lineage`: Chứa các tham chiếu (ID) của Task cha, Dataset đầu vào và Model sử dụng.
Điều này giúp hệ thống có khả năng truy vết (Traceability) 100%.

## 3. Data Exchange qua ClearML Dataset

Thay vì truyền file trực tiếp, hệ thống sử dụng **Dataset Versioning**:

1. Bước `Extract Data` tạo version Dataset thô.
2. Bước `Feature Engineering` kế thừa version đó và tạo version Dataset đặc trưng.
3. Các bước huấn luyện và đánh giá sẽ tải bản sao local của dataset để xử lý.
