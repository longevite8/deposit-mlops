# Xóa tài nguyên của dự án

## 1. Tổng quan

Công cụ `del_clearml_project.py` được thiết kế để xóa triệt để các tài nguyên liên quan đến dự án trên ClearML Server. Script này sử dụng `APIClient` để can thiệp sâu vào hệ thống, đảm bảo không để lại các rác (artifacts) hoặc các Dataset bị khóa.

Sử dụng công cụ này khi:

* Cần xóa sạch môi trường để khởi tạo lại từ đầu của 1 dự án.
* Giải phóng dung lượng lưu trữ (Models và Datasets chiếm nhiều không gian nhất).
* Sắp xếp lại cấu trúc thư mục Project khi có sự thay đổi trong `config.py`.
* Chỉ nên áp dụng khi đang ở pha thử nghiệm dự án, khi chúng ta cần khởi tạo lại dự án từ đầu.

### Lưu ý an toàn quan trọng

**Không thể khôi phục**: Hành động này sẽ xóa vĩnh viễn dữ liệu trên ClearML Server. Hãy đảm bảo bạn đã sao lưu các thông tin quan trọng của dự án (như Champion Model ...) nếu cần.

---

## 2. Điều kiện tiên quyết

* Đã cài đặt ClearML SDK: `pip install clearml`.
* Tệp `config.py` phải chứa các thông tin định danh project chính xác:
  * `PROJECT_TEMPLATE`
  * `PROJECT_PIPELINE`
  * `PROJECT_DATASET`

---

## 3. Cách sử dụng

Chạy script trực tiếp từ thư mục gốc của dự án:

```bash
python del_clearml_project.py
```

**Quy trình thực thi:**

1. Script sẽ liệt kê các Project mục tiêu dựa trên cấu hình trong `config.py`.
2. Hệ thống yêu cầu xác nhận: `Are you sure you want to delete ALL resources in these projects? (y/N)`.
3. Nếu chọn `y`, script sẽ tiến hành xóa tuần tự.

---

## 4. Cơ chế hoạt động của Script

Script thực hiện xóa theo phân cấp từ thấp đến cao để tránh lỗi ràng buộc dữ liệu:

1. **Bước 1: Tìm kiếm Project**: Lấy danh sách tất cả các projects và lọc ra các project có tên bắt đầu bằng các prefix được định nghĩa trong `config.py`.
2. **Bước 2: Xóa Models**: Truy quét và xóa toàn bộ Models thuộc các projects mục tiêu.
3. **Bước 3: Xóa Tasks**: Xóa toàn bộ Tasks (bao gồm cả Experiments, Pipelines, và Datasets). Sử dụng flag `force=True` để cưỡng chế xóa các task đang chạy hoặc bị treo.
4. **Bước 4: Xóa Project Folder**: Sau khi đã sạch tài nguyên bên trong, script thực hiện xóa thư mục Project với `force=True` để giải phóng hoàn toàn ID trên hệ thống.

---

## 5. Cấu hình mục tiêu xóa

Danh sách các dự án bị ảnh hưởng được quản lý tập trung tại `config.py`. Script sẽ xóa bất kỳ dự án nào có tên **khớp hoặc bắt đầu bằng** các prefix sau:

* `PROJECT_TEMPLATE`: Chứa các bản mẫu công việc.
* `PROJECT_PIPELINE`: Chứa lịch sử chạy các quy trình tự động.
* `PROJECT_DATASET`: Chứa các phiên bản dữ liệu thô và dữ liệu đặc trưng.

---

## 6. Lưu ý an toàn quan trọng

* **Không thể khôi phục**: Hành động này sẽ xóa vĩnh viễn dữ liệu trên ClearML Server. Hãy đảm bảo bạn đã sao lưu các model quan trọng (Champion Model) nếu cần.
* **Xác nhận thủ công**: Script luôn yêu cầu nhập `y` để bắt đầu. Nếu bạn nhấn `Enter` hoặc nhập bất kỳ ký tự nào khác, quá trình sẽ bị hủy bỏ (Aborted).
* **Quyền hạn**: Tài khoản API cấu hình trong `clearml.conf` phải có quyền quản trị (Admin) hoặc quyền xóa đối với các dự án này.

---

## 7. Xử lý lỗi thường gặp

### Lỗi không xóa được Project Folder

Nếu bạn xóa thủ công trên UI thường gặp lỗi do dự án còn chứa Datasets ẩn.
**Giải pháp:** Script `del_clearml_project.py` đã xử lý việc này bằng cách gọi `client.tasks.delete(force=True)` trước khi xóa project, giúp loại bỏ các "Dataset Tasks" vốn là nguyên nhân gây khóa thư mục.

### Không tìm thấy Project

**Triệu chứng:** Thông báo `No matching projects found`.
**Giải pháp:** Kiểm tra lại biến `PROJECT_TEMPLATE`, `PROJECT_PIPELINE`, `PROJECT_DATASET` trong `config.py` xem có khớp với tên hiển thị trên ClearML Web UI hay không.

---

## 8. Tích hợp Pipeline

Sau khi dọn dẹp, bạn nên khởi tạo lại môi trường theo trình tự sau:

1. **Dọn dẹp**: `python del_clearml_project.py`
2. **Đăng ký lại**: `python register_templates.py`
3. **Chạy Pipeline**: `python -m pipelines.trainning_pipeline`
