# Quy trình Huấn luyện (Training Pipeline)

Pipeline này thực hiện việc tìm kiếm và nâng cấp mô hình mới theo trình tự:

1. **Extract Data**: Lấy dữ liệu thô từ nguồn và tạo Dataset.
2. **Feature Engineering**: Chuyển đổi dữ liệu và xử lý đặc trưng.
3. **Validate Data**: Kiểm tra integrity của dữ liệu.
4. **Drift Detection**: So sánh dữ liệu mới với dữ liệu nền của mô hình hiện tại.
5. **HPO (Hyperparameter Optimization)**: Dùng Optuna để tìm bộ tham số tốt nhất.
6. **Train Model**: Huấn luyện LightGBM với tham số vừa tìm được.
7. **Evaluate Model**: Tính toán MAPE, R2 trên tập Test.
8. **Register Model**: Đăng ký mô hình lên hệ thống nếu vượt qua Quality Gate.
9. **Explain Model**: Sử dụng SHAP để giải thích tầm quan trọng của các đặc trưng.
10. **Compare Champion**: So sánh trực tiếp Candidate Model với Champion hiện tại.
11. **Promote Champion**: Nếu Candidate thắng, thực hiện đổi Tag trên Model Registry.
