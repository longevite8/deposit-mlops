# Quy trình Vận hành (Production Pipeline)

Pipeline này chạy định kỳ (Scheduled) để thực hiện dự báo và giám sát chất lượng mô hình:

1. **Extract & Feature**: Tương tự như quy trình huấn luyện nhưng trên dữ liệu thực tế mới nhất.
2. **Drift Detection**: Kiểm tra xem dữ liệu đầu vào hiện tại có bị thay đổi phân phối so với tập huấn luyện không.
3. **Inference Layer**: Tải **Champion Model** từ Registry và thực hiện dự báo Batch.
4. **Monitoring**: So sánh dự báo với giá trị thực tế (Ground Truth) khi có dữ liệu phản hồi.
5. **Alerting**: Gửi Email thông báo qua SMTP nếu các chỉ số vượt ngưỡng an toàn.
6. **Auto-Retraining**: Nếu điều kiện Retrain thỏa mãn, Task này sẽ tự động trigger một thực thể mới của **Training Pipeline**.
