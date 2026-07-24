# Quy trình Vận hành (Production Pipeline)

Pipeline này chạy định kỳ (Scheduled) để thực hiện dự báo và giám sát chất lượng mô hình:

1. **Extract & Feature**: Tương tự như quy trình huấn luyện nhưng trên dữ liệu thực tế mới nhất.
2. **Drift Detection**: Kiểm tra xem dữ liệu đầu vào hiện tại có bị thay đổi phân phối so với tập huấn luyện không.
3. **Inference Layer**: Tải **Champion Model** từ Registry và thực hiện dự báo Batch.
4. **Monitoring**: So sánh dự báo với giá trị thực tế (Ground Truth) khi có dữ liệu phản hồi.
5. **Alerting**: Gửi Email thông báo qua SMTP nếu các chỉ số vượt ngưỡng an toàn.
6. **Auto-Retraining**: Nếu điều kiện Retrain thỏa mãn, Task này sẽ tự động trigger một thực thể mới của **Training Pipeline**.

## ClearML Daily Scheduler

Sau khi chạy Production Pipeline thủ công thành công ít nhất một lần, lấy Task ID
của Pipeline Controller trên ClearML UI rồi cấu hình:

```bash
export PRODUCTION_PIPELINE_ID="<production-pipeline-controller-task-id>"
```

Đăng ký lịch chạy hằng ngày:

```bash
python -m scripts.py.schedule_production_pipeline
```

Mặc định scheduler sẽ:

- clone `PRODUCTION_PIPELINE_ID` và enqueue vào queue `mco-services`;
- chạy mỗi ngày lúc `19:00 UTC`, tương ứng `02:00 Asia/Ho_Chi_Minh`;
- bật `single_instance=True` để không tạo run mới nếu run trước vẫn đang chạy;
- enqueue scheduler controller remote trên queue `mco-services`.

Các biến cấu hình chính:

```bash
PRODUCTION_PIPELINE_ID="<production-pipeline-controller-task-id>"
PRODUCTION_SCHEDULER_QUEUE="mco-services"
PRODUCTION_SCHEDULER_DAILY_UTC_HOUR="19"
PRODUCTION_SCHEDULER_DAILY_UTC_MINUTE="0"
PRODUCTION_SCHEDULER_START_REMOTELY="true"
PRODUCTION_SCHEDULER_EXECUTE_IMMEDIATELY="false"
```

Nếu muốn kiểm thử bằng cách chạy ngay một lần sau khi đăng ký:

```bash
python -m scripts.py.schedule_production_pipeline \
  --pipeline_task_id "<production-pipeline-controller-task-id>" \
  --execute_immediately true
```

Máy/agent chạy scheduler cần ClearML credentials và phải có agent đang listen
queue `mco-services`.

Nếu một lần đăng ký scheduler bị fail trước đó, archive/stop task scheduler cũ
trên ClearML UI trước khi đăng ký lại để tránh nhiều scheduler controller cùng
tên.
