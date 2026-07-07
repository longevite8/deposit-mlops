"""
ClearML Helper Utilities — Xử lý các tác vụ phổ biến trong MLOps Pipeline.
Giải quyết vấn đề Race Condition và đảm bảo Artifact/Metadata sẵn sàng.
"""

import time
from typing import Any, Optional
from clearml import Task


def wait_for_artifact(
    task_obj: Task,
    artifact_name: str,
    max_retries: int = 10,
    wait_interval: float = 2.0,
    logger_obj: Optional[Task] = None,
) -> Any:
    """
    Đợi cho đến khi artifact của Task cha sẵn sàng và không rỗng.

    Hàm này sẽ thực hiện retry logic với exponential backoff để xử lý
    hiện tượng Race Condition khi Task cha vừa mới finalize Dataset/Model.

    Args:
        task_obj (Task): Task object của Task cha
        artifact_name (str): Tên artifact cần chờ (ví dụ: "raw_dataset_id")
        max_retries (int): Số lần thử lại tối đa (mặc định: 10 lần)
        wait_interval (float): Thời gian chờ giữa các lần thử (giây). Mặc định: 2s
        logger_obj (Task, optional): Task hiện tại để log thông báo. Nếu None, in console.

    Returns:
        Any: Giá trị của artifact nếu thành công

    Raises:
        ValueError: Nếu timeout hoặc artifact không tìm thấy

    Example:
        >>> feature_task = Task.get_task(task_id="abc123")
        >>> dataset_id = wait_for_artifact(
        ...     feature_task,
        ...     "feature_dataset_id",
        ...     max_retries=10,
        ...     logger_obj=current_task
        ... )
    """

    def _log(message: str):
        """Helper function để log thông báo."""
        if logger_obj:
            logger_obj.get_logger().report_text(message)
        else:
            print(message)

    for attempt in range(max_retries):
        try:
            # Làm mới dữ liệu từ Server
            task_obj.reload()

            # Kiểm tra xem artifact có tồn tại không
            if artifact_name in task_obj.artifacts:
                artifact_value = task_obj.artifacts[artifact_name].get()

                # Kiểm tra artifact không rỗng
                if artifact_value is not None:
                    _log(
                        f"✅ Artifact '{artifact_name}' ready on attempt {attempt + 1}/{max_retries}"
                    )
                    return artifact_value
                else:
                    _log(
                        f"⏳ Artifact '{artifact_name}' is None (attempt {attempt + 1}/{max_retries})"
                    )
            else:
                _log(
                    f"⏳ Artifact '{artifact_name}' not found yet (attempt {attempt + 1}/{max_retries})"
                )

        except Exception as e:
            _log(
                f"⚠️ Error retrieving artifact '{artifact_name}' (attempt {attempt + 1}/{max_retries}): {str(e)}"
            )

        # Nếu không phải lần cuối cùng, chờ trước khi retry
        if attempt < max_retries - 1:
            time.sleep(wait_interval)

    # Nếu vẫn không lấy được sau tất cả retry
    raise ValueError(
        f"❌ Artifact '{artifact_name}' not ready after {max_retries} attempts "
        f"({max_retries * wait_interval} seconds total)"
    )


def wait_for_metadata(
    model_obj: Any,
    metadata_key: str,
    max_retries: int = 10,
    wait_interval: float = 2.0,
    logger_obj: Optional[Task] = None,
) -> Any:
    """
    Đợi cho đến khi metadata của Model/Task sẵn sàng.

    Hàm này giúp xử lý trường hợp Model vừa mới được register/promote,
    metadata chưa kịp được index trên backend.

    Args:
        model_obj: Model object hoặc dict metadata
        metadata_key (str): Khóa cần lấy từ metadata (ví dụ: "feature_dataset_id")
        max_retries (int): Số lần thử lại tối đa
        wait_interval (float): Thời gian chờ giữa các lần thử (giây)
        logger_obj (Task, optional): Task hiện tại để log thông báo

    Returns:
        Any: Giá trị của metadata nếu thành công

    Raises:
        ValueError: Nếu timeout hoặc metadata key không tìm thấy

    Example:
        >>> model = Model(model_id="xyz789")
        >>> feature_id = wait_for_metadata(
        ...     model,
        ...     "feature_dataset_id",
        ...     logger_obj=current_task
        ... )
    """

    def _log(message: str):
        """Helper function để log thông báo."""
        if logger_obj:
            logger_obj.get_logger().report_text(message)
        else:
            print(message)

    for attempt in range(max_retries):
        try:
            # Nếu là Model object, gọi get_all_metadata()
            if hasattr(model_obj, "get_all_metadata"):
                metadata = model_obj.get_all_metadata()
            else:
                # Nếu là dict, dùng trực tiếp
                metadata = model_obj

            # Kiểm tra metadata key tồn tại và có giá trị
            if metadata_key in metadata:
                value_obj = metadata[metadata_key]

                # Xử lý cấu trúc {"value": "..."} từ ClearML
                if isinstance(value_obj, dict) and "value" in value_obj:
                    metadata_value = value_obj["value"]
                else:
                    metadata_value = value_obj

                if metadata_value is not None and metadata_value != "":
                    _log(
                        f"✅ Metadata '{metadata_key}' ready on attempt {attempt + 1}/{max_retries}"
                    )
                    return metadata_value
                else:
                    _log(
                        f"⏳ Metadata '{metadata_key}' is empty (attempt {attempt + 1}/{max_retries})"
                    )
            else:
                _log(
                    f"⏳ Metadata key '{metadata_key}' not found (attempt {attempt + 1}/{max_retries})"
                )

        except Exception as e:
            _log(
                f"⚠️ Error retrieving metadata '{metadata_key}' (attempt {attempt + 1}/{max_retries}): {str(e)}"
            )

        # Nếu không phải lần cuối cùng, chờ trước khi retry
        if attempt < max_retries - 1:
            time.sleep(wait_interval)

    # Nếu vẫn không lấy được sau tất cả retry
    raise ValueError(
        f"❌ Metadata key '{metadata_key}' not ready after {max_retries} attempts "
        f"({max_retries * wait_interval} seconds total)"
    )


def wait_for_task_completion(
    task_obj: Task,
    max_retries: int = 30,
    wait_interval: float = 1.0,
    logger_obj: Optional[Task] = None,
) -> bool:
    """
    Đợi cho đến khi Task hoàn thành 100% (status = completed).

    Hàm này cho phép một Task con chủ động chờ Task cha hoàn tất
    trước khi bắt đầu xử lý (thay vì dựa hoàn toàn vào PipelineController).

    Args:
        task_obj (Task): Task object cần chờ
        max_retries (int): Số lần check tối đa
        wait_interval (float): Thời gian chờ giữa các lần check (giây)
        logger_obj (Task, optional): Task hiện tại để log thông báo

    Returns:
        bool: True nếu Task completed, False nếu failed/aborted

    Raises:
        TimeoutError: Nếu Task vẫn chưa completed sau timeout

    Example:
        >>> parent_task = Task.get_task(task_id="parent_123")
        >>> wait_for_task_completion(parent_task, logger_obj=current_task)
    """

    def _log(message: str):
        if logger_obj:
            logger_obj.get_logger().report_text(message)
        else:
            print(message)

    for attempt in range(max_retries):
        try:
            task_obj.reload()
            status = task_obj.status

            if status == "completed":
                _log("✅ Parent task completed successfully")
                return True
            elif status in ["failed", "aborted"]:
                _log(f"❌ Parent task {status}")
                return False
            else:
                _log(
                    f"⏳ Parent task status: {status} (check {attempt + 1}/{max_retries})"
                )

        except Exception as e:
            _log(f"⚠️ Error checking task status: {str(e)}")

        if attempt < max_retries - 1:
            time.sleep(wait_interval)

    raise TimeoutError(
        f"Task did not complete after {max_retries * wait_interval} seconds"
    )


def safe_get_artifact_then_metadata(
    task_obj: Task,
    artifact_name: str,
    fallback_metadata_key: Optional[str] = None,
    max_retries: int = 10,
    wait_interval: float = 2.0,
    logger_obj: Optional[Task] = None,
) -> Any:
    """
    Lấy Artifact trước, nếu không có thì fallback sang Metadata.

    Hàm này rất hữu ích khi Task cha có thể lưu dữ liệu dưới dạng
    Artifact hoặc Metadata tùy theo ngữ cảnh.

    Args:
        task_obj (Task): Task object
        artifact_name (str): Tên artifact ưu tiên
        fallback_metadata_key (str, optional): Key metadata nếu artifact không có
        max_retries (int): Số lần retry tối đa
        wait_interval (float): Thời gian chờ giữa retry
        logger_obj (Task, optional): Task để log

    Returns:
        Any: Giá trị artifact hoặc metadata

    Raises:
        ValueError: Nếu không tìm được artifact và metadata
    """

    def _log(message: str):
        if logger_obj:
            logger_obj.get_logger().report_text(message)
        else:
            print(message)

    # Thử lấy Artifact trước
    try:
        artifact_value = wait_for_artifact(
            task_obj,
            artifact_name,
            max_retries=max_retries,
            wait_interval=wait_interval,
            logger_obj=logger_obj,
        )
        return artifact_value
    except ValueError as e:
        _log(f"⚠️ Artifact '{artifact_name}' failed: {str(e)}")

    # Nếu artifact không có, fallback sang metadata
    if fallback_metadata_key:
        try:
            _log(f"🔄 Fallback to metadata key '{fallback_metadata_key}'")
            metadata_value = wait_for_metadata(
                task_obj,
                fallback_metadata_key,
                max_retries=max_retries,
                wait_interval=wait_interval,
                logger_obj=logger_obj,
            )
            return metadata_value
        except ValueError as e:
            _log(f"❌ Metadata '{fallback_metadata_key}' also failed: {str(e)}")
            raise ValueError(
                f"Could not retrieve '{artifact_name}' (artifact) or '{fallback_metadata_key}' (metadata)"
            )

    raise ValueError(
        f"Artifact '{artifact_name}' not found and no fallback metadata key provided"
    )
