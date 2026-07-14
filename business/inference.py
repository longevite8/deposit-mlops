import time
import numpy as np


def run_model_inference(model, X):
    """
    Thực hiện dự báo và đo lường hiệu năng xử lý.
    """
    start_time = time.time()
    prediction = model.predict(X)
    inference_time = time.time() - start_time

    # Tính latency (ms per sample)
    inference_latency_ms = (inference_time / len(X)) * 1000 if len(X) > 0 else 0

    return prediction, inference_time, inference_latency_ms


def calculate_prediction_statistics(prediction):
    """
    Tính toán các thông số thống kê cơ bản của bộ kết quả dự báo.
    """
    return {
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction)),
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
    }


def build_output_dataframe(original_df, prediction, column_name="prediction"):
    """
    Hợp nhất kết quả dự báo vào DataFrame gốc.
    """
    output_df = original_df.copy()
    output_df[column_name] = prediction
    return output_df
