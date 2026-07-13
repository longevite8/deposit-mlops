import pandas as pd
from config import (
    TARGET_COLUMN,
    MIN_VALUE,
    MAX_VALUE,
    MAX_MISSING_RATE,
    REQUIRED_COLUMNS,
)


def validate_data(df: pd.DataFrame) -> dict:
    # Kiểm tra các điều kiện dữ liệu

    msg = ""
    # =====================================================
    # Schema validation
    # =====================================================

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    schema_ok = len(missing_columns) == 0
    schema_validate = {
        "schema_ok": schema_ok,
        "missing_columns": missing_columns,
        "message": "Schema validation passed." if schema_ok else "",
    }

    if not schema_ok:
        schema_validate["message"] = (
            f"Schema validation failed: Missing columns {missing_columns}"
        )
        msg = msg + f". {schema_validate['message']}"

    # =====================================================
    # Missing value validation
    # =====================================================

    missing_rate = df[REQUIRED_COLUMNS].isna().mean()

    missing_ok = missing_rate.max() <= MAX_MISSING_RATE
    missing_validate = {
        "missing_ok": missing_ok,
        "missing_rate_max": missing_rate.max(),
        "message": "Missing value validation passed." if missing_ok else "",
    }

    if not missing_ok:
        missing_validate["message"] = (
            f". Missing value validation failed: {missing_rate[missing_rate > MAX_MISSING_RATE]}"
        )
        msg = msg + f". {missing_validate['message']}"

    # =====================================================
    # Dtype validation
    # =====================================================

    bad_columns = []

    for col in REQUIRED_COLUMNS:
        if str(df[col].dtype) not in (
            "int64",
            "float64",
            "bool",
        ):
            bad_columns.append(col)

    dtype_ok = len(bad_columns) == 0

    dtype_validate = {
        "dtype_ok": dtype_ok,
        "bad_columns": bad_columns,
        "message": "Dtype validation passed." if dtype_ok else "",
    }
    if not dtype_ok:
        dtype_validate["message"] = f"Dtype validation failed: {bad_columns}"
        msg = msg + f". {dtype_validate['message']}"

    # =====================================================
    # Range validation
    # =====================================================

    range_ok = df[TARGET_COLUMN].between(MIN_VALUE, MAX_VALUE).all()

    range_validate = {
        "range_ok": range_ok,
        "message": "Range validation passed." if range_ok else "",
    }
    if not range_ok:
        range_validate["message"] = (
            f"Range validation failed: {TARGET_COLUMN} contains values outside [{MIN_VALUE}, {MAX_VALUE}]"
        )
        msg = msg + f". {range_validate['message']}"

    # =====================================================
    # Final result
    # =====================================================

    passed = schema_ok and missing_ok and dtype_ok and range_ok

    return {
        "passed": passed,
        "schema_check": schema_validate,
        "missing_check": missing_validate,
        "dtype_check": dtype_validate,
        "range_check": range_validate,
        "message": msg,
    }
