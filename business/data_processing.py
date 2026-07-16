"""Database-backed raw cashflow extraction for deposit forecasting."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import pandas as pd
from sqlalchemy.engine import URL

from config import (
    DATE_COLUMN,
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_SCHEMA,
    DB_USER,
    SOURCE_APPROVAL_STATUS,
    SOURCE_FLOW_TYPE,
    SOURCE_FROM_DATE,
    SOURCE_PROJECT_NAME,
    SOURCE_TO_DATE,
    TARGET_COLUMN,
)


class DataExtractionError(RuntimeError):
    """Base error for raw cashflow extraction failures."""


class DataExtractionConfigError(DataExtractionError):
    """Required source database configuration is missing."""


class DataExtractionNotFoundError(DataExtractionError):
    """The source query succeeded but returned no rows."""


class DataExtractionDatabaseError(DataExtractionError):
    """The source query failed before extraction could complete."""


@dataclass(frozen=True)
class SourceConfig:
    """Source database and filter configuration."""

    db_user: str
    db_password: str
    db_host: str
    db_port: int
    db_name: str
    db_schema: str
    project_name: str
    flow_type: str
    approval_status: str
    from_date: str
    to_date: str


def get_source_config() -> SourceConfig:
    """Build source configuration from central config values."""

    config = SourceConfig(
        db_user=DB_USER,
        db_password=DB_PASSWORD,
        db_host=DB_HOST,
        db_port=DB_PORT,
        db_name=DB_NAME,
        db_schema=DB_SCHEMA,
        project_name=SOURCE_PROJECT_NAME,
        flow_type=SOURCE_FLOW_TYPE,
        approval_status=SOURCE_APPROVAL_STATUS,
        from_date=SOURCE_FROM_DATE,
        to_date=SOURCE_TO_DATE,
    )
    validate_source_config(config)
    return config


def validate_source_config(config: SourceConfig) -> None:
    """Fail fast when required DB/source fields are not configured."""

    required_values = {
        "DB_USER": config.db_user,
        "DB_PASSWORD": config.db_password,
        "DB_HOST": config.db_host,
        "DB_NAME": config.db_name,
        "DB_SCHEMA": config.db_schema,
        "SOURCE_PROJECT_NAME": config.project_name,
        "SOURCE_FLOW_TYPE": config.flow_type,
        "SOURCE_APPROVAL_STATUS": config.approval_status,
        "SOURCE_FROM_DATE": config.from_date,
        "SOURCE_TO_DATE": config.to_date,
    }
    missing = [name for name, value in required_values.items() if not str(value).strip()]
    if missing:
        raise DataExtractionConfigError(
            "Missing required source configuration: " + ", ".join(missing)
        )

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", config.db_schema):
        raise DataExtractionConfigError(
            "DB_SCHEMA must be a simple SQL identifier such as 'public'."
        )

    from_date = pd.to_datetime(config.from_date, errors="coerce")
    to_date = pd.to_datetime(config.to_date, errors="coerce")
    if pd.isna(from_date) or pd.isna(to_date):
        raise DataExtractionConfigError(
            "SOURCE_FROM_DATE and SOURCE_TO_DATE must be valid dates."
        )
    if from_date > to_date:
        raise DataExtractionConfigError(
            "SOURCE_FROM_DATE must be earlier than or equal to SOURCE_TO_DATE."
        )


def build_database_url(config: SourceConfig) -> URL:
    """Build a SQLAlchemy PostgreSQL connection URL."""

    return URL.create(
        "postgresql+psycopg2",
        username=config.db_user,
        password=config.db_password,
        host=config.db_host,
        port=config.db_port,
        database=config.db_name,
    )


def build_cashflow_query(config: SourceConfig) -> tuple[str, dict[str, Any]]:
    """Return the aggregate daily cashflow query and bound parameters."""

    query = f"""
        SELECT
            prc.date::date AS {DATE_COLUMN},
            SUM(prc.amount)::float AS {TARGET_COLUMN}
        FROM {config.db_schema}.project_real_cashflow AS prc
        JOIN {config.db_schema}.project AS p
            ON p.id = prc.project_id
        JOIN {config.db_schema}.cashflow_type AS ct
            ON ct.id = prc.cashflow_type_id
        WHERE p.name = :project_name
          AND ct.flow_type = :flow_type
          AND prc.approval_status = :approval_status
          AND prc.date >= :from_date
          AND prc.date <= :to_date
        GROUP BY prc.date::date
        ORDER BY prc.date::date
    """
    params = {
        "project_name": config.project_name,
        "flow_type": config.flow_type,
        "approval_status": config.approval_status,
        "from_date": config.from_date,
        "to_date": config.to_date,
    }
    return query, params


def normalize_cashflow_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize query output to the existing deposit raw-data contract."""

    if df.empty:
        return pd.DataFrame(columns=[DATE_COLUMN, TARGET_COLUMN])

    normalized = df[[DATE_COLUMN, TARGET_COLUMN]].copy()
    normalized[DATE_COLUMN] = pd.to_datetime(normalized[DATE_COLUMN]).dt.normalize()
    normalized[TARGET_COLUMN] = pd.to_numeric(
        normalized[TARGET_COLUMN], errors="coerce"
    ).astype(float)
    normalized = (
        normalized.groupby(DATE_COLUMN, as_index=False)[TARGET_COLUMN]
        .sum()
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    return normalized


def extract_data(config: SourceConfig | None = None) -> pd.DataFrame:
    """Extract approved daily deposit cashflow from PostgreSQL."""

    source_config = config or get_source_config()
    validate_source_config(source_config)
    query, params = build_cashflow_query(source_config)

    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(build_database_url(source_config), pool_pre_ping=True)
        with engine.connect() as connection:
            df = pd.read_sql_query(text(query), connection, params=params)
    except DataExtractionError:
        raise
    except Exception as exc:
        raise DataExtractionDatabaseError(
            "Database extraction failed for "
            f"{source_config.from_date} to {source_config.to_date}."
        ) from exc

    normalized = normalize_cashflow_frame(df)
    if normalized.empty:
        raise DataExtractionNotFoundError(
            "No approved deposit cashflows were found for "
            f"project '{source_config.project_name}' between "
            f"{source_config.from_date} and {source_config.to_date}."
        )

    return normalized


def build_extract_source_summary(config: SourceConfig, df: pd.DataFrame) -> dict[str, Any]:
    """Build source metadata for the extract task summary."""

    date_values = pd.to_datetime(df[DATE_COLUMN]) if not df.empty else pd.Series()
    return {
        "source_type": "postgresql",
        "source_project_name": config.project_name,
        "source_flow_type": config.flow_type,
        "source_approval_status": config.approval_status,
        "source_from_date": config.from_date,
        "source_to_date": config.to_date,
        "min_date": date_values.min().date().isoformat() if not df.empty else None,
        "max_date": date_values.max().date().isoformat() if not df.empty else None,
    }
