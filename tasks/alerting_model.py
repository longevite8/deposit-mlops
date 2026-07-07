import os
import smtplib
import textwrap
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
from clearml import Task

from config import (
    ALERT_EMAIL_FROM,
    ALERT_EMAIL_SUBJECT_PREFIX,
    ALERT_EMAIL_TO,
    ALERT_SMTP_HOST,
    ALERT_SMTP_PASSWORD,
    ALERT_SMTP_PORT,
    ALERT_SMTP_USER,
    PROJECT_TEMPLATE,
    TEMPLATE_ALERTING_NAME,
)

from helpers import wait_for_artifact  # THÊM: Import từ helper


# =====================================================
# Email helper
# =====================================================


def _build_html_body(
    alert_summary: dict,
    alert_lineage: dict,
    monitoring_metrics: dict,
) -> str:
    """Build an HTML email body from the alert and monitoring data."""
    status_color = "#d9534f" if alert_summary["alert"] else "#5cb85c"
    rows = "".join(
        f"<tr><td style='padding:4px 8px'>{k}</td>"
        f"<td style='padding:4px 8px'><b>{v:.4f}</b></td></tr>"
        for k, v in monitoring_metrics.items()
    )
    return textwrap.dedent(f"""
        <html><body style="font-family:Arial,sans-serif;font-size:14px">
        <h2 style="color:{status_color}">
            {ALERT_EMAIL_SUBJECT_PREFIX} {alert_summary["severity"]} — {alert_summary["status"]}
        </h2>
        <table border="1" cellspacing="0" style="border-collapse:collapse">
          <tr style="background:#f5f5f5">
            <th style="padding:4px 8px">Item</th>
            <th style="padding:4px 8px">Value</th>
          </tr>
          <tr><td style="padding:4px 8px">Alert</td>
              <td style="padding:4px 8px"><b>{alert_summary["alert"]}</b></td></tr>
          <tr><td style="padding:4px 8px">Severity</td>
              <td style="padding:4px 8px"><b>{alert_summary["severity"]}</b></td></tr>
          <tr><td style="padding:4px 8px">Reason</td>
              <td style="padding:4px 8px">{alert_summary["reason"]}</td></tr>
          <tr><td style="padding:4px 8px">Need Retraining</td>
              <td style="padding:4px 8px"><b>{alert_summary["need_retraining"]}</b></td></tr>
          <tr><td style="padding:4px 8px">Model ID</td>
              <td style="padding:4px 8px">{alert_lineage["model_id"]}</td></tr>
          <tr><td style="padding:4px 8px">Feature Dataset</td>
              <td style="padding:4px 8px">{alert_lineage["feature_dataset_id"]}</td></tr>
          <tr><td style="padding:4px 8px">Timestamp</td>
              <td style="padding:4px 8px">{datetime.utcnow().isoformat()} UTC</td></tr>
        </table>
        <h3>Monitoring Metrics</h3>
        <table border="1" cellspacing="0" style="border-collapse:collapse">
          <tr style="background:#f5f5f5">
            <th style="padding:4px 8px">Metric</th>
            <th style="padding:4px 8px">Value</th>
          </tr>
          {rows}
        </table>
        </body></html>
    """)


def send_alert_email(
    subject: str,
    html_body: str,
    logger,
) -> bool:
    """Send an HTML alert email via SMTP.

    Reads SMTP password from the ``CLEARML_ALERT_SMTP_PASSWORD`` environment
    variable first; falls back to ``ALERT_SMTP_PASSWORD`` in ``config.py``.

    Returns ``True`` on success, ``False`` on failure (non-fatal).
    """
    password = os.environ.get("CLEARML_ALERT_SMTP_PASSWORD", ALERT_SMTP_PASSWORD)

    if not password:
        logger.report_text(
            "SMTP password not configured. "
            "Set the CLEARML_ALERT_SMTP_PASSWORD environment variable. "
            "Skipping email send."
        )
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = ALERT_EMAIL_FROM
    msg["To"] = ", ".join(ALERT_EMAIL_TO)
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(ALERT_SMTP_HOST, ALERT_SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(ALERT_SMTP_USER, password)
            server.sendmail(ALERT_EMAIL_FROM, ALERT_EMAIL_TO, msg.as_string())

        logger.report_text(f"Alert email sent to: {ALERT_EMAIL_TO}")
        return True

    except smtplib.SMTPException as exc:
        logger.report_text(f"Failed to send alert email: {exc}")
        return False


# =====================================================
# Task init
# =====================================================

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_ALERTING_NAME,
    task_type=Task.TaskTypes.qc,
)

# =====================================================
# Parameters
# =====================================================

params = task.connect(
    {
        "monitoring_task_id": "",
    }
)

# =====================================================
# Template creation mode
# =====================================================

if not params["monitoring_task_id"]:
    task.get_logger().report_text("Template creation mode.")

    task.close()

    raise SystemExit(0)


# =====================================================
# Load monitoring report
# =====================================================

monitoring_task = Task.get_task(task_id=params["monitoring_task_id"])

# SỬA: Dùng wait_for_artifact để chắc chắn artifact sẵn sàng
monitoring_summary = wait_for_artifact(
    monitoring_task,
    "monitoring_summary",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

monitoring_metrics = wait_for_artifact(
    monitoring_task,
    "monitoring_metrics",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

monitoring_lineage = wait_for_artifact(
    monitoring_task,
    "monitoring_lineage",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

need_retraining = monitoring_summary["need_retraining"]

alert = bool(need_retraining)

# =====================================================
# Build alert payload
# =====================================================

alert_summary = {
    "alert": alert,
    "status": monitoring_summary["status"],
    "severity": "HIGH" if alert else "INFO",
    "need_retraining": monitoring_summary["need_retraining"],
    "reason": "Monitoring FAIL" if alert else "Monitoring PASS",
    "timestamp": datetime.utcnow().isoformat(),
}

alert_lineage = {
    "model_id": monitoring_summary["model_id"],
    "feature_dataset_id": monitoring_summary["feature_dataset_id"],
    "monitoring_task_id": monitoring_lineage["monitoring_task_id"],
    "alert_task_id": task.id,
}

# =====================================================
# Send email
# =====================================================

subject = (
    f"{ALERT_EMAIL_SUBJECT_PREFIX} "
    f"{'🚨 ALERT — Retraining Required' if alert else '✅ OK — No Action Needed'}"
)

html_body = _build_html_body(alert_summary, alert_lineage, monitoring_metrics)

email_sent = send_alert_email(subject, html_body, task.get_logger())

alert_summary["email_sent"] = email_sent

# =====================================================
# Artifacts
# =====================================================

task.upload_artifact("alert_summary", alert_summary)

task.upload_artifact("alert_lineage", alert_lineage)

# =====================================================
# ClearML logs
# =====================================================

task.get_logger().report_single_value("alert", int(alert))

task.get_logger().report_single_value("email_sent", int(email_sent))

summary_df = pd.DataFrame([alert_summary])
task.get_logger().report_table(
    title="Alert Summary",
    series="summary",
    iteration=0,
    table_plot=summary_df,
)

markdown = f"""
# Alert Dashboard

## Alert Status

| Item | Value |
|------|-------|
| Alert | **{alert_summary["alert"]}** |
| Status | **{alert_summary["status"]}** |
| Severity | **{alert_summary["severity"]}** |
| Need Retraining | **{alert_summary["need_retraining"]}** |
| Email Sent | {alert_summary["email_sent"]} |
| Timestamp | {alert_summary["timestamp"]} UTC |

## Model

| Item | Value |
|------|-------|
| Model ID | {alert_lineage["model_id"]} |
| Feature Dataset | {alert_lineage["feature_dataset_id"]} |

## Reason

{alert_summary["reason"]}
"""

task.get_logger().report_text(markdown)

# THÊM: Đồng bộ hoàn toàn trước khi kết thúc
task.flush()

task.close()
