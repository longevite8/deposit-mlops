import textwrap
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import (
    ALERT_SMTP_HOST,
    ALERT_SMTP_PORT,
    ALERT_SMTP_USER,
    ALERT_SMTP_PASSWORD,
    ALERT_EMAIL_FROM,
    ALERT_EMAIL_TO,
)


def determine_alert_status(monitoring_summary, drift_summary):
    """Xác định trạng thái alert dựa trên monitoring và drift."""
    need_retraining = monitoring_summary.get("need_retraining", False)
    drift_status = drift_summary.get("status", "UNKNOWN")

    if need_retraining:
        reason = (
            "Monitoring FAIL" if drift_status == "FAIL" else "Performance degradation"
        )
        return True, reason
    return False, "Monitoring PASS"


def build_alert_subject(alert_needed, prefix="[MLOps Alert]"):
    """Tạo tiêu đề email dựa trên trạng thái alert."""
    status_str = (
        "🚨 ALERT — Retraining Required" if alert_needed else "✅ OK — No Action Needed"
    )
    return f"{prefix} {status_str}"


def format_email_body(
    monitoring_summary, monitoring_lineage, monitoring_metrics, alert_needed, reason
):
    """Dựng cấu trúc HTML email giống với phiên bản cũ."""
    status_color = "#d9534f" if alert_needed else "#5cb85c"
    severity = "HIGH" if alert_needed else "INFO"
    status_text = monitoring_summary.get("status", "UNKNOWN")

    # Dựng bảng metrics
    rows = "".join(
        f"<tr><td style='padding:4px 8px'>{k}</td>"
        f"<td style='padding:4px 8px'><b>{v:.4f}</b></td></tr>"
        for k, v in monitoring_metrics.items()
    )

    return textwrap.dedent(f"""
        <html><body style="font-family:Arial,sans-serif;font-size:14px">
        <h2 style="color:{status_color}">
            {severity} — {status_text}
        </h2>
        <table border="1" cellspacing="0" style="border-collapse:collapse">
          <tr style="background:#f5f5f5">
            <th style="padding:4px 8px">Item</th>
            <th style="padding:4px 8px">Value</th>
          </tr>
          <tr><td style="padding:4px 8px">Alert</td>
              <td style="padding:4px 8px"><b>{alert_needed}</b></td></tr>
          <tr><td style="padding:4px 8px">Severity</td>
              <td style="padding:4px 8px"><b>{severity}</b></td></tr>
          <tr><td style="padding:4px 8px">Reason</td>
              <td style="padding:4px 8px">{reason}</td></tr>
          <tr><td style="padding:4px 8px">Model ID</td>
              <td style="padding:4px 8px">{monitoring_lineage.get("model_id")}</td></tr>
          <tr><td style="padding:4px 8px">Feature Dataset</td>
              <td style="padding:4px 8px">{monitoring_lineage.get("feature_dataset_id")}</td></tr>
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


def send_email(subject, html_content, logger=None):
    """Gửi email HTML qua SMTP với error handling."""
    # Ưu tiên lấy password từ Env Var giống bản cũ
    password = os.environ.get("CLEARML_ALERT_SMTP_PASSWORD", ALERT_SMTP_PASSWORD)

    if not password:
        if logger:
            logger.report_text("SMTP password missing. Skipping email.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = ALERT_EMAIL_FROM
    msg["To"] = ", ".join(ALERT_EMAIL_TO)
    msg.attach(MIMEText(html_content, "html"))

    try:
        with smtplib.SMTP(ALERT_SMTP_HOST, ALERT_SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(ALERT_SMTP_USER, password)
            server.sendmail(ALERT_EMAIL_FROM, ALERT_EMAIL_TO, msg.as_string())

        if logger:
            logger.report_text(f"✅ Alert email sent to: {ALERT_EMAIL_TO}")
        return True
    except Exception as e:
        if logger:
            logger.report_text(f"❌ Failed to send alert email: {e}")
        return False
