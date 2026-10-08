"""Fail-closed production promotion check for a locally selected model.

The manifest records human approval; this check cannot replace data review,
signature verification, calibration, or infrastructure controls.
"""

import json
from pathlib import Path


def validate_promotion(manifest_path: Path, model_path: Path, model: dict) -> dict:
    approval = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_text = ("approved_by", "approved_at", "deployment_target", "dataset_sha256")
    if approval.get("status") != "approved" or approval.get("data_type") != "real":
        raise ValueError("Production requires approval of a real-data model")
    if any(not str(approval.get(key, "")).strip() for key in required_text):
        raise ValueError("Promotion manifest is missing approval or deployment details")
    approved_model = manifest_path.parent / approval.get("model_path", "")
    if approved_model.resolve() != model_path.resolve():
        raise ValueError("Approved model path does not match the loaded artifact")
    if approval["dataset_sha256"] != model.get("split_sha256"):
        raise ValueError("Approved dataset hash does not match the model")
    if approval.get("model_version") != model.get("version"):
        raise ValueError("Approved model version does not match the artifact")
    if model.get("version") in ("v6", "v7") and approval.get("checkpoint_approved") is not True:
        raise ValueError("Pretrained checkpoint approval is required")
    for partition in ("validation", "test"):
        report_name = approval.get(f"{partition}_report")
        if not report_name:
            raise ValueError(f"Missing {partition} evaluation report")
        report = json.loads((manifest_path.parent / report_name).read_text(encoding="utf-8"))
        if (report.get("partition") != partition or
                report.get("dataset_sha256") != model.get("split_sha256") or
                report.get("version") != model.get("version") or
                report.get("test_count", 0) < 1):
            raise ValueError(f"{partition} report does not match the approved model and data")
    return approval
