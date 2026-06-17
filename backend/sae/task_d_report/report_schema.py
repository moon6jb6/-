"""Unified Attribution Report Schema for ZhiXin Engine.

All architecture-specific attribution reports (Transformer, CfC, LSTM, XGBoost,
generic PyTorch) MUST conform to this schema before being handed to customers.

Provides:
  - UNIFIED_REPORT_SCHEMA: canonical field definitions
  - validate_report(): strict validation returning (is_valid, errors)
  - generate_empty_report(): produce a blank template
  - convert_legacy_report(): convert existing report formats to unified format

Usage:
    from report_schema import validate_report, generate_empty_report
    is_valid, errors = validate_report(my_report)
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime, timezone
from typing import Any


# ═══════════════════════════════════════════════════════════════════════
# Schema Definition
# ═══════════════════════════════════════════════════════════════════════

VALID_ARCHITECTURES = {"transformer", "cfc", "lstm", "xgboost", "generic_pytorch"}
VALID_TASKS = {"二分类", "多分类", "回归", "binary_classification", "multi_classification", "regression"}
VALID_METHODS = {"rome", "patching", "ig", "tree_shap", "tau_analysis", "lime", "attention_rollout"}
VALID_CONFIDENCE = {"high", "medium", "low"}


UNIFIED_REPORT_SCHEMA = {
    "meta": {
        "report_id": "str, UUID format",
        "timestamp": "str, ISO8601 with timezone",
        "platform_version": "str, e.g. v0.1.0",
    },
    "model_info": {
        "architecture": "str, one of: transformer|cfc|lstm|xgboost|generic_pytorch",
        "model_name": "str",
        "task": "str, e.g. 二分类/多分类/回归",
        "input_shape": "list[int]",
        "output_shape": "list[int]",
    },
    "data_info": {
        "dataset_name": "str",
        "sample_count": "int",
        "feature_names": "list[str]",
        "data_hash": "str, SHA256 (optional, may be empty string)",
    },
    "attribution": {
        "method": "str, e.g. rome|patching|ig|tree_shap|tau_analysis",
        "global_importance": {
            "<feature_name>": "float, importance value (sorted descending by value)"
        },
        "single_sample": {
            "input": "list, raw input values",
            "output": "float, model prediction probability",
            "attributions": "list[float], per-feature attribution values",
            "top_features": [
                {"feature": "str", "attribution": "float", "rank": "int"}
            ],
        },
        "completeness_check": {
            "sum_attributions": "float",
            "output_minus_baseline": "float",
            "error_pct": "float, should be < 1%",
        },
    },
    "robustness": {
        "consistency_std": "float, std across N runs",
        "noise_top3_unchanged": "bool, whether top3 unchanged after noise",
        "ood_warning": "bool, whether OOD detected",
    },
    "conclusion": {
        "top3_features": "list[str], top 3 most important feature names",
        "confidence": "str, high|medium|low",
        "one_line_summary": "str, one-line conclusion in Chinese",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════

def _check_type(value: Any, expected_type: str, path: str, errors: list[str]) -> None:
    """Check that *value* matches *expected_type* description; append errors."""
    if expected_type == "str":
        if not isinstance(value, str):
            errors.append(f"{path}: expected str, got {type(value).__name__}")
    elif expected_type == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{path}: expected int, got {type(value).__name__}")
    elif expected_type == "float":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{path}: expected float, got {type(value).__name__}")
    elif expected_type == "bool":
        if not isinstance(value, bool):
            errors.append(f"{path}: expected bool, got {type(value).__name__}")
    elif expected_type == "list":
        if not isinstance(value, list):
            errors.append(f"{path}: expected list, got {type(value).__name__}")
    elif expected_type == "list[str]":
        if not isinstance(value, list):
            errors.append(f"{path}: expected list[str], got {type(value).__name__}")
        elif not all(isinstance(v, str) for v in value):
            errors.append(f"{path}: all elements must be str")
    elif expected_type == "list[int]":
        if not isinstance(value, list):
            errors.append(f"{path}: expected list[int], got {type(value).__name__}")
        elif not all(isinstance(v, int) for v in value):
            errors.append(f"{path}: all elements must be int")
    elif expected_type == "list[float]":
        if not isinstance(value, list):
            errors.append(f"{path}: expected list[float], got {type(value).__name__}")
        elif not all(isinstance(v, (int, float)) for v in value):
            errors.append(f"{path}: all elements must be float")
    elif expected_type == "dict":
        if not isinstance(value, dict):
            errors.append(f"{path}: expected dict, got {type(value).__name__}")


def validate_report(report: dict) -> tuple[bool, list[str]]:
    """Validate a report dict against UNIFIED_REPORT_SCHEMA.

    Returns:
        (is_valid, errors) where is_valid is True if no errors found.
    """
    errors: list[str] = []

    if not isinstance(report, dict):
        return False, ["report must be a dict"]

    # ── meta ───────────────────────────────────────────────────────────
    meta = report.get("meta")
    if not isinstance(meta, dict):
        errors.append("meta: missing or not a dict")
    else:
        _check_type(meta.get("report_id"), "str", "meta.report_id", errors)
        _check_type(meta.get("timestamp"), "str", "meta.timestamp", errors)
        _check_type(meta.get("platform_version"), "str", "meta.platform_version", errors)

    # ── model_info ─────────────────────────────────────────────────────
    mi = report.get("model_info")
    if not isinstance(mi, dict):
        errors.append("model_info: missing or not a dict")
    else:
        _check_type(mi.get("architecture"), "str", "model_info.architecture", errors)
        arch = mi.get("architecture", "")
        if isinstance(arch, str) and arch and arch not in VALID_ARCHITECTURES:
            errors.append(
                f"model_info.architecture: '{arch}' not in {VALID_ARCHITECTURES}"
            )
        _check_type(mi.get("model_name"), "str", "model_info.model_name", errors)
        _check_type(mi.get("task"), "str", "model_info.task", errors)
        _check_type(mi.get("input_shape"), "list[int]", "model_info.input_shape", errors)
        _check_type(mi.get("output_shape"), "list[int]", "model_info.output_shape", errors)

    # ── data_info ──────────────────────────────────────────────────────
    di = report.get("data_info")
    if not isinstance(di, dict):
        errors.append("data_info: missing or not a dict")
    else:
        _check_type(di.get("dataset_name"), "str", "data_info.dataset_name", errors)
        _check_type(di.get("sample_count"), "int", "data_info.sample_count", errors)
        _check_type(di.get("feature_names"), "list[str]", "data_info.feature_names", errors)
        # data_hash is optional
        dh = di.get("data_hash", "")
        if dh is not None:
            _check_type(dh, "str", "data_info.data_hash", errors)

    # ── attribution ────────────────────────────────────────────────────
    attr = report.get("attribution")
    if not isinstance(attr, dict):
        errors.append("attribution: missing or not a dict")
    else:
        _check_type(attr.get("method"), "str", "attribution.method", errors)

        # global_importance
        gi = attr.get("global_importance")
        if not isinstance(gi, dict):
            errors.append("attribution.global_importance: missing or not a dict")
        elif gi:
            for k, v in gi.items():
                if not isinstance(k, str):
                    errors.append(f"attribution.global_importance key '{k}': must be str")
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    errors.append(f"attribution.global_importance['{k}']: must be float")

        # single_sample
        ss = attr.get("single_sample")
        if not isinstance(ss, dict):
            errors.append("attribution.single_sample: missing or not a dict")
        else:
            _check_type(ss.get("input"), "list", "attribution.single_sample.input", errors)
            _check_type(ss.get("output"), "float", "attribution.single_sample.output", errors)
            _check_type(
                ss.get("attributions"), "list[float]",
                "attribution.single_sample.attributions", errors,
            )
            top_feats = ss.get("top_features")
            if not isinstance(top_feats, list):
                errors.append("attribution.single_sample.top_features: must be list")
            else:
                for i, tf in enumerate(top_feats):
                    if not isinstance(tf, dict):
                        errors.append(f"attribution.single_sample.top_features[{i}]: must be dict")
                    else:
                        _check_type(tf.get("feature"), "str", f"top_features[{i}].feature", errors)
                        _check_type(tf.get("attribution"), "float", f"top_features[{i}].attribution", errors)
                        _check_type(tf.get("rank"), "int", f"top_features[{i}].rank", errors)

        # completeness_check
        cc = attr.get("completeness_check")
        if not isinstance(cc, dict):
            errors.append("attribution.completeness_check: missing or not a dict")
        else:
            _check_type(cc.get("sum_attributions"), "float", "completeness_check.sum_attributions", errors)
            _check_type(cc.get("output_minus_baseline"), "float", "completeness_check.output_minus_baseline", errors)
            _check_type(cc.get("error_pct"), "float", "completeness_check.error_pct", errors)

    # ── robustness ─────────────────────────────────────────────────────
    rob = report.get("robustness")
    if not isinstance(rob, dict):
        errors.append("robustness: missing or not a dict")
    else:
        _check_type(rob.get("consistency_std"), "float", "robustness.consistency_std", errors)
        _check_type(rob.get("noise_top3_unchanged"), "bool", "robustness.noise_top3_unchanged", errors)
        _check_type(rob.get("ood_warning"), "bool", "robustness.ood_warning", errors)

    # ── conclusion ─────────────────────────────────────────────────────
    conc = report.get("conclusion")
    if not isinstance(conc, dict):
        errors.append("conclusion: missing or not a dict")
    else:
        _check_type(conc.get("top3_features"), "list[str]", "conclusion.top3_features", errors)
        _check_type(conc.get("confidence"), "str", "conclusion.confidence", errors)
        conf = conc.get("confidence", "")
        if isinstance(conf, str) and conf and conf not in VALID_CONFIDENCE:
            errors.append(f"conclusion.confidence: '{conf}' not in {VALID_CONFIDENCE}")
        _check_type(conc.get("one_line_summary"), "str", "conclusion.one_line_summary", errors)

    is_valid = len(errors) == 0
    return is_valid, errors


# ═══════════════════════════════════════════════════════════════════════
# Empty Template
# ═══════════════════════════════════════════════════════════════════════

def generate_empty_report() -> dict:
    """Return a blank report template with all required fields set to default values."""
    return {
        "meta": {
            "report_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform_version": "v0.1.0",
        },
        "model_info": {
            "architecture": "generic_pytorch",
            "model_name": "",
            "task": "",
            "input_shape": [],
            "output_shape": [],
        },
        "data_info": {
            "dataset_name": "",
            "sample_count": 0,
            "feature_names": [],
            "data_hash": "",
        },
        "attribution": {
            "method": "",
            "global_importance": {},
            "single_sample": {
                "input": [],
                "output": 0.0,
                "attributions": [],
                "top_features": [],
            },
            "completeness_check": {
                "sum_attributions": 0.0,
                "output_minus_baseline": 0.0,
                "error_pct": 0.0,
            },
        },
        "robustness": {
            "consistency_std": 0.0,
            "noise_top3_unchanged": True,
            "ood_warning": False,
        },
        "conclusion": {
            "top3_features": [],
            "confidence": "low",
            "one_line_summary": "",
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Legacy Report Conversion
# ═══════════════════════════════════════════════════════════════════════

def _build_meta() -> dict:
    return {
        "report_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform_version": "v0.1.0",
    }


def _convert_causality_report(legacy: dict) -> dict:
    """Convert causality_report_v1.0.json to unified format.

    Source: causal_ablation.py -> build_report()
    """
    report = generate_empty_report()
    report["meta"] = _build_meta()

    config = legacy.get("experiment_config", {})
    report["model_info"]["architecture"] = "transformer"
    report["model_info"]["model_name"] = config.get("model", "")
    report["model_info"]["task"] = "因果归因"

    report["data_info"]["dataset_name"] = "causal_ablation_prompts"
    report["data_info"]["sample_count"] = config.get("n_prompts", 0)

    report["attribution"]["method"] = "rome"

    # Extract global importance from feature_results
    feature_results = legacy.get("feature_results", [])
    global_imp = {}
    single_attributions = []
    for fr in feature_results:
        fid = fr.get("feature_id", 0)
        mean_rest = fr.get("aggregate_results", {}).get("mean_restoration", 0.0)
        global_imp[f"feature_{fid}"] = round(mean_rest, 6)
        single_attributions.append(mean_rest)

    # Sort by value descending
    report["attribution"]["global_importance"] = dict(
        sorted(global_imp.items(), key=lambda x: x[1], reverse=True)
    )

    report["attribution"]["single_sample"]["attributions"] = single_attributions

    if single_attributions:
        top3 = sorted(global_imp.items(), key=lambda x: x[1], reverse=True)[:3]
        report["attribution"]["single_sample"]["top_features"] = [
            {"feature": name, "attribution": val, "rank": i + 1}
            for i, (name, val) in enumerate(top3)
        ]

    # Completeness: sum of restorations
    sum_attr = sum(single_attributions)
    report["attribution"]["completeness_check"]["sum_attributions"] = round(sum_attr, 6)

    # Robustness from bootstrap
    if feature_results:
        p_values = [
            fr.get("aggregate_results", {}).get("bootstrap_p_value", 1.0)
            for fr in feature_results
        ]
        report["robustness"]["consistency_std"] = round(float(sum(p_values)) / max(len(p_values), 1), 4)

    # Conclusion
    top3_names = list(report["attribution"]["global_importance"].keys())[:3]
    report["conclusion"]["top3_features"] = top3_names
    verdicts = [
        fr.get("aggregate_results", {}).get("verdict", "NOT_SIGNIFICANT")
        for fr in feature_results
    ]
    n_sig = sum(1 for v in verdicts if v == "SIGNIFICANT")
    report["conclusion"]["confidence"] = "high" if n_sig >= len(verdicts) * 0.5 else "low"
    report["conclusion"]["one_line_summary"] = (
        f"共测试{len(feature_results)}个特征，{n_sig}个具有统计显著性。"
    )

    return report


def _convert_bias_report(legacy: dict) -> dict:
    """Convert bias_report.json to unified format.

    Source: bias_detection.py -> build_bias_report()
    """
    report = generate_empty_report()
    report["meta"] = _build_meta()

    report["model_info"]["architecture"] = "transformer"
    report["model_info"]["model_name"] = legacy.get("model", "")
    report["model_info"]["task"] = "公平性检测"

    report["data_info"]["dataset_name"] = "bias_contrastive_pairs"

    report["attribution"]["method"] = "ig"  # Input x Gradient

    # Build global importance from dimension mean biases
    dimensions = legacy.get("dimensions", {})
    global_imp = {}
    for dim_key, dim_data in dimensions.items():
        label = dim_data.get("label", dim_key)
        mean_bias = dim_data.get("mean_max_bias", 0.0)
        global_imp[label] = round(mean_bias, 4)

    report["attribution"]["global_importance"] = dict(
        sorted(global_imp.items(), key=lambda x: x[1], reverse=True)
    )

    # Robustness
    summary = legacy.get("summary", {})
    risk_dist = summary.get("risk_distribution", {})
    severe_count = risk_dist.get("severe", 0)
    high_count = risk_dist.get("high", 0)
    report["robustness"]["ood_warning"] = (severe_count + high_count) > 0

    # Conclusion
    top3 = list(report["attribution"]["global_importance"].keys())[:3]
    report["conclusion"]["top3_features"] = top3
    overall_risk = summary.get("overall_risk", "low")
    report["conclusion"]["confidence"] = (
        "high" if overall_risk in ("low",) else "low"
    )
    report["conclusion"]["one_line_summary"] = (
        f"共检测{summary.get('total_dimensions', 0)}个维度，整体风险等级：{overall_risk}。"
    )

    return report


def _convert_interpret_report(legacy: dict) -> dict:
    """Convert CfC interpret report (from interpret_base.py) to unified format.

    Source: interpret_base.py -> generate_explanation_report()
    The interpret report is typically HTML; this handles the dict form.
    """
    report = generate_empty_report()
    report["meta"] = _build_meta()

    report["model_info"]["architecture"] = "cfc"
    report["model_info"]["model_name"] = legacy.get("model_name", "CfC Model")
    report["model_info"]["task"] = legacy.get("task", "二分类")

    report["data_info"]["dataset_name"] = legacy.get("dataset_name", "")
    report["data_info"]["feature_names"] = legacy.get("feature_names", [])

    report["attribution"]["method"] = "ig"  # Integrated Gradients

    # Feature attributions from the report
    feat_imp = legacy.get("feature_importance", {})
    if feat_imp:
        report["attribution"]["global_importance"] = dict(
            sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
        )
        top3 = list(report["attribution"]["global_importance"].keys())[:3]
        report["conclusion"]["top3_features"] = top3

    top_features = legacy.get("top_features", [])
    if top_features:
        report["attribution"]["single_sample"]["top_features"] = [
            {"feature": name, "attribution": imp, "rank": i + 1}
            for i, (name, imp) in enumerate(top_features)
        ]

    report["conclusion"]["confidence"] = "medium"
    report["conclusion"]["one_line_summary"] = legacy.get(
        "conclusion_template", "模型决策归因分析完成。"
    )

    return report


def _convert_patching_report(legacy: dict) -> dict:
    """Convert patching report to unified format."""
    report = generate_empty_report()
    report["meta"] = _build_meta()

    report["model_info"]["architecture"] = "transformer"
    report["model_info"]["model_name"] = legacy.get("model", "")
    report["attribution"]["method"] = "patching"

    results = legacy.get("results", legacy.get("feature_results", []))
    global_imp = {}
    for r in results:
        fid = r.get("feature_id", r.get("id", 0))
        score = r.get("patching_effect", r.get("effect", 0.0))
        global_imp[f"feature_{fid}"] = round(float(score), 6)

    report["attribution"]["global_importance"] = dict(
        sorted(global_imp.items(), key=lambda x: x[1], reverse=True)
    )

    top3 = list(report["attribution"]["global_importance"].keys())[:3]
    report["conclusion"]["top3_features"] = top3
    report["conclusion"]["confidence"] = "medium"
    report["conclusion"]["one_line_summary"] = f"激活修补分析完成，共分析{len(results)}个特征。"

    return report


def _convert_drift_report(legacy: dict) -> dict:
    """Convert drift detection report to unified format."""
    report = generate_empty_report()
    report["meta"] = _build_meta()

    report["model_info"]["architecture"] = legacy.get("architecture", "generic_pytorch")
    report["model_info"]["model_name"] = legacy.get("model_name", "")
    report["attribution"]["method"] = legacy.get("method", "ig")

    report["robustness"]["ood_warning"] = legacy.get("drift_detected", False)

    report["conclusion"]["confidence"] = "medium"
    report["conclusion"]["one_line_summary"] = legacy.get(
        "summary", "分布漂移检测完成。"
    )

    return report


_LEGACY_CONVERTERS = {
    "rome": _convert_causality_report,
    "bias": _convert_bias_report,
    "cfc_interpret": _convert_interpret_report,
    "patching": _convert_patching_report,
    "drift": _convert_drift_report,
}


def convert_legacy_report(legacy_report: dict, source_type: str) -> dict:
    """Convert an existing report format to the unified format.

    Args:
        legacy_report: The existing report dict.
        source_type: One of "rome", "bias", "cfc_interpret", "patching", "drift".

    Returns:
        A new dict conforming to UNIFIED_REPORT_SCHEMA.

    Raises:
        ValueError: If source_type is not recognised.
    """
    converter = _LEGACY_CONVERTERS.get(source_type)
    if converter is None:
        raise ValueError(
            f"Unknown source_type '{source_type}'. "
            f"Supported: {list(_LEGACY_CONVERTERS.keys())}"
        )
    return converter(legacy_report)


# ═══════════════════════════════════════════════════════════════════════
# CLI quick-test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json, sys

    # Quick self-test with example report
    example_path = __file__.replace("report_schema.py", "example_report.json")
    try:
        with open(example_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        ok, errs = validate_report(report)
        print(f"validate_report(example_report.json): valid={ok}")
        if errs:
            for e in errs:
                print(f"  ERROR: {e}")
    except FileNotFoundError:
        print("example_report.json not found, skipping self-test.")

    # Test empty template
    empty = generate_empty_report()
    ok, errs = validate_report(empty)
    print(f"validate_report(empty_template): valid={ok}")
    if errs:
        for e in errs:
            print(f"  ERROR: {e}")
