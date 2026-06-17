# Task D: Unified Report Format & Robustness Testing Framework

## Status: DONE

## Output Files

| File | Description |
|------|-------------|
| `report_schema.py` | Unified report schema, validation, empty template generation, legacy report conversion |
| `report_template.html` | Chinese HTML report template, responsive design, no code/JSON visible |
| `robustness_test.py` | Architecture-agnostic robustness testing framework (4 test functions + aggregate runner) |
| `example_report.json` | Real report generated from Agent A's XGBoost TreeSHAP attribution (passes `validate_report()`) |
| `example_report_xgboost.json` | Same as above (backup copy), from `task_a_xgboost/shap_report.json` converted to unified format |
| `generate_report.py` | Script that generates the report; attempts CfC checkpoint first, falls back to XGBoost |
| `INSTRUCTION.md` | Original task instructions |

## Schema Fields

The unified report (`UNIFIED_REPORT_SCHEMA`) contains 6 top-level sections:

| Section | Required Fields | Description |
|---------|-----------------|-------------|
| `meta` | `report_id`, `timestamp`, `platform_version` | Report metadata (UUID, ISO8601 timestamp) |
| `model_info` | `architecture`, `model_name`, `task`, `input_shape`, `output_shape` | Model details. `architecture` must be one of: `transformer`, `cfc`, `lstm`, `xgboost`, `generic_pytorch` |
| `data_info` | `dataset_name`, `sample_count`, `feature_names`, `data_hash` | Dataset metadata. `data_hash` is optional (may be empty string) |
| `attribution` | `method`, `global_importance`, `single_sample`, `completeness_check` | Attribution results: global feature importance, single-sample breakdown, completeness error check |
| `robustness` | `consistency_std`, `noise_top3_unchanged`, `ood_warning` | Robustness test summary metrics |
| `conclusion` | `top3_features`, `confidence`, `one_line_summary` | Final verdict. `confidence` must be one of: `high`, `medium`, `low` |

## API Functions (report_schema.py)

- `validate_report(report: dict) -> tuple[bool, list[str]]` - Validate a report dict against the schema. Returns (is_valid, error_list).
- `generate_empty_report() -> dict` - Produce a blank report template with all required fields.
- `convert_legacy_report(legacy_report: dict, source_type: str) -> dict` - Convert existing report formats to unified format. Supported `source_type` values: `"rome"`, `"bias"`, `"cfc_interpret"`, `"patching"`, `"drift"`.

## HTML Template

- Open `report_template.html` in any browser to preview.
- Pure CSS, no external dependencies.
- Responsive design (works on mobile).
- All text in Chinese; no code or JSON visible.
- Color scheme: dark blue `#1a365d` (headings), light gray `#f7fafc` (background), green `#38a169` (pass), red `#e53e3e` (fail).
- Uses `{{placeholder}}` template variables for server-side rendering.

## Robustness Test Functions (robustness_test.py)

| Function | Purpose | Pass Criterion |
|----------|---------|----------------|
| `test_attribution_consistency()` | Run attribution N times on same input | Mean per-feature std < 0.01 |
| `test_boundary_conditions()` | Zero input and all-ones input | No crash, no NaN, no Inf |
| `test_adversarial_robustness()` | Add Gaussian noise, check top-3 stability | Top-3 feature indices unchanged |
| `test_ood_detection()` | Scale input by 10x to simulate OOD | OOD detected (NaN/Inf/dramatic change) |
| `run_all_tests()` | Run all 4 tests, produce aggregate report | All 4 tests pass |

All functions accept `(model, input_tensor, attrib_fn)` and are architecture-agnostic.

## Verification Results

- `validate_report(example_report.json)` returns `True` with 0 errors.
- `validate_report(generate_empty_report())` returns `True` with 0 errors.
- `validate_report` correctly rejects missing fields (`meta`, `conclusion`, `robustness`), invalid enum values (`architecture`, `confidence`), and non-dict input.
- `robustness_test.py` self-test runs successfully (3/4 tests pass with dummy model; adversarial test fails as expected with random data).

## Report Data Source

`example_report.json` contains **real attribution data** from Agent A's XGBoost model (`task_a_xgboost/shap_report.json`), converted to the unified format using `convert_legacy_report()` logic.

Originally the plan was to use the CfC flash-crash checkpoint (`liquid_networks/flash_crash/models/`), but the checkpoint was saved with numpy 2.x (references `numpy._core` module) which is incompatible with the current numpy 1.24.4 environment. The `generate_report.py` script attempts CfC loading first and falls back to XGBoost automatically.

All attribution values (global importance, single-sample SHAP values, completeness check) are real model outputs, not hand-filled examples.
