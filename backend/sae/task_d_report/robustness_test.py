"""Robustness Testing Framework for Attribution Methods.

Architecture-agnostic: works with any model and attribution function via
duck-typing (model.forward / attrib_fn interface).

Test functions:
  - test_attribution_consistency: N-run stability
  - test_boundary_conditions: zero / ones input
  - test_adversarial_robustness: noise perturbation
  - test_ood_detection: out-of-distribution detection
  - run_all_tests: aggregate runner with optional JSON output

Usage:
    from robustness_test import run_all_tests
    results = run_all_tests(model, input_tensor, attrib_fn, report_path="robustness.json")
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Type aliases (duck-typing interfaces)
# ═══════════════════════════════════════════════════════════════════════
# model: any object with a callable forward / __call__ that accepts a tensor
#        and returns predictions (tensor or ndarray).
# attrib_fn: Callable(model, input_tensor) -> np.ndarray
#            Must return a 1-D array of per-feature attribution values.


def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensor or list to numpy array."""
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _top_k_indices(attributions: np.ndarray, k: int = 3) -> list[int]:
    """Return indices of top-k features by absolute attribution value."""
    return list(np.argsort(np.abs(attributions))[::-1][:k])


# ═══════════════════════════════════════════════════════════════════════
# Individual Tests
# ═══════════════════════════════════════════════════════════════════════

def test_attribution_consistency(
    model: Any,
    input_tensor: Any,
    attrib_fn: Callable,
    n_runs: int = 3,
    threshold: float = 0.01,
) -> dict:
    """Attribution consistency test: run the same input *n_runs* times.

    A deterministic attribution method should produce identical results.
    The pass criterion is that the per-feature standard deviation across runs
    is below *threshold* (default 0.01).

    Returns:
        {
            "std": float,           # mean std across features
            "pass": bool,
            "details": [
                {"run": int, "attribution_sum": float, "top3": list[int]},
                ...
            ]
        }
    """
    all_attributions: list[np.ndarray] = []
    details: list[dict] = []

    for i in range(n_runs):
        attr = _to_numpy(attrib_fn(model, input_tensor))
        all_attributions.append(attr)
        details.append({
            "run": i + 1,
            "attribution_sum": round(float(np.sum(attr)), 6),
            "top3": _top_k_indices(attr),
        })

    # Stack and compute per-feature std
    stacked = np.stack(all_attributions, axis=0)  # (n_runs, n_features)
    per_feature_std = np.std(stacked, axis=0)
    mean_std = float(np.mean(per_feature_std))

    return {
        "std": round(mean_std, 6),
        "pass": mean_std < threshold,
        "details": details,
    }


def test_boundary_conditions(
    model: Any,
    attrib_fn: Callable,
    input_shape: tuple,
) -> dict:
    """Boundary condition test: zero input and all-ones input.

    The attribution function must not crash, and should return finite values.

    Returns:
        {
            "zero_input": {
                "crash": bool,
                "attribution_sum": float,
                "has_nan": bool,
                "has_inf": bool,
            },
            "ones_input": {
                "crash": bool,
                "attribution_sum": float,
                "has_nan": bool,
                "has_inf": bool,
            },
            "pass": bool,
        }
    """
    results = {}

    for label, fill_value in [("zero_input", 0.0), ("ones_input", 1.0)]:
        try:
            # Build boundary tensor (numpy or torch — try torch first)
            try:
                import torch
                x = torch.full(input_shape, fill_value, dtype=torch.float32)
            except ImportError:
                x = np.full(input_shape, fill_value, dtype=np.float32)

            attr = _to_numpy(attrib_fn(model, x))
            results[label] = {
                "crash": False,
                "attribution_sum": round(float(np.sum(attr)), 6),
                "has_nan": bool(np.any(np.isnan(attr))),
                "has_inf": bool(np.any(np.isinf(attr))),
            }
        except Exception as e:
            results[label] = {
                "crash": True,
                "error": str(e),
                "attribution_sum": 0.0,
                "has_nan": False,
                "has_inf": False,
            }

    # Pass = no crash, no NaN, no Inf
    pass_conditions = []
    for label in ("zero_input", "ones_input"):
        r = results[label]
        pass_conditions.append(
            not r["crash"] and not r["has_nan"] and not r["has_inf"]
        )

    results["pass"] = all(pass_conditions)
    return results


def test_adversarial_robustness(
    model: Any,
    input_tensor: Any,
    attrib_fn: Callable,
    noise_level: float = 0.1,
    n_perturbations: int = 5,
) -> dict:
    """Adversarial robustness test: add noise and check if top-3 attributions change.

    Adds Gaussian noise at *noise_level* fraction of the input magnitude,
    repeated *n_perturbations* times. The test passes if top-3 features
    remain unchanged across all perturbations.

    Returns:
        {
            "original_top3": list[int],
            "noisy_top3": list[list[int]],
            "unchanged": bool,
            "pass": bool,
        }
    """
    # Original attribution
    orig_attr = _to_numpy(attrib_fn(model, input_tensor))
    orig_top3 = _top_k_indices(orig_attr)

    # Convert input to numpy for noise injection
    if hasattr(input_tensor, "detach"):
        import torch
        base = input_tensor.detach().cpu().numpy()
        is_torch = True
    else:
        base = np.asarray(input_tensor)
        is_torch = False

    noisy_top3_list: list[list[int]] = []
    all_unchanged = True

    for _ in range(n_perturbations):
        noise = np.random.normal(0, noise_level, size=base.shape).astype(np.float32)
        noisy_input = base + noise * np.abs(base)  # relative noise

        if is_torch:
            import torch
            noisy_tensor = torch.tensor(noisy_input, dtype=torch.float32)
            if hasattr(input_tensor, "device"):
                noisy_tensor = noisy_tensor.to(input_tensor.device)
        else:
            noisy_tensor = noisy_input

        noisy_attr = _to_numpy(attrib_fn(model, noisy_tensor))
        noisy_top3 = _top_k_indices(noisy_attr)
        noisy_top3_list.append(noisy_top3)

        if set(noisy_top3) != set(orig_top3):
            all_unchanged = False

    return {
        "original_top3": orig_top3,
        "noisy_top3": noisy_top3_list,
        "unchanged": all_unchanged,
        "pass": all_unchanged,
    }


def test_ood_detection(
    model: Any,
    input_tensor: Any,
    attrib_fn: Callable,
    ood_scale: float = 10.0,
) -> dict:
    """Out-of-distribution detection test.

    Creates an OOD input by scaling the original input by *ood_scale*,
    then checks if the attribution pattern changes dramatically or if
    the method produces warning signals.

    The test passes if OOD is detected (attribution pattern changes
    significantly or values become unreasonable).

    Returns:
        {
            "ood_detected": bool,
            "warning": str,
            "original_attribution_sum": float,
            "ood_attribution_sum": float,
            "pass": bool,
        }
    """
    orig_attr = _to_numpy(attrib_fn(model, input_tensor))
    orig_sum = float(np.sum(np.abs(orig_attr)))

    # Create OOD input by scaling
    if hasattr(input_tensor, "detach"):
        import torch
        ood_input = (input_tensor.detach() * ood_scale).to(input_tensor.device)
        if hasattr(input_tensor, "dtype"):
            ood_input = ood_input.to(input_tensor.dtype)
    else:
        ood_input = np.asarray(input_tensor) * ood_scale

    try:
        ood_attr = _to_numpy(attrib_fn(model, ood_input))
        ood_sum = float(np.sum(np.abs(ood_attr)))

        # Check for OOD signals
        has_nan = bool(np.any(np.isnan(ood_attr)))
        has_inf = bool(np.any(np.isinf(ood_attr)))

        # Attribution pattern change
        if orig_sum > 1e-8:
            ratio = ood_sum / orig_sum
        else:
            ratio = float("inf") if ood_sum > 1e-8 else 1.0

        # OOD detected if: NaN, Inf, or dramatic change (>5x or <0.2x)
        ood_detected = has_nan or has_inf or ratio > 5.0 or ratio < 0.2

        warning = ""
        if has_nan:
            warning = "OOD输入产生NaN归因值"
        elif has_inf:
            warning = "OOD输入产生Inf归因值"
        elif ratio > 5.0:
            warning = f"OOD输入归因值异常增大（{ratio:.1f}倍）"
        elif ratio < 0.2:
            warning = f"OOD输入归因值异常缩小（{ratio:.2f}倍）"
        else:
            warning = "归因值在合理范围内"

        return {
            "ood_detected": ood_detected,
            "warning": warning,
            "original_attribution_sum": round(orig_sum, 6),
            "ood_attribution_sum": round(ood_sum, 6),
            "pass": ood_detected,  # Pass if OOD is correctly detected
        }

    except Exception as e:
        return {
            "ood_detected": True,
            "warning": f"OOD输入导致异常: {str(e)}",
            "original_attribution_sum": round(orig_sum, 6),
            "ood_attribution_sum": 0.0,
            "pass": True,  # Crash = OOD detected
        }


# ═══════════════════════════════════════════════════════════════════════
# Aggregate Runner
# ═══════════════════════════════════════════════════════════════════════

def run_all_tests(
    model: Any,
    input_tensor: Any,
    attrib_fn: Callable,
    n_runs: int = 3,
    noise_level: float = 0.1,
    ood_scale: float = 10.0,
    report_path: Optional[str] = None,
) -> dict:
    """Run all robustness tests and produce an aggregate report.

    Args:
        model: Any model object compatible with attrib_fn.
        input_tensor: Input data (torch Tensor or numpy array).
        attrib_fn: Attribution function: attrib_fn(model, x) -> np.ndarray.
        n_runs: Number of runs for consistency test.
        noise_level: Noise fraction for adversarial test.
        ood_scale: Scale factor for OOD test.
        report_path: If set, write JSON report to this path.

    Returns:
        {
            "timestamp": str,
            "tests": {
                "consistency": {...},
                "boundary": {...},
                "adversarial": {...},
                "ood": {...},
            },
            "summary": {
                "total": int,
                "passed": int,
                "failed": int,
                "pass_rate": float,
            },
            "overall_pass": bool,
        }
    """
    t0 = time.time()

    # Determine input shape
    if hasattr(input_tensor, "shape"):
        input_shape = tuple(input_tensor.shape)
    else:
        input_shape = tuple(np.asarray(input_tensor).shape)

    # Run tests
    consistency = test_attribution_consistency(model, input_tensor, attrib_fn, n_runs=n_runs)
    boundary = test_boundary_conditions(model, attrib_fn, input_shape)
    adversarial = test_adversarial_robustness(
        model, input_tensor, attrib_fn, noise_level=noise_level
    )
    ood = test_ood_detection(model, input_tensor, attrib_fn, ood_scale=ood_scale)

    tests = {
        "consistency": consistency,
        "boundary": boundary,
        "adversarial": adversarial,
        "ood": ood,
    }

    # Summary
    passed = sum(1 for t in tests.values() if t.get("pass", False))
    total = len(tests)
    overall = passed == total

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "tests": tests,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / max(total, 1), 2),
        },
        "overall_pass": overall,
    }

    if report_path:
        import os
        out_dir = os.path.dirname(report_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report


# ═══════════════════════════════════════════════════════════════════════
# CLI Quick Test (requires a simple model for demonstration)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Robustness Test Framework — Self-Test")
    print("=" * 60)

    # Simple numpy model and attribution function for testing
    class SimpleModel:
        """Dummy model: output = sum of input."""
        def __call__(self, x):
            if hasattr(x, "detach"):
                import torch
                return torch.sum(x, dim=-1)
            return np.sum(x, axis=-1)

    def simple_attrib(model, x):
        """Identity attribution (each feature contributes its own value)."""
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy().flatten()
        return np.asarray(x).flatten()

    # Create test input
    np.random.seed(42)
    test_input = np.random.randn(1, 10).astype(np.float32)
    model = SimpleModel()

    results = run_all_tests(
        model, test_input, simple_attrib,
        n_runs=3, noise_level=0.1,
        report_path=None,
    )

    print(f"\nOverall pass: {results['overall_pass']}")
    print(f"Summary: {results['summary']}")
    for name, test in results["tests"].items():
        status = "PASS" if test["pass"] else "FAIL"
        print(f"  {name}: {status}")
