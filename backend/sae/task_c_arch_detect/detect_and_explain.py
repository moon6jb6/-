"""
detect_and_explain.py — Detection + Attribution routing framework.

Workflow:
  1. detect_architecture(model) -> get architecture type
  2. Route to corresponding attribution method based on architecture
  3. Return attribution result

Attribution methods are mock/placeholder implementations.
The actual attribution functions will be implemented by Agent A and B.
This module focuses on the routing logic.
"""

from typing import Any, Optional

from arch_detect import detect_architecture


# ── Attribution Mock/Placeholder Functions ─────────────────────────────
# These mock functions simulate the routing target. The actual implementations
# will be provided by other agents:
#   - CfC/LNN     -> interpret_base.py (Agent B: liquid_networks)
#   - Transformer  -> sae.compliance (Agent A: causal_ablation/activation_patching)
#   - LSTM/GRU     -> ig_generic.py (Agent B)
#   - XGBoost      -> tree_shap.py (Agent B)
#   - Generic      -> ig_generic.py (Agent B)


def _mock_tau_ig_trajectory(model, input_data, **kwargs) -> dict:
    """Mock: CfC/LNN tau + integrated gradients trajectory attribution."""
    return {
        "method": "tau_ig_trajectory",
        "status": "mock",
        "message": "CfC/LNN attribution — to be implemented by liquid_networks agent",
        "attributions": None,
    }


def _mock_rome_and_patching(model, input_data, **kwargs) -> dict:
    """Mock: Transformer ROME causal tracing + activation patching."""
    return {
        "method": "rome_and_patching",
        "status": "mock",
        "message": "Transformer attribution — to be implemented by sae.compliance agent",
        "attributions": None,
    }


def _mock_integrated_gradients(model, input_data, **kwargs) -> dict:
    """Mock: Generic integrated gradients for LSTM/GRU or generic PyTorch."""
    return {
        "method": "integrated_gradients",
        "status": "mock",
        "message": "Integrated gradients — to be implemented by ig_generic agent",
        "attributions": None,
    }


def _mock_tree_shap(model, input_data, **kwargs) -> dict:
    """Mock: XGBoost TreeSHAP attribution."""
    return {
        "method": "tree_shap",
        "status": "mock",
        "message": "TreeSHAP — to be implemented by tree_shap agent",
        "attributions": None,
    }


# ── Attribution Router ─────────────────────────────────────────────────

_ATTRIBUTION_ROUTES = {
    "tau_ig_trajectory": _mock_tau_ig_trajectory,
    "rome_and_patching": _mock_rome_and_patching,
    "integrated_gradients": _mock_integrated_gradients,
    "tree_shap": _mock_tree_shap,
}


def detect_and_explain(model, input_data=None, **kwargs) -> dict:
    """
    Auto-detect model architecture and route to corresponding attribution.

    Flow:
      1. detect_architecture(model) -> architecture type + explain_method
      2. Look up explain_method in attribution router
      3. Call the corresponding attribution function
      4. Return combined result

    Args:
        model: Any supported model object.
        input_data: Input tensor/array for attribution (optional for detection).
        **kwargs: Additional arguments passed to attribution method.

    Returns:
        dict with keys:
            - architecture: str (detected architecture type)
            - architecture_description: str
            - confidence: str ("high" or "low")
            - attribution_method: str (name of attribution method)
            - attributions: attribution result (or None for mock)
            - report: dict with attribution details
    """
    # Step 1: Detect architecture
    arch_info = detect_architecture(model)

    arch_name = arch_info["architecture"]
    explain_method = arch_info["explain_method"]

    # Step 2: Route to attribution method
    attr_fn = _ATTRIBUTION_ROUTES.get(explain_method)
    if attr_fn is None:
        raise ValueError(
            f"No attribution method registered for '{explain_method}'. "
            f"Available: {list(_ATTRIBUTION_ROUTES.keys())}"
        )

    # Step 3: Run attribution
    attr_result = attr_fn(model, input_data, **kwargs)

    # Step 4: Assemble response
    return {
        "architecture": arch_name,
        "architecture_description": arch_info["description"],
        "confidence": arch_info["confidence"],
        "attribution_method": explain_method,
        "attributions": attr_result.get("attributions"),
        "report": {
            "status": attr_result.get("status", "unknown"),
            "message": attr_result.get("message", ""),
            "method": explain_method,
        },
    }
