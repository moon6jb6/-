"""
arch_detect.py — Architecture auto-detection module.

Detects model architecture type from 5 supported categories:
  1. CfC/LNN          (priority 1, highest — also nn.Module)
  2. Transformer       (priority 2)
  3. LSTM/GRU          (priority 3)
  4. XGBoost           (priority 4, no PyTorch dependency)
  5. Generic PyTorch   (priority 99, fallback for any nn.Module)

Priority order: CfC > Transformer > LSTM > XGBoost > Generic PyTorch
"""

import torch.nn as nn

# ── Architecture Registry ──────────────────────────────────────────────

ARCHITECTURE_REGISTRY = {
    "cfc": {
        "priority": 1,
        "check": lambda model: (
            type(model).__name__ == "CfC"
            or hasattr(model, "ncps")
            or ("liquid" in type(model).__module__.lower()
                if hasattr(type(model), "__module__") and type(model).__module__
                else False)
        ),
        "explain_method": "tau_ig_trajectory",
        "description": "CfC/LNN liquid neural network",
    },
    "transformer": {
        "priority": 2,
        "check": lambda model: (
            "GPTNeoX" in type(model).__name__
            or "GPT2" in type(model).__name__
            or "Bert" in type(model).__name__
            or (hasattr(model, "transformer") and hasattr(model, "lm_head"))
            or ("transformer" in type(model).__module__.lower()
                if hasattr(type(model), "__module__") and type(model).__module__
                else False)
        ),
        "explain_method": "rome_and_patching",
        "description": "Transformer architecture (GPT/BERT/etc.)",
    },
    "lstm": {
        "priority": 3,
        "check": lambda model: (
            isinstance(model, (nn.LSTM, nn.GRU))
            or (hasattr(model, "lstm") and isinstance(getattr(model, "lstm", None), (nn.LSTM, nn.GRU)))
            or (hasattr(model, "rnn") and isinstance(getattr(model, "rnn", None), (nn.LSTM, nn.GRU)))
        ),
        "explain_method": "integrated_gradients",
        "description": "LSTM/GRU recurrent neural network",
    },
    "xgboost": {
        "priority": 4,
        "check": lambda model: (
            type(model).__module__.startswith("xgboost")
            if hasattr(type(model), "__module__") and type(model).__module__
            else False
        ),
        "explain_method": "tree_shap",
        "description": "XGBoost gradient boosting trees",
    },
    "generic_pytorch": {
        "priority": 99,
        "check": lambda model: isinstance(model, nn.Module),
        "explain_method": "integrated_gradients",
        "description": "Generic PyTorch model",
    },
}


# ── Core Detection ─────────────────────────────────────────────────────

def detect_architecture(model) -> dict:
    """
    Detect model architecture type.

    Checks architectures in priority order (lower number = higher priority).
    The first match wins. generic_pytorch (priority 99) serves as fallback
    for any nn.Module.

    Args:
        model: Any model object (PyTorch nn.Module, XGBoost booster, etc.)

    Returns:
        dict with keys:
            - architecture: str (e.g. "lstm", "cfc", "transformer", etc.)
            - explain_method: str (recommended attribution method)
            - description: str (human-readable description)
            - confidence: str ("high" for specific match, "low" for generic)

    Raises:
        ValueError: If model does not match any known architecture
                    (i.e. not an nn.Module and not XGBoost).
    """
    # Sort by priority (ascending) so highest-priority (lowest number) is checked first
    sorted_archs = sorted(
        ARCHITECTURE_REGISTRY.items(),
        key=lambda item: item[1]["priority"],
    )

    for arch_name, arch_info in sorted_archs:
        try:
            if arch_info["check"](model):
                confidence = "high" if arch_name != "generic_pytorch" else "low"
                return {
                    "architecture": arch_name,
                    "explain_method": arch_info["explain_method"],
                    "description": arch_info["description"],
                    "confidence": confidence,
                }
        except Exception:
            # If a check function raises, skip to next architecture
            continue

    raise ValueError(
        f"Unrecognized model architecture: {type(model).__name__}. "
        f"Supported: {list(ARCHITECTURE_REGISTRY.keys())}"
    )


def get_supported_architectures() -> list:
    """
    Return a list of all supported architectures with their metadata.

    Returns:
        list of dicts, each containing:
            - name: str
            - priority: int
            - explain_method: str
            - description: str
    """
    return [
        {
            "name": arch_name,
            "priority": arch_info["priority"],
            "explain_method": arch_info["explain_method"],
            "description": arch_info["description"],
        }
        for arch_name, arch_info in sorted(
            ARCHITECTURE_REGISTRY.items(),
            key=lambda item: item[1]["priority"],
        )
    ]


def validate_model_input(model, sample_input) -> bool:
    """
    Validate that the model can accept the given input format.

    Performs a forward pass with torch.no_grad() to check compatibility.
    For non-PyTorch models (e.g. XGBoost), attempts a predict() call.

    Args:
        model: Any model object.
        sample_input: Input tensor or array suitable for the model.

    Returns:
        True if model accepts the input without error.

    Raises:
        RuntimeError: If the forward pass / predict call fails.
    """
    try:
        if isinstance(model, nn.Module):
            model.eval()
            with __import__("torch").no_grad():
                output = model(sample_input)
            return True
        else:
            # XGBoost-style: try predict()
            output = model.predict(sample_input)
            return True
    except Exception as e:
        raise RuntimeError(
            f"Model failed to process sample input: {e}"
        ) from e
