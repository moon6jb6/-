"""
test_arch_detect.py — Unit tests for architecture detection and routing.

Run: python test_arch_detect.py

Covers 9 test cases:
  1. nn.LSTM -> "lstm"
  2. nn.GRU -> "lstm" (GRU classified as LSTM category)
  3. Custom model with .lstm attribute -> "lstm"
  4. nn.Module (MLP) -> "generic_pytorch"
  5. XGBoost XGBClassifier -> "xgboost"
  6. Mock CfC model -> "cfc"
  7. Unknown non-nn.Module type -> ValueError
  8. get_supported_architectures() returns 5 architectures
  9. validate_model_input for correct/incorrect inputs
"""

import sys
import unittest
from pathlib import Path

# Ensure the task_c_arch_detect directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn

from arch_detect import (
    ARCHITECTURE_REGISTRY,
    detect_architecture,
    get_supported_architectures,
    validate_model_input,
)
from detect_and_explain import detect_and_explain


# ── Mock Models for Testing ────────────────────────────────────────────

class SimpleLSTM(nn.Module):
    """Wraps nn.LSTM — should be detected as 'lstm'."""
    def __init__(self, input_size=10, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class SimpleGRU(nn.Module):
    """Wraps nn.GRU — should be detected as 'lstm'."""
    def __init__(self, input_size=10, hidden_size=32):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class SimpleMLP(nn.Module):
    """Plain MLP — should fall through to 'generic_pytorch'."""
    def __init__(self, input_size=10, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.net(x)


class MockCfC(nn.Module):
    """Simulates CfC model with ncps attribute — should be detected as 'cfc'."""
    def __init__(self, input_size=10, hidden_size=32):
        super().__init__()
        self.ncps = True  # Marker attribute from ncps.torch.CfC
        self.linear = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.out(torch.relu(self.linear(x)))


class MockCfCByName(nn.Module):
    """Class named 'CfC' — should be detected as 'cfc' by name."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.linear(x)


class MockTransformer(nn.Module):
    """Has transformer + lm_head attributes — should be detected as 'transformer'."""
    def __init__(self, vocab_size=100, d_model=32):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4), num_layers=1
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        out = self.transformer(x)
        return self.lm_head(out)


class MockXGBoostInstance:
    """Simulates XGBoost model — type().__module__ starts with 'xgboost'."""
    __module__ = "xgboost.sklearn"

    def predict(self, X):
        return [0] * len(X)


class BareObject:
    """Not an nn.Module, not XGBoost — should raise ValueError."""
    pass


# ── Named 'CfC' class for name-based detection ────────────────────────
def _make_cfc():
    """Dynamically create a class named 'CfC' that extends nn.Module."""
    cls = type("CfC", (nn.Module,), {})
    obj = cls()
    obj.linear = nn.Linear(10, 1)
    # Monkey-patch forward
    import types
    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.linear(x)
    obj.forward = types.MethodType(forward, obj)
    return obj


# ── Test Suite ─────────────────────────────────────────────────────────

class TestDetectArchitecture(unittest.TestCase):
    """Tests for detect_architecture()."""

    def test_01_lstm_raw(self):
        """nn.LSTM (raw, not wrapped) -> 'lstm'."""
        model = nn.LSTM(input_size=10, hidden_size=32, batch_first=True)
        result = detect_architecture(model)
        self.assertEqual(result["architecture"], "lstm")
        self.assertEqual(result["explain_method"], "integrated_gradients")
        self.assertEqual(result["confidence"], "high")

    def test_02_gru_raw(self):
        """nn.GRU (raw, not wrapped) -> 'lstm' (GRU classified under LSTM category)."""
        model = nn.GRU(input_size=10, hidden_size=32, batch_first=True)
        result = detect_architecture(model)
        self.assertEqual(result["architecture"], "lstm")
        self.assertEqual(result["confidence"], "high")

    def test_03_custom_model_with_lstm_attr(self):
        """Custom nn.Module with .lstm attribute -> 'lstm'."""
        model = SimpleLSTM()
        result = detect_architecture(model)
        self.assertEqual(result["architecture"], "lstm")

    def test_04_generic_pytorch_mlp(self):
        """Plain MLP (nn.Module) -> 'generic_pytorch' (fallback)."""
        model = SimpleMLP()
        result = detect_architecture(model)
        self.assertEqual(result["architecture"], "generic_pytorch")
        self.assertEqual(result["explain_method"], "integrated_gradients")
        self.assertEqual(result["confidence"], "low")

    def test_05_xgboost(self):
        """XGBoost model with xgboost module -> 'xgboost'."""
        model = MockXGBoostInstance()
        result = detect_architecture(model)
        self.assertEqual(result["architecture"], "xgboost")
        self.assertEqual(result["explain_method"], "tree_shap")
        self.assertEqual(result["confidence"], "high")

    def test_06_cfc_by_ncps_attr(self):
        """CfC model with .ncps attribute -> 'cfc'."""
        model = MockCfC()
        result = detect_architecture(model)
        self.assertEqual(result["architecture"], "cfc")
        self.assertEqual(result["explain_method"], "tau_ig_trajectory")
        self.assertEqual(result["confidence"], "high")

    def test_06b_cfc_by_class_name(self):
        """CfC model detected by class name 'CfC' -> 'cfc'."""
        model = _make_cfc()
        result = detect_architecture(model)
        self.assertEqual(result["architecture"], "cfc")

    def test_06c_cfc_priority_over_generic(self):
        """CfC is nn.Module but should match 'cfc' (priority 1), not 'generic_pytorch'."""
        model = MockCfC()
        self.assertIsInstance(model, nn.Module)  # It IS an nn.Module
        result = detect_architecture(model)
        self.assertEqual(result["architecture"], "cfc")  # But matches cfc first

    def test_07_unknown_raises_valueerror(self):
        """Non-nn.Module, non-XGBoost object -> ValueError."""
        model = BareObject()
        with self.assertRaises(ValueError) as ctx:
            detect_architecture(model)
        self.assertIn("Unrecognized", str(ctx.exception))

    def test_08_get_supported_architectures(self):
        """get_supported_architectures() returns exactly 5 architectures."""
        archs = get_supported_architectures()
        self.assertEqual(len(archs), 5)
        names = [a["name"] for a in archs]
        self.assertIn("cfc", names)
        self.assertIn("transformer", names)
        self.assertIn("lstm", names)
        self.assertIn("xgboost", names)
        self.assertIn("generic_pytorch", names)

    def test_08b_architectures_sorted_by_priority(self):
        """Architectures are returned in priority order."""
        archs = get_supported_architectures()
        priorities = [a["priority"] for a in archs]
        self.assertEqual(priorities, sorted(priorities))

    def test_08c_registry_has_5_entries(self):
        """ARCHITECTURE_REGISTRY has exactly 5 entries."""
        self.assertEqual(len(ARCHITECTURE_REGISTRY), 5)

    def test_09_validate_model_input_correct(self):
        """validate_model_input returns True for compatible input."""
        model = SimpleMLP()
        sample_input = torch.randn(1, 10, 10)  # (batch, seq, features)
        result = validate_model_input(model, sample_input)
        self.assertTrue(result)

    def test_09b_validate_model_input_incorrect(self):
        """validate_model_input raises RuntimeError for incompatible input."""
        model = SimpleMLP(input_size=10)
        bad_input = torch.randn(1, 10, 99)  # Wrong feature dimension
        with self.assertRaises(RuntimeError):
            validate_model_input(model, bad_input)

    def test_09c_validate_gru_input(self):
        """validate_model_input works for GRU model."""
        model = SimpleGRU(input_size=10, hidden_size=16)
        sample_input = torch.randn(2, 5, 10)
        result = validate_model_input(model, sample_input)
        self.assertTrue(result)

    # ── Priority correctness tests ────────────────────────────────────

    def test_priority_cfc_over_transformer(self):
        """If a model matches both CfC and Transformer, CfC wins."""
        # This model has .transformer and .lm_head (transformer attributes)
        # AND .ncps (CfC attribute). CfC should win.
        class AmbiguousModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.ncps = True
                self.transformer = nn.Linear(10, 10)
                self.lm_head = nn.Linear(10, 10)
            def forward(self, x):
                return x

        model = AmbiguousModel()
        result = detect_architecture(model)
        self.assertEqual(result["architecture"], "cfc")

    def test_priority_transformer_over_lstm(self):
        """If a model matches both Transformer and LSTM, Transformer wins."""
        # Not typical, but validates priority ordering
        class AmbiguousModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(10, 10, batch_first=True)
                self.transformer = nn.Linear(10, 10)
                self.lm_head = nn.Linear(10, 10)
            def forward(self, x):
                return x

        model = AmbiguousModel()
        result = detect_architecture(model)
        self.assertEqual(result["architecture"], "transformer")

    def test_priority_lstm_over_generic(self):
        """LSTM model is nn.Module but matches 'lstm' (not 'generic_pytorch')."""
        model = SimpleLSTM()
        self.assertIsInstance(model, nn.Module)
        result = detect_architecture(model)
        self.assertEqual(result["architecture"], "lstm")


class TestDetectAndExplain(unittest.TestCase):
    """Tests for the detect_and_explain routing framework."""

    def test_route_lstm(self):
        """detect_and_explain routes LSTM to integrated_gradients."""
        model = SimpleLSTM()
        input_data = torch.randn(1, 5, 10)
        result = detect_and_explain(model, input_data)
        self.assertEqual(result["architecture"], "lstm")
        self.assertEqual(result["attribution_method"], "integrated_gradients")
        self.assertIn("report", result)
        self.assertEqual(result["report"]["status"], "mock")

    def test_route_cfc(self):
        """detect_and_explain routes CfC to tau_ig_trajectory."""
        model = MockCfC()
        input_data = torch.randn(1, 5, 10)
        result = detect_and_explain(model, input_data)
        self.assertEqual(result["architecture"], "cfc")
        self.assertEqual(result["attribution_method"], "tau_ig_trajectory")

    def test_route_transformer(self):
        """detect_and_explain routes Transformer to rome_and_patching."""
        model = MockTransformer()
        input_data = torch.randn(1, 5, 32)
        result = detect_and_explain(model, input_data)
        self.assertEqual(result["architecture"], "transformer")
        self.assertEqual(result["attribution_method"], "rome_and_patching")

    def test_route_generic(self):
        """detect_and_explain routes generic MLP to integrated_gradients."""
        model = SimpleMLP()
        input_data = torch.randn(1, 5, 10)
        result = detect_and_explain(model, input_data)
        self.assertEqual(result["architecture"], "generic_pytorch")
        self.assertEqual(result["attribution_method"], "integrated_gradients")
        self.assertEqual(result["confidence"], "low")

    def test_route_xgboost(self):
        """detect_and_explain routes XGBoost to tree_shap."""
        model = MockXGBoostInstance()
        result = detect_and_explain(model, input_data=None)
        self.assertEqual(result["architecture"], "xgboost")
        self.assertEqual(result["attribution_method"], "tree_shap")

    def test_route_unknown_raises(self):
        """detect_and_explain raises ValueError for unknown model type."""
        model = BareObject()
        with self.assertRaises(ValueError):
            detect_and_explain(model)


# ── Run Tests ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Architecture Detection Unit Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
