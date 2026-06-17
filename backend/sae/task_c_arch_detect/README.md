# Task C: Architecture Auto-Detection and Attribution Routing

## Status: DONE

## Output Files

| File | Description |
|------|-------------|
| `arch_detect.py` | Core architecture detection module (5 architectures) |
| `detect_and_explain.py` | Detection + attribution routing framework |
| `test_arch_detect.py` | Unit tests (24 test cases) |
| `INSTRUCTION.md` | Original task instructions |

## Run Tests

```bash
cd backend/sae/task_c_arch_detect
python test_arch_detect.py
```

## Test Results

```
24 tests, 0 failures, 0 errors
```

Coverage:
- 5 architecture detection paths (CfC, Transformer, LSTM/GRU, XGBoost, Generic PyTorch)
- Priority ordering correctness (CfC > Transformer > LSTM > XGBoost > Generic)
- 3 priority conflict scenarios (CfC vs Transformer, Transformer vs LSTM, LSTM vs Generic)
- Input validation (correct and incorrect inputs)
- Error handling (unknown model type -> ValueError)
- Routing framework for all 5 attribution methods

## Architecture Detection Logic

| Priority | Architecture | Detection Criteria | Explain Method |
|----------|-------------|-------------------|----------------|
| 1 | `cfc` | Class name == 'CfC', OR has `ncps` attribute, OR module path contains 'liquid' | `tau_ig_trajectory` |
| 2 | `transformer` | Class name contains 'GPTNeoX'/'GPT2'/'Bert', OR has both `transformer`+`lm_head` attributes, OR module path contains 'transformer' | `rome_and_patching` |
| 3 | `lstm` | Is `nn.LSTM`/`nn.GRU`, OR has `.lstm`/`.rnn` attribute that is `nn.LSTM`/`nn.GRU` | `integrated_gradients` |
| 4 | `xgboost` | `type(model).__module__` starts with 'xgboost' (no PyTorch dependency) | `tree_shap` |
| 99 | `generic_pytorch` | Fallback for any `nn.Module` | `integrated_gradients` |

Priority rule: First match wins, checked from lowest priority number (highest priority) to highest.

## Attribution Routing

Attribution methods are mock/placeholder implementations. The routing framework in `detect_and_explain.py` maps:
- `cfc` -> `_mock_tau_ig_trajectory` (to be implemented by liquid_networks agent)
- `transformer` -> `_mock_rome_and_patching` (to be implemented by sae.compliance agent)
- `lstm` -> `_mock_integrated_gradients` (to be implemented by ig_generic agent)
- `xgboost` -> `_mock_tree_shap` (to be implemented by tree_shap agent)
- `generic_pytorch` -> `_mock_integrated_gradients` (to be implemented by ig_generic agent)
