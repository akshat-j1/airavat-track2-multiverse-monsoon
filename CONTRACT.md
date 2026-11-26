# Track 2 Engine Contract

This is the Python interface that `serve.py` uses to implement the [official Track 2 v1.0.0 API contract](https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/download/v1.0.0/API_CONTRACT.md).

```python
# engine/engine.py
class Engine:
    model_name: str

    def __init__(self, weights_dir: str): ...

    def simulate(self, init_states, doy_sincos, steps, perturbation=None): ...
```

The constructor loads the model once for CPU float32 inference. The server can reuse the same instance across calls.

| Argument or result | Contract |
| --- | --- |
| `init_states` | NumPy `float32` array with shape `(3, 6, 32, 36)` in physical units. The three days are consecutive, oldest first. |
| `doy_sincos` | `[sin(2π × doy / 365), cos(2π × doy / 365)]` for the last input day. The API does not send the calendar year. |
| `steps` | Number of future daily states to return. |
| `perturbation` | Optional NumPy `float32` array with shape `(6, 32, 36)` in physical units, aligned with the input channels and grid. The engine adds the delta to a copy of the last input state before forecasting. |
| Return value | NumPy `float32` array with shape `(steps, 6, 32, 36)` in physical units. Index 0 is the first day after the inputs. |

The three trailing axes are channel, latitude, longitude. The 1° cell centers run from 6.5° to 37.5° latitude (south to north) and 65.5° to 100.5° longitude (west to east).

| Channel index | Variable | Unit |
| --- | --- | --- |
| 0 | `t2m` | K |
| 1 | `tp` | mm/day |
| 2 | `u10` | m/s |
| 3 | `v10` | m/s |
| 4 | `msl` | Pa |
| 5 | `tcwv` | kg/m² |

Use `stats.json` only for internal normalization. Bundle `land_sea_mask` and `orography_m` with the model, since the API supplies only the dynamic fields.

Rollouts must be deterministic for identical inputs, contain only finite values, and return nonnegative precipitation. An explicit all-zero perturbation must produce exactly the same output as `None` under `numpy.array_equal`. Inference may write only to `/tmp`.

The development stub in `engine/stub_engine.py` implements the same interface.
