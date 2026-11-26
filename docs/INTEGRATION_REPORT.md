# Part B integration verification

The local `app` branch merged Part A's `9ba482f` without conflicts. Local app
work was preserved first in `593cb66`. Git LFS materialized the 344,546-byte
`weights/airavat.pt` checkpoint. Its SHA-256 is
`30143677fb684a5e3daa92efcc7042de174d63c2ad07d7d509da9b258899796d`.

| Check | Result |
| --- | --- |
| Strict startup in isolated Windows runtime venv | PASS. `airavat-anomaly-unet` ready in 6.43 s. CPU Torch, four threads, xarray and netCDF4 absent. |
| Official schema validator | PASS. All six list/base64 cases, 0.16–0.25 s over loopback IPv4. |
| Contract and impact checks | PASS. 48 contract checks and 10 impact checks. |
| Explicit null vs zero-array and repeat calls | PASS. Exact `np.array_equal` over 10-step outputs. |
| 10-step HTTP wall latency | PASS. Five calls, 0.188 s median, 0.204 s max, 29.93–32.32 ms reported inference. |
| Live dashboard | PASS. Edge showed ready, two live timelines, and response plots. Branch cumulative land rain changed from 84.467 mm on the Bay preset to 84.427 mm on the Arabian preset. |
| Date cutoffs | PASS in executable sources, including training. Splits are computed from dataset years. The dates in `train/validation.json` and checkpoint metadata record fitted periods. |
| Network and offline dashboard | PASS. Source inspection found no external requests in the inference path. Frontend assets are local and its fetches target the local grid and API. |
| Repository contents | PASS. No shipped weather archives. Materialized LFS weight, working tree plus Git about 1.4 MB excluding ignored local environments and fixtures, well below 2 GB. |
| Ignore rules | PASS. `.venv/` and `tests/_windows/` are ignored and untracked. |
| Training dependency split | PASS. The official NetCDF4 archive opened with training dependencies in the local venv. The serving venv used only the runtime dependency set. |
| Official repository scanner | PASS. 15/15 with no flags on deliverable files. The dataset-derived split produced 721 partition-contained validation windows through lead eight. |

## Shock results

First-day locality measures squared response energy within twice the
4.5-cell bump radius. Footprint is the number of cells containing 90% of the
response energy. Both suites passed all 37 checks.

| Window | Shock | First-day locality | Day 1 → 10 footprint | Day 1 → 10 peak |
| --- | --- | ---: | ---: | ---: |
| Synthetic | Bay of Bengal | 100.00% | 69 → 375 | 1.7578 → 0.6719 |
| Synthetic | Arabian Sea | 100.00% | 91 → 396 | 2.7735 → 2.8046 |
| Real | Bay of Bengal | 99.95% | 73 → 533 | 1.1357 → 0.2032 |
| Real | Arabian Sea | 100.00% | 68 → 382 | 4.2500 → 1.2266 |

The real fixture uses three consecutive observations ending at row 195
(day of year 196) of the organizer's NetCDF archive. The row was chosen before
checking the results to match the monsoon-season demo. The all-years checkpoint
was fitted on the archive containing that window. The fixture verifies
counterfactual behavior on observations. Held-out skill figures are available
separately in [train/validation.json](../train/validation.json).

I treated recorded fit periods as provenance for the hardcoded-date audit.
The isolated server used port 8001 to avoid an existing server on port 8000.
Raw suite output and the real fixture are in ignored `tests/_windows/`.
