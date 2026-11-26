# Part B integration report

**Decision: DO NOT PUSH.** Runtime, API, dashboard, and requested shock checks
pass, but two Part A source/packaging findings fail the user's broader repo
rules. No push, force-push, or push to `main` was performed. The final integration
changes and this report are left uncommitted for review. Local `app` HEAD is
`8db2806` (preserved app work); it is not a pushed integration commit.

## Sync and ownership

`CLAUDE.md` and `CONTRACT.md` were missing in the initial unborn checkout, so they
were read from the remote-tracking branches before integration. The checkout was
on an unborn `main`, with all app work untracked. That work was preserved in a
local `app` commit, then rebased onto the fetched shared history as `8db2806`.
`CLAUDE.md` requires rebasing on shared `main` at handoff. No conflicts occurred.

After `git fetch --all --prune`, local `main` was created to track `origin/main`.
Both `origin/main` and `origin/ml` resolve to
`2da1db8c9e39ef5742176c62f5711d045194109c`. Consequently,
`git diff --stat main...origin/ml` is empty and `git merge origin/ml` reported
`Already up to date`. The ML log requested with `-10` contains four commits:

```text
2da1db8 H3: freeze the all-years forecast checkpoint and validation metrics
0b87d20 H2: improve rollout skill, rain spread, and shock dynamics
1cddc40 H1: train and integrate the first forecast checkpoint
92a843c H0: establish the verified Track 2 contract
```

`git lfs install` and `git lfs pull` succeeded. `git lfs ls-files` reports
`53ea2d4486 * weights/airavat.pt`; the file is 344,418 bytes, not a pointer.
Its SHA-256 is
`53ea2d448627f44bf220f147c863e0078514955aa22441d03796d2d2802e8eb3`.

`git diff --name-only origin/ml -- engine/engine.py model/ train/ weights/ assets/ tests/eval_skill.py`
returns no paths. All Part A files are unchanged.

## Part A findings requiring teammate action

### FAIL: hardcoded training dates under the user's repo-wide rule

Evidence: [train/common.py](../train/common.py), lines 11–12:

```python
TRAIN_END = np.datetime64("2020-01-01")
VALIDATION_END = np.datetime64("2022-01-01")
```

These control training/evaluation splits. They are **not inference dates** and
are not an inference anti-cheat violation according to the official scanner,
which checks dates only in inference code. They nevertheless fail the user's
literal instruction to have no hardcoded dates in the repo's runnable source.

Exact requested fix for Part A: accept train and validation cutoff dates as
required training/evaluation arguments or dataset metadata. Pass them into
`window_ends` and the fitting/validation selection logic in `train/common.py`,
`train/train.py`, `train/eval_baselines.py`, and `tests/eval_skill.py`, instead
of importing fixed date constants. Derive evaluation year labels from those
configured splits. Preserve held-out training separation and rerun the training
and evaluation checks. Alternatively, the user must explicitly narrow this rule
to inference code before this finding can be cleared without a source change.

### FAIL: declared dependency outside the supplied sandbox allowlist

Evidence: [train/requirements.txt](../train/requirements.txt), line 4, declares
`netCDF4`. It is not in the supplied sandbox list, is not installed in the test
venv, and no vendored copy exists in the repository. The root runtime
`requirements.txt` added by Part B contains only allowed libraries and no version
pins. Inference itself does not import `netCDF4`.

Exact requested fix for Part A: remove `netCDF4` from the declared training
requirements if it is unnecessary and document a NetCDF3 input conversion so
`xarray` can read training data through the supplied SciPy backend. If reading
NetCDF4 directly remains required, vendor a working backend and its native
dependencies for the organizer target platform, configure its import path, and
verify training with networking disabled. Do not assume this package is present
or downloadable in the offline sandbox. The official archive's container format
was not independently inspected in this integration; this finding concerns the
unvendored declared dependency, not an observed inference import failure.

## Interface and runtime results

Model: **`airavat-anomaly-unet`**. Local testing used activated `.venv`, Python
3.14.10 and PyTorch `2.15.0+cpu` (`torch.version.cuda is None`). No signature adapter
was necessary. The constructor is `Engine(weights_dir)` and simulation is
`simulate(init_states, doy_sincos, steps, perturbation=None)`.

| Check | Result and evidence |
| --- | --- |
| Loads the supplied weights directory | PASS; copied checkpoint into an external temporary directory and constructed Engine from that directory |
| CPU float32, eval mode, no dropout | PASS; all model parameters CPU float32, `.training == False`, no dropout modules |
| Four-thread CPU limit | PASS; `torch.get_num_threads() == 4`, OMP and MKL limits set to 4 |
| Raw engine result | PASS; float32 `(10,6,32,36)`, all finite, rain nonnegative |
| Raw physical ranges | PASS; t2m 274.107–309.258 K; msl 99,695.5–101,682.063 Pa; tcwv 4.554–70.337 kg/m² |
| Determinism and zero/null equality | PASS; raw `np.array_equal`, and HTTP checks also pass |
| No inference randomness or input mutation | PASS; Torch RNG state and original input unchanged |
| No network or file writes during engine load/inference | PASS in guarded direct run; no socket connect/DNS events or file-write opens observed |
| Strict startup, no fallback | PASS; `MONSOON_STRICT=1`, `/health` ready with real model, startup 5.804 s against 120 s limit |
| Official schema validator | PASS; exit 0, all six list/base64/no-shock/warm/zero cases pass; 2.23–2.29 s per localhost call |
| `tests/check_contract.py --require-real` | PASS; 48 checks, exit 0; malformed inputs return 400 |
| `tests/check_shocks.py --require-real` | PASS; 37 checks, exit 0 |
| Indicative impact numerical tests | PASS; 10 tests in `node tests/check_impact.js` |

Ten-step engine-only timings: 0.0384, 0.0306, 0.0299, 0.0293, 0.0285 seconds.
Ten-step full HTTP wall times on explicit IPv4: 0.2426, 0.2034, 0.2026, 0.2050,
0.1894 seconds; **median 0.2034 s, maximum 0.2426 s**, including JSON encoding,
transfer and decoding. Corresponding server inference times were 29.66–30.99 ms.

On this Windows host, `localhost` resolves IPv6 first and adds roughly two
seconds before falling back to IPv4. The mandatory validator command was run
unmodified against localhost and still passed its 5 s gate. Startup measurement
used an IPv4 readiness probe after clearing stale preview listeners, so it does
not incorporate a repeated localhost probe timeout.

### Requested localized shocks

Both use a raised-cosine bump of radius 4.5 cells, ten returned days, and the
contract grid. Step 0 below is API day 1; last step is API day 10.

| Shock | Day-1 energy within 2R | 90% energy footprint, day 1 → 10 | Mean distance, day 1 → 10 | Peak, day 1 → 10 |
| --- | --- | --- | --- | --- |
| Bay t2m +2 K, requested centre 16°N/88°E | 90.07% | 142 → 396 cells | 3.7523 → 13.8072 cells | 2.2500 → 0.6250 |
| Arabian u10 −3 m/s, requested centre 16°N/70°E | 80.07% | 211 → 426 cells | 6.8926 → 16.1918 cells | 2.4687 → 3.2032 |

All response values are finite; both repeat calls are deterministic. Both pass
the ≥80% locality gate, outward-growth gate and last-peak ≤3× first-peak gate.
**Risk:** Arabian locality clears the threshold by only about 0.07 percentage
points on the synthetic window. This is not a guarantee for held-out windows.
The check uses squared differences across physical channels without channel
standardization; the peak is the maximum absolute difference across those
channels in their respective native units. It is not the official standardized
causality score.

## Dashboard verification

PASS: opened `/dashboard/` in the local browser against the strict real server.
The health badge displayed the real model. Running the Bay shock replaced input
previews with live forecast maps, with measured inference times (one UI run:
Prime 43.01 ms, Branch 69.76 ms). The coastline came from the checkpoint mask.

PASS: changing the UI shock to heat dome changed live indicative impact outputs:

| Mean per land cell, full ten-day rollout | Bay shock | Heat dome |
| --- | --- | --- |
| Prime cumulative rain | 85.834 mm | 85.834 mm |
| Branch cumulative rain | 85.890 mm | 86.007 mm |
| Delta cumulative rain | +0.056 mm | +0.174 mm |
| Branch rain-deficit index | −1.147 | −1.150 |

The UI used 736 land cells and excluded ocean. The cumulative-rain layer and
its delta map selected correctly. Policy sentences reflected the computed
values and carried the indicative/non-agronomic labels.

PASS: all frontend source/assets are under `dashboard/`. Source scan found no
CDN, external font, remote script, or external asset references. Only the local
API and local `grid.json` are fetched. `dashboard/grid.json` contains coordinates
and the checkpoint's two permitted static fields, not past weather.

## Repo rules audit

| Requested item | Result |
| --- | --- |
| No weather APIs/network in `serve.py`, `engine/`, `model/` | PASS; source scan has no hits; guarded engine run has no network events |
| No hardcoded dates in inference | PASS |
| No hardcoded dates across runnable source | **FAIL — `train/common.py:11` and `train/common.py:12`, Part A** |
| No shipped past-weather `.nc`, `.npy`, `.npz`, `.zip`, `.csv` archives | PASS; no matching deliverable paths; test data created outside the repo and removed |
| Training code present and runnable | PASS, limited smoke test: 370 synthetic days in temporary NetCDF3, one width-4 CPU epoch, checkpoint written; no held-out skill/retraining claim |
| Requirements match supplied environment | Root runtime PASS; **Part A declaration FAIL — `train/requirements.txt:4`, `netCDF4`** |
| Real weights included via Git LFS | PASS; real file, size and SHA above |
| Repository under 2 GB | PASS; audited deliverable files plus `.git` about 0.98 MB; whole workspace including `.venv` about 1.03 GB |
| `.venv/`, `tests/_windows/`, `__pycache__/`, `*.log` ignored, not staged | PASS; ignore checks succeed and index is empty |
| README explains training and running | PASS; root README includes activated-venv commands, training/evaluation commands, dependency caveat and tests |
| Part A files untouched | PASS; comparison with `origin/ml` is empty |

Official `monsoon_r1.repo_scan.scan` result: **15/15**, anti-cheat 8/8, code health
7/7, no flags. That result does not override the two stricter user-rule failures.

## Part B fixes completed locally

- Corrected fallback coordinates to the contract south-to-north grid, day-of-year
  encoding to the contract's 365-day period, and `tp` labels to mm/day.
- Aligned server/dashboard limits to the official kit's 14-step maximum.
- Added strict malformed/nonfinite-step and unit-circle phase validation.
- Accepted the official validator's optional little-endian float32 base64 input,
  while preserving canonical nested-list output. No official kit import at runtime.
- Exported only checkpoint static fields to `dashboard/grid.json`; exporter also
  supports the official packed NetCDF static-channel layout.
- Added allowed-library runtime requirements, missing ignore rules, and README.

## Assumptions and verification limits

1. Creating `app` and anchoring the unborn checkout to fetched shared `main`
   preserves the intended branch roles; remote `ml`/`main` were already identical.
2. The user's no-hardcoded-dates rule applies to runnable training source too,
   whereas historical metadata/documentation is evidence, not executable date logic.
3. The dependency allowlist applies to declared repository training requirements,
   not just server imports. The official kit and its `requests` client are local
   test tooling, excluded from shipped runtime requirements and source.
4. Windows' external temporary directory is the local equivalent of `/tmp`.
   No inference file writes were observed anywhere; Linux read-only-container
   execution was not reproduced on this Windows machine.
5. API contract, shock, timing and browser tests used synthetic physical windows.
   The small training smoke used synthetic NetCDF3 input. The real training archive
   and blind/held-out weather cases were unavailable and skill was not recomputed.
6. Chosen shock centres are 16°N/88°E and 16°N/70°E, mapped to nearest contract
   cells (15.5°N/87.5°E and 15.5°N/69.5°E). The earlier Arabian 66°E centre was
   moved inside the official sea box to avoid clipping the bump at the domain edge.
7. The checkpoint's `static` entries are mask then orography, as established by
   `model/unet.py` and `train/train.py`. No synthetic land boundary was substituted.
8. Activated-venv `python` was used throughout. Windows child test processes use
   that interpreter's `sys.executable` to avoid Windows resolving a base interpreter
   before the activated PATH. Local library versions are not proof of compatibility
   with every organizer version or platform.

Raw local evidence is in ignored `tests/_windows/`: `engine-evidence.json`,
`startup-evidence.json`, `validator.log`, `contract.log`, `shocks.log`,
`http-evidence.json`, `training-evidence.json`, `audit-evidence.json`, and
`repo-scan.json`. Essential results are reproduced above because these scratch
artifacts are not intended to be committed.

**Next action belongs to Part A or the user:** resolve the two flagged policy
findings, then repeat the checks and only then commit/push `app`. No Part A fix
was attempted by Part B.
