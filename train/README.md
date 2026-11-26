# Local training and evaluation

Install the local training dependencies with `python -m pip install -r requirements-train.txt` from the repository root. The official training archive uses NetCDF-4. Xarray uses the compiled `netCDF4` backend to read it during training and data inspection. The CPU Engine loads its model and static fields from `weights/airavat.pt`. The organizer's offline serving environment needs the runtime dependencies only.

```sh
python -m train.inspect /path/to/india_climate_train.nc
python -m train.eval_baselines /path/to/india_climate_train.nc --val-years 2
python -m train.train /path/to/india_climate_train.nc --split train --val-years 2 --output /tmp/airavat-validation/airavat.pt
python -m tests.eval_skill /path/to/india_climate_train.nc --val-years 2 --weights /tmp/airavat-validation
python -m train.train /path/to/india_climate_train.nc --split all --output weights/airavat.pt
```

`--val-years 2` reserves the last two calendar years represented in the dataset. This includes a partial final year if one is present. Every forecast window stays within its partition. `--split all` fits on the entire time axis for deployment. To run the optional shock diagnostics in `tests.eval_skill`, install the official kit wheel and supply its `stats.json`.
