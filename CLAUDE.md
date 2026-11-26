# Multiverse of Monsoon

Workflow A (`ml`) owns `engine/`, `model/`, `train/`, `weights/`, `assets/`, and `tests/eval_skill.py`. Workflow B (`app`) owns `serve.py`, `dashboard/`, `tests/check_*.py`, `README.md`, and `brief/`.

Workflow B owns `engine/stub_engine.py` so app development can start before the first model checkpoint. Workflow A owns the rest of `engine/`. Each branch edits only its own files.

`CLAUDE.md` and `CONTRACT.md` are shared. Coordinate any edits after H0. The Engine interface freezes at H0. Discuss any later change with the other workflow before editing it.

Merge to `main` at H0 (contract and starter validation), H1 (first model checkpoint), H2 (rollout and shock improvements), and H3 (final checkpoint and audit). Rebase both branches on `main` after each handoff and before pushing. Keep commits small and push work branches often.

Workflow A runs Workflow B’s contract and shock checks before each model handoff and supplies the validation metrics for the brief. Keep the raw weather archive outside Git.
