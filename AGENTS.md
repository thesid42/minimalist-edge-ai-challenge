# Experiment workflow

These preferences come from the team in this chat and apply to future work in this project.

- Train new candidates on the laptop GPU using `.venv-train-gpu`; do local export, accuracy evaluation, latency and energy measurements in CPU `.venv`. AI Hub may use CPU or NPU. Stop local training during CPU benchmarking.
- The organizers clarified that training uses all 27,000 EuroSAT images with our own laptop baseline. Report evaluation on this same pool as training-set accuracy. Keep diagnostic subset and remote scores separate.
- Preserve trained checkpoints. Whenever an experiment is selected as the new best edge candidate, save its ONNX model, recovered weights, recipe, full-data accuracy, hash-matched CPU benchmark and selection reason using `python -m edgeprep.champions` (see `docs/model_preservation.md`). Save the winner before cleanup or publication.
- Keep each winner in a separate versioned `best_models/` bundle. Never delete or overwrite an older winner when promoting a new one. `best_models/current.json` identifies the current selection; `registry.json` keeps winner history.
- A winner is a judged balance of accuracy, worst-class recall, size, CPU latency and energy; do not select by accuracy alone or mix measurements from different devices/protocols. The team explicitly selected recovered192 / FP16-storage as the initial saved winner.
- Update the README, numbered experiment log, notebook and relevant evidence as results complete. Use `scripts/refresh_results.ps1 -ExecuteNotebook` for a verified refresh.
- Selected winner weights under `best_models/` are intended for the public GitHub repository. Credentials, environments, raw data and routine intermediate artifacts remain excluded. Verify model hashes, scan for configured credentials, then push authorized updates.
