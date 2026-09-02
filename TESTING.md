# Testing

## Deterministic suite

The local suite uses Python's standard `unittest` runner and never calls an external model:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Run the same command on Linux or macOS with `PYTHONPATH=src`.

## Core acceptance checks

Validate the Python sources and Compose file without starting services:

```powershell
$env:PYTHONPATH = "src"
python -m compileall -q src scripts tests
$env:NEO4J_PASSWORD = "local-test-password"
docker compose config --quiet
```

The `scripts/dgx_acceptance.sh` entry point additionally creates an environment, installs the package, and runs the deterministic suite on Linux/DGX hosts.

## Live NVIDIA Kimi K3 gate

The live gate uses NVIDIA's hosted OpenAI-compatible endpoint. It is strictly serial and requires both a runtime `NVIDIA_API_KEY` and explicit `LEGALBENCH_LIVE_CONFIRM=true`.

Check the effective configuration first:

```powershell
$env:PYTHONPATH = "src"
python scripts/check_config.py
```

The default output budget is 20000 tokens. Lower it for a short connectivity smoke test, or set `LIVE_LEGALBENCH_MAX_TOKENS=4096` when a faster, smaller evaluation is intentional. Never run the live gate as a load test.

```powershell
$env:PYTHONPATH = "src"
$env:LEGALBENCH_LIVE_CONFIRM = "true"
python scripts/run_k3_serial.py
```

Secrets must remain in the process environment and must not be committed.
