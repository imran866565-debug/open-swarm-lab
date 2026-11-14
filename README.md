# OPEN SWARM lab — first real experiment

**One prompt in, one coherent AI answer out — computed by a swarm, not a server.**

OPEN SWARM is an attempt to prove that independent, voluntary contributor
devices can form **one logical AI cloud**: each node contributes storage,
RAM, CPU and bandwidth, and a user submits a single prompt and gets a
single answer without managing any node.

This repository is experiment #1. It is **not** the Android app
(`open-swarm-android` is a separate, private repo and is untouched by this
work). It runs on free GitHub-hosted Ubuntu runners and on any Linux box.

## What this experiment actually proves (and what it does not)

| Claim | Status in this repo |
|---|---|
| 3 worker processes start automatically, unique peer IDs, all register | Tested (A) |
| Heartbeats, discovery, dead-node detection | Tested (B) |
| Resource advertisement + capacity-based assignment | Tested (C) |
| Real Qwen2.5-0.5B inference distributed across 3 RPC workers | Tested (D) — see evidence rules below |
| One prompt → one real model answer (×3 prompts) | Tested (E) |
| 2 concurrent prompts, no mixing | Tested (F) |
| Kill a node → honest failure → restart → recovery | Tested (G) |
| Chunk hashes, corruption / loss detection, rebuild | Tested (H) |
| CPU / RAM / network / timing metrics | Tested (I) |

**Honest topology label — read this first:** the three workers in this
experiment are **three LOGICAL nodes on ONE physical VM** (one
GitHub-hosted runner, communicating over `127.0.0.1`). They prove the
coordination, distribution and failure logic — they do **not** prove three
independent physical computers. See
[`docs/multi-runner-investigation.md`](docs/multi-runner-investigation.md)
for why separate GitHub jobs cannot simply peer with each other, and what
the real path to multi-machine swarms is (contributor devices running
self-hosted runners).

**Distribution honesty rule:** a copy of the full model on every worker
would *not* count. Here the model weights are split by llama.cpp's RPC
backend: `llama-cli --rpc 127.0.0.1:50052,127.0.0.1:50053,127.0.0.1:50054
-ngl 99 --split-mode layer --device RPC0,RPC1,RPC2 --tensor-split 1,1,1`
offloads transformer layers across the three RPC server processes
(`llama-rpc-server` / `ggml-rpc-server`, name changed across versions).
The `--device` + `--tensor-split 1,1,1` flags matter: without them the
client's local CPU also takes a share. Participation is proven two ways
(recorded in `results/metrics.json` under `distribution`): the client
log's per-RPC `model buffer size` lines when the client prints them, and
— since llama.cpp v1.0.0's `llama-cli` stays silent about them at default
verbosity — every RPC server's own log showing the client handshake plus
repeated `[graph_compute]` executions (`n_nodes`/`n_tensors` per compute).
A server that never received tensors cannot compute graphs. If neither
proof fires, test D is marked FAIL — never faked.
test D is marked FAIL — never faked.

## Architecture

```
user prompt
    │  swarm/client.py  — "one prompt, one answer" (stdout = answer only)
    ▼
coordinator (swarm/coordinator.py, 127.0.0.1:8471)
    │  /register  /heartbeat  /nodes  /submit  /task/<id>/complete|fail
    │  assigns the request to ALL alive RPC nodes as one pool;
    │  fails honestly if fewer than 3 nodes are alive
    ▼
worker-0 ── llama-rpc-server 127.0.0.1:50052 ┐
worker-1 ── llama-rpc-server 127.0.0.1:50053 ├─ llama-cli --rpc <all three>
worker-2 ── llama-rpc-server 127.0.0.1:50054 ┘   -ngl 99  (layers split)
    ▲  each worker: unique peer_id, heartbeats every 2s,
       advertises CPU / RAM / disk, owns its RPC server
```

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct-GGUF`, `Q4_K_M`
  (0.49B params, 24 layers, 491,400,032 bytes, SHA-256 pinned in
  `scripts/run_experiment.py`), downloaded from Hugging Face at run time.
- **Runtime:** llama.cpp, pinned ref `v1.0.0`, built with `-DGGML_RPC=ON`.
- **Code:** pure-stdlib Python 3.10+ — no pip dependencies, no paid APIs,
  no cloud databases.

## Reproduce it

### On GitHub Actions (the intended test environment)

1. Push to `main`, or **Actions → Open Swarm Experiment → Run workflow**.
2. Wait (~15–30 min: llama.cpp build + ~350 MB model download + tests).
3. Download the `open-swarm-results` artifact: `results/metrics.json`,
   `results/TEST_REPORT.md`, `results/coordinator-state.json`, `results/logs/`.

### Locally (same script the workflow runs)

```bash
git clone https://github.com/imran866565-debug/open-swarm-lab.git
cd open-swarm-lab
# needs: python3.10+, git, cmake, build-essential, curl
python3 scripts/run_experiment.py --max-tokens 96
# results in results/metrics.json, results/TEST_REPORT.md, results/logs/
```

Fast checks only (no model, no build):

```bash
python -m unittest discover -s tests -v
```

One prompt, one answer (once a swarm is running):

```bash
python -m swarm.client --prompt "Explain how a decentralized cloud works." \
  --cli-bin vendor/llama.cpp/build/bin/llama-cli \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

## Tests (A–I)

In `tests/test_swarm.py` (unit, stdlib `unittest`) and
`scripts/run_experiment.py` (integration, real model):

- **A** startup · **B** discovery/heartbeats · **C** resource sharing ·
  **D** model distribution · **E** task execution (3 prompts) ·
  **F** concurrent users · **G** node failure + recovery ·
  **H** data integrity (SHA-256 chunks) · **I** resource monitoring.

Each test is reported **PASS / FAIL / BLOCKED** with evidence in
`results/metrics.json` (`tests` section) and `results/TEST_REPORT.md`.
A feature that was not actually run is BLOCKED, never PASS.

## Security

- llama.cpp's RPC backend is documented upstream as proof-of-concept,
  fragile and insecure — so every RPC server binds **`127.0.0.1` only**,
  never a public port. No private prompts, no sensitive data in tests.
- No tokens, passwords, keys or secrets anywhere in this repo.
- Pinned dependencies: llama.cpp `v1.0.0`, actions `checkout@v4`,
  `setup-python@v5`, `upload-artifact@v4`.
- Workflows run with `contents: read` only.

## Limitations (known, by design)

- 3 logical nodes / 1 VM — not 3 physical machines (see the
  multi-runner investigation doc).
- CPU-only inference; GPU contribution is future work.
- RPC backend is experimental upstream; a failed RPC server fails the
  whole distributed inference honestly rather than degrading gracefully.
- No persistence / no real contributor network yet — that is the next
  experiment, once this one is proven.

## Results

Latest verified run: see [`TEST_REPORT.md`](TEST_REPORT.md) (updated only
from a real run — never hand-written) and the workflow artifacts.
