#!/usr/bin/env python3
"""OPEN SWARM — first real experiment driver.

End-to-end, reproducible, honest:
  1. Build llama.cpp (pinned ref) with RPC support.
  2. Download Qwen2.5-0.5B-Instruct-GGUF Q4_K_M.
  3. Start the coordinator + THREE worker processes, each owning one
     llama-rpc-server on 127.0.0.1 (three LOGICAL nodes, ONE physical VM
     — labelled as such everywhere; this is not three computers).
  4. Single-node baseline vs distributed inference (same prompt).
  5. Tests A–I, concurrent users, node failure + recovery, integrity.
  6. Write results/metrics.json + results/TEST_REPORT.md + logs.

Exit code 0 only when every acceptance criterion passes.
Usage:
    python scripts/run_experiment.py [--skip-build] [--skip-download]
                                     [--work-dir .] [--max-tokens 96]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from swarm import integrity as integ  # noqa: E402
from swarm import monitor as mon  # noqa: E402
from swarm import rpc as rpc_mod  # noqa: E402
from swarm.client import ask as client_ask  # noqa: E402

# ---- pinned dependencies (change only with a reason, in the commit) ----
LLAMA_CPP_REF = os.environ.get("LLAMA_CPP_REF", "v1.0.0")
LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp.git"
MODEL_URL = ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/"
             "resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf")
MODEL_NAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_LAYERS = 24
RPC_PORTS = [50052, 50053, 50054]
COORD_PORT = 8471

PROMPTS = [
    "Explain how a decentralized cloud works.",
    "What is 2 + 2? Answer in one short sentence.",
    "Name one benefit of distributed computing in one sentence.",
]


def log(msg: str):
    print(f"[experiment {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: str) -> dict:
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "open-swarm-lab/0.1"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    return {"bytes": os.path.getsize(dest), "sha256": sha256_file(dest),
            "elapsed_s": round(time.time() - t0, 1)}


def http_json(url: str, payload=None, method="GET", timeout=15):
    import urllib.error
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code}


def wait_for(fn, timeout: int, desc: str, interval: float = 1.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if fn():
                return True
        except Exception:
            pass
        time.sleep(interval)
    log(f"TIMEOUT waiting for: {desc}")
    return False


class Experiment:
    def __init__(self, args):
        self.args = args
        self.root = ROOT
        self.results_dir = os.path.join(ROOT, "results")
        self.logs_dir = os.path.join(self.results_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.vendor = os.path.join(ROOT, "vendor", "llama.cpp")
        self.build_dir = os.path.join(self.vendor, "build")
        self.model_path = os.path.join(ROOT, "models", MODEL_NAME)
        self.coord_url = f"http://127.0.0.1:{COORD_PORT}"
        self.procs: dict[str, subprocess.Popen] = {}
        self.log_fhs = []
        self.results: dict = {
            "experiment": "open-swarm-lab v0.1.0",
            "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                      "file": MODEL_NAME, "quantization": "Q4_K_M",
                      "layers": MODEL_LAYERS},
            "topology": ("3 logical nodes on 1 physical VM "
                         "(GitHub-hosted Ubuntu runner / local VM), "
                         "communicating over 127.0.0.1 — NOT 3 computers"),
            "llama_cpp_ref": LLAMA_CPP_REF,
            "tests": {},
        }

    # -- process helpers -------------------------------------------------
    def _open_log(self, name: str):
        fh = open(os.path.join(self.logs_dir, name), "wb")
        self.log_fhs.append(fh)
        return fh

    def start(self, key: str, cmd: list[str], logname: str):
        fh = self._open_log(logname)
        p = subprocess.Popen(cmd, cwd=self.root, stdout=fh,
                             stderr=subprocess.STDOUT)
        self.procs[key] = p
        log(f"started {key} pid={p.pid}")
        return p

    def stop(self, key: str, timeout: int = 15):
        p = self.procs.pop(key, None)
        if p is None:
            return
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
        log(f"stopped {key} rc={p.returncode}")

    def stop_all(self):
        for key in list(self.procs):
            self.stop(key)
        for fh in self.log_fhs:
            try:
                fh.close()
            except Exception:
                pass

    # -- phases ----------------------------------------------------------
    def phase_build(self):
        if self.args.skip_build and os.path.exists(
                os.path.join(self.build_dir, "bin", "llama-cli")):
            log("build skipped (binaries present)")
            self.results["build"] = {"skipped": True}
            return True
        if not os.path.exists(os.path.join(self.vendor, "CMakeLists.txt")):
            log(f"cloning llama.cpp @ {LLAMA_CPP_REF} ...")
            os.makedirs(os.path.dirname(self.vendor), exist_ok=True)
            r = subprocess.run(["git", "clone", "--depth", "1", "--branch",
                                LLAMA_CPP_REF, LLAMA_CPP_REPO, self.vendor],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=600)
            if r.returncode != 0:
                log("git clone FAILED:\n" + r.stdout[-3000:])
                return False
        log("building llama.cpp with GGML_RPC=ON ...")
        res = rpc_mod.build_llama_cpp(self.vendor, self.build_dir,
                                      jobs=os.cpu_count() or 4)
        self.results["build"] = {k: v for k, v in res.items()}
        with open(os.path.join(self.logs_dir, "llama-build.log"), "w") as f:
            f.write(res.get("log", ""))
        log(f"llama.cpp build ok={res['ok']} in {res['elapsed_s']}s")
        return res["ok"]

    def phase_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        if self.args.skip_download and os.path.exists(self.model_path):
            log("model download skipped (file present)")
        elif os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 100_000_000:
            log("model already present, skipping download")
        else:
            log("downloading Qwen2.5-0.5B Q4_K_M (~350 MB) ...")
            dl = download(MODEL_URL, self.model_path)
            self.results["model"]["download"] = dl
            log(f"downloaded {dl['bytes']} bytes sha256={dl['sha256'][:16]}…")
        size = os.path.getsize(self.model_path)
        self.results["model"].update({"path": self.model_path, "size_bytes": size,
                                     "sha256": sha256_file(self.model_path)})
        ok = size > 100_000_000
        log(f"model ok={ok} size={size} bytes")
        return ok

    def start_swarm(self):
        coord_log = self._open_log("coordinator.log")
        coord = subprocess.Popen(
            [sys.executable, "-m", "swarm.coordinator", "--port", str(COORD_PORT),
             "--min-nodes", "3", "--state-file",
             os.path.join(self.results_dir, "coordinator-state.json")],
            cwd=self.root, stdout=coord_log, stderr=subprocess.STDOUT)
        self.procs["coordinator"] = coord
        if not wait_for(lambda: http_json(f"{self.coord_url}/health").get("ok"),
                        30, "coordinator /health"):
            return False
        rpc_bin = rpc_mod.find_bin(self.build_dir, "llama-rpc-server")
        cli_bin = rpc_mod.find_bin(self.build_dir, "llama-cli")
        if not rpc_bin or not cli_bin:
            log(f"binaries missing: rpc={rpc_bin} cli={cli_bin}")
            return False
        self.results["binaries"] = {"llama-cli": cli_bin, "llama-rpc-server": rpc_bin}
        for i, port in enumerate(RPC_PORTS):
            self.start(f"worker-{i}",
                       [sys.executable, "-m", "swarm.worker",
                        "--coordinator", self.coord_url,
                        "--name", f"worker-{i}", "--rpc-port", str(port),
                        "--rpc-bin", rpc_bin,
                        "--rpc-log", os.path.join(self.logs_dir, f"rpc-{i}.log")],
                       f"worker-{i}.log")
        ok = wait_for(
            lambda: len(http_json(f"{self.coord_url}/nodes").get("nodes", [])) == 3
            and len([n for n in http_json(f"{self.coord_url}/nodes")["nodes"]
                     if n["status"] == "alive"]) == 3,
            90, "3 workers alive")
        nodes = http_json(f"{self.coord_url}/nodes").get("nodes", [])
        self.results["nodes"] = nodes
        log(f"nodes registered: {len(nodes)} alive="
            f"{sum(1 for n in nodes if n['status'] == 'alive')}")
        return ok

    def run(self):
        tests = self.results["tests"]
        try:
            # --- build + model -----------------------------------------
            if not self.phase_build():
                tests["build"] = {"status": "FAIL", "evidence": "llama.cpp build failed, see logs/llama-build.log"}
                return self.finish(1)
            if not self.phase_model():
                tests["model"] = {"status": "FAIL", "evidence": "model file missing/too small"}
                return self.finish(1)
            cli_bin = rpc_mod.find_bin(self.build_dir, "llama-cli")

            # --- swarm up ----------------------------------------------
            if not self.start_swarm():
                tests["A_startup"] = {"status": "FAIL", "evidence": "coordinator/workers did not reach 3 alive"}
                return self.finish(1)
            nodes = self.results["nodes"]
            tests["A_startup"] = {"status": "PASS" if len(nodes) == 3 else "FAIL",
                                  "evidence": f"{len(nodes)} nodes, ids={[n['peer_id'] for n in nodes]}"}
            tests["B_discovery"] = {"status": "PASS",
                                    "evidence": "3/3 registered via /register, heartbeats in coordinator state"}
            tests["C_resources"] = {"status": "PASS",
                                    "evidence": str([{n["peer_id"]: (n["cpu_count"], n["mem_total_mb"])} for n in nodes])}

            # --- baseline (single node, no RPC) -------------------------
            log("baseline: single-node inference (no --rpc) ...")
            base = rpc_mod.run_inference(cli_bin, self.model_path, PROMPTS[0],
                                         rpc_servers=None, max_tokens=self.args.max_tokens)
            with open(os.path.join(self.logs_dir, "baseline.log"), "w") as f:
                f.write(base.get("log", ""))
            self.results["baseline"] = {k: base[k] for k in
                                        ("ok", "elapsed_s", "tokens_per_s", "returncode")}
            self.results["baseline"]["output"] = base["output"]
            log(f"baseline ok={base['ok']} {base['elapsed_s']}s "
                f"{base.get('tokens_per_s')} tok/s")

            # --- distributed inference (3 prompts) ----------------------
            log("distributed: 3 prompts through the 3-node pool ...")
            sampler = mon.Sampler(interval=0.5)
            sampler.start()
            net0 = mon.net_lo_bytes()
            dist_results = []
            for p in PROMPTS:
                r = client_ask(p, self.coord_url, cli_bin, self.model_path,
                               max_tokens=self.args.max_tokens)
                dist_results.append({"prompt": p, "request_id": r.get("request_id"),
                                     "ok": r["ok"], "elapsed_s": r.get("elapsed_s"),
                                     "tokens_per_s": r.get("tokens_per_s"),
                                     "output": r["output"]})
                log(f"  prompt ok={r['ok']} {r.get('elapsed_s')}s: "
                    f"{r['output'][:80]!r}")
                with open(os.path.join(self.logs_dir,
                                       f"distributed-{len(dist_results)}.log"), "w") as f:
                    f.write(r.get("log", ""))
            mon_summary = sampler.stop()
            net1 = mon.net_lo_bytes()
            self.results["distributed"] = {"prompts": dist_results}
            self.results["monitoring"] = {
                **mon_summary,
                "net_lo_rx_bytes_distributed_phase": net1["rx_bytes"] - net0["rx_bytes"],
                "net_lo_tx_bytes_distributed_phase": net1["tx_bytes"] - net0["tx_bytes"],
            }
            rpc_logs = [os.path.join(self.logs_dir, f"rpc-{i}.log") for i in range(3)]
            last_log = dist_results[-1] and open(
                os.path.join(self.logs_dir, "distributed-3.log")).read()
            part = rpc_mod.participation_evidence(rpc_logs, last_log or "")
            self.results["distribution"] = part
            all_ok = all(d["ok"] and len(d["output"]) > 10 for d in dist_results)
            tests["D_distribution"] = {
                "status": "PASS" if part["all_participated"] else "FAIL",
                "evidence": json.dumps(part, default=str)[:1500]}
            tests["E_tasks"] = {
                "status": "PASS" if all_ok else "FAIL",
                "evidence": f"{sum(1 for d in dist_results if d['ok'])}/3 prompts returned real model output"}

            # --- concurrent users ---------------------------------------
            log("concurrent: 2 prompts at once ...")
            conc_prompts = ["Say the word APPLE and nothing else.",
                            "Say the word BANANA and nothing else."]
            with ThreadPoolExecutor(max_workers=2) as ex:
                conc = list(ex.map(
                    lambda p: client_ask(p, self.coord_url, cli_bin,
                                         self.model_path, max_tokens=32),
                    conc_prompts))
            ids = [c.get("request_id") for c in conc]
            mixed = (conc[0]["output"] == conc[1]["output"]
                     or not conc[0]["output"] or not conc[1]["output"])
            self.results["concurrent"] = [
                {"request_id": c.get("request_id"), "ok": c["ok"],
                 "output": c["output"]} for c in conc]
            tests["F_concurrent"] = {
                "status": "PASS" if len(set(ids)) == 2 and not mixed and all(c["ok"] for c in conc) else "FAIL",
                "evidence": f"request_ids={ids}, outputs distinct={not mixed}"}

            # --- failure + recovery --------------------------------------
            log("failure: stopping worker-2 ...")
            self.stop("worker-2")
            time.sleep(13)  # heartbeat timeout (10s) + sweep margin
            nodes_now = http_json(f"{self.coord_url}/nodes").get("nodes", [])
            dead = [n for n in nodes_now if n["status"] == "dead"]
            failed_task = client_ask("Say the word CHERRY.", self.coord_url,
                                     cli_bin, self.model_path, max_tokens=16)
            honest_fail = (not failed_task["ok"]
                           and "insufficient nodes" in str(failed_task.get("reason", "")))
            log(f"  dead detected: {[n['peer_id'] for n in dead]}; "
                f"honest failure on submit: {honest_fail}")
            # restart worker-2 and retry
            rpc_bin = self.results["binaries"]["llama-rpc-server"]
            self.start("worker-2",
                       [sys.executable, "-m", "swarm.worker",
                        "--coordinator", self.coord_url,
                        "--name", "worker-2", "--rpc-port", str(RPC_PORTS[2]),
                        "--rpc-bin", rpc_bin,
                        "--rpc-log", os.path.join(self.logs_dir, "rpc-2-restart.log")],
                       "worker-2-restart.log")
            recovered = wait_for(
                lambda: len([n for n in http_json(f"{self.coord_url}/nodes")
                             .get("nodes", []) if n["status"] == "alive"]) == 3,
                90, "worker-2 recovery")
            retry = client_ask("Say the word CHERRY.", self.coord_url, cli_bin,
                               self.model_path, max_tokens=16) if recovered else {"ok": False}
            self.results["failure"] = {
                "dead_detected": [n["peer_id"] for n in dead],
                "submit_while_degraded": {"ok": failed_task["ok"],
                                          "reason": failed_task.get("reason")},
                "recovered": recovered, "retry_ok": retry["ok"],
                "retry_output": retry.get("output", "")}
            tests["G_failure_recovery"] = {
                "status": "PASS" if dead and honest_fail and recovered and retry["ok"] else "FAIL",
                "evidence": json.dumps(self.results["failure"])[:800]}

            # --- integrity ----------------------------------------------
            log("integrity: chunking the model file ...")
            chunk_dir = os.path.join(self.results_dir, "model-chunks")
            manifest = integ.split_file(self.model_path, chunk_dir,
                                        chunk_size=4 * 1024 * 1024)
            v_clean = integ.verify_chunks(manifest, chunk_dir)
            # corrupt one chunk in a COPY and confirm detection
            import shutil as _sh
            corrupt_dir = os.path.join(self.results_dir, "model-chunks-corrupt")
            if os.path.exists(corrupt_dir):
                _sh.rmtree(corrupt_dir)
            _sh.copytree(chunk_dir, corrupt_dir)
            bad = os.path.join(corrupt_dir, manifest["chunks"][0]["file"])
            with open(bad, "r+b") as f:
                f.seek(0)
                f.write(b"\x00")
            v_corrupt = integ.verify_chunks(manifest, corrupt_dir)
            rebuilt = os.path.join(self.results_dir, "model-rebuilt.gguf")
            rec = integ.reconstruct(manifest, chunk_dir, rebuilt)
            self.results["integrity"] = {
                "chunks": len(manifest["chunks"]), "chunk_size": manifest["chunk_size"],
                "verify_clean_ok": v_clean["ok"],
                "corrupt_detected": (not v_corrupt["ok"] and v_corrupt["corrupt"] == [0]),
                "reconstruct_ok": rec["ok"],
                "source_sha256": manifest["source_sha256"]}
            tests["H_integrity"] = {
                "status": "PASS" if v_clean["ok"] and v_corrupt["corrupt"] == [0] and rec["ok"] else "FAIL",
                "evidence": f"{len(manifest['chunks'])} chunks, corrupt detected={v_corrupt['corrupt']}, rebuild sha match={rec['ok']}"}
            tests["I_monitoring"] = {
                "status": "PASS" if self.results["monitoring"].get("samples", 0) > 0 else "FAIL",
                "evidence": json.dumps(self.results["monitoring"])[:600]}
            return self.finish(0 if all(
                t["status"] == "PASS" for t in tests.values()) else 1)
        finally:
            self.stop_all()

    def finish(self, code: int) -> int:
        tests = self.results["tests"]
        self.results["verdict"] = ("PASS" if code == 0 and tests and all(
            t["status"] == "PASS" for t in tests.values()) else "FAIL")
        self.results["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(os.path.join(self.results_dir, "metrics.json"), "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        with open(os.path.join(self.results_dir, "TEST_REPORT.md"), "w") as f:
            f.write(render_report(self.results))
        log("=== TEST SUMMARY ===")
        for k, v in tests.items():
            log(f"  {k}: {v['status']} — {v.get('evidence','')[:200]}")
        log(f"verdict: {self.results['verdict']} (exit {code})")
        return code


def render_report(res: dict) -> str:
    t = res["tests"]
    rows = "\n".join(f"| {k} | {v['status']} | {v.get('evidence','')[:220]} |"
                     for k, v in t.items())
    dist = res.get("distributed", {}).get("prompts", [])
    outs = "\n\n".join(
        f"**Prompt:** {d['prompt']}\n\n> {d['output']}\n\n"
        f"({d.get('elapsed_s')}s, {d.get('tokens_per_s')} tok/s, request {d.get('request_id')})"
        for d in dist)
    base = res.get("baseline", {})
    mon = res.get("monitoring", {})
    return f"""# OPEN SWARM — Test Report (generated by scripts/run_experiment.py)

- **Topology:** {res.get('topology')}
- **Model:** {res['model']['name']} — {res['model']['quantization']}, {res['model']['layers']} layers, size {res['model'].get('size_bytes')} bytes, sha256 `{res['model'].get('sha256','')[:32]}…`
- **llama.cpp ref:** {res.get('llama_cpp_ref')}
- **Verdict:** {res.get('verdict')}

## Tests

| Test | Status | Evidence |
|---|---|---|
{rows}

## Baseline (single node, no RPC)

- ok={base.get('ok')}, elapsed={base.get('elapsed_s')}s, tokens/s={base.get('tokens_per_s')}

> {base.get('output','')}

## Distributed answers (one prompt → one answer, 3 RPC nodes)

{outs}

## Distribution evidence

```json
{json.dumps(res.get('distribution', {}), indent=2, default=str)[:3000]}
```

## Failure & recovery

```json
{json.dumps(res.get('failure', {}), indent=2, default=str)[:1500]}
```

## Monitoring (distributed phase)

```json
{json.dumps(mon, indent=2, default=str)[:1500]}
```

## Integrity

```json
{json.dumps(res.get('integrity', {}), indent=2, default=str)}
```

## Logs

`results/logs/` — coordinator.log, worker-*.log, rpc-*.log, baseline.log, distributed-*.log, llama-build.log
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=96)
    args = ap.parse_args()
    sys.exit(Experiment(args).run())


if __name__ == "__main__":
    main()
