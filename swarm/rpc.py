"""llama.cpp RPC orchestration: build, launch, run, prove participation.

Distribution model (llama.cpp RPC backend):
  * ``llama-rpc-server`` runs on each node (127.0.0.1 only).
  * ``llama-cli --rpc <host:port,...> -ngl 99 -m <model>`` offloads
    transformer layers to the RPC servers; llama.cpp splits the layer
    set across the listed servers. A single node running the same
    command with no --rpc flag is the single-node baseline.
  * Participation is proven from the RPC servers' own logs + the
    client's log (which lists the RPC servers and the offloaded layers),
    never from our own claims.

Build flag: ``-DGGML_RPC=ON`` (llama.cpp CMake option).
Pinned llama.cpp ref lives in scripts/run_experiment.py (LLAMA_CPP_REF).
"""
from __future__ import annotations

import os
import re
import subprocess
import time

# Log markers that prove a process really did distributed work.
RPC_LOG_MARKERS = [
    re.compile(r"rpc", re.I),
    re.compile(r"server.*listening|listening on", re.I),
    re.compile(r"tensor|layer|offload", re.I),
]


def run(cmd: list[str], cwd: str | None = None, timeout: int | None = None,
        env: dict | None = None) -> subprocess.CompletedProcess:
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    return subprocess.run(cmd, cwd=cwd, timeout=timeout, env=merged_env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)


def build_llama_cpp(src_dir: str, build_dir: str, jobs: int = 4) -> dict:
    """cmake configure + build with RPC support. Returns {ok, log}.

    Target names changed across llama.cpp history (llama-rpc-server ->
    ggml-rpc-server), so try the known target sets in order, then fall
    back to a full build.
    """
    os.makedirs(build_dir, exist_ok=True)
    t0 = time.time()
    cfg = run(["cmake", "-B", build_dir, "-S", src_dir,
               "-DGGML_RPC=ON", "-DCMAKE_BUILD_TYPE=Release"], timeout=600)
    if cfg.returncode != 0:
        return {"ok": False, "stage": "cmake-configure",
                "log": cfg.stdout, "elapsed_s": round(time.time() - t0, 1)}
    target_sets = [
        ["llama-cli", "ggml-rpc-server", "llama-server"],
        ["llama-cli", "llama-rpc-server", "llama-server"],
        None,  # full build
    ]
    last_log = ""
    for targets in target_sets:
        cmd = ["cmake", "--build", build_dir, "-j", str(jobs)]
        if targets:
            cmd += ["--target"] + targets
        bld = run(cmd, timeout=1800)
        last_log = bld.stdout
        if bld.returncode == 0:
            return {"ok": True, "stage": "done",
                    "log": cfg.stdout[-2000:] + "\n--- build ---\n" + bld.stdout[-6000:],
                    "elapsed_s": round(time.time() - t0, 1),
                    "targets": targets or "all"}
    return {"ok": False, "stage": "cmake-build",
            "log": cfg.stdout[-2000:] + "\n--- build ---\n" + last_log[-6000:],
            "elapsed_s": round(time.time() - t0, 1)}


def find_bin(build_dir: str, name: str) -> str | None:
    """Find a built binary, trying current and legacy names."""
    aliases = {"llama-rpc-server": ["ggml-rpc-server", "llama-rpc-server",
                                    "rpc-server"],
               "llama-cli": ["llama-cli"],
               "llama-server": ["llama-server"]}.get(name, [name])
    for alias in aliases:
        for cand in (os.path.join(build_dir, "bin", alias),
                     os.path.join(build_dir, alias)):
            if os.path.exists(cand):
                return cand
    return None


def parse_tokens_per_s(stderr_text: str) -> float | None:
    m = re.search(r"eval time.*?([\d.]+)\s*ms\s*/\s*(\d+)\s*runs.*?([\d.]+)\s*tokens per second",
                  stderr_text, re.S)
    if m:
        return float(m.group(3))
    m = re.search(r"([\d.]+)\s*tokens per second", stderr_text)
    return float(m.group(1)) if m else None


def run_inference(cli_bin: str, model_path: str, prompt: str,
                  rpc_servers: list[str] | None = None,
                  max_tokens: int = 128, seed: int = 42,
                  timeout: int = 600) -> dict:
    """Run one real inference. Returns metrics + raw logs (never faked)."""
    cmd = [cli_bin, "-m", model_path, "-p", prompt, "-n", str(max_tokens),
           "--seed", str(seed), "--temp", "0.7", "--no-display-prompt",
           "-c", "2048"]
    if rpc_servers:
        # Pure 3-way split: WITHOUT --device the client's local CPU also
        # takes a share (~1/4 each). --device must come AFTER --rpc.
        # --tensor-split values are ratios, normalised: 1,1,1 = equal thirds.
        # --split-mode layer is the only mode that works across hosts.
        cmd += ["--rpc", ",".join(rpc_servers), "-ngl", "99",
                "--split-mode", "layer",
                "--device", ",".join(f"RPC{i}" for i in range(len(rpc_servers))),
                "--tensor-split", ",".join("1" for _ in rpc_servers)]
    else:
        cmd += ["-ngl", "0"]  # pure CPU baseline, no offload at all
    t0 = time.time()
    try:
        proc = run(cmd, timeout=timeout, env={"GGML_RPC_DEBUG": "1"})
        rc, out = proc.returncode, proc.stdout
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "reason": "timeout", "elapsed_s": round(time.time() - t0, 1),
                "output": "", "log": (e.stdout or b"").decode(errors="replace")
                if isinstance(e.stdout, bytes) else (e.stdout or "")}
    elapsed = time.time() - t0
    # llama-cli prints the completion after the prompt echo / timings.
    output = extract_completion(out)
    return {"ok": rc == 0 and bool(output.strip()),
            "returncode": rc, "elapsed_s": round(elapsed, 2),
            "tokens_per_s": parse_tokens_per_s(out),
            "output": output.strip(), "log": out,
            "rpc_servers": rpc_servers or []}


def extract_completion(text: str) -> str:
    """Best-effort extraction of the generated text from llama-cli output."""
    # Cut the llama.cpp timing / perf footer.
    cut = re.split(r"\nllama_perf_|^\s*perf\s", text, flags=re.M | re.S)
    body = cut[0]
    # Drop the common llama-cli header lines.
    lines = [ln for ln in body.splitlines()
             if not ln.startswith(("llama_model_loader", "llama_context",
                                  "system_info", "print_info", "load_backend",
                                  "llama_", "common_", "main: ", "sampler",
                                  "generate:", "build:", "commit:", "ggml_"))]
    return "\n".join(lines).strip()


def participation_evidence(rpc_log_paths: list[str], client_log: str) -> dict:
    """Score each RPC server's log for real participation markers.

    PRIMARY evidence lives in the CLIENT log (upstream llama.cpp):
      load_tensors: offloaded 24/24 layers to GPU
      load_tensors: RPC0[127.0.0.1:50052] model buffer size = ~160 MiB
      load_tensors: RPC1[127.0.0.1:50053] model buffer size = ~160 MiB
      load_tensors: RPC2[127.0.0.1:50054] model buffer size = ~160 MiB
    Three non-zero RPC buffer lines = the model really is split across
    the three workers. All weight on CPU (or one line only) = FAIL.
    Server logs are secondary (banner + GGML_RPC_DEBUG activity).
    """
    per_server = {}
    for path in rpc_log_paths:
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            per_server[path] = {"present": False, "markers": {}, "bytes": 0}
            continue
        markers = {pat.pattern: bool(pat.search(text)) for pat in RPC_LOG_MARKERS}
        transfer_lines = len(re.findall(r"(?i)tensor|compute|layer \d+", text))
        per_server[path] = {"present": True, "bytes": len(text),
                            "markers": markers,
                            "transfer_lines": transfer_lines,
                            "participated": transfer_lines > 0 or any(markers.values())}
    # --- client log: the authoritative distribution record ---
    offloaded = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers", client_log)
    buf_lines = re.findall(
        r"RPC(\d+)\[([0-9.]+:\d+)\]\s+model buffer size\s*=\s*([~\d.]+)\s*MiB",
        client_log)
    cpu_buf = re.search(r"CPU\s+model buffer size\s*=\s*([~\d.]+)\s*MiB", client_log)
    rpc_buffers = {f"RPC{i}": {"endpoint": ep, "mib": size}
                   for i, ep, size in buf_lines}
    client_markers = {pat.pattern: bool(pat.search(client_log)) for pat in RPC_LOG_MARKERS}
    rpc_mentioned = re.findall(r"127\.0\.0\.1:\d+", client_log)
    distributed = (len(rpc_buffers) >= 3
                   and all(float(v["mib"].lstrip("~")) > 0 for v in rpc_buffers.values()))
    return {"servers": per_server,
            "client_log_markers": client_markers,
            "client_log_rpc_endpoints": sorted(set(rpc_mentioned)),
            "offloaded_layers": offloaded.groups() if offloaded else None,
            "rpc_model_buffers_mib": rpc_buffers,
            "cpu_model_buffer_mib": cpu_buf.group(1) if cpu_buf else None,
            "distributed_by_client_log": distributed,
            "all_participated": bool(per_server) and all(
                v.get("participated") for v in per_server.values())}
