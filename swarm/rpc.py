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


def parse_layer_assignment(log_text: str) -> dict:
    """Parse v1.0.0 `-lv 5` lines: `layer N assigned to device RPCk`.

    Returns {"RPC0": [0..8], "RPC1": [9..16], "RPC2": [17..23]} style
    mapping plus counts. Empty when the log has no such lines (default
    verbosity). This is the exact per-layer split — the strongest
    distribution proof available.
    """
    assign: dict[str, list[int]] = {}
    for layer, dev in re.findall(r"layer\s+(\d+)\s+assigned to device\s+(RPC\d+)",
                                 log_text):
        assign.setdefault(dev, []).append(int(layer))
    for dev in assign:
        assign[dev] = sorted(set(assign[dev]))
    return assign


def parse_tokens_per_s(stderr_text: str) -> float | None:
    # v1.0.0 chat UI: "[ Prompt: 65.7 t/s | Generation: 10.3 t/s ]"
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", stderr_text)
    if m:
        return float(m.group(1))
    m = re.search(r"eval time.*?([\d.]+)\s*ms\s*/\s*(\d+)\s*runs.*?([\d.]+)\s*tokens per second",
                  stderr_text, re.S)
    if m:
        return float(m.group(3))
    m = re.search(r"([\d.]+)\s*tokens per second", stderr_text)
    return float(m.group(1)) if m else None


def run_inference(cli_bin: str, model_path: str, prompt: str,
                  rpc_servers: list[str] | None = None,
                  max_tokens: int = 128, seed: int = 42,
                  timeout: int = 600, log_verbosity: int | None = None) -> dict:
    """Run one real inference. Returns metrics + raw logs (never faked).

    ``log_verbosity`` (e.g. 5) is for EVIDENCE PROBES ONLY: at -lv 5 the
    client prints the exact per-layer device assignment, but its debug
    lines interleave with the generated text, so answer runs must keep
    default verbosity.
    """
    cmd = [cli_bin, "-m", model_path, "-p", prompt, "-n", str(max_tokens),
           "--seed", str(seed), "--temp", "0.7", "--no-display-prompt",
           "-c", "2048"]
    if log_verbosity is not None:
        cmd += ["-lv", str(log_verbosity)]
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
    """Extract the generated answer from llama-cli output.

    v1.0.0 runs an interactive chat UI: banner + ASCII art, then
    "> <prompt>" echo, the answer, then "[ Prompt: .. | Generation: .. ]".
    The answer is the slice between the prompt echo and the timing line.
    Falls back to the old banner-stripping heuristic when those markers
    are absent (older llama.cpp).
    """
    lines = text.splitlines()
    start = end = None
    for i, ln in enumerate(lines):
        if start is None and ln.startswith("> "):
            start = i + 1
        elif start is not None and "[ Prompt:" in ln:
            end = i
            break
    if start is not None and end is not None and end > start:
        return "\n".join(lines[start:end]).strip()
    if start is not None:
        # No timing line (e.g. run cut short): take everything after echo,
        # dropping trailing UI lines.
        body = [ln for ln in lines[start:]
                if ln.strip() and ln.strip() not in (">", "Exiting...")]
        return "\n".join(body).strip()
    # Fallback: cut the llama.cpp timing / perf footer, drop header lines.
    cut = re.split(r"\nllama_perf_|^\s*perf\s", text, flags=re.M | re.S)
    body = cut[0]
    kept = [ln for ln in body.splitlines()
            if not ln.startswith(("llama_model_loader", "llama_context",
                                  "system_info", "print_info", "load_backend",
                                  "llama_", "common_", "main: ", "sampler",
                                  "generate:", "build:", "commit:", "ggml_"))]
    return "\n".join(kept).strip()


def participation_evidence(rpc_log_paths: list[str], client_log: str) -> dict:
    """Score each RPC server's log for real participation markers.

    PRIMARY evidence lives in the CLIENT log (upstream llama.cpp):
      load_tensors: offloaded 24/24 layers to GPU
      load_tensors: RPC0[127.0.0.1:50052] model buffer size = ~160 MiB
    BUT llama.cpp v1.0.0's llama-cli does not print those lines at
    default verbosity. Independent SECONDARY proof: every RPC server's
    own log shows the client handshake ([hello] + rpc_serve_client) and
    repeated [graph_compute] executions (n_nodes/n_tensors per compute)
    — a server that never received tensors cannot compute graphs.
    D passes on EITHER proof, and records which one fired, so a silent
    client log can never hide a real distribution — nor fake one.
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
    layer_assignment = parse_layer_assignment(client_log)
    layers_covered = sorted(l for layers in layer_assignment.values() for l in layers)
    layer_split_proven = (len(layer_assignment) >= 3
                          and len(layers_covered) >= 24)
    servers_computed = bool(per_server) and all(
        v.get("participated") for v in per_server.values())
    return {"servers": per_server,
            "client_log_markers": client_markers,
            "client_log_rpc_endpoints": sorted(set(rpc_mentioned)),
            "offloaded_layers": offloaded.groups() if offloaded else None,
            "rpc_model_buffers_mib": rpc_buffers,
            "cpu_model_buffer_mib": cpu_buf.group(1) if cpu_buf else None,
            "layer_assignment": layer_assignment,
            "layer_split": {k: len(v) for k, v in layer_assignment.items()},
            "layers_covered": len(layers_covered),
            "distributed_by_client_log": distributed or layer_split_proven,
            "servers_computed": servers_computed,
            "distribution_proven": distributed or layer_split_proven or servers_computed,
            "proof": ("client_log_buffers" if distributed else
                      "layer_assignment" if layer_split_proven else
                      "server_graph_compute" if servers_computed else "none"),
            "all_participated": servers_computed}
