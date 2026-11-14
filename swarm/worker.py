"""Swarm worker: one logical node.

- Generates a unique peer_id (uuid).
- Advertises CPU / RAM / disk to the coordinator.
- Heartbeats every 2 s; if the worker dies, the coordinator marks it
  DEAD after the heartbeat timeout — that is the failure detector.
- Optionally owns one llama.cpp RPC server (llama-rpc-server) bound to
  127.0.0.1 on its own port. The RPC server is what actually holds a
  share of the model weights during distributed inference.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid


def log(msg: str):
    print(f"[worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mem_total_mb() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def resources() -> dict:
    du = shutil.disk_usage(".")
    return {"cpu_count": os.cpu_count() or 1,
            "mem_total_mb": mem_total_mb(),
            "disk_free_mb": du.free // (1024 * 1024)}


def http_json(url: str, payload: dict, timeout: int = 10) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


class Worker:
    def __init__(self, coordinator: str, name: str, rpc_port: int,
                 rpc_bin: str | None, rpc_log: str | None):
        self.coordinator = coordinator.rstrip("/")
        self.peer_id = f"node-{uuid.uuid4().hex[:12]}"
        self.name = name
        self.rpc_port = rpc_port
        self.rpc_bin = rpc_bin
        self.rpc_log = rpc_log
        self.rpc_proc: subprocess.Popen | None = None
        self.rpc_log_fh = None
        self._stop = threading.Event()

    def start_rpc_server(self) -> bool:
        """Launch llama-rpc-server on 127.0.0.1:<rpc_port>. Returns False
        (swarm-only mode) when no binary is available — never fake it."""
        if not self.rpc_bin or not os.path.exists(self.rpc_bin):
            log(f"{self.name}: no llama-rpc-server binary, swarm-only mode")
            return False
        cmd = [self.rpc_bin, "--host", "127.0.0.1", "--port", str(self.rpc_port)]
        if self.rpc_log:
            os.makedirs(os.path.dirname(self.rpc_log) or ".", exist_ok=True)
            self.rpc_log_fh = open(self.rpc_log, "wb")
            out = self.rpc_log_fh
        else:
            out = subprocess.DEVNULL
        env = dict(os.environ, GGML_RPC_DEBUG="1")
        self.rpc_proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                         env=env)
        # wait until it answers / is listening (up to 30 s)
        for _ in range(60):
            if self.rpc_proc.poll() is not None:
                log(f"{self.name}: llama-rpc-server exited early rc={self.rpc_proc.returncode}")
                return False
            try:
                with open(f"/proc/net/tcp", encoding="utf-8") as f:
                    pass
            except OSError:
                pass
            time.sleep(0.5)
            # cheap readiness probe: process alive + port open via socket
            import socket
            try:
                with socket.create_connection(("127.0.0.1", self.rpc_port), timeout=1):
                    log(f"{self.name}: llama-rpc-server up on 127.0.0.1:{self.rpc_port} pid={self.rpc_proc.pid}")
                    return True
            except OSError:
                continue
        log(f"{self.name}: llama-rpc-server did not become ready in 30 s")
        return False

    def register(self) -> bool:
        payload = {"peer_id": self.peer_id, "name": self.name,
                   "rpc_host": "127.0.0.1", "rpc_port": self.rpc_port,
                   **resources()}
        try:
            http_json(f"{self.coordinator}/register", payload)
            log(f"{self.name}: registered as {self.peer_id} "
                f"rpc=127.0.0.1:{self.rpc_port} cpu={payload['cpu_count']} "
                f"mem={payload['mem_total_mb']}MB")
            return True
        except Exception as e:
            log(f"{self.name}: register FAILED: {e}")
            return False

    def heartbeat_loop(self):
        while not self._stop.wait(2.0):
            try:
                http_json(f"{self.coordinator}/heartbeat", {"peer_id": self.peer_id})
            except Exception as e:
                log(f"{self.name}: heartbeat failed: {e}")

    def run(self):
        # Graceful SIGTERM/SIGINT so a killed worker also stops its RPC
        # server (failure test depends on the port actually going away).
        def _handler(signum, frame):
            self._stop.set()
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
        rpc_ok = self.start_rpc_server()
        if not self.register():
            self.shutdown()
            return 1
        hb = threading.Thread(target=self.heartbeat_loop, daemon=True)
        hb.start()
        log(f"{self.name}: running (peer {self.peer_id}, rpc_ready={rpc_ok})")
        try:
            while not self._stop.wait(1.0):
                if self.rpc_proc is not None and self.rpc_proc.poll() is not None:
                    log(f"{self.name}: llama-rpc-server DIED rc={self.rpc_proc.returncode}")
                    return 2
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
        return 0

    def shutdown(self):
        self._stop.set()
        if self.rpc_proc is not None and self.rpc_proc.poll() is None:
            self.rpc_proc.terminate()
            try:
                self.rpc_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.rpc_proc.kill()
        if self.rpc_log_fh:
            self.rpc_log_fh.close()


def main():
    ap = argparse.ArgumentParser(description="Open Swarm worker node")
    ap.add_argument("--coordinator", default="http://127.0.0.1:8471")
    ap.add_argument("--name", required=True)
    ap.add_argument("--rpc-port", type=int, required=True)
    ap.add_argument("--rpc-bin", default=None,
                    help="path to llama-rpc-server (omit = swarm-only mode)")
    ap.add_argument("--rpc-log", default=None)
    args = ap.parse_args()
    w = Worker(args.coordinator, args.name, args.rpc_port, args.rpc_bin, args.rpc_log)
    sys.exit(w.run())


if __name__ == "__main__":
    main()
