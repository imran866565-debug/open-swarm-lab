"""Swarm coordinator: node registry, heartbeats, resource view, task assignment.

Stdlib only. Binds 127.0.0.1 only — this is an experiment, never public.
A node is ALIVE if it heartbeated within HEARTBEAT_TIMEOUT seconds.
Inference tasks are assigned to *all* alive RPC nodes as one logical
compute pool; if fewer than ``min_nodes`` are alive the task FAILS
honestly instead of pretending a partial swarm did the work.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HEARTBEAT_TIMEOUT = 10.0  # seconds without heartbeat -> DEAD
SWEEP_INTERVAL = 2.0


def now() -> float:
    return time.time()


class CoordinatorState:
    def __init__(self, min_nodes: int = 3, state_file: str | None = None):
        self.min_nodes = min_nodes
        self.state_file = state_file
        self.lock = threading.Lock()
        self.nodes: dict[str, dict] = {}   # peer_id -> node record
        self.tasks: dict[str, dict] = {}   # request_id -> task record
        self._stop = threading.Event()
        self._sweeper = threading.Thread(target=self._sweep_loop, daemon=True)
        self._sweeper.start()

    def _sweep_loop(self):
        while not self._stop.wait(SWEEP_INTERVAL):
            self.sweep()

    def sweep(self):
        with self.lock:
            for node in self.nodes.values():
                alive = (now() - node["last_seen"]) <= HEARTBEAT_TIMEOUT
                node["status"] = "alive" if alive else "dead"
            self._persist_locked()

    def stop(self):
        self._stop.set()

    # -- persistence ---------------------------------------------------
    def _persist_locked(self):
        if not self.state_file:
            return
        try:
            tmp = self.state_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"nodes": list(self.nodes.values()),
                           "tasks": list(self.tasks.values())},
                          f, indent=2)
            os.replace(tmp, self.state_file)
        except OSError:
            pass  # state file is best-effort diagnostics, never fatal

    # -- nodes ---------------------------------------------------------
    def register(self, info: dict) -> dict:
        peer_id = info.get("peer_id") or f"node-{uuid.uuid4().hex[:12]}"
        rec = {
            "peer_id": peer_id,
            "name": info.get("name", peer_id),
            "rpc_host": info.get("rpc_host", "127.0.0.1"),
            "rpc_port": info.get("rpc_port"),
            "cpu_count": info.get("cpu_count"),
            "mem_total_mb": info.get("mem_total_mb"),
            "disk_free_mb": info.get("disk_free_mb"),
            "status": "alive",
            "registered_at": now(),
            "last_seen": now(),
            "heartbeats": 1,
        }
        with self.lock:
            self.nodes[peer_id] = rec
            self._persist_locked()
        return dict(rec)

    def heartbeat(self, peer_id: str) -> dict | None:
        with self.lock:
            rec = self.nodes.get(peer_id)
            if rec is None:
                return None
            rec["last_seen"] = now()
            rec["heartbeats"] += 1
            rec["status"] = "alive"
            return dict(rec)

    def list_nodes(self) -> list[dict]:
        with self.lock:
            return [dict(n) for n in self.nodes.values()]

    def alive_nodes(self) -> list[dict]:
        self.sweep()
        return [n for n in self.list_nodes() if n["status"] == "alive"]

    # -- tasks ---------------------------------------------------------
    def submit(self, prompt: str, max_tokens: int = 128) -> dict:
        """Assign one inference request to the whole alive pool.

        Returns the task record. If too few nodes are alive the task is
        created in state 'failed' with an honest reason — never faked.
        """
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        alive = self.alive_nodes()
        rpc_endpoints = [f"{n['rpc_host']}:{n['rpc_port']}"
                         for n in alive if n.get("rpc_port")]
        if len(alive) < self.min_nodes or len(rpc_endpoints) < self.min_nodes:
            task = {"request_id": request_id, "prompt": prompt,
                    "status": "failed",
                    "reason": (f"insufficient nodes: {len(alive)} alive, "
                               f"{len(rpc_endpoints)} RPC endpoints, "
                               f"need {self.min_nodes}"),
                    "assigned_nodes": [n["peer_id"] for n in alive],
                    "rpc_endpoints": rpc_endpoints,
                    "created_at": now()}
        else:
            # Assign by capacity: most CPU first (all still participate —
            # llama.cpp RPC splits layers across every listed server).
            ranked = sorted(alive, key=lambda n: (n.get("cpu_count") or 0),
                            reverse=True)
            task = {"request_id": request_id, "prompt": prompt,
                    "status": "assigned",
                    "reason": None,
                    "assigned_nodes": [n["peer_id"] for n in ranked],
                    "rpc_endpoints": rpc_endpoints,
                    "max_tokens": max_tokens,
                    "created_at": now()}
        with self.lock:
            self.tasks[request_id] = task
            self._persist_locked()
        return dict(task)

    def complete(self, request_id: str, output: str, metrics: dict | None = None) -> dict | None:
        with self.lock:
            task = self.tasks.get(request_id)
            if task is None:
                return None
            task["status"] = "completed"
            task["output"] = output
            task["metrics"] = metrics or {}
            task["completed_at"] = now()
            self._persist_locked()
            return dict(task)

    def fail(self, request_id: str, reason: str) -> dict | None:
        with self.lock:
            task = self.tasks.get(request_id)
            if task is None:
                return None
            task["status"] = "failed"
            task["reason"] = reason
            task["completed_at"] = now()
            self._persist_locked()
            return dict(task)

    def get_task(self, request_id: str) -> dict | None:
        with self.lock:
            t = self.tasks.get(request_id)
            return dict(t) if t else None


def make_handler(state: CoordinatorState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep stdout clean; we log ourselves
            pass

        def _send(self, code: int, obj: dict):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8") or "{}")

        def do_GET(self):
            p = urlparse(self.path).path
            if p == "/health":
                self._send(200, {"ok": True, "nodes": len(state.nodes),
                                 "alive": len(state.alive_nodes())})
            elif p == "/nodes":
                self._send(200, {"nodes": state.list_nodes()})
            elif p.startswith("/task/"):
                t = state.get_task(p[len("/task/"):])
                self._send(200 if t else 404, t or {"error": "no such task"})
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self):
            p = urlparse(self.path).path
            try:
                body = self._read_json()
            except Exception as e:
                self._send(400, {"error": f"bad json: {e}"})
                return
            if p == "/register":
                self._send(200, state.register(body))
            elif p == "/heartbeat":
                rec = state.heartbeat(body.get("peer_id", ""))
                self._send(200 if rec else 404, rec or {"error": "unknown peer"})
            elif p == "/submit":
                if not body.get("prompt"):
                    self._send(400, {"error": "prompt required"})
                    return
                self._send(200, state.submit(body["prompt"],
                                             int(body.get("max_tokens", 128))))
            elif p.startswith("/task/") and p.endswith("/complete"):
                rid = p[len("/task/"):-len("/complete")]
                t = state.complete(rid, body.get("output", ""),
                                   body.get("metrics"))
                self._send(200 if t else 404, t or {"error": "no such task"})
            elif p.startswith("/task/") and p.endswith("/fail"):
                rid = p[len("/task/"):-len("/fail")]
                t = state.fail(rid, body.get("reason", "unknown"))
                self._send(200 if t else 404, t or {"error": "no such task"})
            else:
                self._send(404, {"error": "unknown path"})
    return Handler


def serve(host: str = "127.0.0.1", port: int = 8471,
          min_nodes: int = 3, state_file: str | None = None) -> ThreadingHTTPServer:
    state = CoordinatorState(min_nodes=min_nodes, state_file=state_file)
    srv = ThreadingHTTPServer((host, port), make_handler(state))
    srv.coordinator_state = state  # type: ignore[attr-defined]
    return srv


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Open Swarm coordinator (127.0.0.1 only)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8471)
    ap.add_argument("--min-nodes", type=int, default=3)
    ap.add_argument("--state-file", default="results/coordinator-state.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.state_file) or ".", exist_ok=True)
    srv = serve(args.host, args.port, args.min_nodes, args.state_file)
    print(f"[coordinator] listening on http://{args.host}:{args.port} "
          f"min_nodes={args.min_nodes}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.coordinator_state.stop()


if __name__ == "__main__":
    main()
