"""OPEN SWARM automated tests (A–I).

Fast unit tests run everywhere with the stdlib unittest runner:
    python -m unittest discover -s tests -v
Integration tests (D/E/F/G with a real model) are gated on
OPEN_SWARM_INTEGRATION=1 plus OPEN_SWARM_MODEL + OPEN_SWARM_CLI_BIN,
and are normally driven by scripts/run_experiment.py, which owns the
llama.cpp build, the three RPC servers and the coordinator.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swarm import coordinator as coord_mod  # noqa: E402
from swarm import integrity, monitor  # noqa: E402
from swarm.coordinator import CoordinatorState, serve  # noqa: E402

INTEGRATION = os.environ.get("OPEN_SWARM_INTEGRATION") == "1"


def http(url, payload=None, method="GET", timeout=10):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class CoordinatorFixture(unittest.TestCase):
    def setUp(self):
        self.port = free_port()
        self.srv = serve("127.0.0.1", self.port, min_nodes=3)
        self.t = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.t.start()
        self.base = f"http://127.0.0.1:{self.port}"
        for _ in range(50):
            try:
                http(f"{self.base}/health")
                break
            except Exception:
                time.sleep(0.1)

    def tearDown(self):
        self.srv.shutdown()
        self.srv.coordinator_state.stop()

    def register_node(self, name, rpc_port):
        return http(f"{self.base}/register", {
            "name": name, "rpc_port": rpc_port, "cpu_count": 4,
            "mem_total_mb": 8192, "disk_free_mb": 10240}, "POST")


class TestNodeStartup(CoordinatorFixture):
    """A. Node startup — three nodes launch, unique IDs, all register."""

    def test_three_nodes_register_with_unique_ids(self):
        ids = set()
        for i in range(3):
            rec = self.register_node(f"worker-{i}", 50052 + i)
            ids.add(rec["peer_id"])
            self.assertEqual(rec["status"], "alive")
        self.assertEqual(len(ids), 3)
        nodes = http(f"{self.base}/nodes")["nodes"]
        self.assertEqual(len(nodes), 3)


class TestDiscovery(CoordinatorFixture):
    """B. Discovery — coordinator found, heartbeats work, dead detected."""

    def test_heartbeat_keeps_alive_and_timeout_marks_dead(self):
        rec = self.register_node("worker-0", 50052)
        pid = rec["peer_id"]
        http(f"{self.base}/heartbeat", {"peer_id": pid}, "POST")
        nodes = {n["peer_id"]: n for n in http(f"{self.base}/nodes")["nodes"]}
        self.assertEqual(nodes[pid]["status"], "alive")
        self.assertGreaterEqual(nodes[pid]["heartbeats"], 2)
        # Simulate a dead node: stop heartbeating, shrink the timeout.
        old = coord_mod.HEARTBEAT_TIMEOUT
        coord_mod.HEARTBEAT_TIMEOUT = 0.3
        try:
            time.sleep(0.6)
            self.srv.coordinator_state.sweep()
            nodes = {n["peer_id"]: n for n in http(f"{self.base}/nodes")["nodes"]}
            self.assertEqual(nodes[pid]["status"], "dead")
        finally:
            coord_mod.HEARTBEAT_TIMEOUT = old

    def test_unknown_peer_heartbeat_is_404(self):
        with self.assertRaises(urllib.error.HTTPError):
            http(f"{self.base}/heartbeat", {"peer_id": "node-doesnotexist"}, "POST")


class TestResourceSharing(CoordinatorFixture):
    """C. Resource sharing — capacity advertised, coordinator sees all,
    work assigned by capacity; honest failure when nodes are missing."""

    def test_resources_visible_and_task_assigned_by_capacity(self):
        for i, cpu in enumerate((2, 8, 4)):
            http(f"{self.base}/register", {
                "name": f"worker-{i}", "rpc_port": 50052 + i, "cpu_count": cpu,
                "mem_total_mb": 4096, "disk_free_mb": 10240}, "POST")
        nodes = http(f"{self.base}/nodes")["nodes"]
        self.assertEqual(len(nodes), 3)
        self.assertTrue(all(n["cpu_count"] for n in nodes))
        task = http(f"{self.base}/submit", {"prompt": "hello"}, "POST")
        self.assertEqual(task["status"], "assigned")
        self.assertEqual(len(task["rpc_endpoints"]), 3)
        # Highest-CPU node first in the assignment order.
        self.assertEqual(task["assigned_nodes"][0],
                         [n for n in nodes if n["cpu_count"] == 8][0]["peer_id"])

    def test_submit_with_too_few_nodes_fails_honestly(self):
        self.register_node("worker-0", 50052)
        task = http(f"{self.base}/submit", {"prompt": "hello"}, "POST")
        self.assertEqual(task["status"], "failed")
        self.assertIn("insufficient nodes", task["reason"])


class TestIntegrity(unittest.TestCase):
    """H. Data integrity — chunk hashes, corruption + loss detection."""

    def test_split_verify_reconstruct_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "payload.bin")
            data = os.urandom(3 * 1024 * 1024 + 123)
            with open(src, "wb") as f:
                f.write(data)
            chunks = os.path.join(d, "chunks")
            manifest = integrity.split_file(src, chunks, chunk_size=1024 * 1024)
            self.assertEqual(len(manifest["chunks"]), 4)
            self.assertTrue(integrity.verify_chunks(manifest, chunks)["ok"])
            # Corrupt one chunk -> detected, reconstruct refuses.
            bad = os.path.join(chunks, manifest["chunks"][1]["file"])
            with open(bad, "r+b") as f:
                f.seek(10)
                f.write(b"X")
            v = integrity.verify_chunks(manifest, chunks)
            self.assertFalse(v["ok"])
            self.assertEqual(v["corrupt"], [1])
            self.assertFalse(integrity.reconstruct(manifest, chunks,
                                                   os.path.join(d, "out.bin"))["ok"])
            # Missing chunk -> detected.
            os.remove(os.path.join(chunks, manifest["chunks"][2]["file"]))
            v = integrity.verify_chunks(manifest, chunks)
            self.assertIn(2, v["missing"])
            # Clean copy reconstructs byte-identical.
            manifest2 = integrity.split_file(src, os.path.join(d, "chunks2"),
                                             chunk_size=1024 * 1024)
            out = os.path.join(d, "rebuilt.bin")
            r = integrity.reconstruct(manifest2, os.path.join(d, "chunks2"), out)
            self.assertTrue(r["ok"])
            self.assertEqual(r["sha256"], manifest2["source_sha256"])


class TestMonitoring(unittest.TestCase):
    """I. Resource monitoring — snapshots + sampler produce real numbers."""

    def test_snapshot_and_sampler(self):
        s0 = monitor.snapshot()
        self.assertGreater(s0["mem"]["total_mb"], 0)
        self.assertIn("rx_bytes", s0["net_lo"])
        sampler = monitor.Sampler(interval=0.2)
        sampler.start()
        time.sleep(0.5)
        summary = sampler.stop()
        self.assertGreaterEqual(summary["samples"], 1)
        self.assertIn("cpu_percent_avg", summary)


@unittest.skipUnless(INTEGRATION, "integration: needs built llama.cpp + model")
class TestDistributedInference(unittest.TestCase):
    """D/E/F/G (integration) — real Qwen inference over 3 RPC servers.
    Driven by scripts/run_experiment.py; kept here so CI can run the
    same assertions when the model + binaries are present."""

    @classmethod
    def setUpClass(cls):
        cls.model = os.environ["OPEN_SWARM_MODEL"]
        cls.cli = os.environ["OPEN_SWARM_CLI_BIN"]
        cls.rpc = os.environ["OPEN_SWARM_RPC_SERVERS"].split(",")
        assert len(cls.rpc) == 3, "need exactly 3 RPC servers"
        assert os.path.exists(cls.model), f"model missing: {cls.model}"
        assert os.path.exists(cls.cli), f"llama-cli missing: {cls.cli}"

    def _infer(self, prompt, rpc=True):
        from swarm.rpc import run_inference
        return run_inference(self.cli, self.model, prompt,
                             rpc_servers=self.rpc if rpc else None,
                             max_tokens=64, timeout=600)

    def test_distributed_inference_returns_real_answer(self):
        res = self._infer("Explain how a decentralized cloud works.")
        self.assertTrue(res["ok"], res.get("log", "")[-2000:])
        self.assertGreater(len(res["output"]), 20)

    def test_three_prompts_one_answer_each(self):
        for p in ("What is 2 + 2? Answer briefly.",
                  "Name one benefit of distributed computing.",
                  "Explain how a decentralized cloud works."):
            res = self._infer(p)
            self.assertTrue(res["ok"], p)
            self.assertGreater(len(res["output"]), 10)

    def test_concurrent_prompts_do_not_mix(self):
        from concurrent.futures import ThreadPoolExecutor
        prompts = ["Say the word APPLE and nothing else.",
                   "Say the word BANANA and nothing else."]
        with ThreadPoolExecutor(max_workers=2) as ex:
            results = list(ex.map(self._infer, prompts))
        self.assertTrue(all(r["ok"] for r in results))
        # Request isolation: both answers non-empty and different from
        # each other (a 0.5B model does not reliably obey "say only X",
        # so keyword presence is recorded, not asserted).
        self.assertTrue(all(len(r["output"]) > 5 for r in results))
        self.assertNotEqual(results[0]["output"], results[1]["output"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
