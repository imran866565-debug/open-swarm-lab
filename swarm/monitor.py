"""Resource monitoring (I) — stdlib only, reads /proc (Linux).

Records CPU %, memory, localhost network bytes, per-process RSS, and
load average. Falls back gracefully on non-Linux systems.
"""
from __future__ import annotations

import os
import threading
import time


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def cpu_times() -> tuple[int, int]:
    """(total_jiffies, idle_jiffies) from /proc/stat."""
    for line in _read("/proc/stat").splitlines():
        if line.startswith("cpu "):
            parts = [int(x) for x in line.split()[1:]]
            return sum(parts), parts[3]
    return 0, 0


def mem_info_mb() -> dict:
    info = {}
    for line in _read("/proc/meminfo").splitlines():
        k, _, v = line.partition(":")
        try:
            info[k.strip()] = int(v.split()[0]) // 1024
        except (ValueError, IndexError):
            pass
    return {"total_mb": info.get("MemTotal", 0),
            "available_mb": info.get("MemAvailable", 0),
            "used_mb": info.get("MemTotal", 0) - info.get("MemAvailable", 0)}


def net_lo_bytes() -> dict:
    """Loopback rx/tx bytes from /proc/net/dev — the RPC traffic path."""
    for line in _read("/proc/net/dev").splitlines():
        parts = line.split()
        if parts and parts[0].rstrip(":") == "lo":
            return {"rx_bytes": int(parts[1]), "tx_bytes": int(parts[9])}
    return {"rx_bytes": 0, "tx_bytes": 0}


def proc_rss_mb(pid: int) -> int:
    for line in _read(f"/proc/{pid}/status").splitlines():
        if line.startswith("VmRSS:"):
            try:
                return int(line.split()[1]) // 1024
            except (ValueError, IndexError):
                return 0
    return 0


def snapshot(pids: list[int] | None = None) -> dict:
    total, idle = cpu_times()
    return {"ts": time.time(),
            "cpu_total": total, "cpu_idle": idle,
            "mem": mem_info_mb(),
            "net_lo": net_lo_bytes(),
            "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
            "procs": {str(p): proc_rss_mb(p) for p in (pids or [])}}


def cpu_percent(s0: dict, s1: dict) -> float:
    dt = (s1["cpu_total"] - s0["cpu_total"])
    di = (s1["cpu_idle"] - s0["cpu_idle"])
    return round(100.0 * (1 - di / dt), 1) if dt > 0 else 0.0


class Sampler(threading.Thread):
    """Sample the system every ``interval`` seconds while work runs."""

    def __init__(self, interval: float = 0.5, pids: list[int] | None = None):
        super().__init__(daemon=True)
        self.interval = interval
        self.pids = pids or []
        self.samples: list[dict] = []
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.wait(self.interval):
            self.samples.append(snapshot(self.pids))

    def stop(self) -> dict:
        self._stop_event.set()
        self.join(timeout=5)
        if len(self.samples) < 2:
            return {"samples": len(self.samples)}
        first, last = self.samples[0], self.samples[-1]
        peak_rss = {}
        for s in self.samples:
            for pid, rss in s["procs"].items():
                peak_rss[pid] = max(peak_rss.get(pid, 0), rss)
        return {
            "samples": len(self.samples),
            "duration_s": round(last["ts"] - first["ts"], 2),
            "cpu_percent_avg": cpu_percent(first, last),
            "mem_used_mb_max": max(s["mem"]["used_mb"] for s in self.samples),
            "net_lo_rx_bytes": last["net_lo"]["rx_bytes"] - first["net_lo"]["rx_bytes"],
            "net_lo_tx_bytes": last["net_lo"]["tx_bytes"] - first["net_lo"]["tx_bytes"],
            "peak_rss_mb_by_pid": peak_rss,
        }
