"""One prompt, one answer — the user-facing CLI.

The user submits ONE prompt. The client asks the coordinator for an
assignment (which RPC nodes form the pool), runs the real distributed
inference through llama.cpp, reports the single completed answer back
to the coordinator, and prints ONLY the final answer on stdout.
All internal node operations go to stderr / the log file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

from swarm.rpc import find_bin, run_inference


def http_json(url: str, payload: dict | None = None, method: str = "GET",
              timeout: int = 15) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def ask(prompt: str, coordinator: str, cli_bin: str, model: str,
        max_tokens: int = 128, timeout: int = 600) -> dict:
    coordinator = coordinator.rstrip("/")
    task = http_json(f"{coordinator}/submit",
                     {"prompt": prompt, "max_tokens": max_tokens}, "POST")
    request_id = task["request_id"]
    if task["status"] != "assigned":
        return {"request_id": request_id, "ok": False,
                "reason": task.get("reason", "not assigned"),
                "output": ""}
    t0 = time.time()
    res = run_inference(cli_bin, model, prompt,
                        rpc_servers=task["rpc_endpoints"],
                        max_tokens=max_tokens, timeout=timeout)
    res["request_id"] = request_id
    res["elapsed_s"] = round(time.time() - t0, 2)
    if res["ok"]:
        http_json(f"{coordinator}/task/{request_id}/complete",
                  {"output": res["output"],
                   "metrics": {"elapsed_s": res["elapsed_s"],
                               "tokens_per_s": res.get("tokens_per_s"),
                               "rpc_endpoints": res["rpc_servers"]}}, "POST")
    else:
        http_json(f"{coordinator}/task/{request_id}/fail",
                  {"reason": f"inference rc={res.get('returncode')}"}, "POST")
    return res


def main():
    ap = argparse.ArgumentParser(description="Open Swarm client: one prompt, one answer")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--coordinator", default="http://127.0.0.1:8471")
    ap.add_argument("--cli-bin", required=True, help="path to llama-cli")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()
    res = ask(args.prompt, args.coordinator, args.cli_bin, args.model,
              args.max_tokens)
    if not res["ok"]:
        print(f"ERROR: {res.get('reason', 'inference failed')}", file=sys.stderr)
        sys.exit(1)
    # stdout carries ONLY the final answer.
    print(res["output"])


if __name__ == "__main__":
    main()
