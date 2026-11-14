# Multi-runner investigation — can three GitHub jobs be three swarm nodes?

**Question:** can the three workers run as three *separate* GitHub Actions
jobs/runners, so the swarm spans three machines instead of three processes
on one VM — without paid infrastructure or unsafe public ports?

**Short answer: not with GitHub-hosted runners alone. BLOCKED for now,
by architecture, not by effort.** The one-VM experiment in this repo is the
honest first step; the real multi-machine path is contributor devices.

## What was investigated

GitHub-hosted runners (`ubuntu-latest`) are ephemeral, isolated VMs:

1. **No peer-to-peer connectivity.** Each job gets its own VM behind NAT.
   There is no documented way for job A to open a TCP connection to job B.
   Runner IPs are not published to the workflow, and inbound ports are not
   reachable. Assuming otherwise would be pretending.
2. **No shared localhost.** `127.0.0.1` in job A is a different machine
   from `127.0.0.1` in job B. The llama.cpp RPC backend binds localhost by
   design (and must stay localhost-only for security — it is documented
   upstream as insecure/proof-of-concept).
3. **Workflow artifacts / caches are not a live channel.** They are
   async file handoffs (upload → later download). You cannot run live RPC
   inference or heartbeats through them.
4. **Service containers** run on the *same* runner as the job — they do
   not span jobs.

## Candidate bridges — and why each is rejected or deferred

| Bridge | Verdict | Reason |
|---|---|---|
| Public port on a runner + GitHub API for IP discovery | **Rejected** | Unsafe: exposes the insecure RPC backend to the internet; runner IP is not reliably discoverable; ephemeral. |
| Cloudflare Tunnel (quick) from each job, coordinator via tunnel URLs | **Not viable in CI** | Each job would need to publish its tunnel URL somewhere the others can read *live* (artifacts are async); jobs in one workflow cannot rendezvous in real time. Also adds a third-party MITM to model traffic. |
| Tailscale / WireGuard mesh between jobs | **Blocked** | Needs an account + auth key in secrets, and a coordination server; free-tier tailnets exist, but storing a network key in CI secrets for an experiment that moves model tensors is out of scope and still does not solve job-to-job discovery timing. Revisit only with an explicit decision. |
| Matrix job + `workflow_call` chaining | **Not live** | Sequential, not concurrent; no live socket between runs. |
| Paid cloud VM / VPS as rendezvous | **Rejected** | Mission rule: no paid hosting. |
| **Self-hosted runners on contributor devices** | **The real path** | This is the actual Open Swarm vision: a volunteer's own machine runs a self-hosted runner (or the future node agent), registers with the coordinator over an outbound-only connection, and contributes CPU/RAM/storage. No inbound ports on the contributor, no paid cloud. Deferred to the next experiment — it needs a real contributor device, not CI. |

## Conclusion

- **This experiment:** 3 logical nodes / 1 runner VM, localhost RPC.
  Proves coordination, distribution, failure handling.
- **Next experiment (not in this repo yet):** one GitHub-hosted runner
  as coordinator + one self-hosted runner on a contributor device as a
  remote node, coordinator reached via the contributor's *outbound*
  connection only. That is the smallest honest multi-machine step.
- Any claim of "three separate GitHub jobs peered live" without one of
  the bridges above would be fabricated. This document exists so the
  limitation stays visible.
