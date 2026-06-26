# RunPod Debug Log — WS1 launch (Family B & C divergence)

Setup gotchas hit while launching `scripts/ws1_pod.sh` on a fresh H100 pod
(`slgqwiibziprz8`, EU-FR-1, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`),
2026-06-23. Complements the env-recipe / RunPod-ops notes in `RUNLOG.md` and
the upstream-bug docs. Newest first.

---

## Image is PEP 668 "externally-managed" → pip refuses
The `...-ubuntu2404` image's system Python blocks `pip install`
("externally-managed-environment"). The older Phase-2 image didn't.
**Fix:** `export PIP_BREAK_SYSTEM_PACKAGES=1` (or `--break-system-packages`)
before any pip in the pod script.

## This image already ships RELEASE torch 2.8.0 → skip the recipe-v2 reinstall
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` has release torch 2.8.0+cu128
preinstalled, so the RUNLOG "recipe v2" torch reinstall (which existed to undo a
*dev* torch in the old image) is unnecessary here. Just install
transformers/peft/accelerate/kagglehub + mamba-ssm/causal-conv1d
(`--no-deps --no-build-isolation`) and verify kernels import.

## `pkill -f ws1_pod.sh` killed its own SSH shell (exit 255)
The remote `bash -c '… pkill -f ws1_pod.sh …'` command line *contains* the
string `ws1_pod.sh`, so `pkill -f` matched and killed the shell running it →
connection died (exit 255), and the intended relaunch in the same command never
ran. (Same "self-match curse" noted for `pkill/pgrep` in project memory.)
**Fix:** bracket the pattern — `pkill -9 -f '[w]s1_pod.sh'` — and keep kill,
verify, and relaunch in **separate** SSH calls.

## kagglehub stdout polluted a captured shell variable
`B_DIR=$(python3 -c "... print(path)")` captured kagglehub's progress lines
("Download already complete (1046 bytes).", "Downloading to …") *into the
variable*, so the adapter-config existence check failed →
`FATAL: B adapter_config missing` and the script `exit 1`'d (before reaching
the self-stop logic, so the pod stayed up — idle billing).
**Fix:** have Python **write the resolved path to a file**
(`open('/workspace/ws1/B_DIR','w').write(path)`), then `B_DIR=$(cat …)`. No
stdout capture.

## Partial kagglehub downloads re-download fully
A version pulled with a timeout (no `.complete` marker) re-downloads the whole
version next time — including the big bundled checkpoints/optimizer states
(lora3 v3 is ~6.5 GB total though the adapter itself is ~65 MB). Let the first
download finish, or it restarts from scratch.

## Captures are SLOW — budget reality (not a bug, a constraint)
`capture_routing.py` MAX_TOKENS are large (factual 1536, computational 2048,
reasoning/symbolic 3072) and it's a thinking model, so per-token routing capture
(23 hooks/token) runs ~3.2-3.7 tok/s → **~15-23 min per 8-problem batch**.
Family B ≈ 5-6 h; B+C ≈ 11-12 h ≈ $37-40 at $3.29/h. Original "$15-20" estimate
was too low. If budget-bound, options: cut MAX_TOKENS (routing distributions are
fairly robust, but base npz were full-length → looser JSD comparability), run
B-only, or raise the cap. (Chosen: full budgets, $50 cap, stop-after-B rule if
nearing cap.)

## Reusable launch recipe (worked)
1. Create pod via REST `POST /v1/pods` with `networkVolumeId` (pins DC),
   `ports:["22/tcp"]`, `env:{PUBLIC_KEY: <throwaway pubkey>}` — **no API key in
   env** (secret scanner). Crib `imageName`/mount from a prior pod via
   `GET /v1/pods/{id}`.
2. Poll `GET /v1/pods/{id}` for `portMappings` → `{"22": <port>}` + `publicIp`.
3. `ssh -p <port> root@<ip> -i <key>`; write `/root/.podenv` (umask 077) with
   `POD_ID` + `RP_KEY` for self-delete; copy/neuter `capture_routing_h100.py`
   (`patch_ptxas` → insert `return` first line, Hopper doesn't need it); `nohup`
   the run.
4. Script self-DELETEs the pod on full success, self-STOPs on failure.
