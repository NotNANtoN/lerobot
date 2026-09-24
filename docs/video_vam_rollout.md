# Video-VAM robot rollout (Mac ↔ abakus RPC) and physical testing

Merges the former `video_vam_rpc.md`, `video_vam_rollout.md` (Mac quickstart) and `ROBOT_TESTING_PLAN.md` (now in `archive/`). Code contract for the policy path: [`video_vam_rollout_path.md`](./video_vam_rollout_path.md).

## Architecture

- **Mac** owns the SO-101 follower (Feetech serial) and the front camera (OpenCV 640×480 @ 10 fps), runs the 10 Hz control loop and RTC action queue: `scripts/video_vam/run_mac_vam_rpc.py`.
- **abakus** (RTX 4090) hosts backbone + action expert: `scripts/video_vam/rpc_server.py`, bound to `127.0.0.1:8765` (Mac launcher default port 8766), one request at a time, reached over SSH.
- The Mac launcher starts/reuses a managed tmux server instance on abakus (`video-vam-rpc-<port>`), identified by resolved checkpoint, file fingerprint, protocol and config. Mismatches replace only the launcher-owned instance; unmanaged listeners are never killed. Startup logs: `/tmp/video-vam-rpc-PORT-INSTANCE.log` on abakus.
- No exclusive GPU lock for RPC; concurrent inference is permitted (do not run it next to a training job you care about).

## Setup (Mac)

```sh
uv sync --locked --extra feetech
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=no -o ControlPath=none abakus true
```

Keep Mac and abakus checkouts at the same commit (`sync-to-abakus.sh` / `sync-from-abakus.sh`). A SOCKS/SSH failure during POST is a transport failure, not an inference result; predictions are not retried.

## Checkpoint aliases (`--checkpoint`)

Defined in `run_mac_vam_rpc.py` (paths on abakus). Status as of 2026-09-24:

| Alias                                               | Run dir                                          | Status                                                              |
| :-------------------------------------------------- | :----------------------------------------------- | :------------------------------------------------------------------ |
| `cosmos3_lora` (default)                            | `v2-cosmos3-edge-lora-smolexpert`                | Scale-100 leader (provisional dataset)                              |
| `cosmos3_aug_v1` / `cosmos3_aug_v2`                 | `v{1,2}-cosmos3-edge-lora-online-aug-smolexpert` | online-aug runs                                                     |
| `cosmos3_base`                                      | `cosmos3-edge-undseq-smolexpert`                 | pre-routing-fix zero-shot                                           |
| `cosmos2b_t16`                                      | `cube-out-of-box-cosmos-pool2-smolexpert`        | T=16, ~1.2 s per chunk                                              |
| `cosmos2b_t2_undistilled` / `cosmos2b_t2_distilled` | `cosmos2b-t2-{undistilled,distilled}-smolexpert` | weights were missing on 09-08; check they were retrained before use |
| `smolvla_v1` / `smolvla_v2`                         | SmolVLA 29.2k (v1) / 25k (Scale-100)             | baselines                                                           |

Action flow steps: checkpoints default to 10 Euler steps. For lower latency start the server with `--euler-steps 3` (explicit; record it in the test log).

## Dry run (no camera / motors)

```sh
uv run python scripts/video_vam/run_mac_vam_rpc.py --checkpoint cosmos3_lora --dry-run \
  --inference.type rtc --ready-timeout 180 --rpc-timeout 120 --stop-server
```

RTC dry-run sends two requests with synthetic grey images and zero state (no-prefix, then prefix-guided with `inference_delay=1`) and checks finite chunks of the advertised shape. It does **not** validate cameras, calibration, safety, control rate or task success. `--inference.type sync` sends a single request. SmolVLA originals are normalized model actions; Video-VAM originals are physical actions.

## Hardware rollout

Requirements before removing `--dry-run`:

- follower port + robot id with existing calibration; front camera ≥ 10 fps (box camera usually index 1);
- `--robot.use_degrees=true` and **six measured bounds each** in `--joint-limits-min/--joint-limits-max` (5 joints in degrees, gripper 0–100). The limits in `run_rpc_server.sh` are placeholders, not calibrated limits;
- complete per-motor `--robot.max_relative_target`; support the arm on disconnect (torque release).

```sh
uv run python scripts/video_vam/run_mac_vam_rpc.py \
  --checkpoint cosmos3_lora \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5A460820701 --robot.id=so101 \
  --robot.use_degrees=true \
  --joint-limits-min <6 measured values> --joint-limits-max <6 measured values> \
  --duration=30 --task="take cube out of box"
```

The loop aborts on invalid actions, stale timing, RTC underruns or clipping. Useful flags: `--keep-server`, `--stop-server`, `--duration`, `--robot.cameras=...` to change camera index. Low-level JSON debugging: `python scripts/video_vam/rpc_client.py --policy video_vam --dry-json` (request: `state`, `images_front_u8_png_b64` — 1 PNG for SmolVLA, 5 for Video-VAM — optional `feature_seed`; response: `action_chunk [30][6]` in physical units, `policy`, `latency_s`).

## Physical test protocol (proposed standard)

Offline RMSE has not predicted grasp success (diary 09-11). Record every session as a row in the leaderboard's future "physical" table:

1. Fixed set of ≥ 10 cube start positions (marked on the table), same lighting notes, camera index, commit, checkpoint alias, `--euler-steps`.
2. Per trial: success (cube out of box), grasp attempted, grasp success, time to grasp, abort reason.
3. Order: SmolVLA baseline first, then VAM candidates, interleaved to average drift in lighting/battery.

## Starting the server manually on abakus

```bash
cd /home/anton/lerobot-video-vam && source scripts/video_vam/cosmos_cuda_env.sh
.venv/bin/python scripts/video_vam/rpc_server.py --policy video_vam \
  --checkpoint outputs/train/v2-cosmos3-edge-lora-smolexpert \
  --joint-limits-min <...> --joint-limits-max <...> [--euler-steps 3] [--no-compile]
# SmolVLA: --policy smolvla --checkpoint <run>/checkpoints/<step>/pretrained_model (no joint limits)
```
