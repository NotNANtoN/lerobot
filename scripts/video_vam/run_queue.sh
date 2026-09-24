#!/usr/bin/env bash
# Run several presets sequentially (replaces the dated run_*_queue.sh scripts).
#
#   bash scripts/video_vam/run_queue.sh <preset> [<preset> ...]
#   CONTINUE_ON_ERROR=1 bash scripts/video_vam/run_queue.sh a b c
#
# Launch detached on abakus so the queue survives SSH drops (kill sessions by name, never the
# tmux server PID):
#   tmux new -d -s vam_queue 'bash scripts/video_vam/run_queue.sh a b c'
set -Euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
(( $# > 0 )) || { printf 'usage: %s <preset> [<preset> ...]\n' "$0" >&2; exit 64; }

for preset in "$@"; do
    [[ -f "${SCRIPT_DIR}/presets/${preset}.args" ]] || { printf 'Unknown preset: %s\n' "$preset" >&2; exit 64; }
done

failed=()
for preset in "$@"; do
    printf '[queue] %s starting %s\n' "$(date)" "$preset"
    if ! bash "${SCRIPT_DIR}/run_experiment.sh" "$preset"; then
        failed+=("$preset")
        printf '[queue] %s FAILED %s\n' "$(date)" "$preset" >&2
        [[ ${CONTINUE_ON_ERROR:-0} == 1 ]] || exit 1
    fi
done
printf '[queue] done; failed: %s\n' "${failed[*]:-none}"
(( ${#failed[@]} == 0 ))
