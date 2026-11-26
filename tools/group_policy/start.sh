#!/usr/bin/env bash
set -euo pipefail
ALPHA_ROOT=/root/autodl-tmp/alphadefect
algorithm=${1:?Provide grpo, dapo or dspo}
case "$algorithm" in grpo|dapo|dspo) ;; *) exit 2 ;; esac
mkdir -p "$ALPHA_ROOT/rl-runs/$algorithm"
exec 9>"$ALPHA_ROOT/rl-runs/$algorithm/run.lock"
if ! flock -n 9; then
  echo "The experiment already holds its run lock."
  exit 0
fi
nohup setsid bash "$ALPHA_ROOT/src/tools/group_policy/run.sh" "$ALPHA_ROOT/rl-configs/$algorithm.json" \
  >> "$ALPHA_ROOT/rl-runs/$algorithm/console.log" 2>&1 < /dev/null &
echo "$!" > "$ALPHA_ROOT/rl-runs/$algorithm/launcher.pid"
cat "$ALPHA_ROOT/rl-runs/$algorithm/launcher.pid"
