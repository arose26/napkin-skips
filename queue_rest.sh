#!/usr/bin/env bash
# Everything after the seeds 0-2 shard, in one local queue.
#
# Deliberately does NOT call run_shard.sh, for two reasons:
#   1. bash reads a script incrementally, so editing run_shard.sh while the live
#      0-2 driver is still inside it can resume that driver mid-line.
#   2. run_shard.sh's sweep gate is a single global `.done.sweep`, so a second
#      shard would see the first shard's sentinel and skip its own sweep. The
#      sentinels here are per-shard.
set -u
cd "$(dirname "$0")"
ARMS="full zeros lo-only detach narrow"

# The 0-2 shard signals completion by writing .done.sweep=0 as its last gated step.
until [ "$(cat .done.sweep 2>/dev/null)" = "0" ]; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] shard 0-2 complete; starting seeds 3-4"

for s in 3 4; do
  for a in $ARMS; do
    f=".done.train.$a-$s"
    [ "$(cat "$f" 2>/dev/null)" = "0" ] && continue
    echo "[$(date -u +%H:%M:%S)] train $a seed $s"
    python3 napkin_skips.py train --arm "$a" --seed "$s" --epochs 30 \
      > "logs/train-$a-$s.log" 2>&1
    echo $? > "$f"
  done
done

if [ "$(cat .done.sweep.34 2>/dev/null)" != "0" ]; then
  echo "[$(date -u +%H:%M:%S)] sweep seeds 3 4"
  python3 napkin_skips.py sweep --seedlist 3 4 > logs/sweep-34.log 2>&1
  echo $? > .done.sweep.34
fi
echo "[$(date -u +%H:%M:%S)] grid complete (25 runs)"

# Bounds the open question: does narrow asymptote above full's ~1.1 plateau
# (a real ceiling) or reach it (only slower)? Same machine as every grid run,
# because this number is only meaningful next to full's.
if [ "$(cat .done.longnarrow 2>/dev/null)" != "0" ]; then
  echo "[$(date -u +%H:%M:%S)] long narrow probe (120 epochs)"
  python3 napkin_skips.py train --arm narrow --seed 98 --epochs 120 --every 2000 \
    > logs/long-narrow.log 2>&1
  echo $? > .done.longnarrow
fi
echo "[$(date -u +%H:%M:%S)] all local work complete"
