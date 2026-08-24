#!/usr/bin/env bash
# One shard of the grid, for the second compute lane. Idempotent: every phase and
# every run is existence-checked, so a re-run costs at most the run in flight.
#
#   SEEDS="3 4" EPOCHS=30 setsid nohup ./run_shard.sh > driver.log 2>&1 &
#
# Liveness is file counts increasing between two ticks, never "is a process alive".
set -u
SEEDS=${SEEDS:-"3 4"}
EPOCHS=${EPOCHS:-30}
ARMS=${ARMS:-"full zeros lo-only detach narrow"}
mkdir -p out logs

if [ ! -f .done.selfcheck ]; then
  echo "[$(date -u +%H:%M:%S)] selfcheck"
  python3 napkin_skips.py selfcheck > logs/selfcheck.log 2>&1
  rc=$?; echo $rc > .done.selfcheck
  [ "$rc" -ne 0 ] && { echo "SELFCHECK FAILED rc=$rc -- refusing to train"; exit 1; }
fi

for s in $SEEDS; do
  for a in $ARMS; do
    f=".done.train.$a-$s"
    [ -f "$f" ] && continue
    echo "[$(date -u +%H:%M:%S)] train $a seed $s"
    python3 napkin_skips.py train --arm "$a" --seed "$s" --epochs "$EPOCHS" \
      > "logs/train-$a-$s.log" 2>&1
    echo $? > "$f"
  done
done

if [ ! -f .done.sweep ]; then
  echo "[$(date -u +%H:%M:%S)] sweep"
  python3 napkin_skips.py sweep --seedlist $SEEDS > logs/sweep.log 2>&1
  echo $? > .done.sweep
fi
echo "[$(date -u +%H:%M:%S)] shard complete"
tar czf shard-out.tgz out/res logs 2>/dev/null
echo "wrote shard-out.tgz"
