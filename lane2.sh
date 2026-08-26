#!/usr/bin/env bash
# Second lane: resolve the one unresolved registered prediction (detach vs full).
#
# Runs ONLY full and detach at seeds 5-9, so that pair reaches n=10 while the
# five-arm table stays at n=5. These are deliberately NOT swept into out/res --
# merging them would give two arms n=10 against three arms n=5, which report()
# already refuses to tabulate ("never compare at mismatched n").
#
# Memory-capped so it cannot starve the long full-98 run that is already in
# flight. That job's allocator holds its high-water mark, but a greedy second
# process could still take headroom it needs; capping is cheaper than finding out.
set -u
cd "$(dirname "$0")"
mkdir -p out logs
FRAC=${FRAC:-0.38}
for s in 5 6 7 8 9; do
  for a in full detach; do
    f=".done.pair.$a-$s"
    [ "$(cat "$f" 2>/dev/null)" = "0" ] && continue
    echo "[$(date -u +%H:%M:%S)] lane2 train $a seed $s (cap ${FRAC})"
    python3 capped_train.py "$FRAC" --arm "$a" --seed "$s" --epochs 30 \
      > "logs/train-$a-$s.log" 2>&1
    echo $? > "$f"
  done
done
echo "[$(date -u +%H:%M:%S)] lane2 complete"
