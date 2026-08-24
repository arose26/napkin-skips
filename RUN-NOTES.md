# Run notes

Watchers do not survive a context refresh or a machine restart, so the arming commands live
here rather than being reconstructed.

## Lane allocation (revised 2026-08-24 after two second-lane failures)

**The published grid runs entirely on the local RTX 4050.** All 25 runs, one machine.

The original plan sharded the grid across a second lane to halve wall-clock. That plan is
abandoned, and the reasons are worth keeping:

- **Colab free tier reaped the VM twice inside an hour**, the second time losing a partial
  run outright (checkpoints are only written at the end, so ~1 h of T4 time produced nothing).
  It does real work while alive — 2000 steps in 214 s at 99% GPU — but disconnects whenever
  the browser is idle, and its limits are duration-based as well as inactivity-based.
- **A rented RTX 4090 (RunPod, $0.34/h community) could not use its GPU.** `nvidia-smi` saw
  the card; `torch.cuda.is_available()` returned `False` with "CUDA unknown error", and a pod
  restart did not clear it. Terminated after ~10 min, ~$0.06 spent.
- **The second lane introduced a correctness bug**, not just unreliability: `.done.*`
  sentinels were git-tracked, so the local lane's passing selfcheck synced to Colab and
  switched Colab's own selfcheck off. See INSIGHTS #7.

The second lane was only ever a speed optimisation. Dropping it costs time and nothing else —
and it removes the hardware variable from the grid entirely, which is strictly better for the
comparison than balancing it was.

**The Colab lane is dropped entirely.** It was reaped a third time within minutes of being
repointed, and — more decisively — there is no job left that it can correctly do.

The long `narrow` probe looked disposable, but it is not hardware-free: its whole purpose is
to be compared against `full`'s plateau, which is a **cross-arm** comparison. Running it on a
T4 while every `full` run is on the 4050 would reintroduce exactly the arm↔hardware confound
that sharding by seed was designed to avoid. A measurement that is only meaningful next to
another arm has to run on the same machine as that arm.

So everything runs local, in one queue:

| lane | hardware | job |
|---|---|---|
| local | RTX 4050 Laptop, 6 GB, torch 2.4.1+cu121, numpy 1.26.4 | **all 25 grid runs, then the long `narrow` probe** |
| Colab | — | **nothing.** Not viable, and no job is hardware-free enough to give it |

Idle capacity is not a reason to run something on it. Correctness beats utilisation.

One practical note for any future relaunch on a stale clone: `git pull` aborts with *"Your
local changes to driver.log would be overwritten by merge"*, because `driver.log` was tracked
before INSIGHTS #7 and the clone predates its removal. `git fetch && git reset --hard
origin/main` is the fix; a fresh clone does not have the problem.

Measured local rate: 14,040 steps in ~1,290 s = **21.5 min/run** including 7 FMD probes.
Seeds 0–2 (15 runs) ETA ~14:25; seeds 3–4 (10 runs, launched after) ETA ~18:00.

## Local lane

```bash
cd /home/bob/napkin-skips
SEEDS="0 1 2" EPOCHS=30 setsid nohup ./run_shard.sh > driver.log 2>&1 &
```

then, when that shard's `.done.shard` lands:

```bash
SEEDS="3 4" EPOCHS=30 setsid nohup ./run_shard.sh > driver2.log 2>&1 &
```

## Colab lane

Notebook `Untitled14.ipynb`, drive id `1-8eQj6JK9wk0X5z22BTrNUirEDm8NiF3`, account K.
Working copy `/content/napkin-skips`, cloned from GitHub — code is never pasted through the
browser, the cell just pulls.

```bash
%%bash
cd /content
[ -d napkin-skips ] || git clone -q https://github.com/arose26/napkin-skips.git
cd napkin-skips
git pull -q
SEEDS="3 4" EPOCHS=30 setsid nohup ./run_shard.sh > driver.log 2>&1 &
sleep 8
echo launched; tail -5 driver.log 2>/dev/null; true
```

`setsid` matters: without it the driver dies with the cell. A `%%bash` cell **raises** on any
nonzero exit, so such a cell must end in something that exits 0 (`true`).

**Never `git pull` while the driver is running** — bash reads a script incrementally, so
rewriting `run_shard.sh` under a live driver can resume it at a byte offset that is now the
middle of a different line.

Cell outputs are not scrapable over CDP (sandboxed cross-origin iframes); the read channel is
a screenshot of the poll cell. Liveness is **file counts increasing between two ticks**, never
"is a process alive".

## Sentinels

`.done.selfcheck`, `.done.train.<arm>-<seed>`, `.done.sweep` each carry the phase's **exit
status**, and every gate tests the contents rather than the file's existence — a failed phase
must not be skipped on relaunch. `rc=1` is therefore distinguishable from `rc=0` and from
"still going".

## The budget decision

`--epochs 30` is inherited from napkin-diffusion, where it was chosen for `full` alone. It is
**not** taken on trust: an onset probe runs `full` and `narrow` (the two extreme arms) for 45
epochs at seed 99, probing FMD every 1000 steps, to check that every arm has stopped
improving before the grid's budget ends.

```bash
for arm in full narrow; do
  { python3 napkin_skips.py train --arm $arm --seed 99 --epochs 45 --every 1000 \
      > out/probe-$arm.log 2>&1; echo $? > .done.probe.$arm; } &
done
```

Seed 99 is outside the grid's 0–4 range, so the probe checkpoints never collide with grid
runs and are never swept into the results.

If the curves are still falling at 14,040 steps (30 epochs), both lanes restart at the longer
budget and the sentinels must be cleared first. The alternative — publishing a grid that ended
before the arms converged — reports a ranking of whichever arm merely started faster, which is
the mistake napkin-gamemaster made and documented.

## Reproducing the environment

The local lane needs no `.deps` directory: this interpreter (python 3.11.10) already has
torch 2.4.1 with numpy 1.26.4. An earlier `.deps` symlink borrowed from napkin-diffusion
broke the import outright — that tree's numpy was built for a different interpreter, and
`PYTHONPATH=.deps` shadowed the working one with a source tree that refuses to import.

## Queued: the long `narrow` run (bounds the open question)

The probe left one thing unresolved and the README says so: `narrow` is still falling at
21,060 steps, and a descending curve can asymptote anywhere. "Has not saturated" is not "will
catch up", so the repo claims a **convergence penalty** and explicitly declines to claim a
**quality ceiling**.

To bound it, one long single-seed run of `narrow`. **This is the Colab lane's only job** —
it is outside the grid, so losing it to a reaped VM costs nothing but a relaunch:

```bash
cd /content/napkin-skips && git pull -q
python3 napkin_skips.py train --arm narrow --seed 98 --epochs 120 --every 2000 \
  > logs/long-narrow.log 2>&1 &
```

120 epochs = 56,160 steps, ~4x the grid budget. Seed 98 keeps it out of the grid's 0–4 range
and out of the probe's 99, so it can never be swept into the published table.

What it decides: if `narrow` levels out above `full`'s ~1.1 plateau, the arrows set a quality
ceiling at this scale. If it reaches ~1.1, they only cost convergence speed. Either answer is
publishable; the current state — not knowing — is the one thing that cannot be written up as
a finding.
