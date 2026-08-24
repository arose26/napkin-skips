# Insights from building napkin-skips

Written in the order I hit them, same convention as the rest of the series.

## 1. The thing named `skip` in the source is not the thing being ablated

`napkin_diffusion.py` contains, verbatim:

```python
self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
```

That is the **residual projection inside a ResBlock**. The subject of this repo is the three
`torch.cat` arrows from encoder to decoder, which the source never names at all — they are
anonymous expressions inside `forward`. So a repo about "ablating the skips" starts by
touching none of the code that says `skip`.

This is worth more than a naming pedantry note, because it changed a measurement. The first
sketch of the no-skip arm simply narrowed the decoder blocks from `2c` to `c` input channels,
which reads like "remove the concatenation". It removes three things:

| what | at cin=2c | at cin=c | delta |
|---|---|---|---|
| `GroupNorm(8, cin)` affine | `2·2c` | `2·c` | `2c` |
| `Conv2d(cin, c, 3)` weight | `c·2c·9` | `c·c·9` | `9c²` |
| residual `Conv2d(2c, c, 1)` | `2c² + c` | **`Identity`, 0** | `2c² + c` |

The third row is the trap: `ResBlock` projects its residual **only when `cin != cout`**, so
halving the input width makes that projection disappear entirely. The "remove the arrows" arm
therefore also silently removes three 1×1 convs — 237,216 parameters in total, 8.4% of the
model, and a structural change to every one of the three decoder stages.

That is why `narrow` is not the control. The arm the headline rests on is `zeros`: the same
network, byte for byte, with the arrows concatenating zeros. Its parameter count is identical
to `full` by construction, and `selfcheck` asserts it rather than trusting it.

**Takeaway:** before ablating a component, print the parameter count of the ablated model. If
it moved, you changed capacity too, and the honest fix is a shape-matched control arm rather
than a paragraph explaining why the capacity difference probably doesn't matter. (This is
napkin-gamemaster's `no-motion` lesson arriving in a completely different architecture.)

## 2. Assert the severance, don't assume it

An ablation arm can silently fail to ablate. The cheap proof here is exact rather than
tolerance-based: if an arrow really carries zeros, then the weight block reading those input
channels gets **exactly** zero gradient — not 1e-8, zero.

```python
gz = skip_grads("zeros")
assert all(v == 0.0 for v in gz.values())     # every arrow dead
gf = skip_grads("full")
assert all(v > 0.0 for v in gf.values())      # every arrow live
gl = skip_grads("lo-only")
assert gl[3] > 0.0 and gl[2] == 0.0 and gl[1] == 0.0   # exactly one arrow live
```

The `detach` arm gets the complementary assert, and it is the one I'd have skipped: with the
same weights, `detach`'s forward pass must be **bit-identical** to `full`'s, because detaching
changes the graph and not the value. If that assert ever fails, `detach` has quietly become an
information ablation and no longer isolates the gradient path.

I first wrote that assert on the GPU with `torch.equal` and a note to myself that there was
"no tolerance to argue about". That was wrong for a reason worth reading entry 5 for: on some
devices two identical forward calls are not bitwise equal either, so the assert was testing
the platform rather than the arm. It now runs on CPU, where `torch.equal` means what I thought
it meant everywhere.

**Takeaway:** for every ablation arm, write the assert that would fail if the arm were secretly
still connected (or secretly disconnecting more than intended). "I removed the line, so it must
be removed" is not a measurement, and an arm that doesn't do what its name says produces a
number that looks exactly like a real result.

## 3. A sentinel that carries an exit status must not be gated on existence

`run_shard.sh` writes each phase's exit code to `.done.<phase>` so `rc=1` is distinguishable
from `rc=0` and from "still going". The gate, though, was:

```bash
if [ ! -f .done.selfcheck ]; then ... fi
```

which means a shard whose **selfcheck failed** skips the selfcheck on the next launch and
proceeds to train. The status was recorded faithfully and then never read. Both properties —
"records why it failed" and "refuses to continue after a failure" — were needed, and having the
first one made the second one look already handled.

Fixed by testing the contents:

```bash
if [ "$(cat .done.selfcheck 2>/dev/null)" != "0" ]; then ... fi
```

A third variant of the same bug turned up later, and it is the one I would least have
predicted: `.done.sweep` has **no shard in its name**. Two shards of the same grid — seeds
0–2 and seeds 3–4 — share that one filename, so the second shard would have read the first
shard's sentinel and skipped its own sweep entirely, publishing five arms at three seeds while
believing it had five. The per-run sentinels are keyed `.done.train.<arm>-<seed>` and are
fine; the aggregate step was the one that forgot it could be run more than once.

**Takeaway:** the moment a sentinel carries information richer than "done", every gate that
reads it has to be re-checked. An existence test against a status-bearing file treats failure
as success, which is the one direction a resumable pipeline must never round. And a sentinel's
*name* is part of its correctness: if a phase can run twice with different inputs, its
filename must contain those inputs, or the second run inherits the first one's completion.

## 4. Borrowing a sibling repo's `.deps` broke the import outright

The house convention is `pip install --target .deps "numpy<2"` plus `PYTHONPATH=.deps`,
because torch 2.x wheels are built against the numpy-1 ABI. Symlinking napkin-diffusion's
`.deps` into this repo — apparently the laziest possible reuse — produced:

```
ImportError: Error importing numpy: you should not try to import numpy from
        its source directory
```

That tree was populated for a different interpreter, and `PYTHONPATH` **shadows** rather than
supplements: a working numpy 1.26.4 was already installed in this python 3.11.10, and the
shim hid it behind a source tree that refuses to import. Deleting the symlink fixed it.

**Takeaway:** a dependency shim is only correct for the interpreter it was built for, and
`PYTHONPATH` is a shadow, not a fallback. Check whether the environment already satisfies the
constraint before importing a sibling's workaround for a problem you may not have.

## 5. The second lane found a bug in my assert within six minutes of existing

Adding a Colab T4 as a second compute lane was supposed to be a wall-clock optimisation. Its
first act was to fail the selfcheck that had passed on the laptop every time:

```
assert torch.equal(mf(xb, tb), md(xb, tb)), "detach must not change the forward pass"
AssertionError: detach must not change the forward pass
```

`detach()` cannot change a forward value — it changes the autograd graph. And the arm was
fine. The assert was wrong, in a way a single machine could never have shown me: it compared
**two separate GPU forward calls** and demanded they be bitwise equal. On the RTX 4050 they
were. On the T4 they are not, so the assert was silently testing kernel determinism and
calling it an architecture claim.

The fix separates the two questions. "Detach changes the graph, not the value" is an
architecture claim, so it is asserted on **CPU**, where the arithmetic is reproducible and
`torch.equal` means what it says. The GPU then gets a *relative* check against its own noise
floor — the same model run twice — so `full` vs `detach` only has to be no further apart than
`full` is from itself:

```python
noise = (r1 - r2).abs().max().item()   # same model, twice
delta = (r1 - rd).abs().max().item()   # full vs detach
assert delta <= max(noise, 1e-7)
```

Two things worth keeping. napkin-gamemaster spent an afternoon discovering that
nondeterministic CUDA kernels perturb weights at ~1e-7 and blamed its async collector for it;
the same phenomenon reappeared here as a *false assert failure* rather than a false result,
which is the cheap way to meet it. And the honest framing of the run: this bug was latent in
the repo, would have shipped, and the only reason it surfaced is that the code ran on hardware
it was not written on. **A second device is a test, not just a second worker.**

Postscript on the thing that went right: `run_shard.sh` refused to train after the selfcheck
failed, and printed the traceback into the driver log. That gate had been fixed an hour
earlier (entry 3) for a hypothetical. It was not hypothetical.

## 6. The ablation's headline number is a function of when you stop

The 45-epoch onset probe was meant to answer a yes/no question — is 30 epochs enough? — and
instead invalidated the shape of the claim I was going to make.

| step | 3k | 6k | 9k | 12k | **14k** | 17k | **21k** |
|---|---|---|---|---|---|---|---|
| `full` | 4.14 | 1.77 | 2.07 | 1.39 | **1.50** | 1.26 | **1.10** |
| `narrow` | 133 | 73.3 | 43.4 | 28.3 | **22.3** | 16.4 | **10.9** |

`full`'s values from 11k onward wobble inside 1.10–1.68 with no trend — that is the sampling
noise of a 2,000-sample FMD estimate, and the arm is saturated. `narrow` falls monotonically
at roughly −8% per 1000 steps for the entire probe and is still falling at the end.

One arm converged and the other did not, so the measured penalty depends on the budget:

```
at 14,040 steps (30 epochs):  1.50 vs 22.34  ->  14.9x
at 21,060 steps (45 epochs):  1.10 vs 10.88  ->   9.9x
```

The ratio **shrinks as the budget grows**. So "removing the skips costs 15× FMD" is not a
statement about quality at all — it is a statement about convergence speed wearing quality's
clothes, and its value is set by where I chose to stop. napkin-gamemaster hit the same wall
from the other side ("every surviving arm was rising at the buzzer") and the lesson generalises
past RL: *a single-budget endpoint from arms with different convergence rates measures the
rates, not the ceilings.*

**And here is the inference I nearly published.** Having seen `narrow` still descending, I
drafted the headline as "the arrows don't cap quality, they cost convergence speed". That does
not follow. A monotonically falling curve can asymptote anywhere — `narrow` could level out at
8 while `full` sits at 1.1, which would be a real ceiling and not slowness at all. "Has not
saturated" and "will reach the same place" are different claims, and only the first one is
measured. The repo now states the gap at each budget, states that the gap shrinks, and states
that **whether `narrow` ever reaches `full`'s quality is unresolved by this grid** — with a
long single-seed run queued to put a bound on it rather than a paragraph of extrapolation.

**Takeaway:** when ablation arms converge at different rates, report the curve and treat the
endpoint as a budget-dependent slice of it. And before writing "X is only slower, not worse",
check whether you have observed X's asymptote or merely its descent.

## 7. Committing the sentinels made one lane skip the work the other had done

`run_shard.sh` gates every phase on `.done.<phase>` holding `0`. Entry 3 fixed the gate. What
neither the gate nor the fix noticed is that **`.done.*` was tracked by git**, because
`.gitignore` listed `out/` and `.deps` and nothing else.

So the sentinels rode the repo between machines. The local lane finished its selfcheck and
wrote `.done.selfcheck=0`; a routine `git add -A` committed it; the Colab lane pulled and
`run_shard.sh` correctly concluded the selfcheck had passed — **on a different machine, with a
different GPU, a different torch and a different numpy.** Its driver log goes straight to
`train full seed 3` with no selfcheck line at all.

That is the exact failure the second lane had already caught once (entry 5): the selfcheck is
the thing that discovered a T4/4050 disagreement, and this bug silently switched it off for
the T4. The two entries together make the point sharper than either does alone — a
cross-platform check is worthless if the "already checked" flag is itself cross-platform.

The blast radius was larger than the selfcheck. `.done.train.<arm>-<seed>` is the same kind of
file: had the shards overlapped on any seed, one lane would have skipped a run it never
performed and the grid would have quietly published an arm with a missing seed.

Fixed by making the sentinels local-only — `.done.*`, `driver.log` and `logs/` are now
gitignored and untracked.

**Takeaway:** completion state describes *a machine*, not *a codebase*, so it must never live
in the artifact that syncs between machines. When adding a resumability sentinel, add its
gitignore entry in the same commit — and when a repo starts being used from two places, audit
what `git add -A` has been sweeping up.
