# napkin-skips

**Do a UNet's skip connections earn their keep?**

Every diffusion tutorial draws the UNet with its long horizontal arrows and every one of
them says the arrows "preserve high-frequency detail". Almost none of them ablate the
arrows. This repo does — a straight leave-one-out at **2.8M parameters** on MNIST, holding
data, sampler, NFE axis and metric identical to
[napkin-diffusion](https://github.com/arose26/napkin-diffusion). This file *is* that file
plus an arm knob, which is the point: nothing else moved.

## The naming trap, first

`ResBlock.skip` in the source is the 1×1 residual projection **inside** every block. It is
**not** what is ablated here. The subject is the three `torch.cat` arrows from encoder to
decoder. **Every arm keeps every residual connection.** "A denoiser with no skips" would be
a different and much sicker network than anything in this grid.

## The five arms

Only `narrow` changes the parameter count. The rest are shape- and param-matched, so a gap
between them is *information* and cannot be capacity — the control-arm discipline
[napkin-gamemaster](https://github.com/arose26/napkin-gamemaster) learned the hard way when
its `no-stack` arm turned out to change two variables at once.

| arm | the arrow becomes | params | isolates |
|---|---|---|---|
| `full` | `cat([up(u), h_k])` | 2.813M | the textbook UNet — control |
| `zeros` | `cat([up(u), 0*h_k])` | 2.813M | **information**: same net, arrows carry nothing |
| `lo-only` | `h3` live, `h1`/`h2` zeroed | 2.813M | is it only the low-res arrow that matters? |
| `detach` | `cat([up(u), h_k.detach()])` | 2.813M | **gradient path** vs feature path |
| `narrow` | no `cat` at all | 2.576M | the plain encoder–decoder CNN |

`zeros` is the arm the headline claim rests on. It is byte-for-byte the same network as
`full` with the arrows zeroed, so it cannot be dismissed as a capacity difference.

`narrow` is deliberately *not* the control, because halving a decoder block's input width
from 2c to c removes three things, not one — and the third is easy to miss. `ResBlock`
projects its residual with a 1×1 conv only when `cin != cout`, so at `cin == cout` that
projection becomes an `Identity` and vanishes. `narrow` therefore also loses three 1×1
convs and some GroupNorm affines: **237,216 parameters**, a number `selfcheck` derives from
the layer shapes rather than reading off the model.

## Registered hypothesis

Written before the grid ran; `git log` on this file is the timestamp.

The tempting information-theoretic argument says the skips should be *unnecessary* here:
the 8×8×128 bottleneck holds 8,192 values against a 1,024-pixel output, so it is not a
capacity bottleneck at all. I predict that argument is wrong, because the bottleneck is a
*spatial* one — a 4× downsample destroys the pixel-accurate alignment that ε-prediction
needs at low noise, and no count of channels at 8px restores it.

1. **`zeros` is decisively worse than `full`** — at least 2× the FMD at 100 NFE. The skips
   are load-bearing.
2. **`lo-only` lands nearer `zeros` than `full`.** The 8px arrow is at the *same resolution*
   as the mid block, so it is close to a plain residual around three blocks; the useful
   arrows should be `h1`/`h2`, the ones that restore resolution.
3. **`detach` ≈ `full`.** The value is the forward feature path, not the gradient shortcut.
   At 2.8M parameters and 14k steps there is no vanishing-gradient problem for a shortcut
   to solve.
4. **`narrow` ≈ `zeros`.** Its 8% parameter deficit should be second-order next to losing
   the arrows. If `narrow` is much worse than `zeros`, capacity mattered too, and claim 1's
   attribution needs the `zeros` arm to survive on its own.

Prediction 1 is the one worth being wrong about. Predictions 2–4 are mechanism claims, and
any of them failing is more interesting than all of them holding.

## Results

The grid (5 arms × 5 seeds) is running. What is already measured is the **convergence probe**,
and it changed the shape of the question.

| step | 3k | 6k | 9k | 12k | 14k | 17k | 21k |
|---|---|---|---|---|---|---|---|
| `full` | 4.14 | 1.77 | 2.07 | 1.39 | 1.50 | 1.26 | 1.10 |
| `narrow` | 133 | 73.3 | 43.4 | 28.3 | 22.3 | 16.4 | 10.9 |

`full` saturates around 11k steps (the wobble after it is FMD sampling noise). `narrow` is
still falling at 21k. Because one arm converged and the other did not, the penalty depends on
where you stop:

```
30 epochs (14,040 steps):  1.50 vs 22.34  ->  14.9x
45 epochs (21,060 steps):  1.10 vs 10.88  ->   9.9x
```

So the honest primary artifact is the **FMD-vs-step curve**, not an endpoint table — every run
in the grid records one. The endpoint table still ships, labelled as the budget-dependent
slice it is.

![full vs narrow at 45 epochs](assets/probe-full-vs-narrow.png)

Both panes are the same sampler, the same seed and the same 21,060 training steps — `full`
left, `narrow` right. This is what FMD 1.10 against 10.88 looks like: not "slightly softer
digits", but digits against squiggles. Worth putting next to the numbers, because a 10x FMD
ratio sounds like a matter of degree and is not one here. (`assets/probe-*.log` hold the raw
curves.)

**What is not established:** whether `narrow` ever reaches `full`'s quality. A curve that is
still descending can asymptote anywhere, and "has not saturated" is not "will catch up". A
long single-seed run is queued to bound it. Until that lands, this repo claims a convergence
penalty and explicitly does not claim a quality ceiling.

## What this does *not* show

It is tempting to read this repo as "why a UNet beats a transformer at small scale". It
cannot say that. A UNet-vs-DiT comparison is a separate, unfinished experiment
([napkin-dit](https://github.com/arose26/napkin-dit)), and until it produces a result there
is no measured win here for skips to explain. If that comparison does land UNet-first, this
grid is a *candidate mechanism* for it — nothing more.

Scale caveat, stated once: every number here is at 2.8M parameters on 32×32 grayscale
digits. Skips plausibly matter less as the bottleneck widens and more as resolution grows,
and this repo measures one point, not a trend.

## Running it

```bash
python3 napkin_skips.py selfcheck
```

```bash
python3 napkin_skips.py grid --seeds 5
```

```bash
python3 napkin_skips.py sweep --seeds 5 && python3 napkin_skips.py report
```

Needs `torch`, `torchvision`, `matplotlib`, `imageio`, and `numpy<2` (torch 2.x wheels are
built against the numpy-1 ABI). `out/` is gitignored; every published number has its own
JSON under `assets/`.

## Metric

**FMD**, not FID: a Fréchet distance in the feature space of a small MNIST CNN, inherited
unchanged from napkin-diffusion. Comparable *within* this repo only — never against a
published FID. Inception features on upscaled grayscale digits look authoritative and mean
very little, so this repo does not pretend otherwise.
