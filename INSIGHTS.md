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
information ablation and no longer isolates the gradient path. `torch.equal`, not `allclose` —
there is no tolerance to argue about.

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

**Takeaway:** the moment a sentinel carries information richer than "done", every gate that
reads it has to be re-checked. An existence test against a status-bearing file treats failure
as success, which is the one direction a resumable pipeline must never round.

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
