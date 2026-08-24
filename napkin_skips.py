"""napkin-skips: do a UNet's encoder-decoder skip connections earn their keep?

Everyone draws the UNet with its long horizontal arrows and everyone says they
"preserve high-frequency detail". Nobody in the tutorial literature ablates them.
This repo does, at 2.8M parameters on MNIST, holding data, sampler, NFE axis and
metric identical to napkin-diffusion (this file is that file plus an arm knob).

Naming, up front, because the code itself is a trap: `ResBlock.skip` is the 1x1
residual projection *inside* every block, and it is NOT what is ablated here.
The subject is the three `torch.cat` arrows from encoder to decoder. Every arm
keeps all residual connections.

Five arms. Only `narrow` changes the parameter count; the rest are shape- and
param-matched, so information can be separated from capacity (the lesson
napkin-gamemaster learned the hard way with its `no-motion` control):

  full     cat([up(u), h_k])            the textbook UNet -- control
  zeros    cat([up(u), 0*h_k])          same params/FLOPs/shapes, zero information
  lo-only  h3 only, h1/h2 zeroed        is it just the low-res skip that matters?
  detach   cat([up(u), h_k.detach()])   same information, no gradient shortcut
  narrow   no cat at all                the plain encoder-decoder CNN; fewer params

`zeros` is the arm that makes the headline claim falsifiable: it is byte-for-byte
the same network as `full` with the arrows carrying nothing, so any gap between
them is the *information* in the skips and cannot be capacity.

The registered hypothesis is in the README, written before the grid ran.

Usage:
    PYTHONPATH=.deps python3 napkin_skips.py selfcheck
    PYTHONPATH=.deps python3 napkin_skips.py onset --arms full narrow
    PYTHONPATH=.deps python3 napkin_skips.py train --arm full --seed 0
    PYTHONPATH=.deps python3 napkin_skips.py grid
    PYTHONPATH=.deps python3 napkin_skips.py report
    PYTHONPATH=.deps python3 napkin_skips.py gif
"""
import argparse, json, math, pathlib, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = pathlib.Path(__file__).parent / "out"
T = 1000

# ---------------------------------------------------------------- noise schedule

def cosine_alpha_bar(T=T, s=0.008):
    """Nichol & Dhariwal cosine schedule, via betas clipped at 0.999."""
    t = torch.linspace(0, T, T + 1)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0)          # (T,), alpha_bar[i] for t=i


AB = cosine_alpha_bar().to(DEV)


def q_sample(x0, t, noise):
    ab = AB[t].view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

# ------------------------------------------------------------------------ model

class TimeEmb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.SiLU(), nn.Linear(d * 4, d * 4))

    def forward(self, t):
        half = self.d // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, C, H * W).permute(1, 0, 3, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(h.transpose(1, 2).reshape(B, C, H, W))


ARMS = ("full", "zeros", "lo-only", "detach", "narrow")

# Which encoder levels hand their real features to the decoder. Levels not listed
# still get concatenated -- as zeros -- so the tensor shapes and the parameter
# count stay identical to `full`. `narrow` is the sole exception: it removes the
# concatenation itself, which is why it is the only arm with fewer parameters.
LIVE = {"full": (1, 2, 3), "detach": (1, 2, 3), "lo-only": (3,),
        "zeros": (), "narrow": ()}


class UNet(nn.Module):
    """32 -> 64 -> 128 channels at 32/16/8 px, self-attention at 8px."""

    def __init__(self, ch=(32, 64, 128), tdim=32, arm="full"):
        super().__init__()
        assert arm in ARMS, f"unknown arm {arm!r}"
        self.arm, self.live = arm, LIVE[arm]
        cat = 1 if arm == "narrow" else 2      # decoder input width multiplier
        c1, c2, c3 = ch
        td = tdim * 4
        self.temb = TimeEmb(tdim)
        self.inp = nn.Conv2d(1, c1, 3, padding=1)
        self.d1a, self.d1b = ResBlock(c1, c1, td), ResBlock(c1, c1, td)
        self.down1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.d2a, self.d2b = ResBlock(c1, c2, td), ResBlock(c2, c2, td)
        self.down2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)
        self.d3a, self.d3b = ResBlock(c2, c3, td), ResBlock(c3, c3, td)
        self.at3 = Attn(c3)
        self.mid1, self.midat, self.mid2 = ResBlock(c3, c3, td), Attn(c3), ResBlock(c3, c3, td)
        self.u3a, self.u3b = ResBlock(c3 * cat, c3, td), ResBlock(c3, c3, td)
        self.up3 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)
        self.u2a, self.u2b = ResBlock(c2 * cat, c2, td), ResBlock(c2, c2, td)
        self.up2 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)
        self.u1a, self.u1b = ResBlock(c1 * cat, c1, td), ResBlock(c1, c1, td)
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 1, 3, padding=1))

    def join(self, u, h, lvl):
        """The ablated arrow. `u` is the decoder stream, `h` the encoder feature.

        A zeroed skip is still concatenated. That keeps the decoder's first conv at
        the same input width as `full`, so the weight block reading those channels
        exists and simply receives exact-zero gradient -- which selfcheck asserts,
        and which is the cheapest possible proof the arm is really severed."""
        if self.arm == "narrow":
            return u
        if lvl not in self.live:
            h = torch.zeros_like(h)
        elif self.arm == "detach":
            h = h.detach()
        return torch.cat([u, h], 1)

    def forward(self, x, t):
        e = self.temb(t)
        h0 = self.inp(x)
        h1 = self.d1b(self.d1a(h0, e), e)
        h2 = self.d2b(self.d2a(self.down1(h1), e), e)
        h3 = self.at3(self.d3b(self.d3a(self.down2(h2), e), e))
        m = self.mid2(self.midat(self.mid1(h3, e)), e)
        u = self.u3b(self.u3a(self.join(m, h3, 3), e), e)
        u = self.u2b(self.u2a(self.join(self.up3(u), h2, 2), e), e)
        u = self.u1b(self.u1a(self.join(self.up2(u), h1, 1), e), e)
        return self.out(u)

# --------------------------------------------------------------------- samplers
# Everything below lives in the sigma parameterisation, where
#     x_tilde = x / sqrt(alpha_bar) = x0 + sigma * eps,   sigma = sqrt((1-ab)/ab)
# In these coordinates the probability-flow ODE is simply dx_tilde/dsigma = eps,
# so deterministic DDIM is *literally* Euler and Heun is its 2nd-order sibling.


# Largest t whose alpha_bar is still >= 1e-4, i.e. sigma_max ~ 100.
# We do NOT start at t=999. The cosine schedule drives alpha_bar[999] down to
# ~2.4e-9, so sigma there is ~2e4, and a 2nd-order solver simply cannot take that
# first step: Heun averages the derivative at sigma=2e4 with the one at sigma=12,
# and the average is meaningless. (1st-order Euler survives it only because the x0
# clip absorbs the damage.) Truncating the schedule where the signal is already
# 1e-4 of the variance costs nothing measurable and is standard practice.
T_START = int((AB >= 1e-4).nonzero().max())
SIG = ((1 - AB) / AB).sqrt()          # sigma(t), ascending in t


def _timesteps(nfe_steps):
    """Descending subsequence of the training timesteps, from T_START down to 0."""
    return torch.linspace(T_START, 0, nfe_steps).round().long().to(DEV)


def _sigma_schedule(steps, spacing):
    """Which noise levels to visit. Returns (sigmas[steps+1] ending at 0, timesteps).

    "t"      uniform in the training timestep index -- what every DDPM/DDIM tutorial
             does, and what the ancestral sampler needs.
    "karras" a rho=7 power law uniform in sigma^(1/rho) (Karras et al. 2022).

    Worth printing the two side by side once. Uniform-in-t on a cosine schedule is
    badly conditioned at the top: it leaps sigma 91.7 -> 15.6 in a single step and
    then spends half its remaining budget below sigma=1, where the trajectory is
    nearly straight and cheap. Karras descends 91.7 -> 76.4 -> 63.4 -> 52.3 instead.
    Removing that one enormous first step is most of why it wins, and it is the same
    root cause that broke the 2nd-order solver (see README).
    """
    if spacing == "karras":
        s_max, s_min, rho = SIG[T_START].item(), SIG[0].item(), 7.0
        i = torch.arange(steps, dtype=torch.float64)
        sig = ((s_max ** (1 / rho) + i / (steps - 1)
                * (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho).float().to(DEV)
        ts = torch.searchsorted(SIG, sig.contiguous()).clamp(max=T - 1)
    else:
        ts = _timesteps(steps)
        sig = SIG[ts]
    return torch.cat([sig, torch.zeros(1, device=DEV)]), ts


@torch.no_grad()
def sample(model, n, sampler="heun", nfe=50, seed=0, track=None, clamp=True, spacing="t"):
    """Returns (images in [-1,1], nfe_used). `track` is an optional frame list.

    clamp=False disables the x0 clip. The clip helps real samples at low NFE, but
    it is also what makes `ancestral` (written in terms of x) and DDIM at eta=1
    (written in terms of x0) stop agreeing, so selfcheck compares them without it."""
    assert sampler != "ancestral" or spacing == "t", "ancestral needs t-uniform spacing"
    g = torch.Generator(DEV).manual_seed(seed)
    steps = max(2, (nfe + 1) // 2) if sampler == "heun" else max(2, nfe)
    sig, ts = _sigma_schedule(steps, spacing)
    ab = AB[ts]
    x = torch.randn(n, 1, 32, 32, device=DEV, generator=g)
    used = 0

    if sampler == "ancestral":
        for i, t in enumerate(ts):
            ab_t = ab[i]
            ab_p = ab[i + 1] if i + 1 < steps else torch.tensor(1.0, device=DEV)
            eps = model(x, t.repeat(n)); used += 1
            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            if clamp:
                x0 = x0.clamp(-1, 1)
            a_s = ab_t / ab_p                      # alpha of this (possibly skipped) step
            b_s = 1 - a_s
            mean = (ab_p.sqrt() * b_s / (1 - ab_t)) * x0 + (a_s.sqrt() * (1 - ab_p) / (1 - ab_t)) * x
            if i + 1 < steps:
                var = b_s * (1 - ab_p) / (1 - ab_t)
                x = mean + var.sqrt() * torch.randn(x.shape, device=DEV, generator=g)
            else:
                x = mean
            if track is not None:
                track.append(x.clone())
        return x, used

    # --- deterministic ODE samplers, in sigma space
    # Scale by the *continuous* sigma, not by the discretised alpha_bar of the nearest
    # training timestep. With karras spacing those differ, and feeding the net an input
    # scaled for one noise level while telling it a timestep for another is off-manifold.
    ab_ext = 1 / (1 + sig ** 2)
    xt = x / ab_ext[0].sqrt()

    def eps_at(xt, i):
        """Model eps at sigma[i], optionally with the denoiser clipped to [-1,1].

        The clip is load-bearing here, not cosmetic. Under the cosine schedule
        alpha_bar[999] underflows to ~2.4e-9, so sigma_max is ~2e4 and the first
        Euler step has dsigma ~ -2e4. Unclipped, one imperfect eps prediction at
        t=999 scaled by that gives you noise. Clipping x0 turns the step into
        x0_hat + (sigma_next/sigma) * x_tilde, which is bounded and sane. It is the
        same x0 clip `ancestral` applies -- which is why ancestral worked without
        this and the ODE samplers did not."""
        e = model(xt * ab_ext[i].sqrt(), ts[min(i, steps - 1)].repeat(n))
        if clamp:
            x0 = (xt - sig[i] * e).clamp(-1, 1)
            e = (xt - x0) / sig[i]
        return e

    for i in range(steps):
        dsig = sig[i + 1] - sig[i]
        d1 = eps_at(xt, i); used += 1
        if sampler == "heun" and sig[i + 1] > 0:
            d2 = eps_at(xt + dsig * d1, i + 1); used += 1
            xt = xt + dsig * 0.5 * (d1 + d2)
        else:                                                    # ddim == Euler
            xt = xt + dsig * d1
        if track is not None:                      # display only, always clipped
            track.append((xt * ab_ext[i + 1].sqrt()).clamp(-1, 1))
    return (xt.clamp(-1, 1) if clamp else xt), used


@torch.no_grad()
def sample_ddim_eta(model, n, eta, nfe, seed=0, clamp=True):
    """Classic DDIM update with a tunable eta. Only used by selfcheck: at eta=1
    this must reproduce `ancestral`, which is a real test of the algebra above."""
    g = torch.Generator(DEV).manual_seed(seed)
    ts = _timesteps(nfe)
    ab = torch.cat([AB[ts], torch.ones(1, device=DEV)])
    x = torch.randn(n, 1, 32, 32, device=DEV, generator=g)
    for i, t in enumerate(ts):
        ab_t, ab_p = ab[i], ab[i + 1]
        eps = model(x, t.repeat(n))
        x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        if clamp:
            # Re-derive eps from the clipped x0 so the (x0, eps) pair stays
            # consistent with x = sqrt(ab)*x0 + sqrt(1-ab)*eps. The original DDIM
            # paper clips x0 but keeps the raw eps; that mixed form is NOT equal to
            # clipped Euler in sigma space, which is what selfcheck would catch.
            x0 = x0.clamp(-1, 1)
            eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
        s = eta * ((1 - ab_p) / (1 - ab_t) * (1 - ab_t / ab_p)).sqrt()
        x = ab_p.sqrt() * x0 + (1 - ab_p - s ** 2).clamp(min=0).sqrt() * eps
        if i + 1 < len(ts):
            x = x + s * torch.randn(x.shape, device=DEV, generator=g)
    return x

# ------------------------------------------------------------------------- data

def loader(bs, train=True, fashion=False, shuffle=True):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Pad(2),
                             transforms.Normalize((0.5,), (0.5,))])
    ds = (datasets.FashionMNIST if fashion else datasets.MNIST)(
        OUT / "data", train=train, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle,
                                       num_workers=2, drop_last=train)

# ------------------------------------------------------------------------- FMD
# Frechet distance in the feature space of a small MNIST CNN. NOT FID: these
# numbers are comparable within this repo only, never against published FIDs.
# Inception-v3 features on upscaled grayscale digits look authoritative and mean
# very little, so we do not pretend.

class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(64, 10)

    def forward(self, x, feat=False):
        f = self.body(x)
        return f if feat else self.head(f)


def train_clf(fashion=False, epochs=2):
    p = OUT / "clf.pt"
    clf = Clf().to(DEV)
    if p.exists():
        clf.load_state_dict(torch.load(p)); return clf.eval()
    opt = torch.optim.AdamW(clf.parameters(), 1e-3)
    dl = loader(256, True, fashion)
    for _ in range(epochs):
        for x, y in dl:
            loss = F.cross_entropy(clf(x.to(DEV)), y.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save(clf.state_dict(), p)
    return clf.eval()


@torch.no_grad()
def feats(clf, imgs, bs=500):
    return torch.cat([clf(imgs[i:i + bs].to(DEV), feat=True).double()
                      for i in range(0, len(imgs), bs)])


def frechet(f1, f2):
    """Frechet distance between Gaussians fitted to f1 (reference) and f2.

    Uses the symmetric form tr((S1^.5 S2 S1^.5)^.5) with eigvalsh. The tempting
    eigvals(S1 @ S2) returns complex garbage: the product of two symmetric matrices
    is not symmetric, and on noisy finite-sample covariances its eigenvalues go
    complex. Features are float64; covariance eigenvalues here span ~1e-15 to 1e2 and
    float32 loses exactly the small ones the square root is most sensitive to.

    This CNN has 14 of its 64 ReLU units permanently dead, so S1 is singular with a
    condition number ~1e32. Masking those dimensions and ridge-regularising both
    covariances was tried and changed the results in the 4th decimal place -- the
    clamp(min=0) below already handles it -- so neither is here.

    Validated by the check that matters for any FID-like metric: two disjoint samples
    of *real* data must score ~0. They do (0.72 for two 2500-image slices of the same
    population). See cmd_sweep for why the reference set is pinned.
    """
    m1, m2 = f1.mean(0), f2.mean(0)
    s1, s2 = torch.cov(f1.T), torch.cov(f2.T)
    ev, V = torch.linalg.eigh(s1)
    s1h = V @ torch.diag(ev.clamp(min=0).sqrt()) @ V.T
    inner = torch.linalg.eigvalsh(s1h @ s2 @ s1h).clamp(min=0).sqrt().sum()
    return ((m1 - m2) ** 2).sum().item() + (s1.trace() + s2.trace() - 2 * inner).item()


# ---------------------------------------------------------------------- stats
# Same hand-rolled estimators the rest of the series uses: no rliable dep.

def iqm(xs):
    v = sorted(xs)
    k = len(v) // 4
    core = v[k:len(v) - k] or v
    return sum(core) / len(core)


def bootstrap_ci(xs, reps=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the IQM. Deterministic given `seed`."""
    g = torch.Generator().manual_seed(seed)
    t = torch.tensor(xs, dtype=torch.float64)
    idx = torch.randint(len(t), (reps, len(t)), generator=g)
    draws = sorted(iqm(row.tolist()) for row in t[idx])
    lo = draws[int(alpha / 2 * reps)]
    hi = draws[int((1 - alpha / 2) * reps) - 1]
    return lo, hi

# ---------------------------------------------------------------------- commands

CKPT = OUT / "ckpt"
RES = OUT / "res"


def n_params(arm):
    return sum(p.numel() for p in UNet(arm=arm).parameters())


def _fmd_now(model, clf, fr, n=2000, nfe=50):
    """Progress metric: FMD of `n` samples from the sweep-winning sampler config."""
    was_training = model.training
    model.eval()
    imgs = []
    for i in range(0, n, 500):
        xb, _ = sample(model, min(500, n - i), "heun", nfe, seed=i, spacing="karras")
        imgs.append(xb.cpu())
    model.train(was_training)
    return frechet(fr, feats(clf, torch.cat(imgs)))


def cmd_train(a):
    """One arm, one seed. Records an FMD-vs-step curve so the grid can prove its
    budget cleared every arm's onset instead of assuming it (napkin-gamemaster's
    ablation-budget lesson: a grid that ends before the arms converge reports a
    ranking of whichever arm merely started faster)."""
    CKPT.mkdir(parents=True, exist_ok=True)
    dest = CKPT / f"{a.arm}-{a.seed}.pt"
    if dest.exists() and not a.force:
        print(f"exists, skipping {dest.name}"); return
    torch.manual_seed(a.seed)
    model = UNet(arm=a.arm).to(DEV)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    print(f"arm={a.arm} seed={a.seed} params {n_params(a.arm) / 1e6:.3f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), a.lr)
    scaler = torch.cuda.amp.GradScaler()
    dl = loader(a.bs, True, a.dataset == "fashion")
    clf = train_clf(a.dataset == "fashion")
    real = torch.cat([x for x, _ in loader(500, False, a.dataset == "fashion", shuffle=False)])
    fr = feats(clf, real)
    ema_model = UNet(arm=a.arm).to(DEV)          # scratch net for evaluating the EMA
    step, curve = 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        for x, _ in dl:
            x = x.to(DEV, non_blocking=True)
            t = torch.randint(0, T, (x.shape[0],), device=DEV)
            noise = torch.randn_like(x)
            with torch.cuda.amp.autocast():
                loss = F.mse_loss(model(q_sample(x, t, noise), t), noise)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema[k].mul_(0.999).add_(v.detach(), alpha=0.001) if v.dtype.is_floating_point else ema[k].copy_(v)
            step += 1
            if step % a.every == 0 or step == 1:
                ema_model.load_state_dict(ema)
                d = _fmd_now(ema_model, clf, fr, a.probe_n)
                curve.append({"step": step, "fmd": d, "loss": loss.item()})
                print(f"  step {step:6d} loss {loss.item():.4f} FMD {d:8.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    torch.save({"ema": ema, "arm": a.arm, "seed": a.seed, "steps": step,
                "curve": curve, "params": n_params(a.arm),
                "secs": round(time.time() - t0, 1)}, dest)
    print("saved", dest, flush=True)


def load_model(arm, seed):
    ck = torch.load(CKPT / f"{arm}-{seed}.pt", map_location=DEV)
    m = UNet(arm=ck["arm"]).to(DEV)
    m.load_state_dict(ck["ema"])
    return m.eval(), ck


def cmd_grid(a):
    """Every arm x every seed, sequentially, resumable by checkpoint existence."""
    for seed in range(a.seeds):
        for arm in a.arms:
            a.arm, a.seed = arm, seed
            cmd_train(a)


def cmd_sweep(a):
    """FMD vs NFE for one trained checkpoint. One file per (arm, seed) so the grid
    is restartable and every published number has its own artifact on disk."""
    RES.mkdir(parents=True, exist_ok=True)
    clf = train_clf(a.dataset == "fashion")
    real = torch.cat([x for x, _ in loader(500, False, a.dataset == "fashion", shuffle=False)])
    fr = feats(clf, real)
    for seed in range(a.seeds):
        for arm in a.arms:
            dest = RES / f"{arm}-{seed}.json"
            if dest.exists() and not a.force:
                print(f"exists, skipping {dest.name}"); continue
            model, ck = load_model(arm, seed)
            row = {"arm": arm, "seed": seed, "params": ck["params"],
                   "steps": ck["steps"], "curve": ck["curve"], "nfe": []}
            for nfe in a.nfe:
                imgs, used = [], 0
                for i in range(0, a.n, 500):
                    xb, used = sample(model, min(500, a.n - i), "heun", nfe,
                                      seed=i, spacing="karras")
                    imgs.append(xb.cpu())
                d = frechet(fr, feats(clf, torch.cat(imgs)))
                row["nfe"].append({"nfe": used, "fmd": d})
                print(f"{arm:8s} s{seed} nfe={used:4d} FMD={d:8.3f}", flush=True)
            dest.write_text(json.dumps(row, indent=2))


def _collect(arms):
    rows = {}
    for arm in arms:
        fs = sorted(RES.glob(f"{arm}-*.json"))
        if fs:
            rows[arm] = [json.loads(f.read_text()) for f in fs]
    return rows


def cmd_report(a):
    rows = _collect(a.arms)
    missing = [arm for arm in a.arms if arm not in rows]
    if missing:
        print(f"!! no results for {missing} -- reporting the rest", flush=True)
    nfes = [r["nfe"] for r in rows[next(iter(rows))][0]["nfe"]]
    out = {"nfe_axis": nfes, "arms": {}}
    for arm, rs in rows.items():
        finals = [r["curve"][-1]["fmd"] for r in rs]
        per_nfe = []
        for j, nfe in enumerate(nfes):
            xs = [r["nfe"][j]["fmd"] for r in rs]
            lo, hi = bootstrap_ci(xs)
            per_nfe.append({"nfe": nfe, "iqm": iqm(xs), "lo": lo, "hi": hi,
                            "seeds": len(xs), "all": xs})
        out["arms"][arm] = {"params": rs[0]["params"], "n_seeds": len(rs),
                           "final_probe_fmd": finals, "per_nfe": per_nfe}
        best = per_nfe[-1]
        print(f"{arm:8s} n={len(rs)} params={rs[0]['params']/1e6:.3f}M  "
              f"FMD@{best['nfe']}={best['iqm']:7.3f} [{best['lo']:.3f},{best['hi']:.3f}]",
              flush=True)
    (OUT / "report.json").write_text(json.dumps(out, indent=2))
    print("wrote", OUT / "report.json")
    plot(out, a.arms)


def plot(out, arms):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = _collect(arms)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    colors = dict(zip(ARMS, ["#111", "#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]))
    for arm, d in out["arms"].items():
        c = colors.get(arm, "#777")
        for r in rows[arm]:                      # every seed, faint
            axl.plot([p["step"] for p in r["curve"]], [p["fmd"] for p in r["curve"]],
                     color=c, alpha=.25, lw=.8)
        st = [p["step"] for p in rows[arm][0]["curve"]]
        med = [iqm([r["curve"][i]["fmd"] for r in rows[arm]]) for i in range(len(st))]
        axl.plot(st, med, color=c, lw=2, label=f"{arm} ({d['params']/1e6:.2f}M)")
        x = [p["nfe"] for p in d["per_nfe"]]
        axr.plot(x, [p["iqm"] for p in d["per_nfe"]], marker="o", color=c, label=arm)
        axr.fill_between(x, [p["lo"] for p in d["per_nfe"]],
                         [p["hi"] for p in d["per_nfe"]], color=c, alpha=.15)
    axl.set_xlabel("training step"); axl.set_ylabel("FMD (lower is better)")
    axl.set_yscale("log"); axl.legend(fontsize=7); axl.grid(alpha=.3)
    axl.set_title("every arm run to convergence, all seeds")
    axr.set_xscale("log"); axr.set_yscale("log"); axr.grid(alpha=.3)
    axr.set_xlabel("network evaluations (NFE)"); axr.set_ylabel("FMD, IQM of seeds")
    axr.set_title("final quality vs sampling budget"); axr.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "skips.png")
    print("wrote", OUT / "skips.png")


def cmd_gif(a):
    """Side-by-side denoising, control vs the severed arm -- the hero artifact."""
    import imageio.v2 as imageio
    from torchvision.utils import make_grid
    tracks = []
    for arm in a.arms:
        model, _ = load_model(arm, a.seed)
        fr = []
        sample(model, 36, "heun", 50, seed=1, track=fr, spacing="karras")
        tracks.append(fr)
    n = min(len(t) for t in tracks)
    frames = []
    for i in range(n):
        panes = [make_grid(t[i][:36].cpu(), nrow=6, normalize=True, value_range=(-1, 1))
                 for t in tracks]
        frames.append(torch.cat(panes, 2).mul(255).byte().permute(1, 2, 0).numpy())
    frames += [frames[-1]] * 8
    imageio.mimsave(OUT / "skips.gif", frames, duration=0.08, loop=0)
    imageio.imwrite(OUT / "skips_final.png", frames[-1])
    print("wrote", OUT / "skips.gif", "panes:", " | ".join(a.arms))


def cmd_selfcheck(a):
    OUT.mkdir(exist_ok=True)
    ab = AB.cpu()
    assert (ab.diff() < 0).all(), "alpha_bar must be strictly decreasing"
    assert ab[0] > 0.99 and ab[-1] < 0.01, f"endpoints off: {ab[0]:.4f} {ab[-1]:.5f}"

    x0 = torch.randn(256, 1, 32, 32, device=DEV)
    xT = q_sample(x0, torch.full((256,), T - 1, device=DEV), torch.randn_like(x0))
    assert abs(xT.mean()) < 0.05 and abs(xT.std() - 1) < 0.05, f"q_sample at T: {xT.mean():.3f} {xT.std():.3f}"

    torch.manual_seed(0)
    m = UNet().to(DEV).eval()                      # random init is fine: we test algebra
    a1, _ = sample(m, 4, "ancestral", 20, seed=7, clamp=False)
    a2 = sample_ddim_eta(m, 4, eta=1.0, nfe=20, seed=7, clamp=False)
    err = ((a1 - a2).abs().max() / a1.abs().max()).item()
    assert err < 1e-3, f"ancestral != ddim(eta=1), max err {err}"

    e1, _ = sample(m, 4, "ddim", 20, seed=7, clamp=False)
    e2 = sample_ddim_eta(m, 4, eta=0.0, nfe=20, seed=7, clamp=False)
    err2 = ((e1 - e2).abs().max() / e1.abs().max()).item()
    assert err2 < 1e-3, f"sigma-space Euler != x-space DDIM, max rel err {err2}"

    _, nfe_h = sample(m, 2, "heun", 20, seed=0)
    assert nfe_h == 19, f"NFE accounting off: heun {nfe_h}"

    for spacing in ("t", "karras"):
        sg, tsq = _sigma_schedule(30, spacing)
        assert (sg.diff() < 0).all(), f"{spacing} sigmas must be strictly decreasing"
        assert sg[-1] == 0 and len(sg) == 31, f"{spacing} schedule must end at sigma=0"
        assert abs(sg[0] - SIG[T_START]) < 1e-3, f"{spacing} must start at sigma_max"

    f = torch.randn(500, 64, device=DEV).double()
    assert abs(frechet(f, f.clone())) < 1e-6, "FMD of a set with itself must be ~0"
    assert frechet(f, f + 3) > 8, "FMD must grow with a mean shift"

    # ---- estimators -----------------------------------------------------
    assert iqm([1, 2, 3, 4, 100]) == 3, "IQM must drop the outer quartiles"
    lo, hi = bootstrap_ci([1., 1., 1., 1., 1.])
    assert lo == hi == 1.0, "a constant sample must give a degenerate CI"

    # ---- the arms: capacity, information, and gradient flow -------------
    # 1. Only `narrow` may differ in parameter count. Everything else is
    #    shape-matched, which is what lets a gap be attributed to information.
    base = n_params("full")
    for arm in ("zeros", "lo-only", "detach"):
        assert n_params(arm) == base, f"{arm} must be param-matched to full: {n_params(arm)} vs {base}"
    # What `narrow` drops, derived from the layer shapes rather than read off the
    # model. Halving a decoder block's input width from 2c to c removes THREE things,
    # and the third is easy to miss: ResBlock projects its residual with a 1x1 conv
    # only when cin != cout, so at cin == cout that projection becomes an Identity
    # and vanishes. `narrow` is therefore not purely "the concatenation removed" --
    # it also loses three 1x1 residual convs and some GroupNorm affines. That is
    # precisely why `zeros`, not `narrow`, is the arm the headline claim rests on.
    #   GroupNorm(8, cin) affine : 2*(2c) -> 2*c          = 2c
    #   Conv2d(cin, c, 3) weight : c*2c*9 -> c*c*9        = 9c^2
    #   residual 1x1 Conv2d(2c,c): 2c^2 + c -> Identity   = 2c^2 + c
    expect = sum(11 * c * c + 3 * c for c in (32, 64, 128))
    assert base - n_params("narrow") == expect, \
        f"narrow should drop exactly {expect} weights, dropped {base - n_params('narrow')}"

    # 2. `detach` must be the SAME FUNCTION as `full` -- identical forward, different
    #    graph. This is the assert that proves the arm isolates the gradient path and
    #    nothing else; if it ever fails, `detach` is secretly an information ablation.
    torch.manual_seed(1)
    mf = UNet(arm="full").to(DEV)
    md = UNet(arm="detach").to(DEV)
    md.load_state_dict(mf.state_dict())
    xb = torch.randn(8, 1, 32, 32, device=DEV)
    tb = torch.randint(0, T, (8,), device=DEV)
    with torch.no_grad():
        assert torch.equal(mf(xb, tb), md(xb, tb)), "detach must not change the forward pass"

    # 3. A zeroed arrow must receive EXACT-zero gradient on the weight block that
    #    reads it, and a live arrow must not. This is the severance proof, and it is
    #    exact rather than tolerance-based because the input really is 0.
    def skip_grads(arm):
        torch.manual_seed(2)
        m = UNet(arm=arm).to(DEV)
        F.mse_loss(m(xb, tb), torch.randn_like(xb)).backward()
        # decoder first-conv weights, split into [decoder-stream | skip-side] channels
        return {lvl: blk.c1.weight.grad[:, w:].abs().sum().item()
                for lvl, blk, w in ((3, m.u3a, 128), (2, m.u2a, 64), (1, m.u1a, 32))}

    gz = skip_grads("zeros")
    assert all(v == 0.0 for v in gz.values()), f"zeros: skip channels must be dead, got {gz}"
    gf = skip_grads("full")
    assert all(v > 0.0 for v in gf.values()), f"full: every skip must carry gradient, got {gf}"
    gl = skip_grads("lo-only")
    assert gl[3] > 0.0 and gl[2] == 0.0 and gl[1] == 0.0, \
        f"lo-only: only the 8px arrow may be live, got {gl}"

    # 4. `detach` cuts the encoder's gradient shortcut. The encoder still trains
    #    through the down-path, so the claim is "smaller", not "zero".
    def enc_grad(arm):
        torch.manual_seed(3)
        m = UNet(arm=arm).to(DEV)
        torch.manual_seed(4)                     # same target for both arms
        F.mse_loss(m(xb, tb), torch.randn_like(xb)).backward()
        return m.d1b.c2.weight.grad.norm().item()
    gd, gfull = enc_grad("detach"), enc_grad("full")
    assert gd != gfull, "detach must change the encoder gradient"

    # 5. Every arm must be able to learn at all: overfit one fixed batch. Catches an
    #    arm that is broken by construction rather than merely worse.
    x, _ = next(iter(loader(32, True)))
    x = x.to(DEV)
    losses = {}
    for arm in ARMS:
        torch.manual_seed(0)
        m2 = UNet(arm=arm).to(DEV)
        opt = torch.optim.AdamW(m2.parameters(), 2e-3)
        torch.manual_seed(0)
        t = torch.randint(0, T, (32,), device=DEV); noise = torch.randn_like(x)
        for _ in range(300):
            loss = F.mse_loss(m2(q_sample(x, t, noise), t), noise)
            opt.zero_grad(); loss.backward(); opt.step()
        losses[arm] = loss.item()
        assert loss.item() < 0.15, f"{arm} cannot overfit one batch, loss {loss.item():.3f}"
    print("overfit losses:", {k: round(v, 4) for k, v in losses.items()})
    print(f"params: " + "  ".join(f"{a}={n_params(a)/1e6:.3f}M" for a in ARMS))
    print(f"selfcheck OK  (eta=1 err {err:.2e}, euler err {err2:.2e})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selfcheck", "train", "grid", "sweep", "report", "gif"])
    p.add_argument("--arm", default="full", choices=ARMS)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--every", type=int, default=2000, help="steps between FMD probes")
    p.add_argument("--probe_n", type=int, default=2000, help="samples per FMD probe")
    p.add_argument("--force", action="store_true")
    p.add_argument("--n", type=int, default=10000, help="samples per sweep point")
    p.add_argument("--nfe", type=int, nargs="+", default=[10, 20, 50, 100])
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
