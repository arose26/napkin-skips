"""Run a training job with a hard cap on its share of VRAM.

Exists so a second job can use spare GPU capacity WITHOUT being able to starve a
job that is already running. The running job's allocator already holds its
high-water mark, so it does not ask the driver for more between probes -- but a
greedy second process could still take the headroom its next probe needs, and a
crash there costs whatever that job has not checkpointed. Capping is cheaper than
finding out.

    python3 capped_train.py <fraction> <args to napkin_skips train...>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

frac = float(sys.argv[1])
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    print(f"[capped_train] limited to {frac:.0%} of {total:.0f} MiB = {frac*total:.0f} MiB", flush=True)

import napkin_skips as S
import argparse
p = argparse.ArgumentParser()
p.add_argument("--arm", default="full", choices=S.ARMS)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--dataset", default="mnist")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--every", type=int, default=2000)
p.add_argument("--probe_n", type=int, default=2000)
p.add_argument("--force", action="store_true")
S.cmd_train(p.parse_args(sys.argv[2:]))
