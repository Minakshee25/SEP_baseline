"""Reserve (fence) GPU memory on a shared card so co-tenants cannot grab the slack while our job
runs. Allocates `--hold_mib` MiB as a uint8 tensor on the CUDA_VISIBLE_DEVICES-relative `--device`
and holds it until: a SIGTERM/SIGINT arrives, the `--parent_pid` process dies, or the `--sentinel`
file disappears. Backs off the request if the card can't give the full amount.

The point: our build reserves its own working set (~model + activations); this holder grabs the
*remaining* free memory so another user's job OOMs on THIS card instead of landing next to us and
starving/OOM-ing our run mid-flight (which is exactly what killed the Llama-3 n2000 build once).

Standalone helper — imported by nothing in the pipeline; safe/additive. The waiter sizes the hold as
(free_now - job_budget - safety) and kills this process when the build attempt exits, and this process
also self-exits if its parent dies, so an orphaned holder can never block the GPU indefinitely.

    CUDA_VISIBLE_DEVICES=$PICK python -m amortized_ue.gpu_reserve --device 0 --hold_mib 7000 --parent_pid $$
"""
from __future__ import annotations

import os
import time
import signal
import argparse


def main():
    ap = argparse.ArgumentParser(description="Fence spare GPU memory for the lifetime of a job.")
    ap.add_argument("--device", type=int, default=0, help="CUDA_VISIBLE_DEVICES-relative index")
    ap.add_argument("--hold_mib", type=int, required=True, help="MiB to reserve")
    ap.add_argument("--parent_pid", type=int, default=None, help="exit when this pid dies")
    ap.add_argument("--sentinel", default=None, help="exit when this file is removed")
    ap.add_argument("--poll", type=float, default=5.0)
    a = ap.parse_args()

    import torch
    torch.cuda.set_device(a.device)
    hold, x = a.hold_mib, None
    while hold > 0:                                            # back off if the card is tighter than asked
        try:
            x = torch.empty(hold * 1024 * 1024, dtype=torch.uint8, device="cuda")
            break
        except RuntimeError:
            hold -= 512
    if x is None:
        print("gpu_reserve: nothing to hold (no free memory)", flush=True)
        return
    print(f"gpu_reserve: holding {hold} MiB on cuda:{a.device} (pid {os.getpid()}, parent {a.parent_pid})", flush=True)

    running = {"v": True}
    def _stop(*_):
        running["v"] = False
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while running["v"]:
        if a.parent_pid is not None:
            try:
                os.kill(a.parent_pid, 0)                       # parent gone -> release, never orphan-block
            except OSError:
                break
        if a.sentinel is not None and not os.path.exists(a.sentinel):
            break
        time.sleep(a.poll)
    print("gpu_reserve: releasing", flush=True)


if __name__ == "__main__":
    main()
