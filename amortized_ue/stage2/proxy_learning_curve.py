"""Does the 3B PROXY (not ridge) benefit from more data?

Every learning curve so far measured ridge/MLP, which are linear and plateau early by
nature. The proxy has far more capacity and might behave differently. This drives the
existing `Trainer` READ-ONLY (no change to train.py / model.py): for each training-set
size it fits a fresh proxy on a fixed subsample of the TRAIN split and scores the SAME
held-out val/test splits.

  curve still climbing at 1440  -> the proxy IS data-hungry; building 10-20k is justified.
  curve flat by ~1000           -> the proxy plateaus at the same scale ridge does; do NOT
                                   build more data (SEP itself used only 2000 across tasks).

Subsamples are nested (smaller ⊂ larger) and drawn from the train split only, so val/test
are untouched. Run in the amortized_stage2 env, pin a free GPU:
    CUDA_VISIBLE_DEVICES=0 python -m amortized_ue.stage2.proxy_learning_curve \
        --z_inputs TBG:22,SLT:15 --projector_hidden_dim 1024 --sizes 500,1000,1440 --seeds 3
"""
from __future__ import annotations

import json
import argparse
import logging

import numpy as np

from amortized_ue.stage2.config import Stage2Config
from amortized_ue.stage2.data import Stage2Data
from amortized_ue.stage2.train import Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--z_inputs", default="TBG:22,SLT:15")
    p.add_argument("--projector_hidden_dim", type=int, default=1024)
    p.add_argument("--selected_k", type=int, default=4)
    p.add_argument("--sizes", default="500,1000,1440", help="comma-separated train sizes")
    p.add_argument("--seeds", type=int, default=3, help="trial seeds per size")
    p.add_argument("--arm", default="z")
    p.add_argument("--out", default="amortized_ue/stage2/runs/proxy_learning_curve.json")
    a = p.parse_args()

    z_inputs = tuple(s.strip() for s in a.z_inputs.split(","))
    sizes = [int(s) for s in a.sizes.split(",")]
    pos, layer = Stage2Data.parse_z_inputs(z_inputs)[0]   # nominal label for logging

    cfg = Stage2Config(
        z_inputs=z_inputs, projector_hidden_dim=a.projector_hidden_dim,
        k_soft_tokens=a.selected_k, arms=(a.arm,),
    )
    data = Stage2Data(cfg)
    trainer = Trainer(cfg, data)
    train_pool = data.split_indices("train")
    full_n = len(train_pool)
    logging.info("train pool = %d rows; sizes = %s; %d seeds each", full_n, sizes, a.seeds)

    results = {}
    for n in sizes:
        n = min(n, full_n)
        te_sp, tr_sp = [], []
        for seed in range(a.seeds):
            # nested subsample: same RNG per seed, take the first n of a fixed permutation,
            # so size=500 ⊂ size=1000 ⊂ ... within a seed (isolates data quantity, not identity)
            rng = np.random.default_rng(1000 + seed)
            rows = np.sort(rng.permutation(train_pool)[:n])
            # fresh proxy init per (size, seed), decoupled like train_arms_trial
            trainer.reseed(trainer._derive_seed(cfg.seed, seed, "init"))
            trainer.model.reinit_trainable()
            trainer._fresh_state = trainer._snapshot_trainable()
            trainer.reset_trainable()
            trainer.reseed(trainer._derive_seed(cfg.seed, seed, a.arm))
            trainer.train_arm(pos, layer, arm=a.arm, train_rows=rows, epochs=cfg.epochs)
            te = trainer.evaluate(pos, layer, a.arm, "test")["spearman"]
            tr = trainer.evaluate(pos, layer, a.arm, "train")["spearman"]
            te_sp.append(te); tr_sp.append(tr)
            logging.info("n=%4d seed=%d  train_sp=%.4f  test_sp=%.4f", n, seed, tr, te)
        results[n] = {
            "test_mean": float(np.mean(te_sp)), "test_std": float(np.std(te_sp, ddof=1) if len(te_sp) > 1 else 0),
            "train_mean": float(np.mean(tr_sp)), "test_values": te_sp,
        }
        logging.info(">>> n=%4d  TEST %.4f ± %.4f  (train %.4f)",
                     n, results[n]["test_mean"], results[n]["test_std"], results[n]["train_mean"])

    with open(a.out, "w") as f:
        json.dump({"z_inputs": list(z_inputs), "sizes": sizes, "seeds": a.seeds, "results": results}, f, indent=2)
    print("\n=== PROXY LEARNING CURVE (test Spearman) ===")
    print(f"{'train rows':>11}{'TEST':>10}{'std':>8}{'TRAIN':>9}")
    for n in sizes:
        r = results[min(n, full_n)]
        print(f"{min(n,full_n):>11}{r['test_mean']:>10.3f}{r['test_std']:>8.3f}{r['train_mean']:>9.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
