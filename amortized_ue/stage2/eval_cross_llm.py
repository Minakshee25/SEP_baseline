"""Cross-LLM transfer eval (E20) -- the thesis experiment.

Score the FROZEN Llama-2-trained proxy on a DIFFERENT target LLM's Stage-1 records, with
NO retraining. For each saved checkpoint (arm, seed) we load only its trainable params,
point a Stage2Data at the OTHER LLM's records, and evaluate on ALL rows (split='all',
OOD-style rebinarisation over the eval labels). Predictions are decoded with the
checkpoint's OWN training transform (exactly as at train time). Metrics are aggregated per
arm across seeds (mean +/- std), matching the in-distribution reporting so the transfer
numbers sit directly beside the ID numbers.

Which arms transfer:
  * text-only arms (q_only, q_resp_only) transfer to ANY target -- no hidden states.
  * z-arms (z, z_q, z_q_resp) transfer only if the target's hidden size == the projector's
    trained h_in. Llama-3-8B is 4096-dim (= Llama-2-7b), so all 5 arms transfer here.
    The z-arm transfer number is the Platonic-Representation-Hypothesis test: does a
    projector fit on Llama-2's hidden geometry predict Llama-3's semantic entropy?

Reuses Stage2Data + Trainer.evaluate_on READ-ONLY; trains nothing, edits no training code.
Run in the `amortized_stage2` env (the proxy backbone Llama-3.2-3B needs transformers 4.52.4).
The target-LLM records themselves are plain tensors -- loading them is env-agnostic.
"""
from __future__ import annotations

import os
import glob
import json
import logging
import argparse
import dataclasses
from collections import defaultdict

from amortized_ue.stage2.data import Stage2Data
from amortized_ue.stage2.train import Trainer
from amortized_ue.stage2.checkpoint import load_checkpoint, read_meta, _cfg_from_meta
from amortized_ue.stage2.run import _summarize_by_arm, _paired_by_arm


def run(ckpt_dir: str, target_model: str, dataset: str, num_samples: int,
        out_path: str | None = None) -> dict:
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    assert paths, f"no *.pt checkpoints in {ckpt_dir}"
    logging.info("Cross-LLM eval: %d checkpoints in %s", len(paths), ckpt_dir)

    # The proxy architecture (z_inputs, projector width, k, max_seq_len, batch...) comes from
    # the checkpoint's own stored config; we only repoint the Stage-1 source at the new target.
    base_cfg = _cfg_from_meta(read_meta(paths[0]))
    src_model = base_cfg.stage1_model_name
    eval_cfg = dataclasses.replace(
        base_cfg, stage1_model_name=target_model, stage1_dataset=dataset,
        stage1_num_samples=num_samples, ood_dataset=None, smoke=False)
    logging.info("Transfer: proxy trained on %s -> evaluated on %s (%s, N=%d, split=all)",
                 src_model, target_model, dataset, num_samples)

    eval_data = Stage2Data(eval_cfg)
    logging.info("Loaded %d target records | hidden_size=%d n_layers=%d",
                 len(eval_data.ids), eval_data.hidden_size, eval_data.n_layers)

    model, trainer = None, None
    by_arm = defaultdict(list)
    for p in paths:
        model, meta, transform = load_checkpoint(p, model=model)   # reuse backbone across ckpts
        if trainer is None:
            trainer = Trainer(eval_cfg, eval_data, model=model)
        trainer.data = eval_data
        # z-arms need matching hidden size; skip (don't crash) if the target differs.
        if meta["h_in"] % eval_data.hidden_size != 0 and _arm_needs_z(meta["arm"]):
            logging.warning("skip %s seed=%s: h_in=%d not compatible with target H=%d",
                            meta["arm"], meta["seed"], meta["h_in"], eval_data.hidden_size)
            continue
        m = trainer.evaluate_on(meta["position"], meta["layer"], meta["arm"],
                                eval_data, transform, split="all")
        by_arm[meta["arm"]].append({"seed": meta["seed"], **m})
        logging.info("  %-11s seed=%s  spearman=%.4f auroc=%.4f rmse=%.4f",
                     meta["arm"], meta["seed"], m["spearman"], m["auroc"], m["rmse"])

    summary = _summarize_by_arm(by_arm)
    out = {
        "ckpt_dir": ckpt_dir, "source_model": src_model, "target_model": target_model,
        "dataset": dataset, "num_samples": num_samples, "n_eval_records": len(eval_data.ids),
        "split": "all", "summary": summary, "paired": _paired_by_arm(by_arm, ref="z"),
        "per_seed": {a: rows for a, rows in by_arm.items()},
    }

    print("\n" + "=" * 78)
    print(f"CROSS-LLM TRANSFER: {src_model} proxy -> {target_model} ({dataset}, N={len(eval_data.ids)})")
    print("=" * 78)
    print(f"{'arm':12s} {'transfer Spearman':>20s} {'transfer AUROC':>18s}")
    for arm in ("z", "z_q", "z_q_resp", "q_only", "q_resp_only"):
        if arm not in summary:
            continue
        sp, au = summary[arm]["spearman"], summary[arm]["auroc"]
        print(f"{arm:12s} {sp['mean']:8.4f} ± {sp['std']:.4f}   {au['mean']:8.4f} ± {au['std']:.4f}")
    print("=" * 78 + "\n")

    out_path = out_path or os.path.join(ckpt_dir, f"cross_llm_{target_model}_{dataset}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logging.info("Wrote cross-LLM eval -> %s", out_path)
    return out


def _arm_needs_z(arm: str) -> bool:
    return arm in ("z", "z_q", "z_q_resp")


def _parse():
    ap = argparse.ArgumentParser(description="Cross-LLM transfer eval (E20).")
    ap.add_argument("--ckpt_dir", required=True,
                    help="dir of trained *.pt checkpoints (the frozen source-LLM proxy)")
    ap.add_argument("--target_model", required=True,
                    help="the DIFFERENT target LLM whose Stage-1 records to score, e.g. "
                         "Meta-Llama-3-8B-Instruct")
    ap.add_argument("--dataset", default="trivia_qa")
    ap.add_argument("--num_samples", type=int, required=True,
                    help="drives the target Stage-1 run-dir name (model_dataset_nN_full)")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = _parse()
    run(a.ckpt_dir, a.target_model, a.dataset, a.num_samples, a.out)
