"""E62 — clean, NON-ENSEMBLED head-to-head: the REFERENCE proxy's `q_resp_only` arm ALONE
(question + canonical answer text, no hidden states, no fusion, no aligned-z) vs each target's
OWN supervised SEP, for all 4 alignment targets (Llama-2 / Mistral / Llama-3 / DeepSeek).

Motivation. E27/E29/E30 established this comparison for MISTRAL (Llama-2-trained REFERENCE proxy's
q_resp_only vs Mistral's matched SEP), but the analogous stand-alone number was never cleanly
pinned for the other targets:
  * Llama-3 / DeepSeek: a `q_resp_only_alone` point estimate exists inside
    procrustes_e30_ens_vs_qresp_<slug>.json, but its paired-bootstrap CI there is vs the ENSEMBLE,
    never vs SEP.
  * Llama-2: never computed at all as a stand-alone reference-proxy-vs-own-SEP number (Llama-2 is
    the reference model, so this is the in-distribution baseline of the whole cross-model line).
  * No target has a paired-bootstrap (q_resp_only - SEP) delta on SE-fidelity.

Method (E27/E29 methodology minus the ensemble parts):
  * proxy  = REFERENCE_multipos_p1024_5arm_ckpt `q_resp_only` seeds, seed-averaged, ONE forward
             pass per record, NO target hidden states / labels / sampling. (arm_preds_per_seed with
             ckpt_dir=REF -> reuses se_fidelity_proxy_vs_sep.score_block for the paired bootstrap.)
  * SEP    = per-(position,layer) LogisticRegression on best_split-binarised SE, fit on the
             target's OWN trivia_qa n2000 TRAIN split, evaluated on the SAME fresh disjoint n1000.
             Layer = E41 fixed TBG layer (exp2_run.BEST_TBG); Llama-2 = TBG:30. Matches
             results/sep_reference_values.json exactly.
  * eval   = fresh trivia_qa n1000, 0 id-overlap with the n2000 training set (asserted per target).
  * metric = Spearman(pred, continuous CAE) and AUROC(pred, SE>thr) on the same rows; paired
             bootstrap (shared resample indices) for (proxy - SEP), 10000 resamples.

Env: `amortized_stage2` + a free GPU (reference proxy forward pass).
    python -m amortized_ue.e62_qresp_alone_vs_sep --data_dir /data2/mn1025/stage1
Writes amortized_ue/results/e62_qresp_alone_vs_sep.json
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np

from amortized_ue.config import Stage1Config
from amortized_ue import exp2_run as E2
from amortized_ue.se_fidelity_proxy_vs_sep import (
    compute_sep, arm_preds_per_seed, score_block)

REF_CKPT = "amortized_ue/stage2/runs/REFERENCE_multipos_p1024_5arm_ckpt/checkpoints"
OUT = "amortized_ue/results/e62_qresp_alone_vs_sep.json"


def fresh_ok(target, data_dir):
    """assert the n1000 eval set shares 0 ids with the n2000 fit set for this target."""
    fit = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=2000,
                       **({"output_dir": data_dir} if data_dir else {}))
    ev = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=1000,
                      **({"output_dir": data_dir} if data_dir else {}))
    with open(fit.manifest_path()) as f:
        fit_ids = set(json.load(f)["records"].keys())
    with open(ev.manifest_path()) as f:
        ev_ids = set(json.load(f)["records"].keys())
    ov = fit_ids & ev_ids
    assert not ov, f"{target}: n1000 overlaps n2000 by {len(ov)} ids -- not fresh"
    return len(fit_ids), len(ev_ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/data2/mn1025/stage1")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default=OUT)
    p.add_argument("--dry_run", action="store_true",
                   help="everything except the proxy GPU forward pass: SEP (CPU), fresh-overlap "
                        "check, checkpoint discovery, Stage2Data build. Verifies setup, writes nothing.")
    args = p.parse_args()

    if args.dry_run:
        import glob as _glob
        import dataclasses as _dc
        from amortized_ue.stage2.data import Stage2Data
        from amortized_ue.stage2.checkpoint import read_meta, _cfg_from_meta
        ok = True
        for target in E2.MODELS:
            short = E2.SHORT[target]
            layer = E2.BEST_TBG[target]
            try:
                nfit, nev = fresh_ok(target, args.data_dir)
                sep = compute_sep(target, eval_dataset="trivia_qa", eval_num_samples=1000,
                                  data_dir=args.data_dir, fit_num_samples=2000,
                                  use_test_split_as_eval=False, layer=layer)
                paths = sorted(_glob.glob(os.path.join(REF_CKPT, "*q_resp_only_seed*.pt")))
                assert paths, f"no q_resp_only checkpoints under {REF_CKPT}"
                meta = read_meta(paths[0])
                cfg = _dc.replace(_cfg_from_meta(meta), stage1_model_name=target,
                                  stage1_dataset="trivia_qa", stage1_num_samples=1000,
                                  ood_dataset=None, smoke=False, stage1_output_dir=args.data_dir)
                data = Stage2Data(cfg)
                rows = data.split_indices("all")
                proxy_ids = set(data.ids[r] for r in rows)
                missing = [i for i in sep["ids"] if i not in proxy_ids]
                assert not missing, f"{len(missing)} SEP ids missing from proxy records"
                from amortized_ue.se_fidelity_proxy_vs_sep import spearman as _sp
                sep_rho = _sp(sep["pred"], sep["y"])
                print(f"  [OK] {short:8s} n2000={nfit} freshN1000={nev}  SEP TBG:{layer} "
                      f"rho={sep_rho:+.4f} auroc_se={sep['auroc_se']:.4f}  "
                      f"proxy seeds={len(paths)} proxy records={len(rows)}  ids aligned OK "
                      f"(SEP eval n={len(sep['ids'])})")
            except Exception as e:
                ok = False
                print(f"  [FAIL] {short}: {type(e).__name__}: {e}")
        print(f"\nDRY RUN {'PASSED — ready for GPU' if ok else 'FAILED'}. "
              f"proxy meta: position={meta['position']} layer={meta['layer']}")
        return

    out = {"_meta": {
        "experiment": "E62",
        "what": "REFERENCE proxy q_resp_only arm ALONE (no fusion) vs own matched SEP, 4 targets",
        "proxy_ckpt": REF_CKPT,
        "proxy_provenance": "Llama-2-trained REFERENCE_multipos_p1024_5arm_ckpt, q_resp_only seeds, seed-averaged",
        "sep": "E41 fixed TBG layer (exp2_run.BEST_TBG); LogisticRegression on best_split-binarised SE; fit target OWN n2000 train",
        "eval": "fresh trivia_qa n1000, 0 id-overlap with n2000 (asserted)",
        "bootstrap": args.bootstrap,
        "note": "score_block's 'proxy_ensemble' == mean over reference q_resp_only seeds == q_resp_only ALONE (no z, no rank-fusion).",
    }, "targets": {}}

    table = []
    for target in E2.MODELS:
        short = E2.SHORT[target]
        layer = E2.BEST_TBG[target]
        nfit, nev = fresh_ok(target, args.data_dir)
        print(f"\n{'='*78}\n{short}  ({target})   SEP TBG:{layer}   n2000={nfit} n1000={nev} (fresh)\n{'='*78}")

        sep = compute_sep(target, eval_dataset="trivia_qa", eval_num_samples=1000,
                          data_dir=args.data_dir, fit_num_samples=2000,
                          use_test_split_as_eval=False, layer=layer)
        ids, P = arm_preds_per_seed("q_resp_only", target, "trivia_qa", 1000,
                                    ckpt_dir=REF_CKPT, data_dir=args.data_dir)
        print(f"  reference q_resp_only: {P.shape[0]} seeds, {P.shape[1]} records")

        block = score_block(sep, ids, P, bootstrap=args.bootstrap, tag=f"e62/{short}")
        block["target"] = short
        block["target_full"] = target
        block["proxy_provenance"] = ("Llama-2-trained REFERENCE proxy q_resp_only arm, seed-averaged; "
                                     "NO target hidden states / labels / sampling / fusion")
        out["targets"][short] = block

        m = block["metrics"]
        d = block["bootstrap_vs_sep"]["proxy_ensemble"]
        sp, au = d["spearman_delta"], d.get("auroc_se_delta")
        row = {
            "target": short,
            "qresp_spearman": m["proxy_ensemble"]["spearman"],
            "qresp_auroc_se": m["proxy_ensemble"]["auroc_se"],
            "sep_spearman": m["sep"]["spearman"],
            "sep_auroc_se": m["sep"]["auroc_se"],
            "sep_layer": f"TBG:{layer}",
            "delta_spearman": sp["mean"],
            "delta_spearman_ci": [sp["lo95"], sp["hi95"]],
            "delta_spearman_excl0": sp["ci_excludes_zero"],
            "delta_auroc_se": au["mean"] if au else None,
            "delta_auroc_se_ci": [au["lo95"], au["hi95"]] if au else None,
            "delta_auroc_se_excl0": au["ci_excludes_zero"] if au else None,
        }
        table.append(row)
        print(f"  q_resp_only : rho={row['qresp_spearman']:+.3f}  auroc_se={row['qresp_auroc_se']:.3f}")
        print(f"  own SEP     : rho={row['sep_spearman']:+.3f}  auroc_se={row['sep_auroc_se']:.3f}")
        print(f"  Δ(qresp-SEP): rho {sp['mean']:+.3f} [{sp['lo95']:+.3f},{sp['hi95']:+.3f}] "
              f"({'excl0' if sp['ci_excludes_zero'] else 'incl0'})   "
              f"auroc {au['mean']:+.3f} [{au['lo95']:+.3f},{au['hi95']:+.3f}] "
              f"({'excl0' if au['ci_excludes_zero'] else 'incl0'})")

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out["_table"] = table
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)

    print("\n\n" + "#" * 96)
    print("E62 CONSOLIDATED TABLE  --  q_resp_only ALONE (reference proxy) vs own SEP, fresh trivia n1000")
    print("#" * 96)
    hdr = (f"{'target':10s}| {'q_resp_only rho/auc':>21s} | {'own SEP rho/auc':>21s} | "
           f"{'Δrho [95% CI]':>26s} | {'Δauc [95% CI]':>26s}")
    print(hdr)
    print("-" * len(hdr))
    for r in table:
        drho = f"{r['delta_spearman']:+.3f} [{r['delta_spearman_ci'][0]:+.3f},{r['delta_spearman_ci'][1]:+.3f}]"
        dauc = f"{r['delta_auroc_se']:+.3f} [{r['delta_auroc_se_ci'][0]:+.3f},{r['delta_auroc_se_ci'][1]:+.3f}]"
        star_r = "*" if r["delta_spearman_excl0"] else " "
        star_a = "*" if r["delta_auroc_se_excl0"] else " "
        print(f"{r['target']:10s}| {r['qresp_spearman']:+.3f} / {r['qresp_auroc_se']:.3f}     | "
              f"{r['sep_spearman']:+.3f} / {r['sep_auroc_se']:.3f} ({r['sep_layer']}) | "
              f"{drho:>25s}{star_r} | {dauc:>25s}{star_a}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
