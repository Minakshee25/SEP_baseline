"""E33 — does z_aligned catch WRONG answers the text proxy misses? Paired-bootstrap of
Δ(ensemble - q_resp) and Δ(aligned_z - q_resp) on the INCORRECT label, per target (all fresh n1000).
Reuses correctness_eval.py building blocks read-only. `amortized_stage2` env (q_resp needs the proxy).
NOTE: arm_preds can run pathologically slow on a busy box (~30 min/target); the Mistral/DeepSeek
point estimates are already in correctness_eval_<model>.json (E31, fresh n1000). Writes
correctness_ens_vs_qresp.json."""
import sys, json
sys.path.append('/vol/bitbucket/mn1025/individual_project/semantic-entropy-probes')
import numpy as np
from sklearn.metrics import roc_auc_score
from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe
from amortized_ue.correctness_eval import REF, build_aligned_z, load_accuracy, paired_bootstrap_auc, ci
from amortized_ue.procrustes_e27_rank_fusion import arm_preds, ecdf

# target: (fit_n, eval_n)  -- all now have a fresh disjoint n1000
TARGETS = {
    "Mistral-7B-Instruct-v0.2":   (2000, 1000),
    "deepseek-llm-7b-chat":       (2000, 1000),
    "Meta-Llama-3-8B-Instruct":   (2000, 1000),   # NEW: fresh n1000 just built
}
master = {}
for target, (fit_n, eval_n) in TARGETS.items():
    print("\n" + "#"*80 + f"\n# {target}  fit n{fit_n} -> eval fresh n{eval_n}\n" + "#"*80)
    sh, s_y, s_ids = load_matrix(Stage1Config(model_name=target, dataset="trivia_qa", num_samples=fit_n), ["TBG","SLT"])
    rh, r_y, r_ids = load_matrix(Stage1Config(model_name=REF, dataset="trivia_qa", num_samples=fit_n), ["TBG","SLT"])
    assert s_ids == r_ids
    tr, va, te = splits(len(s_ids))
    esh, ey, eval_ids = load_matrix(Stage1Config(model_name=target, dataset="trivia_qa", num_samples=eval_n), ["TBG","SLT"])
    eval_rows = np.arange(len(eval_ids))

    acc = np.array([load_accuracy(Stage1Config(model_name=target, dataset="trivia_qa", num_samples=eval_n))[i] for i in eval_ids], float)
    incorrect = (1 - (acc >= 0.5).astype(int))

    z_tr, z_eval = build_aligned_z(sh, rh, r_y, tr, va, esh, eval_rows)
    qr_fit = arm_preds("q_resp_only", target, "trivia_qa", fit_n)
    qr_tr  = np.array([qr_fit[i] for i in s_ids])[tr]
    qr_ev_map = arm_preds("q_resp_only", target, "trivia_qa", eval_n)   # ONE call, then index (was: recomputed per-id -> ~1000x slower)
    qr_eval = np.array([qr_ev_map[i] for i in eval_ids])
    cz, cr = ecdf(z_tr), ecdf(qr_tr)
    preds = {
        "q_resp_only":    qr_eval,
        "aligned_z":      z_eval,
        "rank_fusion_ens": 0.5*(cz(z_eval) + cr(qr_eval)),
    }
    au = {k: float(roc_auc_score(incorrect, v)) for k,v in preds.items()}
    print(f"  incorrect_rate={incorrect.mean():.3f}  N={len(incorrect)}")
    for k in preds: print(f"  {k:16s} AUROC_incorrect {au[k]:.3f}")
    boot = paired_bootstrap_auc(preds, incorrect, B=10000)
    d_ens = ci(boot["rank_fusion_ens"] - boot["q_resp_only"])
    d_z   = ci(boot["aligned_z"]      - boot["q_resp_only"])
    for nm,d in [("ensemble - q_resp", d_ens), ("aligned_z - q_resp", d_z)]:
        excl = "excludes 0" if (d["lo95"]>0 or d["hi95"]<0) else "includes 0"
        print(f"  Δ({nm:20s}) {d['mean']:+.3f} [{d['lo95']:+.3f},{d['hi95']:+.3f}] ({excl})")
    master[target] = {"N": int(len(incorrect)), "incorrect_rate": float(incorrect.mean()),
                      "auroc_incorrect": au,
                      "delta_ensemble_minus_qresp": d_ens, "delta_alignedz_minus_qresp": d_z}
json.dump(master, open("/vol/bitbucket/mn1025/individual_project/semantic-entropy-probes/amortized_ue/correctness_ens_vs_qresp.json","w"), indent=2)
print("\nwrote amortized_ue/correctness_ens_vs_qresp.json")
