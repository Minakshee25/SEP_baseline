"""E53b addendum: the ORIGINAL per-target ridge (plain Ridge on the target's OWN hidden states,
trivia n2000 fit -> squad n1000 eval) -- the project's linear ceiling diagnostic, same recipe as
e53_full_comparison.py's ridge_context, only eval set swapped to squad. CPU, no GPU, no retraining
of anything else.
"""
import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, rho

DATA = "/data2/mn1025/stage1"
TARGETS = {"Llama-2-7b-chat": {"sep_tbg": 30, "ref_tbg": 22, "ref_slt": 15},
           "Mistral-7B-Instruct-v0.2": {"sep_tbg": 31, "ref_tbg": 31, "ref_slt": 20}}
POS = sorted({"TBG", "SLT"})


def ridge_at(hf, yf, tr, va, he, ye, pls):
    Xtr = np.concatenate([hf[p][l] for p, l in pls], axis=1)
    Xev = np.concatenate([he[p][l] for p, l in pls], axis=1)
    model, scaler, alpha, val_rho = fit_probe(Xtr, yf, tr, va)
    return model.predict(scaler.transform(Xev)), alpha


out = {}
for t, L in TARGETS.items():
    fit_cfg = Stage1Config(model_name=t, dataset="trivia_qa", num_samples=2000, output_dir=DATA)
    ev_cfg = Stage1Config(model_name=t, dataset="squad", num_samples=1000, output_dir=DATA)
    hf, yf, idf = load_matrix(fit_cfg, POS)
    he, ye, ide = load_matrix(ev_cfg, POS)
    tr, va, te = splits(len(idf))

    from amortized_ue.loaders import load_records
    recs = load_records(ev_cfg)
    acc = np.array([recs[i]["canonical"]["accuracy"] for i in ide], dtype=float)
    incorrect = (acc < 0.5).astype(int)

    for tag, pls in {"same_as_sep_TBG": [("TBG", L["sep_tbg"])],
                     "reference_ceiling_TBG+SLT": [("TBG", L["ref_tbg"]), ("SLT", L["ref_slt"])]}.items():
        pred, alpha = ridge_at(hf, yf, tr, va, he, ye, pls)
        sp = float(spearmanr(pred, ye).correlation)
        au = float(roc_auc_score(incorrect, pred))
        out.setdefault(t, {})[tag] = {"spearman_vs_se": sp, "auroc_incorrect": au,
                                      "layers": pls, "alpha": alpha,
                                      "n_fit": len(idf), "n_eval": len(ide),
                                      "id_overlap_fit_eval": len(set(idf) & set(ide))}
        print(f"{t:26s} {tag:26s} spearman={sp:+.4f}  auroc_inc={au:.4f}  alpha={alpha}  "
              f"overlap={len(set(idf) & set(ide))}")

json.dump(out, open("/vol/bitbucket/mn1025/individual_project/semantic-entropy-probes/"
                    "amortized_ue/results/e53b_original_ridge_squad.json", "w"), indent=2)
print("wrote results/e53b_original_ridge_squad.json")
