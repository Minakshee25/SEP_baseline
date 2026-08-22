"""E49 -- does excluding the 58 degenerate "<think>" leak records change Qwen3.5-9B's headline
numbers from E45 (correctness AUROC), E47 (SE-fidelity), and E48 (frozen-backbone ablation)?

These 58/1000 rows have canonical_response literally "<think>" or "<think>\n\n</think>" -- the
model exhausted its 250-token thinking budget without producing a real answer. accuracy=0 for all
by construction (not a valid answer), and because the model tends to repeat the same broken string
across samples, these often get LOW true SE too (a degenerate-but-consistent failure, the same
qualitative pattern as the "confidently wrong" Last Tango in Paris example from E47, just more
extreme). Re-scores q_only/q_resp_only (deploy proxy, E45/E47) and the frozen-backbone baseline
(E48) BOTH on the full 1000 and on the clean 942, side by side.

Env: `amortized_stage2_v5` + a free GPU. Run from the repo root:
    /data2/mn1025/conda_envs/amortized_stage2_v5/bin/python -m amortized_ue.e49_qwen35_9b_think_leak_check
"""
from __future__ import annotations

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.correctness_eval import load_accuracy
from amortized_ue.procrustes_e27_rank_fusion import arm_preds
from amortized_ue.e48_frozen_backbone_baseline import PROMPT_HEADER, BACKBONE

TARGET = "Qwen3.5-9B"
DATA_DIR = "/data2/mn1025/stage1"
DEPLOY_CKPT = "/data2/mn1025/stage2_checkpoints/deploy_checkpoints"
NUM_SAMPLES = 1000


def rho(a, b):
    r = spearmanr(a, b).correlation
    return 0.0 if (r is None or np.isnan(r)) else float(r)


def score(name, s, incorrect, y_se, mask):
    au = float(roc_auc_score(incorrect[mask], s[mask]))
    r = rho(s[mask], y_se[mask])
    return au, r


def main():
    cfg = Stage1Config(model_name=TARGET, dataset="trivia_qa", num_samples=NUM_SAMPLES, output_dir=DATA_DIR)
    recs = load_records(cfg)
    ids = sorted(recs.keys())
    acc = load_accuracy(cfg)
    y_se = np.array([recs[i]["labels"]["cluster_assignment_entropy"] for i in ids])
    incorrect = np.array([1.0 - acc[i] for i in ids])

    is_bad = np.array([
        (not recs[i]["canonical"]["response"].strip())
        or ("<think>" in recs[i]["canonical"]["response"].lower())
        for i in ids
    ])
    print(f"N={len(ids)}, bad (<think> leak) = {is_bad.sum()} ({100*is_bad.mean():.1f}%)")
    clean_mask = ~is_bad
    all_mask = np.ones(len(ids), dtype=bool)

    # ---- deploy proxy: q_only, q_resp_only (E45/E47) --------------------------------------------
    print("scoring deploy proxy (q_only, q_resp_only) ...")
    q_only = arm_preds("q_only", TARGET, "trivia_qa", NUM_SAMPLES, ckpt_dir=DEPLOY_CKPT, data_dir=DATA_DIR)
    q_resp = arm_preds("q_resp_only", TARGET, "trivia_qa", NUM_SAMPLES, ckpt_dir=DEPLOY_CKPT, data_dir=DATA_DIR)
    p_only = np.array([q_only[i] for i in ids])
    p_resp = np.array([q_resp[i] for i in ids])

    # ---- true SE (as its own "predictor" baseline) -----------------------------------------------
    p_truese = y_se

    # ---- frozen backbone baseline (E48) ------------------------------------------------------------
    print(f"loading frozen {BACKBONE} (no LoRA) ...")
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BACKBONE, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    id_a = tok.encode(" A", add_special_tokens=False)[-1]
    id_b = tok.encode(" B", add_special_tokens=False)[-1]

    prompts = [
        PROMPT_HEADER
        + f"Question: {recs[i]['question']}\nProposed answer: {recs[i]['canonical']['response']}\n"
          f"Is the proposed answer correct?\nA) True\nB) False\nAnswer:"
        for i in ids
    ]
    p_false = []
    with torch.no_grad():
        for i in range(0, len(prompts), 16):
            batch = prompts[i:i + 16]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
            out = model(**enc)
            lengths = enc["attention_mask"].sum(dim=1) - 1
            last_logits = out.logits[torch.arange(len(batch)), lengths]
            probs = torch.softmax(last_logits[:, [id_a, id_b]].float(), dim=-1)
            p_false.extend(probs[:, 1].cpu().tolist())
    p_false = np.array(p_false)

    # ---- report: full 1000 vs clean 942, all 4 predictors -----------------------------------------
    predictors = {"true_SE": p_truese, "q_only": p_only, "q_resp_only": p_resp,
                  "frozen_backbone_p_false": p_false}
    print(f"\n{'predictor':26s}{'AUROC(all)':>12s}{'AUROC(clean)':>14s}{'d':>7s}"
          f"{'rho(all)':>10s}{'rho(clean)':>12s}{'d':>7s}")
    out = {}
    for name, s in predictors.items():
        au_all, r_all = score(name, s, incorrect, y_se, all_mask)
        au_clean, r_clean = score(name, s, incorrect, y_se, clean_mask)
        print(f"{name:26s}{au_all:>12.3f}{au_clean:>14.3f}{au_clean-au_all:>+7.3f}"
              f"{r_all:>10.3f}{r_clean:>12.3f}{r_clean-r_all:>+7.3f}")
        out[name] = {"auroc_all": au_all, "auroc_clean": au_clean, "auroc_delta": au_clean - au_all,
                     "rho_all": r_all, "rho_clean": r_clean, "rho_delta": r_clean - r_all}

    out["n_total"] = len(ids)
    out["n_bad"] = int(is_bad.sum())
    with open("amortized_ue/results/e49_qwen35_9b_think_leak_check.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote amortized_ue/results/e49_qwen35_9b_think_leak_check.json")


if __name__ == "__main__":
    main()
