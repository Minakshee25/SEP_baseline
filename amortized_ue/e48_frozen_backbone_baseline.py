"""E48 -- is the LoRA training on OUR SE-labeled data actually contributing anything, or is
q_resp_only's zero-shot performance on Qwen/Gemma mostly just the FROZEN Llama-3.2-3B backbone's
own pretrained knowledge (a fact-checker it already had before we trained anything)?

Direct test: skip the trained proxy entirely. Use the SAME frozen backbone (meta-llama/Llama-3.2-3B,
no LoRA, no projector, no head -- just the raw pretrained model) with a standard few-shot
"Is this answer True or False" prompt (the classic p_true self-verification format, adapted here
as a CROSS-model judge: the backbone never generated these answers, it's just reading them). Score
= P(the answer is False) from reading off the model's own next-token logits for "A"/"B", no
training at all.

If this untrained baseline performs comparably to our trained q_resp_only proxy (E45/E47), that
means the LoRA training on SE labels isn't adding much -- the generalization is mostly free,
pretrained knowledge. If the trained proxy clearly wins, training learned something real from the
SE-labeled data beyond what raw pretrained knowledge gives for free.

Few-shot examples are drawn from Llama-2's OWN records (a model none of Qwen/Gemma's test
questions come from) -- unrelated to the evaluated targets, just illustrating the answer format.

Env: `amortized_stage2_v5` + a free GPU (loads the same 3B backbone as the proxy, ~6GB bf16).
    /data2/mn1025/conda_envs/amortized_stage2_v5/bin/python -m amortized_ue.e48_frozen_backbone_baseline
"""
from __future__ import annotations

import json
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.correctness_eval import load_accuracy

TARGETS = ["Qwen3-8B", "Qwen3.5-9B", "gemma-7b-it", "gemma-2-9b-it"]
DATA_DIR = "/data2/mn1025/stage1"
BACKBONE = "meta-llama/Llama-3.2-3B"
NUM_SAMPLES = 1000

# Few-shot examples: clean, unambiguous, from Llama-2's own records (disjoint model/purpose).
FEWSHOT = [
    ("Which US property tycoon bought Turnberry Golf Course in April?", "donald trump", True),
    ("Which 1st World War battle of 1916 saw 60,000 British casualties on the first day?", "the somme", True),
    ("What is the common name of the laryngeal prominence?", "Adam's apple", False),
    ('"Who said, ""To err is human but it feels divine""?"', "Oscar Wilde", False),
]

PROMPT_HEADER = ""
for q, a, is_true in FEWSHOT:
    PROMPT_HEADER += (
        f"Question: {q}\nProposed answer: {a}\n"
        f"Is the proposed answer correct?\nA) True\nB) False\nAnswer:"
        f"{' A' if is_true else ' B'}\n\n"
    )


def rho(a, b):
    r = spearmanr(a, b).correlation
    return 0.0 if (r is None or np.isnan(r)) else float(r)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out", default="amortized_ue/results/e48_frozen_backbone_baseline.json")
    args = p.parse_args()

    print(f"loading frozen {BACKBONE} (no LoRA) ...")
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    tok.padding_side = "right"          # required for the `lengths-1` last-real-token index below
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BACKBONE, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    id_a = tok.encode(" A", add_special_tokens=False)[-1]
    id_b = tok.encode(" B", add_special_tokens=False)[-1]
    print(f"token ids: A={id_a} ({tok.decode([id_a])!r})  B={id_b} ({tok.decode([id_b])!r})")

    results = {}
    for t in TARGETS:
        cfg = Stage1Config(model_name=t, dataset="trivia_qa", num_samples=NUM_SAMPLES, output_dir=DATA_DIR)
        recs = load_records(cfg)
        ids = sorted(recs.keys())
        acc = load_accuracy(cfg)
        y_se = np.array([recs[i]["labels"]["cluster_assignment_entropy"] for i in ids])
        incorrect = np.array([1.0 - acc[i] for i in ids])

        prompts = [
            PROMPT_HEADER
            + f"Question: {recs[i]['question']}\nProposed answer: {recs[i]['canonical']['response']}\n"
              f"Is the proposed answer correct?\nA) True\nB) False\nAnswer:"
            for i in ids
        ]

        p_false = []
        print(f"scoring {t} ({len(prompts)} prompts) ...")
        with torch.no_grad():
            for i in range(0, len(prompts), args.batch_size):
                batch = prompts[i:i + args.batch_size]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
                out = model(**enc)
                # last non-pad token's logits per row
                lengths = enc["attention_mask"].sum(dim=1) - 1
                last_logits = out.logits[torch.arange(len(batch)), lengths]
                ab_logits = last_logits[:, [id_a, id_b]].float()
                probs = torch.softmax(ab_logits, dim=-1)
                p_false.extend(probs[:, 1].cpu().tolist())          # P(B=False) = predicted incorrectness

        p_false = np.array(p_false)
        au = float(roc_auc_score(incorrect, p_false))
        r_se = rho(p_false, y_se)
        print(f"  {t:16s}  AUROC_incorrect={au:.3f}   SE-fidelity rho={r_se:.3f}")
        results[t] = {"n": len(ids), "auroc_incorrect": au, "se_fidelity_rho": r_se,
                      "mean_p_false": float(p_false.mean())}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
