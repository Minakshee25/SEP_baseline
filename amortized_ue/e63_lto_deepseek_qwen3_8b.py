"""E63 — leave-TWO-out cross-model test, single held-out pair: DeepSeek-LLM-7B-Chat vs Qwen3-8B.

Question (E40/E40b framing, one pair, TEXT proxy instead of the aligned-hidden-state ridge):
train ONE `q_resp_only` proxy on 6 target LLMs, hold out 2, and ask whether the proxy
reproduces the held-out pair's genuine SE disagreement — or is only a "hard question" detector.

Why leave-TWO-out and why the null is 0 here (E40b): a single proxy scores BOTH held-out
models, and `q_resp_only` reads only `Question: {q}\nAnswer: {canonical response}` — the
question is identical for the two models, so a pure question-difficulty predictor emits
`predicted_diff == 0` exactly. Any systematic `predicted_diff` vs `true_diff` correlation is
carried by the model-specific *response text* (E40b finding #5: response text is a far more
model-specific channel than the aligned hidden state). No fold-composition artifact — the
proxy never saw either held-out model in any form, and `q_resp_only` never touches z.

Train models (6):  Llama-2-7b-chat, Mistral-7B-Instruct-v0.2, Meta-Llama-3-8B-Instruct,
                   Qwen3.5-9B, gemma-7b-it, gemma-2-9b-it   (each: trivia_qa n2000 _full)
Held out (2):      deepseek-llm-7b-chat, Qwen3-8B            (each: trivia_qa n1000 _full)
                   predicted_diff = proxy_pred(DeepSeek) - proxy_pred(Qwen3-8B)
                   true_diff      = true_SE(DeepSeek)     - true_SE(Qwen3-8B)

Recipe: q_resp_only arm, 3 seeds, batch_size 8 x grad_accum 4 = effective batch 32, 10 epochs,
projector_hidden_dim 1024, k=4 — identical to E53 / the deploy checkpoint (E53 dropped the
effective batch from 32 to a micro-batch of 8 via grad-accum for the same reason: Qwen3.5-9B's
long <think> traces hit max_seq_len=256, OOM-ing a true batch=32; grad-accum reproduces the
batch=32 gradient exactly — no batchnorm in ProxyModel).

Env: amortized_stage2 (GPU). Run from the repo root:
    python -m amortized_ue.e63_lto_deepseek_qwen3_8b --stage overlap   # just the id-overlap check (CPU)
    python -m amortized_ue.e63_lto_deepseek_qwen3_8b --stage train     # pool + train 3 seeds (GPU)
    python -m amortized_ue.e63_lto_deepseek_qwen3_8b --stage eval      # score + analysis + examples (GPU)
    python -m amortized_ue.e63_lto_deepseek_qwen3_8b --stage all       # train then eval
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata, norm

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.linear_ceiling_probe import splits, rho
from amortized_ue.correctness_eval import load_accuracy

TRAIN_MODELS = ["Llama-2-7b-chat", "Mistral-7B-Instruct-v0.2", "Meta-Llama-3-8B-Instruct",
                "Qwen3.5-9B", "gemma-7b-it", "gemma-2-9b-it"]
A_NAME, B_NAME = "deepseek-llm-7b-chat", "Qwen3-8B"      # predicted/true diff = A - B
ARM = "q_resp_only"
TRAIN_N = 2000
EVAL_N = 1000

DEFAULT_DATA_DIR = "/data2/mn1025/stage1"
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT_DIR = os.path.join(_HERE, "stage2", "runs", "E63_lto_6model_qresp", "checkpoints")
TAG = "lto_6model"
RESULTS_DIR = os.path.join(_HERE, "results")
OUT_MAIN = os.path.join(RESULTS_DIR, "e63_lto_deepseek_qwen3_8b.json")
OUT_TABLE = os.path.join(RESULTS_DIR, "e63_lto_disagreement_table.json")
OUT_CURATED = os.path.join(RESULTS_DIR, "e63_lto_examples_curated.json")
OUT_CURVES = os.path.join(RESULTS_DIR, "e63_lto_train_curves.json")


# ----------------------------------------------------------------- helpers ----
def qnorm(v):
    """Within-model rank -> normal quantile (E40/E40b: kills per-model scale/offset)."""
    r = rankdata(np.asarray(v, dtype=np.float64))
    return norm.ppf((r - 0.5) / len(r))


def _record_ids(model, num_samples, data_dir):
    recs = load_records(Stage1Config(model_name=model, dataset="trivia_qa",
                                     num_samples=num_samples, output_dir=data_dir))
    return recs


def overlap_check(data_dir, verbose=True):
    recA = _record_ids(A_NAME, EVAL_N, data_dir)
    recB = _record_ids(B_NAME, EVAL_N, data_dir)
    iA, iB = set(recA), set(recB)
    shared = sorted(iA & iB)
    info = {"A": A_NAME, "B": B_NAME, "n_A": len(iA), "n_B": len(iB),
            "overlap": len(shared), "A_only": len(iA - iB), "B_only": len(iB - iA),
            "identical": iA == iB}
    if verbose:
        print(f"[overlap] {A_NAME} n={len(iA)}  {B_NAME} n={len(iB)}  "
              f"overlap={len(shared)}  A_only={len(iA - iB)}  B_only={len(iB - iA)}  "
              f"identical={iA == iB}")
    if len(shared) == 0:
        raise SystemExit("STOP: DeepSeek and Qwen3-8B eval record sets have ZERO shared "
                         "question ids — a leave-two-out difference test is impossible.")
    return shared, info


# ------------------------------------------------------------------ train -----
def load_pool(data_dir):
    """Pool (question, canonical response, per-model TRAIN-z-scored SE label) train/val rows
    from the 6 TRAIN_MODELS' n2000 trivia_qa records. Mirrors e53_train_qwengemma_deploy.load_pool
    (per-model splits(2000) on that model's own sorted-id order; SE z-scored per model with
    TRAIN-ONLY mean/std applied to both tr and va -> no val leakage into the normalizer).
    No hidden states are loaded (q_resp_only never reads z)."""
    ptr = {"q": [], "r": [], "y": []}
    pva = {"q": [], "r": [], "y": []}
    stats = {}
    for m in TRAIN_MODELS:
        recs = load_records(Stage1Config(model_name=m, dataset="trivia_qa",
                                         num_samples=TRAIN_N, output_dir=data_dir))
        ids = sorted(recs.keys())
        assert len(ids) == TRAIN_N, f"{m}: expected {TRAIN_N} records, got {len(ids)}"
        tr, va, te = splits(len(ids))
        q = [recs[i]["question"] for i in ids]
        r = [recs[i]["canonical"]["response"] for i in ids]
        y = np.array([recs[i]["labels"]["cluster_assignment_entropy"] for i in ids], dtype=np.float32)
        mu, sd = float(y[tr].mean()), float(y[tr].std() + 1e-12)
        ptr["q"] += [q[i] for i in tr]; ptr["r"] += [r[i] for i in tr]; ptr["y"] += list((y[tr] - mu) / sd)
        pva["q"] += [q[i] for i in va]; pva["r"] += [r[i] for i in va]; pva["y"] += list((y[va] - mu) / sd)
        stats[m] = {"n": len(ids), "n_tr": int(len(tr)), "n_va": int(len(va)),
                    "mean_CAE_train": mu, "std_CAE_train": sd}
        print(f"  {m:26s} n={len(ids)} tr={len(tr)} va={len(va)} mean_CAE(train)={mu:.3f}")
    train = {"y": np.array(ptr["y"], dtype=np.float32), "q": ptr["q"], "r": ptr["r"]}
    val = {"y": np.array(pva["y"], dtype=np.float32), "q": pva["q"], "r": pva["r"]}
    return train, val, stats


def do_train(data_dir, ckpt_dir, seeds, batch_size, grad_accum):
    print(f"Pooling {ARM} training data from {len(TRAIN_MODELS)} models "
          f"(data_dir={data_dir}) ...")
    train, val, stats = load_pool(data_dir)
    print(f"pooled: train rows={len(train['y'])}  val rows={len(val['y'])}")

    import torch
    import torch.nn as nn
    from transformers import get_cosine_schedule_with_warmup
    from amortized_ue.stage2.config import Stage2Config
    from amortized_ue.stage2.model import ProxyModel
    from amortized_ue.stage2.train import _tokenize_arm, _arm_uses_z, _arm_text
    from amortized_ue.exp2_run import train_arm

    # unused z filler: q_resp_only never reads z -> h_in=1 keeps the (never-trained) projector tiny
    train["z"] = np.zeros((len(train["y"]), 1), dtype=np.float32)
    val["z"] = np.zeros((len(val["y"]), 1), dtype=np.float32)
    tgt = dict(val)                       # in-dist val-pool sanity target (build_deploy convention)

    cfg = Stage2Config(projector_hidden_dim=1024, k_soft_tokens=4, epochs=10,
                       batch_size=batch_size, grad_accum=grad_accum)
    model = ProxyModel(cfg, h_in=1).to("cuda" if torch.cuda.is_available() else "cpu")

    lens = [len(model.tokenizer(_arm_text(ARM, q, r), add_special_tokens=False)["input_ids"])
            for q, r in zip(train["q"], train["r"])]
    n_cap = sum(1 for l in lens if l >= cfg.max_seq_len)
    print(f"tokenized {ARM} length over pooled train: max={max(lens)} "
          f"p99={int(np.percentile(lens, 99))} at/over cap({cfg.max_seq_len})={n_cap}/{len(lens)}")

    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"\nTraining {ARM} on {len(train['y'])} pooled rows, seeds={seeds}, "
          f"batch_size={batch_size} x grad_accum={grad_accum} "
          f"(effective batch={batch_size * grad_accum}), ckpt -> {ckpt_dir}", flush=True)
    res = train_arm(train, val, tgt, ARM, seeds, cfg, model, torch, nn,
                    get_cosine_schedule_with_warmup, _tokenize_arm, _arm_uses_z,
                    ckpt_dir=ckpt_dir, tag=TAG)
    print(f"\nDONE. val-pool sanity Spearman per seed: "
          f"{[round(s, 3) for s in res['te_spearman']]}  mean={np.mean(res['te_spearman']):.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_CURVES, "w") as f:
        json.dump({"arm": ARM, "train_models": TRAIN_MODELS, "held_out": [A_NAME, B_NAME],
                   "seeds": list(seeds), "data_dir": data_dir, "ckpt_dir": ckpt_dir,
                   "per_model_stats": stats, "train_config": cfg.as_dict(),
                   "n_train": len(train["y"]), "n_val": len(val["y"]),
                   "val_pool_sanity_spearman_by_seed": res["te_spearman"],
                   "curves_by_seed": res["curves_by_seed"],
                   "val_pool_pred_by_seed": [[float(v) for v in p] for p in res["te_pred_by_seed"]],
                   "val_pool_y": [float(v) for v in tgt["y"]]}, f)
    print(f"training curves saved to {OUT_CURVES}")


# ------------------------------------------------------------------- eval -----
def boot_ci(fn, n, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array([fn(rng.integers(0, n, n)) for _ in range(B)])
    return {"mean": float(v.mean()), "lo95": float(np.percentile(v, 2.5)),
            "hi95": float(np.percentile(v, 97.5))}


def sign_acc(dP, dY, idx=None):
    """Sign-agreement on non-tie true-diff rows; exact prediction ties score 0.5 (E40 convention)."""
    dP = dP if idx is None else dP[idx]
    dY = dY if idx is None else dY[idx]
    k = dY != 0
    if k.sum() == 0:
        return 0.5
    h = np.where(dP[k] == 0, 0.5, (np.sign(dP[k]) == np.sign(dY[k])).astype(float))
    return float(h.mean())


def quartile_table(dP, dY, absdY):
    """Disjoint quartiles by |true_diff|, largest gap (Q4) -> smallest (Q1)."""
    order = np.argsort(absdY)                       # ascending
    n = len(order)
    edges = [0, n // 4, n // 2, 3 * n // 4, n]
    out = []
    for qi, (lab, lo, hi) in enumerate([("Q4_largest", 3, 4), ("Q3", 2, 3),
                                        ("Q2", 1, 2), ("Q1_smallest", 0, 1)]):
        idx = order[edges[lo]:edges[hi]]
        k = dY[idx] != 0
        out.append({"quartile": lab,
                    "n": int(len(idx)), "n_nontie": int(k.sum()),
                    "abs_true_diff_range": [float(absdY[idx].min()), float(absdY[idx].max())],
                    "sign_agreement": sign_acc(dP[idx], dY[idx])})
    return out


def do_eval(data_dir, ckpt_dir, bootstrap):
    import torch  # noqa: F401  (arm_preds needs it)
    from amortized_ue.procrustes_e27_rank_fusion import arm_preds

    shared, ov = overlap_check(data_dir)
    ids = shared
    n = len(ids)
    print(f"\n[eval] full shared question set: N={n}")

    recA = load_records(Stage1Config(model_name=A_NAME, dataset="trivia_qa",
                                     num_samples=EVAL_N, output_dir=data_dir))
    recB = load_records(Stage1Config(model_name=B_NAME, dataset="trivia_qa",
                                     num_samples=EVAL_N, output_dir=data_dir))
    accA = load_accuracy(Stage1Config(model_name=A_NAME, dataset="trivia_qa",
                                      num_samples=EVAL_N, output_dir=data_dir))
    accB = load_accuracy(Stage1Config(model_name=B_NAME, dataset="trivia_qa",
                                      num_samples=EVAL_N, output_dir=data_dir))

    seA = np.array([recA[i]["labels"]["cluster_assignment_entropy"] for i in ids], dtype=float)
    seB = np.array([recB[i]["labels"]["cluster_assignment_entropy"] for i in ids], dtype=float)

    print(f"[eval] running proxy arm={ARM} (3 seeds, seed-averaged) on {A_NAME} ...")
    mpA = arm_preds(ARM, A_NAME, "trivia_qa", EVAL_N, ckpt_dir=ckpt_dir, data_dir=data_dir)
    print(f"[eval] running proxy arm={ARM} (3 seeds, seed-averaged) on {B_NAME} ...")
    mpB = arm_preds(ARM, B_NAME, "trivia_qa", EVAL_N, ckpt_dir=ckpt_dir, data_dir=data_dir)
    pA = np.array([mpA[i] for i in ids], dtype=float)
    pB = np.array([mpB[i] for i in ids], dtype=float)

    # ---- per-question diffs: raw (task formula, PRIMARY) + qnorm (E40b convention) --------
    dP_raw, dY_raw = pA - pB, seA - seB
    dP_qn = qnorm(pA) - qnorm(pB)
    dY_qn = qnorm(seA) - qnorm(seB)

    R = {"config": {"train_models": TRAIN_MODELS, "held_out_A": A_NAME, "held_out_B": B_NAME,
                    "arm": ARM, "ckpt_dir": ckpt_dir, "data_dir": data_dir,
                    "n_shared_questions": n, "bootstrap": bootstrap},
         "overlap_check": ov}

    # ---- (a) overall correlation of predicted_diff vs true_diff --------------------------
    def corr_block(dP, dY):
        return {
            "spearman": float(spearmanr(dP, dY).correlation),
            "spearman_ci95": boot_ci(lambda ix: float(spearmanr(dP[ix], dY[ix]).correlation), n, bootstrap),
            "pearson": float(pearsonr(dP, dY)[0]),
            "pearson_ci95": boot_ci(lambda ix: float(pearsonr(dP[ix], dY[ix])[0]), n, bootstrap),
            "overall_sign_agreement": sign_acc(dP, dY),
            "overall_sign_agreement_ci95": boot_ci(lambda ix: sign_acc(dP[ix], dY[ix]), n, bootstrap),
            "n_nontie": int((dY != 0).sum()),
        }
    R["a_overall_correlation"] = {
        "raw_diff_PRIMARY": corr_block(dP_raw, dY_raw),
        "qnorm_diff_E40b": corr_block(dP_qn, dY_qn),
        "predicted_diff_summary": {"mean": float(dP_raw.mean()), "std": float(dP_raw.std()),
                                   "min": float(dP_raw.min()), "max": float(dP_raw.max())},
        "true_diff_summary": {"mean": float(dY_raw.mean()), "std": float(dY_raw.std()),
                              "min": float(dY_raw.min()), "max": float(dY_raw.max()),
                              "abs_mean": float(np.abs(dY_raw).mean()),
                              "abs_median": float(np.median(np.abs(dY_raw))),
                              "frac_abs_gt_0.5": float((np.abs(dY_raw) > 0.5).mean())},
    }

    # ---- (b) sign-agreement by quartile of |true_diff| ----------------------------------
    R["b_quartiles"] = {
        "raw_diff_PRIMARY": {
            "quartiles": quartile_table(dP_raw, dY_raw, np.abs(dY_raw)),
            "overall_sign_agreement": sign_acc(dP_raw, dY_raw),
            "overall_sign_agreement_ci95": boot_ci(lambda ix: sign_acc(dP_raw[ix], dY_raw[ix]), n, bootstrap),
        },
        "qnorm_diff_E40b": {
            "quartiles": quartile_table(dP_qn, dY_qn, np.abs(dY_qn)),
            "overall_sign_agreement": sign_acc(dP_qn, dY_qn),
        },
    }

    # ---- (c) per-model proxy-vs-true-SE Spearman sanity check ----------------------------
    R["c_per_model_spearman"] = {
        A_NAME: {"spearman_proxy_vs_true_se": rho(pA, seA),
                 "mean_true_SE": float(seA.mean()),
                 "mean_accuracy": float(np.mean([accA[i] for i in ids]))},
        B_NAME: {"spearman_proxy_vs_true_se": rho(pB, seB),
                 "mean_true_SE": float(seB.mean()),
                 "mean_accuracy": float(np.mean([accB[i] for i in ids]))},
    }

    # ---- console report ----------------------------------------------------------------
    print("\n" + "=" * 84)
    print(f"E63 leave-TWO-out: {A_NAME}  vs  {B_NAME}   (proxy trained on the other 6, {ARM})")
    print("=" * 84)
    for key, lab in [("raw_diff_PRIMARY", "raw diff (task formula)"), ("qnorm_diff_E40b", "qnorm diff (E40b)")]:
        cb = R["a_overall_correlation"][key]
        print(f"\n[a] {lab}")
        print(f"    Spearman(dP,dY) = {cb['spearman']:+.3f}  "
              f"[{cb['spearman_ci95']['lo95']:+.3f}, {cb['spearman_ci95']['hi95']:+.3f}]")
        print(f"    Pearson (dP,dY) = {cb['pearson']:+.3f}  "
              f"[{cb['pearson_ci95']['lo95']:+.3f}, {cb['pearson_ci95']['hi95']:+.3f}]")
        print(f"    overall sign-agreement = {cb['overall_sign_agreement']:.3f}  "
              f"[{cb['overall_sign_agreement_ci95']['lo95']:.3f}, {cb['overall_sign_agreement_ci95']['hi95']:.3f}]"
              f"   (n_nontie={cb['n_nontie']}, chance 0.500)")
    print("\n[b] sign-agreement by |true_diff| quartile (raw; largest gap -> smallest):")
    for q in R["b_quartiles"]["raw_diff_PRIMARY"]["quartiles"]:
        print(f"    {q['quartile']:12s} n={q['n']:4d}  n_nontie={q['n_nontie']:4d}  "
              f"|dY| in [{q['abs_true_diff_range'][0]:.3f}, {q['abs_true_diff_range'][1]:.3f}]  "
              f"sign-agreement={q['sign_agreement']:.3f}")
    print("\n[c] per-model proxy-vs-true-SE Spearman (sanity):")
    for m, d in R["c_per_model_spearman"].items():
        print(f"    {m:26s} rho={d['spearman_proxy_vs_true_se']:+.3f}  "
              f"mean_SE={d['mean_true_SE']:.3f}  mean_acc={d['mean_accuracy']:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_MAIN, "w") as f:
        json.dump(R, f, indent=1)
    print(f"\nwrote {OUT_MAIN}")

    # ---- example artifact 1: FULL disagreement table (all rows, sorted by |true_diff|) ---
    rows = []
    for k, i in enumerate(ids):
        correct = bool(np.sign(dP_raw[k]) == np.sign(dY_raw[k])) if dY_raw[k] != 0 else None
        rows.append({
            "id": i, "question": recA[i]["question"],
            "deepseek_response": recA[i]["canonical"]["response"],
            "qwen3_8b_response": recB[i]["canonical"]["response"],
            "deepseek_true_se": float(seA[k]), "qwen3_8b_true_se": float(seB[k]),
            "deepseek_correct": bool(accA[i] >= 0.5), "qwen3_8b_correct": bool(accB[i] >= 0.5),
            "true_diff": float(dY_raw[k]), "predicted_diff": float(dP_raw[k]),
            "true_diff_qnorm": float(dY_qn[k]), "predicted_diff_qnorm": float(dP_qn[k]),
            "abs_true_diff": float(abs(dY_raw[k])), "correct": correct,
        })
    rows.sort(key=lambda r: -r["abs_true_diff"])
    with open(OUT_TABLE, "w") as f:
        json.dump({"pair": f"{A_NAME} (A) vs {B_NAME} (B)",
                   "diff_convention": "A - B (raw CAE, and within-model qnorm)",
                   "n_rows": len(rows),
                   "n_correct": int(sum(1 for r in rows if r["correct"] is True)),
                   "n_incorrect": int(sum(1 for r in rows if r["correct"] is False)),
                   "n_tie": int(sum(1 for r in rows if r["correct"] is None)),
                   "rows": rows}, f, indent=1)
    print(f"wrote {OUT_TABLE}  ({len(rows)} rows)")

    # ---- example artifact 2: curated top-5 correct / top-5 incorrect by |true_diff| ------
    corr_rows = [r for r in rows if r["correct"] is True][:5]
    inc_rows = [r for r in rows if r["correct"] is False][:5]
    with open(OUT_CURATED, "w") as f:
        json.dump({"pair": f"{A_NAME} (A) vs {B_NAME} (B)",
                   "note": ("top 5 correctly-called and top 5 incorrectly-called disagreement "
                            "examples by |true_diff|, pulled from e63_lto_disagreement_table.json"),
                   "top5_correct": corr_rows, "top5_incorrect": inc_rows}, f, indent=1)
    print(f"wrote {OUT_CURATED}")

    print("\n=== curated examples ===")
    for tag, rr in [("CORRECTLY CALLED", corr_rows), ("INCORRECTLY CALLED", inc_rows)]:
        print(f"\n--- top 5 {tag} (by |true_diff|) ---")
        for r in rr:
            print(f"\nQ: {r['question']}")
            print(f"  DeepSeek: \"{r['deepseek_response']}\"  SE={r['deepseek_true_se']:.3f} "
                  f"({'correct' if r['deepseek_correct'] else 'WRONG'})")
            print(f"  Qwen3-8B: \"{r['qwen3_8b_response']}\"  SE={r['qwen3_8b_true_se']:.3f} "
                  f"({'correct' if r['qwen3_8b_correct'] else 'WRONG'})")
            print(f"  true_diff={r['true_diff']:+.3f}  predicted_diff={r['predicted_diff']:+.3f}  "
                  f"-> {'HIT' if r['correct'] else 'MISS'}")


# ----------------------------------------------------------- wandb push ------
def do_push_wandb(ckpt_dir):
    """Push the 3-seed checkpoint dir as a versioned W&B model artifact (extra copy; local
    disk stays source of truth). Mirrors stage2.run._push_checkpoints_wandb — `train_arm`
    was called directly here so that hook never fired."""
    import glob
    import wandb
    paths = sorted(glob.glob(os.path.join(ckpt_dir, f"*{ARM}_seed*.pt")))
    assert len(paths) == 3, f"expected 3 seed checkpoints in {ckpt_dir}, found {len(paths)}"
    name = "stage2_ckpts_E63_lto_6model_qresp"
    run = wandb.init(project="amortized_ue_stage2", entity=os.environ.get("WANDB_ENT"),
                     name=name, job_type="checkpoint",
                     config={"arm": ARM, "train_models": TRAIN_MODELS, "held_out": [A_NAME, B_NAME],
                             "recipe": "q_resp_only, 3 seeds, batch 8 x grad_accum 4 (eff 32), "
                                       "projector_hidden_dim 1024, k=4, 10 epochs"})
    art = wandb.Artifact(name, type="model",
                         metadata={"train_models": TRAIN_MODELS, "held_out_pair": [A_NAME, B_NAME],
                                   "arm": ARM, "n_seeds": 3, "proxy_model": "meta-llama/Llama-3.2-3B"})
    art.add_dir(ckpt_dir)
    run.log_artifact(art)
    run.finish()
    print(f"pushed {len(paths)} checkpoints to W&B artifact {name!r} (project amortized_ue_stage2)")
    api = wandb.Api()
    a = api.artifact(f"{os.environ['WANDB_ENT']}/amortized_ue_stage2/{name}:latest")
    print(f"  verified: {name}:{a.version}  size={a.size} bytes  files={[f.name for f in a.files()]}")


# ------------------------------------------------------------------- main -----
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["overlap", "train", "eval", "all", "push_wandb"], default="all")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args()

    if args.stage == "overlap":
        overlap_check(args.data_dir)
        return
    if args.stage == "push_wandb":
        do_push_wandb(args.ckpt_dir)
        return
    if args.stage in ("train", "all"):
        do_train(args.data_dir, args.ckpt_dir, args.seeds, args.batch_size, args.grad_accum)
    if args.stage in ("eval", "all"):
        do_eval(args.data_dir, args.ckpt_dir, args.bootstrap)


if __name__ == "__main__":
    main()
