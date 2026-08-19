"""E40 — Does the pooled multi-model ridge preserve MODEL-SPECIFIC uncertainty?

The question (user's framing): find questions where the target LLMs genuinely DISAGREE in
semantic entropy (e.g. SE_Llama2(x)=1.8 but SE_Mistral(x)=1.2) and ask whether the shared
probe reproduces that disagreement. If it does, the probe has not merely learned "this is a
hard question" — it has preserved "THIS model is uncertain about this question".

Design (reuses E35/E37 machinery verbatim — same alignment, layers, splits, normalization):
  * 4 target LLMs x the SAME 2000 trivia questions, id-joined; splits(2000) -> tr/va/te.
    `te` (200 questions) is IDENTICAL across models, which is what makes the matrix analysis valid.
  * Leave-one-LLM-out: for target T, one ridge is trained on the OTHER 3 models' aligned states
    (label-free Procrustes into the Llama-2 best-TBG frame, per-model feature scaler + per-model
    SE z-score, alpha on the pooled source val) and predicts T's te. So every column of the
    prediction matrix comes from a probe that never saw that model. -> P[4, 200], Y[4, 200].

Why per-model normalization is mandatory (not a choice): the ridge is trained on per-model
z-scored SE labels, so the per-model SE *offset* (mean CAE: Llama-3 .48, Mistral .48, Llama-2 .58,
DeepSeek .78) is unrepresentable by construction. Comparing raw SE across models would score the
probe on a constant it was explicitly built not to emit. Primary normalization = within-model
rank -> normal quantile over te (nonparametric, identical transform applied to Y and P);
robustness = z-score with the target's own VAL stats (never test).

Analyses:
  [A] variance decomposition of Y: question main effect vs model-specific residual, + how often
      the models actually disagree (there must be something to find).
  [B] SPLIT-HALF CEILING: SE is a 10-sample estimate, so the model-specific residual is a
      difference of noisy quantities. Recompute SE from two disjoint halves of the stored
      `labels.semantic_ids` (5+5), residualize each, correlate -> reliability r5, Spearman-Brown
      to r10; the max attainable correlation for a PERFECT predictor is sqrt(r10).
  [C] HEADLINE: corr(model-specific residual of P, model-specific residual of Y), pooled + per
      model; permutation null (shuffle model labels of P within each question) + bootstrap over
      questions; reported as a fraction of the [B] ceiling.
  [D] DISCORDANT PAIRS (the literal test): for each question and each of the 6 model pairs, does
      the probe order the pair the way the true SEs do? Stratified by the size of the true gap.
  [E] DIFFICULTY-ORACLE SEMI-PARTIAL: the strongest possible pure-difficulty predictor is the mean
      TRUE SE of the other 3 models on that question. Does P still correlate with Y after
      partialling it out?
  [F] COMPARISON ARMS from E37 (`results/exp2_lolo_full.json`, same te rows, seed-averaged preds):
      q_only is the CONTROL — its input is identical for every model, so its residual correlation
      MUST be ~0; anything else means the pipeline leaks. q_resp_only / z / fuse are real arms
      (does response TEXT carry more model-specificity than the hidden state?).

Audits printed before any headline number:
  * ids identical across all 4 models (assert), te derived exactly as E35/E37.
  * rebuilt ridge te-Spearman vs the values E37/E38 saved, and the correlation of the rebuilt
    per-example preds with E38's saved `ridge_te_preds`.
  * E37 `target_y` vs this script's Y row (max abs deviation must be 0).

Env: se_probes (CPU). Run: python -m amortized_ue.e40_model_specificity
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata, norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.linear_ceiling_probe import splits
from amortized_ue.exp2_run import (ANCHOR, MODELS, SHORT, BEST_TBG, ALPHAS,
                                   fit_alignment, rho)

PAIRS = [(a, b) for a in range(4) for b in range(4) if a < b]
E37_JSON = "results/exp2_lolo_full.json"
E38_JSON = "results/correctness_eval_e37.json"


# ---------------------------------------------------------------- data --------
def load_all_plus(anchor_layer=30, data_dir=None):
    """exp2_run.load_all + the per-sample semantic_ids needed for the split-half ceiling."""
    data, ids0 = {}, None
    for m in MODELS:
        kw = {"output_dir": data_dir} if data_dir else {}
        recs = load_records(Stage1Config(model_name=m, dataset="trivia_qa", num_samples=2000, **kw))
        ids = sorted(recs.keys())
        assert ids0 is None or ids == ids0, f"id order differs for {m} (join-by-id broken)"
        ids0 = ids
        layer = anchor_layer if m == ANCHOR else BEST_TBG[m]
        data[m] = {
            "H": np.stack([recs[i]["canonical"]["hidden_states"]["TBG"][layer].squeeze().float().numpy()
                           for i in ids]).astype(np.float32),
            "y": np.array([recs[i]["labels"]["cluster_assignment_entropy"] for i in ids], dtype=np.float64),
            "sids": [list(recs[i]["labels"]["semantic_ids"]) for i in ids],
            "acc": np.array([recs[i]["canonical"]["accuracy"] for i in ids], dtype=np.float64),
        }
        del recs
    return data, ids0, splits(len(ids0))


def save_ridge_bundle(ckpt_dir, data, ids, al, fsc, tr, va, te, folds, pair_folds, anchor_layer):
    """Persist the multi-model ridge so it never has to be refit (it never was saved before: E35's
    scripts wrote JSON only and exp2_run's --ckpt_dir covers the SLM proxy, not `ridge_on_z`).

    Lives at stage2/runs/<RUN>/checkpoints/ like every other trained artifact in this repo (that path
    is gitignored wholesale; the bundle goes to W&B, not git).

    Saves the whole deployable chain — the per-model label-free Procrustes alignment (the expensive
    part: one 4096^2 orthogonal fit per model), the per-model feature scalers, the per-model SE
    label z-stats, and every fitted ridge (4 leave-one-out + 6 leave-two-out) with its alpha.
    `load_ridge_bundle` rebuilds a callable predictor from it.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    align = {}
    for m in MODELS:
        mean_m, W = al[m]
        align[f"{m}__mean"] = mean_m.astype(np.float32)
        align[f"{m}__W"] = W.astype(np.float32)
        align[f"{m}__scaler_mean"] = fsc[m].mean_.astype(np.float32)
        align[f"{m}__scaler_scale"] = fsc[m].scale_.astype(np.float32)
        align[f"{m}__label_mu"] = np.float32(data[m]["y"][tr].mean())
        align[f"{m}__label_sd"] = np.float32(data[m]["y"][tr].std() + 1e-12)
    np.savez_compressed(os.path.join(ckpt_dir, "alignment.npz"), **align)

    ridges = {}
    for name, (model, alpha, held) in {**folds, **pair_folds}.items():
        ridges[f"{name}__coef"] = model.coef_.astype(np.float32)
        ridges[f"{name}__intercept"] = np.float32(model.intercept_)
        ridges[f"{name}__alpha"] = np.float32(alpha)
    np.savez_compressed(os.path.join(ckpt_dir, "ridges.npz"), **ridges)

    meta = {
        "what": "pooled multi-model ridge: aligned hidden state -> per-model z-scored semantic entropy",
        "models": MODELS, "short": SHORT,
        "anchor": ANCHOR, "anchor_layer": anchor_layer, "source_layers": BEST_TBG,
        "dataset": "trivia_qa", "n_questions": len(ids),
        "splits": {"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()},
        "ids": ids,
        "folds": {name: {"held_out": [SHORT[h] for h in held],
                         "sources": [SHORT[m] for m in MODELS if m not in held],
                         "alpha": float(alpha)}
                  for name, (_, alpha, held) in {**folds, **pair_folds}.items()},
        "inference": ("z = scaler_m((h_m - mean_m) @ W_m); yhat = z @ coef + intercept; yhat is in "
                      "per-model z-scored SE units (rank-comparable, not raw CAE)"),
    }
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return ckpt_dir


def load_ridge_bundle(ckpt_dir):
    """Rebuild {'predict': fn(model_name, H[n,4096], fold) -> yhat, 'meta': ...} from a saved bundle."""
    A = np.load(os.path.join(ckpt_dir, "alignment.npz"))
    Rg = np.load(os.path.join(ckpt_dir, "ridges.npz"))
    meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))

    def predict(model_name, H, fold):
        z = (np.asarray(H, dtype=np.float32) - A[f"{model_name}__mean"]) @ A[f"{model_name}__W"]
        z = (z - A[f"{model_name}__scaler_mean"]) / A[f"{model_name}__scaler_scale"]
        return z @ Rg[f"{fold}__coef"] + Rg[f"{fold}__intercept"]

    return {"predict": predict, "meta": meta, "alignment": A, "ridges": Rg}


def make_feat(data, al, fsc):
    def feat(m, idx):
        mean_m, W = al[m]
        return fsc[m].transform((data[m]["H"][idx] - mean_m) @ W).astype(np.float32)
    return feat


def fit_loo_ridges(data, feat, tr, va, te):
    """For each target T: ridge on the other 3 aligned models (alpha on pooled source val).

    Returns P_te[4, n_te], P_va[4, n_va], alphas. Identical construction to exp2_run.build_fold
    + ridge_on_z, extended to also emit the target's VAL predictions (for the z-score robustness
    variant). The target's own SE labels are never used to fit anything.
    """
    P_te = np.zeros((len(MODELS), len(te)))
    P_va = np.zeros((len(MODELS), len(va)))
    alphas, spear, folds = {}, {}, {}
    for ti, T in enumerate(MODELS):
        srcs = [m for m in MODELS if m != T]
        Xtr, ytr, Xva, yva = [], [], [], []
        for m in srcs:                                     # SAME questions to every source
            mu, sd = data[m]["y"][tr].mean(), data[m]["y"][tr].std() + 1e-12
            Xtr.append(feat(m, tr)); ytr.append((data[m]["y"][tr] - mu) / sd)
            Xva.append(feat(m, va)); yva.append((data[m]["y"][va] - mu) / sd)
        Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
        Xva, yva = np.vstack(Xva), np.concatenate(yva)
        best = None
        for a in ALPHAS:
            r = Ridge(alpha=a).fit(Xtr, ytr)
            s = rho(r.predict(Xva), yva)                   # alpha on VAL (sources only)
            if best is None or s > best[0]:
                best = (s, a, r)
        _, a_best, model = best
        P_te[ti] = model.predict(feat(T, te))
        P_va[ti] = model.predict(feat(T, va))
        alphas[SHORT[T]] = a_best
        spear[SHORT[T]] = rho(P_te[ti], data[T]["y"][te])
        folds[f"loo_{SHORT[T]}"] = (model, a_best, [T])
    return P_te, P_va, alphas, spear, folds


def fit_loo_ridges_pair(data, feat, tr, va, te, held_out):
    """Leave-TWO-out: ONE ridge trained on the models NOT in `held_out`, scoring both of them.

    Same construction as fit_loo_ridges (alignment/scalers on tr, per-model label z-score, alpha on
    the pooled source val); the only change is that two models are held out and read by one probe,
    which removes the per-column ridge confound from the pairwise test.
    """
    srcs = [m for m in MODELS if m not in held_out]
    Xtr, ytr, Xva, yva = [], [], [], []
    for m in srcs:
        mu, sd = data[m]["y"][tr].mean(), data[m]["y"][tr].std() + 1e-12
        Xtr.append(feat(m, tr)); ytr.append((data[m]["y"][tr] - mu) / sd)
        Xva.append(feat(m, va)); yva.append((data[m]["y"][va] - mu) / sd)
    Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
    Xva, yva = np.vstack(Xva), np.concatenate(yva)
    best = None
    for a in ALPHAS:
        r = Ridge(alpha=a).fit(Xtr, ytr)
        s = rho(r.predict(Xva), yva)
        if best is None or s > best[0]:
            best = (s, a, r)
    _, a_best, model = best
    P = np.vstack([model.predict(feat(T, te)) for T in held_out])
    sp = {SHORT[T]: rho(P[i], data[T]["y"][te]) for i, T in enumerate(held_out)}
    return P, model, a_best, sp


# ------------------------------------------------------- normalization --------
def qnorm(v):
    """Within-model rank -> normal quantile (nonparametric; kills per-model scale/offset)."""
    r = rankdata(np.asarray(v, dtype=np.float64))
    return norm.ppf((r - 0.5) / len(r))


def rownorm(M, how="qnorm", mu=None, sd=None):
    M = np.asarray(M, dtype=np.float64)
    if how == "qnorm":
        return np.vstack([qnorm(row) for row in M])
    return (M - np.asarray(mu)[:, None]) / np.asarray(sd)[:, None]


def resid(M):
    """Remove the question main effect -> model-specific component."""
    return M - M.mean(axis=0, keepdims=True)


# ------------------------------------------------------------ statistics ------
def cell_corr(Rp, Ry, kind="pearson"):
    a, b = Rp.ravel(), Ry.ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(pearsonr(a, b)[0] if kind == "pearson" else spearmanr(a, b).correlation)


def perm_null(Rp_source, Ry, B=2000, seed=0, kind="pearson"):
    """Null = the probe carries NO model-specific info: shuffle model labels within each question.

    Shuffles the normalized predictions (not residuals), then re-residualizes, so the null keeps
    the question main effect intact and destroys only the model->cell assignment.
    """
    rng = np.random.default_rng(seed)
    M, N = Rp_source.shape
    out = np.empty(B)
    for b in range(B):
        S = np.empty_like(Rp_source)
        for j in range(N):
            S[:, j] = Rp_source[rng.permutation(M), j]
        out[b] = cell_corr(resid(S), Ry, kind)
    return out


def boot_questions(Pn, Yn, B=2000, seed=0, kind="pearson"):
    """Bootstrap over QUESTIONS (columns) — cells within a question are not independent."""
    rng = np.random.default_rng(seed)
    N = Pn.shape[1]
    out = np.empty(B)
    for b in range(B):
        j = rng.integers(0, N, N)
        out[b] = cell_corr(resid(Pn[:, j]), resid(Yn[:, j]), kind)
    return out


def ci(v):
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y given z (rank-linear residualization)."""
    rx, ry, rz = (rankdata(v).astype(float) for v in (x, y, z))
    Z = np.c_[np.ones_like(rz), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return 0.0
    return float(pearsonr(ex, ey)[0])


# --------------------------------------------------- split-half ceiling -------
def cae(sub):
    """cluster_assignment_entropy on a subset of samples (same formula as the SEP baseline)."""
    c = np.bincount(np.asarray(sub, dtype=int))
    c = c[c > 0].astype(float)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def split_half_ceiling(data, te, K=200, seed=0, kind="pearson"):
    """Reliability of the MODEL-SPECIFIC residual, from disjoint 5+5 halves of the 10 samples.

    r5 = corr of the two halves' residual matrices; Spearman-Brown -> r10 (the reliability of the
    residual we actually observe); a perfect predictor can attain at most sqrt(r10) against it.
    Also returns the same for the un-residualized (marginal) SE, for contrast.
    """
    rng = np.random.default_rng(seed)
    r5_res, r5_raw = [], []
    for _ in range(K):
        A = np.zeros((len(MODELS), len(te)))
        B = np.zeros((len(MODELS), len(te)))
        for mi, m in enumerate(MODELS):
            for k, idx in enumerate(te):
                s = np.asarray(data[m]["sids"][idx], dtype=int)
                perm = rng.permutation(len(s))
                h = len(s) // 2
                A[mi, k] = cae(s[perm[:h]])
                B[mi, k] = cae(s[perm[h:2 * h]])
        An, Bn = rownorm(A), rownorm(B)
        r5_raw.append(cell_corr(An, Bn, kind))
        r5_res.append(cell_corr(resid(An), resid(Bn), kind))

    def sb(r):
        r = float(np.mean(r))
        return r, (2 * r / (1 + r) if r > -1 else 0.0)

    r5r, r10r = sb(r5_res)
    r5m, r10m = sb(r5_raw)
    return {
        "n_splits": K,
        "residual_r5": r5r, "residual_r10_spearman_brown": r10r,
        "residual_ceiling_sqrt_r10": float(np.sqrt(max(r10r, 0.0))),
        "marginal_r5": r5m, "marginal_r10_spearman_brown": r10m,
        "marginal_ceiling_sqrt_r10": float(np.sqrt(max(r10m, 0.0))),
    }


# ------------------------------------------------------- discordant pairs -----
def discordant(Pn, Yn, quantiles=(0.0, 0.5, 0.75, 0.9)):
    """Sign-agreement on within-question model pairs, stratified by the size of the TRUE gap."""
    dY = np.concatenate([Yn[a] - Yn[b] for a, b in PAIRS])
    dP = np.concatenate([Pn[a] - Pn[b] for a, b in PAIRS])
    out = {}
    for q in quantiles:
        thr = np.quantile(np.abs(dY), q)
        keep = (np.abs(dY) >= thr) & (dY != 0)      # exact ties carry no ordering to predict
        # A predictor with NO model-specific information emits dP == 0 for every pair; scoring that
        # as a miss would report 0.0 instead of chance, so ties count as a coin flip (0.5).
        hits = np.where(dP[keep] == 0, 0.5,
                        (np.sign(dY[keep]) == np.sign(dP[keep])).astype(float))
        n = int(keep.sum())
        acc = float(hits.mean())
        se = float(np.sqrt(acc * (1 - acc) / max(n, 1)))
        out[f"top_{int((1 - q) * 100)}pct"] = {
            "n_pairs": n, "gap_threshold_normed": float(thr), "accuracy": acc,
            "ci95": [acc - 1.96 * se, acc + 1.96 * se]}
    return out


# ------------------------------------------------------------------ main ------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor_layer", type=int, default=30)
    ap.add_argument("--data_dir", default=None, help="override Stage1 output_dir (e.g. /data2 copy)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--ceiling_splits", type=int, default=200)
    ap.add_argument("--out", default="results/e40_model_specificity.json")
    ap.add_argument("--ckpt_dir", default="stage2/runs/E40_pooled_multimodel_ridge/checkpoints",
                    help="where to persist the pooled multi-model ridge (alignment + scalers + ridges); "
                         "follows the repo convention stage2/runs/<RUN>/checkpoints/")
    args = ap.parse_args()

    R = {"config": vars(args), "audits": {}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def dump():
        with open(args.out, "w") as f:
            json.dump(R, f, indent=1)

    print("=" * 92)
    print("E40 — is the pooled multi-model ridge MODEL-SPECIFIC, or only question-difficulty?")
    print("=" * 92)

    # ---- load + fit ----------------------------------------------------------
    data, ids, (tr, va, te) = load_all_plus(args.anchor_layer, args.data_dir)
    print(f"loaded 4 models x {len(ids)} id-joined questions; tr/va/te = {len(tr)}/{len(va)}/{len(te)}")
    print("mean SE by model:", {SHORT[m]: round(float(data[m]['y'].mean()), 3) for m in MODELS})
    al, fsc = fit_alignment(data, tr, args.anchor_layer)      # label-free, fit on tr, ONCE
    feat = make_feat(data, al, fsc)
    P_te, P_va, alphas, spear, folds = fit_loo_ridges(data, feat, tr, va, te)
    Y_te = np.vstack([data[m]["y"][te] for m in MODELS])
    Y_va = np.vstack([data[m]["y"][va] for m in MODELS])
    print("\nrebuilt LOO ridge te-Spearman:", {k: round(v, 4) for k, v in spear.items()}, "alphas:", alphas)

    # ---- audits --------------------------------------------------------------
    aud = R["audits"]
    aud["ridge_spearman_rebuilt"] = spear
    aud["ridge_alpha"] = alphas
    if os.path.exists(E37_JSON):
        e37 = {f["info"]["target"]: f for f in json.load(open(E37_JSON))}
        aud["ridge_spearman_e37"] = {k: round(e37[k]["ridge_z"], 4) for k in e37}
        dev = {}
        for ti, m in enumerate(MODELS):
            dev[SHORT[m]] = float(np.abs(np.array(e37[SHORT[m]]["target_y"]) - Y_te[ti]).max())
        aud["e37_target_y_max_abs_dev"] = dev
        print("E37 ridge te-Spearman :", aud["ridge_spearman_e37"])
        print("E37 target_y vs ours  : max abs dev", dev, "(must be 0.0)")
    if os.path.exists(E38_JSON):
        e38 = json.load(open(E38_JSON))
        agree = {}
        for ti, m in enumerate(MODELS):
            k = SHORT[m]
            if k in e38 and "ridge_te_preds" in e38[k]:
                agree[SHORT[m]] = round(float(pearsonr(np.array(e38[k]["ridge_te_preds"]), P_te[ti])[0]), 6)
        aud["ridge_pred_corr_with_e38"] = agree
        print("rebuilt ridge preds vs E38 saved preds (Pearson):", agree)
    dump()

    # ---- [A] is there anything to find? --------------------------------------
    Yn = rownorm(Y_te)                       # within-model rank -> normal quantile
    Pn = rownorm(P_te)
    Ry, Rp = resid(Yn), resid(Pn)
    tot = float(np.var(Yn))
    qvar = float(np.var(Yn.mean(axis=0)))    # variance of the question main effect
    A = {"total_var_normed": tot, "question_effect_var": qvar,
         "model_specific_resid_var": float(np.var(Ry)),
         "frac_question_effect": qvar / tot,
         "raw_SE_mean_by_model": {SHORT[m]: float(data[m]["y"][te].mean()) for m in MODELS},
         "raw_SE_sd_by_model": {SHORT[m]: float(data[m]["y"][te].std()) for m in MODELS}}
    dY_raw = np.concatenate([np.abs(Y_te[a] - Y_te[b]) for a, b in PAIRS])
    A["raw_abs_gap_between_models"] = {"mean": float(dY_raw.mean()),
                                       "median": float(np.median(dY_raw)),
                                       "p90": float(np.percentile(dY_raw, 90)),
                                       "frac_gap_gt_0.5": float((dY_raw > 0.5).mean())}
    A["cross_model_SE_spearman"] = {
        f"{SHORT[MODELS[a]]}~{SHORT[MODELS[b]]}": round(rho(Y_te[a], Y_te[b]), 3) for a, b in PAIRS}
    R["A_variance"] = A
    print("\n[A] question effect explains "
          f"{qvar / tot:.1%} of the normalized SE variance; model-specific residual {np.var(Ry) / tot:.1%}")
    print("    cross-model SE Spearman:", A["cross_model_SE_spearman"])
    print("    raw |SE_a - SE_b|: mean %.3f, median %.3f, p90 %.3f, %.1f%% of pairs > 0.5"
          % (dY_raw.mean(), np.median(dY_raw), np.percentile(dY_raw, 90), 100 * (dY_raw > 0.5).mean()))
    dump()

    # ---- [B] ceiling ---------------------------------------------------------
    print(f"\n[B] split-half ceiling ({args.ceiling_splits} random 5+5 splits of the 10 samples)...")
    C = split_half_ceiling(data, te, K=args.ceiling_splits)
    R["B_ceiling"] = C
    print(f"    marginal SE  : r5={C['marginal_r5']:.3f} r10={C['marginal_r10_spearman_brown']:.3f} "
          f"ceiling={C['marginal_ceiling_sqrt_r10']:.3f}")
    print(f"    model-specific residual: r5={C['residual_r5']:.3f} "
          f"r10={C['residual_r10_spearman_brown']:.3f} ceiling={C['residual_ceiling_sqrt_r10']:.3f}")
    dump()

    # ---- [C] headline --------------------------------------------------------
    ceil = C["residual_ceiling_sqrt_r10"]
    obs = cell_corr(Rp, Ry)
    nul = perm_null(Pn, Ry, B=args.perm)
    bs = boot_questions(Pn, Yn, B=args.boot)
    pval = float((np.abs(nul) >= abs(obs)).mean())
    per_model = {SHORT[m]: {"pearson": float(pearsonr(Rp[i], Ry[i])[0]),
                            "spearman": rho(Rp[i], Ry[i])} for i, m in enumerate(MODELS)}
    Cc = {"residual_corr_pearson": obs,
          "residual_corr_spearman": cell_corr(Rp, Ry, "spearman"),
          "bootstrap_mean_ci": ci(bs),
          "perm_null_mean_sd": [float(nul.mean()), float(nul.std())],
          "perm_p_value": pval,
          "frac_of_ceiling": float(obs / ceil) if ceil > 0 else None,
          "per_model": per_model,
          "marginal_corr_pearson": cell_corr(Pn, Yn)}
    R["C_headline"] = Cc
    b = ci(bs)
    print(f"\n[C] MODEL-SPECIFIC residual corr = {obs:+.3f}  boot95 [{b[1]:+.3f}, {b[2]:+.3f}]  "
          f"perm-null {nul.mean():+.3f}+-{nul.std():.3f}  p={pval:.4f}")
    print(f"    = {obs / ceil:.1%} of the attainable ceiling ({ceil:.3f})" if ceil > 0 else "")
    print("    per-model residual corr:", {k: round(v['pearson'], 3) for k, v in per_model.items()})
    dump()

    # ---- [D] discordant pairs ------------------------------------------------
    D = discordant(Pn, Yn)
    R["D_discordant_pairs"] = D
    print("\n[D] pairwise ordering accuracy (does the probe rank the pair like the true SEs?):")
    for k, v in D.items():
        print(f"    {k:>10}: n={v['n_pairs']:>5}  acc={v['accuracy']:.3f} "
              f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]   (chance 0.500)")
    dump()

    # ---- [E] difficulty-oracle semi-partial ----------------------------------
    # NOTE this test is BIASED UPWARD and must not be read alone: the oracle D is itself a noisy
    # (3-sample) estimate of question difficulty, so even a NOISELESS pure-difficulty predictor
    # retains a positive partial correlation (+0.27 in synthetic control). The matched empirical
    # null `r_PbarY_given_D` uses Pbar_i = mean of the OTHER models' predictions — a difficulty-only
    # predictor built from the probe itself, carrying no model-i-specific information but the same
    # noise characteristics. Only r_PY_given_D > r_PbarY_given_D is evidence of model-specificity.
    E = {}
    for i, m in enumerate(MODELS):
        others = [j for j in range(len(MODELS)) if j != i]
        diff_oracle = Yn[others].mean(axis=0)          # other 3 models' TRUE SE = pure difficulty
        p_bar = Pn[others].mean(axis=0)                # matched difficulty-only null predictor
        E[SHORT[m]] = {"r_PY": rho(Pn[i], Yn[i]),
                       "r_DY": rho(diff_oracle, Yn[i]),
                       "r_PY_given_D": partial_spearman(Pn[i], Yn[i], diff_oracle),
                       "r_PbarY_given_D_NULL": partial_spearman(p_bar, Yn[i], diff_oracle),
                       "r_PD": rho(Pn[i], diff_oracle)}
    R["E_semipartial"] = E
    print("\n[E] vs the difficulty ORACLE (mean true SE of the other 3 models) — BIASED UP, read the null:")
    print(f"    {'target':>9} | {'r(P,Y)':>7} {'r(D,Y)':>7} {'r(P,Y|D)':>9} {'NULL r(Pbar,Y|D)':>17} {'r(P,D)':>7}")
    for k, v in E.items():
        print(f"    {k:>9} | {v['r_PY']:>7.3f} {v['r_DY']:>7.3f} {v['r_PY_given_D']:>9.3f} "
              f"{v['r_PbarY_given_D_NULL']:>17.3f} {v['r_PD']:>7.3f}")
    dump()

    # ---- [F] comparison arms from E37 ---------------------------------------
    if os.path.exists(E37_JSON):
        e37 = {f["info"]["target"]: f for f in json.load(open(E37_JSON))}
        F = {}
        arms = list(e37[SHORT[MODELS[0]]]["arms"].keys())
        for arm in arms:
            M = np.vstack([np.mean(np.array(e37[SHORT[m]]["arms"][arm]["te_pred_by_seed"]), axis=0)
                           for m in MODELS])                     # seed-averaged, same te rows
            Mn = rownorm(M)
            o = cell_corr(resid(Mn), Ry)
            nu = perm_null(Mn, Ry, B=max(args.perm // 2, 500))
            F[arm] = {"residual_corr": o, "marginal_corr": cell_corr(Mn, Yn),
                      "perm_p": float((np.abs(nu) >= abs(o)).mean()),
                      "boot_ci": ci(boot_questions(Mn, Yn, B=max(args.boot // 2, 500))),
                      "frac_of_ceiling": float(o / ceil) if ceil > 0 else None,
                      "discordant_top25pct": discordant(Mn, Yn)["top_25pct"]["accuracy"]}
        F["ridge_z_pooled"] = {"residual_corr": obs, "marginal_corr": Cc["marginal_corr_pearson"],
                               "perm_p": pval, "boot_ci": b,
                               "frac_of_ceiling": Cc["frac_of_ceiling"],
                               "discordant_top25pct": D["top_25pct"]["accuracy"]}
        R["F_arms"] = F
        print("\n[F] model-specificity by predictor (q_only is the CONTROL — must be ~0):")
        print(f"    {'arm':>16} | {'resid corr':>10} {'boot95':>18} {'perm p':>7} {'%ceil':>7} {'pair-acc(top25)':>16}")
        for k, v in F.items():
            cc = v["boot_ci"]
            print(f"    {k:>16} | {v['residual_corr']:>+10.3f} [{cc[1]:+.3f},{cc[2]:+.3f}] "
                  f"{v['perm_p']:>7.4f} {(v['frac_of_ceiling'] or 0):>6.1%} {v['discordant_top25pct']:>16.3f}")
        dump()

    # ---- [G] leave-TWO-out: both members of a pair read by the SAME probe ----
    # Confound in [C]/[D]: each column of P comes from a different LOO ridge, so a per-model
    # difference could be ridge-to-ridge variation rather than a model-specific read. Here one
    # ridge (trained on the other 2 models) scores BOTH members of the held-out pair, so the only
    # thing that differs within a pair is the hidden state + its alignment map.
    G, pair_folds = {}, {}
    for a, b in PAIRS:
        T2 = [MODELS[a], MODELS[b]]
        Pp, mdl2, al2, sp2 = fit_loo_ridges_pair(data, feat, tr, va, te, T2)
        pair_folds[f"lto_{SHORT[T2[0]]}_{SHORT[T2[1]]}"] = (mdl2, al2, T2)
        Ya, Yb = qnorm(Y_te[a]), qnorm(Y_te[b])
        Pa, Pb = qnorm(Pp[0]), qnorm(Pp[1])
        dY, dP = Ya - Yb, Pa - Pb
        keep = dY != 0
        thr = np.quantile(np.abs(dY[keep]), 0.75)
        top = keep & (np.abs(dY) >= thr)
        acc_all = float((np.sign(dY[keep]) == np.sign(dP[keep])).mean())
        acc_top = float((np.sign(dY[top]) == np.sign(dP[top])).mean())
        G[f"{SHORT[T2[0]]}_vs_{SHORT[T2[1]]}"] = {
            "sources": [SHORT[m] for m in MODELS if m not in T2],
            "alpha": al2, "te_spearman": sp2,
            "diff_corr_pearson": float(pearsonr(dP, dY)[0]),
            "pair_acc_all": acc_all, "n_all": int(keep.sum()),
            "pair_acc_top25": acc_top, "n_top25": int(top.sum())}
    R["G_leave_two_out"] = G
    print("\n[G] leave-TWO-out — one probe scores BOTH models of the pair (no per-column ridge confound):")
    print(f"    {'held-out pair':>22} | {'r(dP,dY)':>9} {'acc all':>8} {'acc top25':>10}")
    for k, v in G.items():
        print(f"    {k:>22} | {v['diff_corr_pearson']:>+9.3f} {v['pair_acc_all']:>8.3f} {v['pair_acc_top25']:>10.3f}")
    dump()

    # ---- SAVE the multi-model ridge (it had never been persisted) ------------
    ck = save_ridge_bundle(args.ckpt_dir, data, ids, al, fsc, tr, va, te, folds, pair_folds,
                           args.anchor_layer)
    # verify the saved bundle reproduces the in-memory predictions before claiming it works
    bundle = load_ridge_bundle(ck)
    dev = max(float(np.abs(bundle["predict"](m, data[m]["H"][te], f"loo_{SHORT[m]}") - P_te[i]).max())
              for i, m in enumerate(MODELS))
    R["audits"]["ckpt_roundtrip_max_abs_dev"] = dev
    print(f"\n[ckpt] saved multi-model ridge bundle -> {ck}  (reload max abs dev {dev:.2e})")
    dump()

    # ---- robustness: val-z-score normalization instead of qnorm --------------
    mu_y, sd_y = Y_va.mean(axis=1), Y_va.std(axis=1) + 1e-12
    mu_p, sd_p = P_va.mean(axis=1), P_va.std(axis=1) + 1e-12
    Yz = rownorm(Y_te, "z", mu_y, sd_y)
    Pz = rownorm(P_te, "z", mu_p, sd_p)
    R["robustness_valzscore"] = {"residual_corr": cell_corr(resid(Pz), resid(Yz)),
                                 "discordant_top25pct": discordant(Pz, Yz)["top_25pct"]["accuracy"]}
    print("\n[robustness] val-z-score normalization (instead of rank-qnorm): resid corr "
          f"{R['robustness_valzscore']['residual_corr']:+.3f}, "
          f"pair-acc(top25) {R['robustness_valzscore']['discordant_top25pct']:.3f}")
    dump()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
