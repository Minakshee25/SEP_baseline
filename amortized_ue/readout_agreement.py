"""Diagnostic: once every model's SE readout is carried into a shared basis, do the
readouts rank questions the same way?

Setup (TBG, trivia_qa n2000, one layer throughout):
  anchor = Llama-2-7b-chat. Models = anchor + Mistral-7B-Instruct-v0.2 +
  Meta-Llama-3-8B-Instruct + deepseek-llm-7b-chat, all on the SAME 2000 questions.

  L_a = the anchor's best TBG ridge layer (auto by val Spearman, as the existing code
  does; overridable). W only connects that one layer, so every model's readout is fit at
  layer L_a and every Procrustes map is fit at layer L_a -> anchor L_a.

For each model X:
  - ridge readout f_X : X's TBG[L_a] -> X's SE  (fit_probe, tr/va, alpha on val)
  - orthogonal Procrustes W_X : X's TBG[L_a] -> anchor's TBG[L_a], TRAIN only, NO labels
    (exactly `orthogonal_procrustes(X_centered, anchor_centered)`, the existing recipe).
    Anchor's own W is the identity.

To "carry a readout into the shared (anchor) basis" we map the ONE fixed set of held-out
states -- the anchor's TBG[L_a] test split -- back into X's space via W_X^T (W orthogonal,
so W^{-1}=W^T) and apply f_X. Every readout then scores the identical inputs, and we take
the pairwise Spearman between the resulting prediction vectors (bootstrap CIs).

Two references, computed the same way so they are comparable to the cross-model numbers:
  - ceiling: split ONE model's train set in half, fit two readouts independently, carry
    both through the same W_X, apply to the same anchor test states, compare them.
  - floor:   replace W_X with a random orthogonal matrix (a few seeds), carry the readout
    through it, apply to the same anchor test states, compare to the anchor readout.

Secondary metrics:
  - pairwise cosine between readouts in standardized space (raw-space direction a_X @ W_X,
    rescaled by the anchor per-dim std -> for the anchor this is coef_ itself).
  - per-pair linear CKA between the two models' held-out states at L_a.
  - reconstruction quality before/after W (row cosine + relative Frobenius error) for each
    non-anchor model vs the anchor.

Sanity checks reported at the end: diagonal == 1.0; split-half ceiling exceeds every
cross-model agreement; floor near zero; each readout scores ~0.5-0.65 against its own SE
labels.

Diagnostic only: reuses linear_ceiling_probe + procrustes_alignment helpers read-only,
trains NO pooled ridge, writes one JSON, touches nothing under semantic_uncertainty/.

Run from the repo root in the `se_probes` env (CPU-only):
    python -m amortized_ue.readout_agreement
"""
from __future__ import annotations

import json
import argparse

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.model_selection import train_test_split

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, rho, SEED
from amortized_ue.procrustes_alignment import (
    best_tbg_layer, row_cosine, rel_recon_error, linear_cka)


def cosine(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def boot_rho(pred_a, pred_b, B=2000, seed=0):
    """Bootstrap of Spearman(pred_a, pred_b) over resampled rows -> (mean, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    n = len(pred_a)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        vals[b] = rho(pred_a[idx], pred_b[idx])
    return float(vals.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def readout_direction(model, scaler, W):
    """Raw-space direction of the readout carried into the ANCHOR basis: a_X @ W_X, where
    a_X = coef_ / scale_ is the readout's raw-space direction in X's own space."""
    a = model.coef_ / scaler.scale_        # raw-space direction in X's space
    return a @ W                            # direction in anchor raw space


def carried_pred(model, scaler, W, m_mean, anchor_mean, Z):
    """Apply readout f_X, carried into the anchor basis, to anchor-space states Z:
    map Z back to X's space  x = (Z - anchor_mean) @ W^T + m_mean, then f_X(x)."""
    x = (Z - anchor_mean) @ W.T + m_mean
    return model.predict(scaler.transform(x))


def run(models, anchor, dataset, num_samples, position, layer, boot, out):
    assert models[0] == anchor, "first model must be the anchor"

    # ---- load TBG[all layers] + SE labels + ids for every model; must be the SAME ids ----
    mats, ys, ids0 = {}, {}, None
    for m in models:
        cfg = Stage1Config(model_name=m, dataset=dataset, num_samples=num_samples)
        hidden, y, ids = load_matrix(cfg, [position])
        mats[m], ys[m] = hidden[position], y            # [L+1, N, H], [N]
        if ids0 is None:
            ids0 = ids
        else:
            assert ids == ids0, f"{m} ids differ from {anchor} -- not the same questions"

    N = len(ids0)
    tr, va, te = splits(N)
    min_layers = min(mats[m].shape[0] for m in models)
    H = mats[anchor].shape[2]
    print(f"paired N={N} (train {len(tr)}/val {len(va)}/test {len(te)}), "
          f"min layers across models={min_layers}, H={H}, position={position}")

    # ---- one layer throughout: the anchor's best TBG ridge layer (auto, or forced) ----
    if layer is None:
        L_a, _ = best_tbg_layer(mats[anchor], ys[anchor], tr, va)
    else:
        L_a = layer
    assert L_a < min_layers, (f"anchor layer L_a={L_a} >= min layers {min_layers} "
                              f"(a model is too shallow to align at this layer)")
    print(f"anchor = {anchor}   layer L_a = {L_a} (used for every readout and every W)\n")

    anchor_mean = mats[anchor][L_a][tr].mean(0, keepdims=True)   # anchor train mean
    anchor_std = mats[anchor][L_a][tr].std(0)                    # anchor per-dim std (standardized dirs)
    Z = mats[anchor][L_a][te]                                    # THE fixed held-out states

    # ---- per model: native readout, Procrustes W, carried prediction on Z ----
    per_model = {}
    for m in models:
        X = mats[m][L_a]
        f, sc, alpha, val = fit_probe(X, ys[m], tr, va)
        native = rho(f.predict(sc.transform(X[te])), ys[m][te])  # readout vs its OWN SE labels
        if m == anchor:
            W = np.eye(H)
            m_mean = anchor_mean
        else:
            A = X[tr] - X[tr].mean(0, keepdims=True)
            B = mats[anchor][L_a][tr] - anchor_mean
            W, _ = orthogonal_procrustes(A, B)
            m_mean = X[tr].mean(0, keepdims=True)
        pred = carried_pred(f, sc, W, m_mean, anchor_mean, Z)
        direction = readout_direction(f, sc, W) * anchor_std  # standardized-space dir
        per_model[m] = {"f": f, "sc": sc, "W": W, "m_mean": m_mean, "alpha": float(alpha),
                        "val": float(val), "native_self_spearman": native,
                        "pred": pred, "dir_std": direction}
        print(f"  {m:32s} alpha={alpha:>7.0f}  native-vs-own-SE rho={native:+.3f}")

    # ---- primary: pairwise Spearman between carried prediction vectors ----
    agree, agree_ci = {}, {}
    for a in models:
        for b in models:
            r, lo, hi = boot_rho(per_model[a]["pred"], per_model[b]["pred"], B=boot)
            agree[(a, b)] = r
            agree_ci[(a, b)] = (lo, hi)

    # ---- ceiling: split-half of each model's train, two readouts, same W, on Z ----
    ceiling = {}
    for m in models:
        X = mats[m][L_a]
        h1, h2 = train_test_split(tr, test_size=0.5, random_state=SEED)
        f1, sc1, _, _ = fit_probe(X, ys[m], np.sort(h1), va)
        f2, sc2, _, _ = fit_probe(X, ys[m], np.sort(h2), va)
        W, m_mean = per_model[m]["W"], per_model[m]["m_mean"]
        p1 = carried_pred(f1, sc1, W, m_mean, anchor_mean, Z)
        p2 = carried_pred(f2, sc2, W, m_mean, anchor_mean, Z)
        r, lo, hi = boot_rho(p1, p2, B=boot)
        # direction-cosine ceiling: same model, two half-data readouts, carried through the same W
        # -> the max readout-direction agreement achievable given estimation noise. Compare this to
        # the cross-model cosines: cross ~= ceiling means "same uncertainty direction (up to noise)";
        # cross << ceiling means the directions differ and agreement rides on a shared subspace only.
        d1 = readout_direction(f1, sc1, W) * anchor_std
        d2 = readout_direction(f2, sc2, W) * anchor_std
        ceiling[m] = {"spearman": r, "lo95": lo, "hi95": hi, "cosine": cosine(d1, d2)}

    # ---- floor: replace each non-anchor W with random orthogonal, compare to anchor ----
    floor = {}
    anchor_pred = per_model[anchor]["pred"]
    for m in models:
        if m == anchor:
            continue
        X, f, sc, m_mean = mats[m][L_a], per_model[m]["f"], per_model[m]["sc"], per_model[m]["m_mean"]
        rng = np.random.default_rng(0)
        vals = []
        for _ in range(5):
            Q, _r = np.linalg.qr(rng.standard_normal((H, H)))
            p = carried_pred(f, sc, Q, m_mean, anchor_mean, Z)
            vals.append(rho(p, anchor_pred))
        floor[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    # ---- secondary: cosine of standardized readouts, CKA, reconstruction ----
    cos = {}
    for a in models:
        for b in models:
            cos[(a, b)] = cosine(per_model[a]["dir_std"], per_model[b]["dir_std"])
    cka, recon = {}, {}
    for m in models:
        if m == anchor:
            continue
        Xa = mats[m][L_a][te]
        cka[m] = linear_cka(Xa, mats[anchor][L_a][te])
        s_c = Xa - per_model[m]["m_mean"]
        t_c = mats[anchor][L_a][te] - anchor_mean
        W = per_model[m]["W"]
        recon[m] = {"cosine_before": row_cosine(s_c, t_c), "cosine_after": row_cosine(s_c @ W, t_c),
                    "recon_err_before": rel_recon_error(s_c, t_c),
                    "recon_err_after": rel_recon_error(s_c @ W, t_c)}

    # ---- print tables --------------------------------------------------------
    short = {m: m.split("-")[0][:8] if m != "deepseek-llm-7b-chat" else "deepseek" for m in models}
    short["Meta-Llama-3-8B-Instruct"] = "Llama-3"
    short["Llama-2-7b-chat"] = "Llama-2"
    short["Mistral-7B-Instruct-v0.2"] = "Mistral"
    labels = [short[m] for m in models]

    print("\n" + "=" * 78)
    print(f"READOUT AGREEMENT in the anchor basis ({anchor}, TBG L{L_a})")
    print("=" * 78)
    print("\nPAIRWISE SPEARMAN between carried prediction vectors (diagonal must be 1.000):")
    print("  " + " " * 10 + "".join(f"{l:>10s}" for l in labels))
    for a in models:
        print("  " + f"{short[a]:<10s}" + "".join(f"{agree[(a,b)]:>10.3f}" for b in models))

    print("\n  cross-model agreement vs anchor  (95% CI)   [ceiling, floor for context]:")
    for m in models:
        if m == anchor:
            continue
        r = agree[(anchor, m)]
        lo, hi = agree_ci[(anchor, m)]
        print(f"    {anchor} <-> {short[m]:<8s}  {r:+.3f}  [{lo:+.3f}, {hi:+.3f}]"
              f"   ceil {ceiling[m]['spearman']:.3f}  floor {floor[m]['mean']:+.3f}")

    print("\n  split-half CEILING (same model, two readouts, same W):")
    for m in models:
        c = ceiling[m]
        print(f"    {short[m]:<8s}  {c['spearman']:.3f}  [{c['lo95']:.3f}, {c['hi95']:.3f}]")

    print("\n  random-orthogonal FLOOR (readout vs anchor, mean+-std over 5 Q):")
    for m in models:
        if m == anchor:
            continue
        print(f"    {short[m]:<8s}  {floor[m]['mean']:+.3f} +- {floor[m]['std']:.3f}")

    print("\nSECONDARY -- cosine between standardized readouts in anchor basis:")
    print("  " + " " * 10 + "".join(f"{l:>10s}" for l in labels))
    for a in models:
        print("  " + f"{short[a]:<10s}" + "".join(f"{cos[(a,b)]:>10.3f}" for b in models))

    print("\nDIRECTION-COSINE CEILING -- does 'same ranking' also mean 'same direction'?")
    print("  (cross = readout-direction cosine vs anchor;  ceiling = same-model split-half cosine)")
    anchor_ceil = ceiling[anchor]["cosine"]
    for m in models:
        if m == anchor:
            continue
        cr, ce = cos[(anchor, m)], ceiling[m]["cosine"]
        verdict = ("~= ceiling: SAME direction (up to noise)" if cr >= 0.85 * min(ce, anchor_ceil)
                   else "<< ceiling: directions DIFFER (shared subspace only)")
        print(f"    {short[m]:<8s}  cross {cr:+.3f}   ceiling(self {ce:+.3f}, anchor {anchor_ceil:+.3f})"
              f"   -> {verdict}")

    print("\nSECONDARY -- CKA(model, anchor) at L_a  and  reconstruction before->after W:")
    for m in models:
        if m == anchor:
            continue
        rc = recon[m]
        print(f"    {short[m]:<8s}  CKA {cka[m]:.3f}   cos {rc['cosine_before']:+.3f}->{rc['cosine_after']:+.3f}"
              f"   recon-err {rc['recon_err_before']:.3f}->{rc['recon_err_after']:.3f}")

    # ---- sanity checks -------------------------------------------------------
    print("\n" + "-" * 78)
    print("SANITY CHECKS")
    diag_ok = all(abs(agree[(m, m)] - 1.0) < 1e-9 for m in models)
    print(f"  diagonal == 1.000                         : {'PASS' if diag_ok else 'FAIL'}")
    cross_vals = [agree[(anchor, m)] for m in models if m != anchor]
    min_ceiling = min(ceiling[m]["spearman"] for m in models)
    ceil_ok = min_ceiling > max(cross_vals)
    print(f"  split-half ceiling > every cross-model      : {'PASS' if ceil_ok else 'FAIL'}"
          f"  (min ceiling {min_ceiling:.3f} vs max cross {max(cross_vals):+.3f})")
    floor_ok = all(abs(floor[m]["mean"]) < 0.10 for m in floor)
    print(f"  floor near zero (|rho|<0.10)                : {'PASS' if floor_ok else 'FAIL'}")
    native_ok = all(0.45 <= per_model[m]["native_self_spearman"] <= 0.70 for m in models)
    natives = ", ".join(f"{short[m]} {per_model[m]['native_self_spearman']:.3f}" for m in models)
    print(f"  each readout ~0.5-0.65 vs own SE            : {'PASS' if native_ok else 'CHECK'}  ({natives})")
    print("=" * 78 + "\n")

    # ---- JSON ----------------------------------------------------------------
    def pair_json(d):
        return {f"{a}|{b}": d[(a, b)] for a in models for b in models}
    result = {
        "anchor": anchor, "models": models, "dataset": dataset, "num_samples": num_samples,
        "position": position, "layer_L_a": int(L_a), "n_test": int(len(te)), "boot": boot,
        "native_self_spearman": {m: per_model[m]["native_self_spearman"] for m in models},
        "readout_alpha": {m: per_model[m]["alpha"] for m in models},
        "pairwise_spearman": pair_json(agree),
        "pairwise_spearman_ci95": {f"{a}|{b}": list(agree_ci[(a, b)]) for a in models for b in models},
        "split_half_ceiling": ceiling,
        "random_orthogonal_floor": floor,
        "pairwise_cosine_std": pair_json(cos),
        "cka_vs_anchor": cka,
        "reconstruction_vs_anchor": recon,
        "sanity": {"diagonal_ok": bool(diag_ok), "ceiling_above_cross": bool(ceil_ok),
                   "floor_near_zero": bool(floor_ok), "native_in_range": bool(native_ok)},
    }
    out = out or "amortized_ue/readout_agreement_result.json"
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {out}")
    return result


def _parse():
    p = argparse.ArgumentParser(description="Readout-agreement diagnostic in the anchor basis (CPU).")
    p.add_argument("--models", nargs="+",
                   default=["Llama-2-7b-chat", "Mistral-7B-Instruct-v0.2",
                            "Meta-Llama-3-8B-Instruct", "deepseek-llm-7b-chat"],
                   help="first entry is the anchor")
    p.add_argument("--dataset", default="trivia_qa")
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--position", default="TBG", choices=["TBG", "SLT"])
    p.add_argument("--layer", type=int, default=None,
                   help="force the shared layer L_a (else the anchor's best TBG layer by val Spearman)")
    p.add_argument("--boot", type=int, default=2000, help="bootstrap resamples for the CIs")
    p.add_argument("--out", default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(a.models, a.models[0], a.dataset, a.num_samples, a.position, a.layer, a.boot, a.out)
