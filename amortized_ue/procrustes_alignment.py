"""E24 — Ridge-level Procrustes alignment test (the surgical PRH probe).

Question: E20-E23 show the hidden-state (z) pathway does NOT transfer across models
(raw z transfer ~chance) even though each model's OWN ridge reads its SE fine (~0.62).
Is that because the two hidden spaces are in different *bases* (fixable by an orthogonal
map -> Platonic), or genuinely non-alignable?

Test (TBG only, never SLT; NO SE labels anywhere in the fitting):
  1. Fit an ORTHOGONAL Procrustes map W from Mistral's TBG -> Llama-2's TBG on the shared
     1440 TRAIN questions (both mean-centered on their own train mean).
  2. Translate Mistral's TBG for the 200 held-out TEST ids: aligned = (x - m_mean) @ W + l_mean.
  3. Feed the translated states through Llama-2's FROZEN ridge probe (fit Llama-2 TBG -> SE),
     Spearman-score vs MISTRAL's SE labels.

Three numbers side by side:
  - raw z transfer   (floor  ~0.04, E21): Llama-2 ridge on Mistral TBG, no alignment.
  - aligned transfer (NEW):               Llama-2 ridge on Procrustes-aligned Mistral TBG.
  - native Mistral ridge (skyline ~0.62, E22): Mistral's own ridge on its own TBG.

Plus a reconstruction-quality diagnostic on the held-out PAIRS so a null is interpretable:
  - per-row cosine (ambient) before vs after   -- moves with W
  - relative Frobenius reconstruction error before vs after   -- moves with W
  - linear CKA (orthogonal-INVARIANT, so before==after by construction) -- "are the two
    spaces alignable-by-rotation at all", independent of whether SE survives.

Interpretation:
  aligned ~ skyline  -> spaces are orthogonally alignable; naive transfer failed only on basis.
  aligned ~ floor    -> alignment does NOT recover SE. Then the reconstruction diagnostic says
                        whether even the GEOMETRY aligned (cosine/recon improved, CKA high) but
                        the SE-relevant directions don't survive, or nothing aligned at all.

CPU-only. Additive/read-only: reuses linear_ceiling_probe helpers (load_matrix/splits/
fit_probe/rho) without modifying them; touches nothing under semantic_uncertainty/.

    python -m amortized_ue.procrustes_alignment            # se_probes env, no GPU
"""
from __future__ import annotations

import json
import argparse

import numpy as np
from scipy.linalg import orthogonal_procrustes

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, rho


def best_tbg_layer(hidden_tbg, y, tr, va):
    """Ridge-best TBG layer by val Spearman (the layer that layer's own ridge reads SE best)."""
    best_layer, best_s = 0, -np.inf
    per_layer = []
    for L in range(hidden_tbg.shape[0]):
        m, sc, a, vs = fit_probe(hidden_tbg[L], y, tr, va)
        per_layer.append((L, float(vs)))
        if vs > best_s:
            best_layer, best_s = L, vs
    return best_layer, per_layer


def row_cosine(A, B):
    """Mean per-row cosine similarity between paired rows of A and B."""
    an = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(an * bn, axis=1)))


def rel_recon_error(Ahat, B):
    """Relative Frobenius reconstruction error ||Ahat - B||_F / ||B||_F."""
    return float(np.linalg.norm(Ahat - B) / (np.linalg.norm(B) + 1e-12))


def linear_cka(X, Y):
    """Linear CKA (feature-centered). Invariant to orthogonal transforms of X or Y and to
    isotropic scaling -> before==after an orthogonal Procrustes map, by construction."""
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    hsic_xy = np.linalg.norm(Yc.T @ Xc) ** 2
    hsic_xx = np.linalg.norm(Xc.T @ Xc)
    hsic_yy = np.linalg.norm(Yc.T @ Yc)
    return float(hsic_xy / (hsic_xx * hsic_yy + 1e-12))


def run(source="Mistral-7B-Instruct-v0.2", target="Llama-2-7b-chat",
        dataset="trivia_qa", num_samples=2000, out=None):
    # ---- load paired TBG states + SE labels (join by id, same order) ----------
    scfg = Stage1Config(model_name=source, dataset=dataset, num_samples=num_samples)
    tcfg = Stage1Config(model_name=target, dataset=dataset, num_samples=num_samples)
    s_hidden, s_y, s_ids = load_matrix(scfg, ["TBG"])
    t_hidden, t_y, t_ids = load_matrix(tcfg, ["TBG"])
    assert s_ids == t_ids, "source/target ids differ -- datasets are not the same questions"
    S, T = s_hidden["TBG"], t_hidden["TBG"]                # [L+1, N, H] each
    tr, va, te = splits(len(s_ids))
    n_layers, N, H = S.shape
    print(f"paired N={N} (train {len(tr)}/val {len(va)}/test {len(te)}), layers={n_layers}, H={H}")
    print(f"source={source}  target={target}  (TBG only)\n")

    # ---- pick TBG layers: target's ridge layer L_a (for floor/aligned); source's best L_m (skyline)
    L_a, _ = best_tbg_layer(T, t_y, tr, va)               # target (Llama-2) ridge layer
    L_m, _ = best_tbg_layer(S, s_y, tr, va)               # source (Mistral) skyline layer
    print(f"target ridge TBG layer L_a = {L_a}   source skyline TBG layer L_m = {L_m}")

    # ---- target frozen ridge probe at L_a (fit target TBG -> target SE) --------
    R_L, sc_L, a_L, _ = fit_probe(T[L_a], t_y, tr, va)
    id_target = rho(R_L.predict(sc_L.transform(T[L_a][te])), t_y[te])     # target in-dist (context)

    # ---- SKYLINE: source's own ridge at its best layer L_m --------------------
    R_M, sc_M, a_M, _ = fit_probe(S[L_m], s_y, tr, va)
    skyline = rho(R_M.predict(sc_M.transform(S[L_m][te])), s_y[te])       # vs source SE

    # ---- FLOOR: target ridge on raw source TBG[L_a], no alignment -------------
    floor = rho(R_L.predict(sc_L.transform(S[L_a][te])), s_y[te])         # vs source SE

    # ---- Procrustes W: source TBG[L_a] -> target TBG[L_a], TRAIN only, NO labels
    m_mean = S[L_a][tr].mean(0, keepdims=True)             # source train mean
    l_mean = T[L_a][tr].mean(0, keepdims=True)             # target train mean
    A = S[L_a][tr] - m_mean                                # source centered (train)
    B = T[L_a][tr] - l_mean                                # target centered (train)
    W, scale = orthogonal_procrustes(A, B)                 # A @ W ~= B, W orthogonal

    # ---- ALIGNED transfer: translate source TEST TBG into target space --------
    aligned_te = (S[L_a][te] - m_mean) @ W + l_mean
    aligned = rho(R_L.predict(sc_L.transform(aligned_te)), s_y[te])       # vs source SE

    # ---- CONTROLS (rule out artifacts) ----------------------------------------
    # (a) mean-shift ONLY, no rotation: isolates rotation vs a trivial offset.
    meanshift_te = S[L_a][te] - m_mean + l_mean
    ctrl_meanshift = rho(R_L.predict(sc_L.transform(meanshift_te)), s_y[te])
    # (b) RANDOM orthogonal in place of W: must stay near the floor (else the ridge reads
    #     source regardless of the map -> artifact). Average over a few seeds.
    rng = np.random.default_rng(0); rand_scores = []
    for _ in range(5):
        Q, _r = np.linalg.qr(rng.standard_normal((H, H)))
        rand_te = (S[L_a][te] - m_mean) @ Q + l_mean
        rand_scores.append(rho(R_L.predict(sc_L.transform(rand_te)), s_y[te]))
    ctrl_random = (float(np.mean(rand_scores)), float(np.std(rand_scores)))

    # ---- reconstruction diagnostic on held-out PAIRS (centered, at L_a) -------
    src_te_c = S[L_a][te] - m_mean
    tgt_te_c = T[L_a][te] - l_mean
    src_aligned_c = src_te_c @ W                           # aligned, still centered
    diag = {
        "cosine_before": row_cosine(src_te_c, tgt_te_c),
        "cosine_after": row_cosine(src_aligned_c, tgt_te_c),
        "recon_err_before": rel_recon_error(src_te_c, tgt_te_c),
        "recon_err_after": rel_recon_error(src_aligned_c, tgt_te_c),
        "cka": linear_cka(src_te_c, tgt_te_c),             # orthogonal-invariant (before==after)
    }

    # ---- report ---------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"PROCRUSTES ALIGNMENT (TBG only): {source} -> {target} ridge, vs {source} SE (N_test={len(te)})")
    print("=" * 74)
    print(f"  raw z transfer   (floor)   : {floor:+.3f}   [target ridge on raw source TBG L{L_a}]")
    print(f"  ctrl: mean-shift only      : {ctrl_meanshift:+.3f}   [no rotation -- must stay ~floor]")
    print(f"  ctrl: random orthogonal    : {ctrl_random[0]:+.3f} ± {ctrl_random[1]:.3f}   [must stay ~floor]")
    print(f"  aligned transfer (NEW)     : {aligned:+.3f}   [+ LEARNED orthogonal Procrustes]")
    print(f"  native source ridge (sky)  : {skyline:+.3f}   [source ridge on own TBG L{L_m}]")
    print(f"  (target in-dist ridge L{L_a})  : {id_target:+.3f}   [context]")
    recovered = (aligned - floor) / (skyline - floor) if skyline > floor else float("nan")
    print(f"  -> fraction of floor->skyline gap recovered by alignment: {recovered:.1%}")
    print("-" * 74)
    print("  reconstruction on held-out pairs (centered TBG L%d):" % L_a)
    print(f"    per-row cosine   before {diag['cosine_before']:+.3f}  ->  after {diag['cosine_after']:+.3f}")
    print(f"    rel recon error  before {diag['recon_err_before']:.3f}  ->  after {diag['recon_err_after']:.3f}")
    print(f"    linear CKA (orthogonal-invariant, before==after): {diag['cka']:.3f}")
    print("=" * 74 + "\n")

    result = {
        "source": source, "target": target, "dataset": dataset, "num_samples": num_samples,
        "n_test": len(te), "L_a_target_ridge": int(L_a), "L_m_source_skyline": int(L_m),
        "raw_z_transfer_floor": floor, "aligned_transfer": aligned,
        "ctrl_meanshift_only": ctrl_meanshift, "ctrl_random_orthogonal": ctrl_random,
        "native_source_ridge_skyline": skyline, "target_in_dist_ridge": id_target,
        "gap_recovered_frac": float(recovered), "reconstruction": diag,
        "alpha_target_ridge": float(a_L), "alpha_source_ridge": float(a_M),
    }
    out = out or "amortized_ue/procrustes_alignment_result.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}")
    return result


def _parse():
    p = argparse.ArgumentParser(description="E24 ridge-level Procrustes alignment test (CPU).")
    p.add_argument("--source", default="Mistral-7B-Instruct-v0.2")
    p.add_argument("--target", default="Llama-2-7b-chat")
    p.add_argument("--dataset", default="trivia_qa")
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--out", default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(a.source, a.target, a.dataset, a.num_samples, a.out)
