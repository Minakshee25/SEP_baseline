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


def bootstrap_diff(pred_a, pred_b, y, B=2000, seed=0):
    """Paired bootstrap of rho(pred_a,y) - rho(pred_b,y) over resampled rows.
    Returns (mean_diff, lo95, hi95, frac_positive)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        diffs[b] = rho(pred_a[idx], y[idx]) - rho(pred_b[idx], y[idx])
    return (float(diffs.mean()), float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)), float(np.mean(diffs > 0)))


def run(source="Mistral-7B-Instruct-v0.2", target="Llama-2-7b-chat",
        dataset="trivia_qa", num_samples=2000, fresh_num_samples=None, out=None,
        position="TBG", source_layer=None, target_layer=None):
    # `position` (default "TBG" -> exact E24 behaviour) selects which stored position to align on;
    # `source_layer`/`target_layer` optionally FORCE the skyline / ridge layer (else auto by val
    # Spearman). Additive: with the defaults this reproduces the original TBG-auto E24/E25 run.
    # ---- load paired states at `position` + SE labels (join by id, same order) ----------
    scfg = Stage1Config(model_name=source, dataset=dataset, num_samples=num_samples)
    tcfg = Stage1Config(model_name=target, dataset=dataset, num_samples=num_samples)
    s_hidden, s_y, s_ids = load_matrix(scfg, [position])
    t_hidden, t_y, t_ids = load_matrix(tcfg, [position])
    assert s_ids == t_ids, "source/target ids differ -- datasets are not the same questions"
    S, T = s_hidden[position], t_hidden[position]         # [L+1, N, H] each
    tr, va, te = splits(len(s_ids))
    n_layers, N, H = S.shape
    print(f"paired N={N} (train {len(tr)}/val {len(va)}/test {len(te)}), layers={n_layers}, H={H}")
    print(f"source={source}  target={target}  (position={position})\n")

    # ---- pick layers: target's ridge layer L_a (for floor/aligned); source's best L_m (skyline)
    L_a = target_layer if target_layer is not None else best_tbg_layer(T, t_y, tr, va)[0]  # target ridge
    L_m = source_layer if source_layer is not None else best_tbg_layer(S, s_y, tr, va)[0]  # source skyline
    print(f"target ridge {position} layer L_a = {L_a}   source skyline {position} layer L_m = {L_m}")

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

    # ---- eval block: all metrics on one eval set (src/tgt paired, at L_a; src also L_m) ----
    def eval_block(src_La, src_Lm, tgt_La, sy, label):
        pred_floor = R_L.predict(sc_L.transform(src_La))                    # raw source -> target ridge
        pred_control = R_L.predict(sc_L.transform(tgt_La))                  # Mech-A: TARGET's OWN states
        aligned_in = (src_La - m_mean) @ W + l_mean
        pred_aligned = R_L.predict(sc_L.transform(aligned_in))             # aligned source -> target ridge
        pred_skyline = R_M.predict(sc_M.transform(src_Lm))                  # source's own ridge
        floor_, control_, aligned_, skyline_ = (rho(pred_floor, sy), rho(pred_control, sy),
                                                rho(pred_aligned, sy), rho(pred_skyline, sy))
        # random-orthogonal control (must stay ~floor)
        rng = np.random.default_rng(0); rand = []
        for _ in range(5):
            Q, _r = np.linalg.qr(rng.standard_normal((H, H)))
            rand.append(rho(R_L.predict(sc_L.transform((src_La - m_mean) @ Q + l_mean)), sy))
        rand_ = (float(np.mean(rand)), float(np.std(rand)))
        # the mechanism test: paired bootstrap of (aligned - control)
        boot = bootstrap_diff(pred_aligned, pred_control, sy)
        recovered = (aligned_ - floor_) / (skyline_ - floor_) if skyline_ > floor_ else float("nan")
        # reconstruction on the paired held-out states (centered at L_a)
        s_c, t_c = src_La - m_mean, tgt_La - l_mean
        recon = {"cosine_before": row_cosine(s_c, t_c), "cosine_after": row_cosine(s_c @ W, t_c),
                 "recon_err_before": rel_recon_error(s_c, t_c), "recon_err_after": rel_recon_error(s_c @ W, t_c),
                 "cka": linear_cka(s_c, t_c)}
        return {"label": label, "n": len(sy), "floor": floor_, "control_mechA": control_,
                "aligned": aligned_, "skyline": skyline_, "target_in_dist": None,
                "ctrl_random_orthogonal": rand_, "gap_recovered_frac": float(recovered),
                "aligned_minus_control": {"mean": boot[0], "lo95": boot[1], "hi95": boot[2],
                                          "frac_positive": boot[3]}, "reconstruction": recon}

    blocks = [eval_block(S[L_a][te], S[L_m][te], T[L_a][te], s_y[te], f"N={len(te)} (n{num_samples} test split)")]

    # ---- optional: evaluate the SAME fitted W/ridges on the E23 fresh batch (tighter CIs) ----
    if fresh_num_samples:
        fscfg = Stage1Config(model_name=source, dataset=dataset, num_samples=fresh_num_samples)
        ftcfg = Stage1Config(model_name=target, dataset=dataset, num_samples=fresh_num_samples)
        fs, fsy, fs_ids = load_matrix(fscfg, [position])
        ft, fty, ft_ids = load_matrix(ftcfg, [position])
        assert fs_ids == ft_ids, "fresh source/target ids differ"
        Sf, Tf = fs[position], ft[position]
        blocks.append(eval_block(Sf[L_a], Sf[L_m], Tf[L_a], fsy, f"N={len(fs_ids)} (fresh n{fresh_num_samples})"))

    # ---- report side by side --------------------------------------------------
    print("\n" + "=" * 82)
    print(f"PROCRUSTES + MECHANISM-A CONTROL (position={position}): {source} -> {target}, vs {source} SE")
    print(f"  target ridge L_a={L_a}   source skyline L_m={L_m}   target in-dist(context)={id_target:+.3f}")
    print("=" * 82)
    hdr = "  {:34s}".format("") + "".join(f"{b['label']:>24s}" for b in blocks)
    print(hdr)
    def row(name, key, fmt="{:+.3f}"):
        print("  {:34s}".format(name) + "".join(f"{fmt.format(b[key]):>24s}" for b in blocks))
    row("raw z transfer (floor)", "floor")
    row("control: target OWN states (Mech-A)", "control_mechA")
    row("aligned transfer (learned W)", "aligned")
    row("native source ridge (skyline)", "skyline")
    print("  {:34s}".format("gap floor->skyline recovered")
          + "".join(f"{b['gap_recovered_frac']:>23.1%} " for b in blocks))
    print("  " + "-" * 80)
    print("  MECHANISM TEST  aligned - control  (paired bootstrap, 95% CI):")
    for b in blocks:
        d = b["aligned_minus_control"]
        sep = "SEPARATED (Mistral-specific)" if d["lo95"] > 0 else "overlaps 0 (shared-difficulty only)"
        print(f"    {b['label']:28s}  {d['mean']:+.3f}  [{d['lo95']:+.3f}, {d['hi95']:+.3f}]  P(>0)={d['frac_positive']:.2f}  -> {sep}")
    print("  " + "-" * 80)
    print("  reconstruction (per-row cosine before->after | CKA):")
    for b in blocks:
        r = b["reconstruction"]
        print(f"    {b['label']:28s}  cos {r['cosine_before']:+.3f}->{r['cosine_after']:+.3f}   CKA {r['cka']:.3f}")
    print("=" * 82 + "\n")

    result = {"source": source, "target": target, "dataset": dataset, "num_samples": num_samples,
              "fresh_num_samples": fresh_num_samples, "position": position, "L_a_target_ridge": int(L_a),
              "L_m_source_skyline": int(L_m), "target_in_dist_ridge": id_target,
              "alpha_target_ridge": float(a_L), "alpha_source_ridge": float(a_M), "evals": blocks}
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
    p.add_argument("--fresh_num_samples", type=int, default=None,
                   help="also evaluate the SAME fitted W/ridges on a disjoint fresh batch of this "
                        "size (e.g. 1000 = the E23 held-out batch) for tighter CIs")
    p.add_argument("--out", default=None)
    p.add_argument("--position", default="TBG", choices=["TBG", "SLT"],
                   help="stored position to align on (default TBG = original E24 behaviour)")
    p.add_argument("--source_layer", type=int, default=None,
                   help="force the source skyline layer L_m (else auto by val Spearman)")
    p.add_argument("--target_layer", type=int, default=None,
                   help="force the target ridge/alignment layer L_a (else auto by val Spearman)")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(a.source, a.target, a.dataset, a.num_samples, a.fresh_num_samples, a.out,
        a.position, a.source_layer, a.target_layer)
