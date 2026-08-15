"""E33 — is z_aligned worth it GIVEN the text proxy? Paired-bootstrap of the label-free ENSEMBLE
minus q_resp_only-ALONE (SE-fidelity), per target. Reuses E30's exact construction (reference=Llama-2
TBG:22/SLT:15, fit n2000 -> eval fresh n1000). Skips the SEP sweep (not needed).
boot_delta(pa=ensemble, pb=q_resp_only) => metric(ens)-metric(qr); CI excludes 0 => z_aligned adds
signal over text alone. Usage:  python -m amortized_ue.procrustes_e33_ens_vs_qresp [<source_model>]
  (default deepseek-llm-7b-chat; also Mistral-7B-Instruct-v0.2, Meta-Llama-3-8B-Instruct).
`amortized_stage2` env (q_resp needs the frozen proxy). Writes procrustes_e30_ens_vs_qresp_<slug>.json."""
import sys, json
sys.path.append('/vol/bitbucket/mn1025/individual_project/semantic-entropy-probes')
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import roc_auc_score
from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, rho
from amortized_ue.stage2.data import best_split, binarize_entropy
from amortized_ue.procrustes_e27_rank_fusion import arm_preds, ecdf, boot_delta

source = sys.argv[1] if len(sys.argv) > 1 else "deepseek-llm-7b-chat"
ref, dataset = "Llama-2-7b-chat", "trivia_qa"
fit_n, eval_n, tbg, slt = 2000, 1000, 22, 15
SLUG = source.replace("/", "_")

# fit set
sh, s_y, s_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=fit_n), ["TBG","SLT"])
rh, r_y, r_ids = load_matrix(Stage1Config(model_name=ref, dataset=dataset, num_samples=fit_n), ["TBG","SLT"])
assert s_ids == r_ids
tr, va, te = splits(len(s_ids))

# eval set (disjoint fresh n1000)
esh, ey, e_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=eval_n), ["TBG","SLT"])
_,   _,  er_ids = load_matrix(Stage1Config(model_name=ref, dataset=dataset, num_samples=eval_n), ["TBG","SLT"])
assert e_ids == er_ids
eval_ids, eval_rows = e_ids, np.arange(len(e_ids))

# Procrustes W (source->ref, train, no labels) + frozen ref ridge
def fit_at(pos, L):
    m = sh[pos][L][tr].mean(0, keepdims=True); l = rh[pos][L][tr].mean(0, keepdims=True)
    W, _ = orthogonal_procrustes(sh[pos][L][tr]-m, rh[pos][L][tr]-l); return m, l, W
mT,lT,WT = fit_at("TBG",tbg); mS,lS,WS = fit_at("SLT",slt)
R, sc, _, _ = fit_probe(np.concatenate([rh["TBG"][tbg], rh["SLT"][slt]],1), r_y, tr, va)
def zpred(TBG_L, SLT_L):
    a = np.concatenate([(TBG_L-mT)@WT+lT, (SLT_L-mS)@WS+lS],1); return R.predict(sc.transform(a))
z_tr   = zpred(sh["TBG"][tbg], sh["SLT"][slt])[tr]
z_eval = zpred(esh["TBG"][tbg], esh["SLT"][slt])[eval_rows]

# q_resp_only from reference proxy on source records
qr_fit = arm_preds("q_resp_only", source, dataset, fit_n)
qr_tr  = np.array([qr_fit[i] for i in s_ids])[tr]
qr_ev_map = arm_preds("q_resp_only", source, dataset, eval_n)
qr_eval   = np.array([qr_ev_map[i] for i in eval_ids])

# labels on eval
thr = best_split(__import__("torch").tensor(s_y[tr]))
ybe = binarize_entropy(__import__("torch").tensor(ey), thr).numpy()
y_ev, yb_ev = ey[eval_rows], ybe[eval_rows]; v = yb_ev >= 0

# predictors
zs = lambda x, refv: (x-refv.mean())/(refv.std()+1e-8)
cz, cr = ecdf(z_tr), ecdf(qr_tr)
ens_std  = 0.5*(zs(z_eval,z_tr)+zs(qr_eval,qr_tr))
ens_rank = 0.5*(cz(z_eval)+cr(qr_eval))

def M(p): return rho(p,y_ev), roc_auc_score(yb_ev[v], p[v])
print(f"N valid = {int(v.sum())}  split thr={float(thr):.3f}")
print(f"{'predictor':32s}{'Spearman':>10s}{'AUROC':>9s}")
for k,p in [("q_resp_only (alone)",qr_eval),("aligned-z (alone)",z_eval),
            ("avg standardized ENS",ens_std),("RANK FUSION ENS",ens_rank)]:
    spm,au = M(p); print(f"{k:32s}{spm:>+10.3f}{au:>9.3f}")

print("\n--- paired bootstrap: ENSEMBLE - q_resp_only-alone (1000 resamples) ---")
out = {"source": source, "ref": ref, "regime": f"{source} fit n{fit_n} -> eval fresh n{eval_n} (N={int(v.sum())})",
       "q_resp_only_alone": {"spearman": float(M(qr_eval)[0]), "auroc": float(M(qr_eval)[1])}}
for name, ens in [("avg standardized",ens_std),("RANK FUSION",ens_rank)]:
    bd = boot_delta(ens, qr_eval, y_ev, yb_ev, v)
    se = "excludes" if (bd["spearman"][1]>0 or bd["spearman"][2]<0) else "includes"
    ae = "excludes" if (bd["auroc"][1]>0 or bd["auroc"][2]<0) else "includes"
    print(f"Δ({name} ENS - q_resp)  Spearman {bd['spearman'][0]:+.3f} [{bd['spearman'][1]:+.3f},{bd['spearman'][2]:+.3f}] ({se} 0)"
          f"   AUROC {bd['auroc'][0]:+.3f} [{bd['auroc'][1]:+.3f},{bd['auroc'][2]:+.3f}] ({ae} 0)")
    out[f"{name}_ENS_minus_qresp"] = {"spearman": {"mean":bd['spearman'][0],"lo95":bd['spearman'][1],"hi95":bd['spearman'][2]},
                                      "auroc": {"mean":bd['auroc'][0],"lo95":bd['auroc'][1],"hi95":bd['auroc'][2]}}
outp = f"/vol/bitbucket/mn1025/individual_project/semantic-entropy-probes/amortized_ue/procrustes_e30_ens_vs_qresp_{SLUG}.json"
json.dump(out, open(outp,"w"), indent=2)
print(f"\nwrote {outp}")
