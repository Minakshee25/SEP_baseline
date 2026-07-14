"""Control for the text-only arms: does a TRIVIAL text model match the 3B?

E12 found the text-only arms surprisingly strong (q_only ID Spearman 0.494 with NO
target-LLM forward pass at all). Before claiming the SLM has a unique capability, rule out
the obvious shortcut: that SE is predictable from surface properties of the question
("long / rare / ambiguous questions are hard"), which a bag-of-words model would capture
just as well.

This is the same discipline that caught the retracted text-arm findings (E6 -> E8):
**establish the cheap exact baseline before believing the deep model.**

Baselines, all on the identical split and metric as Stage 2:
  1. TF-IDF (word + char n-grams) -> ridge         [the real control]
  2. question LENGTH alone -> the crudest shortcut  [sanity floor]

Read against E12:
  TF-IDF ~= 0.494  -> the 3B adds NOTHING over bag-of-words; the "capability" is a surface
                      heuristic and the novelty claim collapses.
  TF-IDF <<  0.494  -> the 3B is reading something semantic that n-grams cannot; the
                      text-only result is real.

Run from the repo root in the `se_probes` env (CPU only):
    python -m amortized_ue.text_baseline_probe
"""
from __future__ import annotations

import json
import argparse
import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_union
from sklearn.model_selection import train_test_split

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records

warnings.filterwarnings("ignore")
SEED, TEST_SIZE, VAL_SIZE = 42, 0.1, 0.2
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


def load(cfg):
    recs = load_records(cfg)
    ids = sorted(recs.keys())                        # same ordering as Stage2Data
    q = [recs[i]["question"] for i in ids]
    r = [recs[i]["canonical"]["response"] for i in ids]
    y = np.array([recs[i]["labels"]["cluster_assignment_entropy"] for i in ids], dtype=np.float32)
    return q, r, y


def rho(a, b) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    s = spearmanr(a, b).correlation
    return 0.0 if (s is None or np.isnan(s)) else float(s)


def splits(n):
    idx = np.arange(n)
    tv, te = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED)
    tr, va = train_test_split(tv, test_size=VAL_SIZE, random_state=SEED)
    return np.sort(tr), np.sort(va), np.sort(te)


def tfidf_union():
    """Word 1-2 grams + char 3-5 grams — a strong, standard bag-of-words text featuriser."""
    return make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
    )


def run(name, texts, ood_texts, y, oy, tr, va, te):
    vec = tfidf_union().fit([texts[i] for i in tr])
    Xtr, Xva, Xte = (vec.transform([texts[i] for i in s]) for s in (tr, va, te))
    XO = vec.transform(ood_texts)
    best = (-np.inf, None)
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(Xtr, y[tr])
        s = rho(m.predict(Xva), y[va])
        if s > best[0]:
            best = (s, m)
    m = best[1]
    id_s, ood_s = rho(m.predict(Xte), y[te]), rho(m.predict(XO), oy)
    print(f"  {name:<34}{Xtr.shape[1]:>8d} feats  val={best[0]:.3f}  ID={id_s:.3f}  OOD={ood_s:.3f}")
    return id_s, ood_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Llama-2-7b-chat")
    p.add_argument("--dataset", default="trivia_qa")
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--ood_dataset", default="squad")
    p.add_argument("--ood_num_samples", type=int, default=1000)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    q, r, y = load(Stage1Config(model_name=a.model_name, dataset=a.dataset,
                                num_samples=a.num_samples))
    oq, orr, oy = load(Stage1Config(model_name=a.model_name, dataset=a.ood_dataset,
                                    num_samples=a.ood_num_samples))
    tr, va, te = splits(len(y))
    print(f"\nID {a.dataset} n={len(y)} split {len(tr)}/{len(va)}/{len(te)} | "
          f"OOD {a.ood_dataset} n={len(oy)} (all rows)")
    print("\nTF-IDF + ridge on the TEXT only — the control for the 3B text-only arms")
    print("  3B reference (E12): q_only ID 0.494±0.049 / OOD 0.259 | "
          "q_resp_only ID 0.521±0.049 / OOD 0.399\n")

    res = {}
    res["q_only"] = run("TF-IDF(question)  [vs q_only]", q, oq, y, oy, tr, va, te)
    qr = [f"{a_} {b_}" for a_, b_ in zip(q, r)]
    oqr = [f"{a_} {b_}" for a_, b_ in zip(oq, orr)]
    res["q_resp_only"] = run("TF-IDF(question+answer)  [vs q_resp_only]", qr, oqr, y, oy, tr, va, te)

    # crudest possible shortcut: is SE just a function of how long the question is?
    qlen = np.array([len(s.split()) for s in q], dtype=np.float32)
    oqlen = np.array([len(s.split()) for s in oq], dtype=np.float32)
    print(f"\n  {'question LENGTH alone (sanity floor)':<34}{1:>8d} feat   "
          f"           ID={rho(qlen[te], y[te]):.3f}  OOD={rho(oqlen, oy):.3f}")
    res["length_only"] = (rho(qlen[te], y[te]), rho(oqlen, oy))

    print("\n  VERDICT: if TF-IDF ~= the 3B, the text-only 'capability' is a bag-of-words")
    print("           shortcut and the novelty claim collapses. If TF-IDF is well below,")
    print("           the 3B is reading something semantic that n-grams cannot.\n")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({k: {"id": v[0], "ood": v[1]} for k, v in res.items()}, f, indent=1)
        print(f"  wrote {a.out}\n")


if __name__ == "__main__":
    main()
