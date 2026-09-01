"""E61 (additive) — efficiency analysis to accompany the RQ1 latency benchmark.

Does NOT touch `rq1_latency.py` or its result JSONs. Writes only `results/e61_efficiency_*.json`.

Adds, for the SAME 200 held-out test questions per target (Llama-2-7b-chat, Mistral-7B-Instruct-v0.2)
that `rq1_latency.py` used (identical `Stage2Data` split: test_size=0.1, seed 42, id-sorted n2000):

  1. exact mean INPUT and GENERATED token counts / question
       - input  = the full Stage-1 prompt (few-shot prefix + brief + context + question), tokenized
                  with the TARGET tokenizer (`build_prompts` reconstructs it verbatim);
       - generated (canonical) = len(record.canonical.token_log_likelihoods);
       - generated (10 samples) = mean over the 10 stored samples of len(token_log_likelihoods).
       token_log_likelihoods has exactly one entry per generated token (Stage-1 scores every
       generated token), so its length IS the generated-token count — no re-generation needed.
  2. target-LLM generations / question: baseline vs proposed.
  3. proxy input-token count (q_resp_only text, Llama-3.2 tokenizer), untruncated + at max_seq_len.
  4. DeBERTa entailment forward passes / question + mean pair token length, obtained by REPLAYING
     `get_semantic_ids` on the stored `semantic_id`s (the algorithm's branching depends only on
     pairwise-equivalence outcomes, which the stored ids give exactly) — no model inference.
  5. estimated FLOPs / question for canonical gen, 10-sample gen, DeBERTa clustering, proxy forward,
     and both complete pipelines. Formula + assumptions documented below and echoed into the JSON.
  6. (--stage proxy) proxy latency FORWARD-ONLY vs TOKENIZER+FORWARD, and peak GPU memory for the
     proxy forward + (optionally) the DeBERTa clustering pass. Target-generation peak memory is not
     cleanly measurable under current GPU contention — the fp32 parameter-memory lower bound is
     reported instead, alongside E61's empirically observed ~33 GB.

Stages:
    # CPU only, both targets in one shot (needs the target + Llama-3.2 tokenizers, all cached):
    python -m amortized_ue.e61_efficiency --stage tokens --data_dir /data2/mn1025/stage1
    # GPU (amortized_stage2), short borrow — proxy latency variants + peak memory:
    python -m amortized_ue.e61_efficiency --stage proxy  --data_dir /data2/mn1025/stage1

`--stage all` runs both (needs the proxy env + a GPU). The two stages merge into the same JSON.
"""
from __future__ import annotations

import os
import json
import time
import argparse
import datetime

import numpy as np

from amortized_ue.rq1_latency import resolve_test_ids, build_prompts, _qr_text

# ----------------------------------------------------------------------------------------------
# FLOPs model — ESTIMATED. Documented here, echoed into the output JSON under "flops_assumptions".
# ----------------------------------------------------------------------------------------------
#   Dense transformer forward  ≈  2 · P · T   FLOPs
#     P = total parameter count, T = number of tokens processed.
#     "2" = one multiply + one add per parameter per token (Kaplan et al. 2020; Chinchilla).
#     This counts every weight once per token; the input-embedding lookup is a gather (~0 FLOPs)
#     so 2·P·T marginally (~2-4 %) over-counts generation. Left in — it is an upper estimate.
#   Attention QK^T + (·V):  ≈ 4 · L · T² · H  total (L layers).  For our T ≲ 250 and a 7B model
#     this is < 1 % of 2·P·T — reported as `attention_term_fraction`, NOT added to the totals.
#   Autoregressive generation of G tokens from a T_in-token prompt, WITH a KV cache:
#     prefill 2·P·T_in  +  G decode steps × 2·P·1  =  2·P·(T_in + G).
#   Stage-1 samples are generated INDEPENDENTLY (no prompt-KV sharing across the 10 samples),
#     so 10-sample gen = 10 × 2·P_target·(T_in + G_sample).   [caveat: an optimised sampler would
#     share the T_in prefill once → up to ~10× less prefill; not what Stage-1 does.]
#   DeBERTa clustering = n_fwd · 2·P_deberta·T_pair   (encoder-only, one bidirectional pass;
#     n_fwd and T_pair measured per question).
#   Proxy forward = 2·P_proxy·T_proxy   (single forward, regression head — no LM head, no vocab
#     projection; LoRA r16 adds ~0.5 % params, folded in).
#
# Parameter counts (total, from HF configs):
P = {
    "Llama-2-7b-chat":            6.738e9,
    "Mistral-7B-Instruct-v0.2":   7.241e9,
    "deberta-v2-xlarge-mnli":     0.886e9,
    "proxy-Llama-3.2-3B":         3.213e9,
}

OUT_TMPL = "amortized_ue/results/e61_efficiency_{target}.json"
DEPLOY_CKPT = "amortized_ue/results/deploy_checkpoints/deploy_q_resp_only_seed0.pt"
TARGET_TOKENIZER = {
    "Llama-2-7b-chat": "NousResearch/Llama-2-7b-chat-hf",
    "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
}


def _stats(a):
    a = np.asarray(a, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


# ----------------------------------------------------------------------------------------------
# replay get_semantic_ids to count entailment forward passes (no model inference)
# ----------------------------------------------------------------------------------------------
def replay_semantic_id_calls(semantic_ids):
    """Faithful replay of semantic_entropy.get_semantic_ids's control flow given the final ids.
    Returns the number of are_equivalent() calls; each = 2 DeBERTa forward passes."""
    n = len(semantic_ids)
    assigned = [False] * n
    are_equiv_calls = 0
    for i in range(n):
        if not assigned[i]:
            assigned[i] = True
            for j in range(i + 1, n):
                are_equiv_calls += 1                      # are_equivalent(i, j) is called unconditionally
                if semantic_ids[j] == semantic_ids[i]:
                    assigned[j] = True
    return are_equiv_calls


# ----------------------------------------------------------------------------------------------
# stage: tokens  (CPU)
# ----------------------------------------------------------------------------------------------
def stage_tokens(target, dataset, num_samples, data_dir, result):
    from transformers import AutoTokenizer

    test_ids, records = resolve_test_ids(target, dataset, num_samples, data_dir)
    prompts, sep_args = build_prompts(target, dataset, num_samples, test_ids)
    id2prompt = {tid: p for tid, _ex, p in prompts}
    id2ex = {tid: ex for tid, ex, _p in prompts}

    ttok = AutoTokenizer.from_pretrained(TARGET_TOKENIZER[target])
    ptok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    max_seq_len = 256   # ProxyModel.cfg.max_seq_len (deploy checkpoint)

    in_toks, g_canon, g_samp = [], [], []
    deb_calls, deb_pair_toks = [], []
    proxy_toks, proxy_toks_capped = [], []
    n_clusters = []

    for tid in test_ids:
        rec = records[tid]
        in_toks.append(len(ttok(id2prompt[tid], add_special_tokens=True)["input_ids"]))
        g_canon.append(len(rec["canonical"]["token_log_likelihoods"] or []))
        gl = [len(s["token_log_likelihoods"] or []) for s in rec["samples"]]
        g_samp.append(float(np.mean(gl)))

        sids = [s["semantic_id"] for s in rec["samples"]]
        n_clusters.append(len(set(sids)))
        calls = replay_semantic_id_calls(sids)
        deb_calls.append(calls)
        # pair text exactly as rq1_latency._one_B builds it: "{question} {response}" both sides
        q = id2ex[tid]["question"]
        cluster_inputs = [f"{q} {s['response']}" for s in rec["samples"]]
        pair_lens = []
        n = len(sids); assigned = [False] * n
        for i in range(n):
            if not assigned[i]:
                assigned[i] = True
                for j in range(i + 1, n):
                    L = len(ttok(cluster_inputs[i], cluster_inputs[j])["input_ids"])
                    pair_lens.append(L)
                    if sids[j] == sids[i]:
                        assigned[j] = True
        if pair_lens:
            deb_pair_toks.append(float(np.mean(pair_lens)))

        qr = _qr_text(rec["question"], rec["canonical"]["response"])
        full = ptok(qr, add_special_tokens=False)["input_ids"]
        proxy_toks.append(len(full))
        proxy_toks_capped.append(min(len(full), max_seq_len))

    tok = {
        "n_questions": len(test_ids),
        "input_tokens_per_question": _stats(in_toks),
        "generated_tokens_canonical": _stats(g_canon),
        "generated_tokens_per_sample": _stats(g_samp),
        "model_max_new_tokens": sep_args.model_max_new_tokens,
        "num_generations": sep_args.num_generations,
        "deberta_forward_passes_per_question": _stats([2 * c for c in deb_calls]),
        "are_equivalent_calls_per_question": _stats(deb_calls),
        "deberta_pair_token_len": _stats(deb_pair_toks),
        "n_clusters_per_question": _stats(n_clusters),
        "proxy_input_tokens": _stats(proxy_toks),
        "proxy_input_tokens_capped_at_max_seq_len": _stats(proxy_toks_capped),
        "max_seq_len": max_seq_len,
        "target_generations_per_question": {
            "baseline": 1 + sep_args.num_generations,
            "proposed": 1,
            "note": "baseline = 1 canonical (low-temp) + 10 high-temp samples; proposed = 1 canonical "
                    "(the proxy consumes its answer text) + 0 extra target generations.",
        },
    }
    result["tokens"] = tok

    # ---- estimated FLOPs / question -------------------------------------------------------------
    Pt = P[target]; Pd = P["deberta-v2-xlarge-mnli"]; Pp = P["proxy-Llama-3.2-3B"]
    Tin = tok["input_tokens_per_question"]["mean"]
    Gc = tok["generated_tokens_canonical"]["mean"]
    Gs = tok["generated_tokens_per_sample"]["mean"]
    nfwd = tok["deberta_forward_passes_per_question"]["mean"]
    Tpair = tok["deberta_pair_token_len"]["mean"]
    Tproxy = tok["proxy_input_tokens"]["mean"]

    f_canon = 2 * Pt * (Tin + Gc)
    f_samples = tok["num_generations"] * 2 * Pt * (Tin + Gs)
    f_cluster = nfwd * 2 * Pd * Tpair
    f_proxy = 2 * Pp * Tproxy
    f_base = f_canon + f_samples + f_cluster
    f_prop = f_canon + f_proxy

    # attention-term fraction (reported, not added), using canonical-gen seq len as representative
    cfg_L = {"Llama-2-7b-chat": 32, "Mistral-7B-Instruct-v0.2": 32}[target]
    cfg_H = 4096
    Tseq = Tin + Gc
    attn_term = 4 * cfg_L * (Tseq ** 2) * cfg_H
    attn_frac = attn_term / f_canon

    result["flops_estimated"] = {
        "_label": "ESTIMATED FLOPs (2·P·T dense-transformer model; see flops_assumptions)",
        "per_question": {
            "canonical_generation": f_canon,
            "ten_sample_generation": f_samples,
            "deberta_clustering": f_cluster,
            "proxy_forward": f_proxy,
            "baseline_pipeline_total": f_base,
            "proposed_pipeline_total": f_prop,
        },
        "ratios": {
            "baseline_total / proposed_total": f_base / f_prop,
            "ten_sample_gen / proxy_forward": f_samples / f_proxy,
            "(samples+clustering) / proxy_forward": (f_samples + f_cluster) / f_proxy,
        },
        "attention_term_fraction_of_canonical_gen": attn_frac,
        "inputs_used": {"P_target": Pt, "P_deberta": Pd, "P_proxy": Pp,
                        "T_in": Tin, "G_canonical": Gc, "G_sample": Gs,
                        "deberta_fwd_per_q": nfwd, "T_pair": Tpair, "T_proxy": Tproxy},
    }
    result["flops_assumptions"] = [
        "Dense transformer forward = 2 * P * T FLOPs (P = total params, T = tokens processed).",
        "Generation of G tokens from a T_in prompt with KV cache = 2 * P * (T_in + G).",
        "10-sample generation = 10 * 2 * P_target * (T_in + G_sample): Stage-1 generates each "
        "sample independently, re-running the T_in prefill every time (no prefix-KV sharing).",
        "DeBERTa clustering = n_fwd * 2 * P_deberta * T_pair; n_fwd = 2 * are_equivalent() calls, "
        "both measured by replaying get_semantic_ids on the stored semantic_ids.",
        "Proxy forward = 2 * P_proxy * T_proxy (regression head; no LM/vocab projection).",
        "Attention QK^T/(.V) term (~4*L*T^2*H) is reported as a fraction, not added (<1%).",
        "Input-embedding lookup treated as ~0 FLOPs (gather); 2*P*T is therefore a slight upper "
        "bound on generation cost.",
        "Parameter counts are TOTAL (incl. embeddings) from the HF configs.",
    ]
    return result


# ----------------------------------------------------------------------------------------------
# stage: proxy  (GPU, amortized_stage2)
# ----------------------------------------------------------------------------------------------
def stage_proxy(target, dataset, num_samples, data_dir, deploy_ckpt, warmup, result):
    import torch
    from amortized_ue.stage2.checkpoint import load_checkpoint

    device = "cuda"
    assert torch.cuda.is_available()
    test_ids, records = resolve_test_ids(target, dataset, num_samples, data_dir)

    model, meta, _t = load_checkpoint(deploy_ckpt, device=device)
    model.eval()
    tok = model.tokenizer
    max_len = model.cfg.max_seq_len

    texts = [_qr_text(records[i]["question"], records[i]["canonical"]["response"]) for i in test_ids]
    pretok = [tok(t, add_special_tokens=False)["input_ids"][:max_len] for t in texts]

    def fwd(ids):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        a = torch.ones_like(x)
        with torch.no_grad():
            return model(None, x, a)

    def tok_fwd(text):
        ids = tok(text, add_special_tokens=False)["input_ids"][:max_len]
        return fwd(ids)

    for i in range(warmup):
        fwd(pretok[i % len(pretok)])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    fo = []
    for ids in pretok:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fwd(ids)
        torch.cuda.synchronize(); fo.append(time.perf_counter() - t0)
    peak_fwd = torch.cuda.max_memory_allocated()

    for i in range(warmup):
        tok_fwd(texts[i % len(texts)])
    torch.cuda.synchronize()
    tf = []
    for t in texts:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        tok_fwd(t)
        torch.cuda.synchronize(); tf.append(time.perf_counter() - t0)

    prox = {
        "n_questions": len(test_ids),
        "gpu_name": torch.cuda.get_device_name(0),
        "dtype": "bf16 backbone (fp32 projector/head)",
        "forward_only_ms": {k: v * 1000 for k, v in _stats(fo).items()},
        "tokenize_plus_forward_ms": {k: v * 1000 for k, v in _stats(tf).items()},
        "tokenizer_overhead_ms_mean": (np.mean(tf) - np.mean(fo)) * 1000,
        "peak_gpu_mem_forward_bytes": int(peak_fwd),
        "peak_gpu_mem_forward_MiB": peak_fwd / 2**20,
    }
    result.setdefault("proxy_latency", {}).update(prox)

    result["target_generation_peak_mem"] = {
        "note": "not cleanly measurable under current GPU contention (both cards ~41/46 GB used by "
                "Stage-1 data-gen). Reported: fp32 parameter-memory lower bound + E61's observed value.",
        "fp32_param_bytes": int(4 * P[target]),
        "fp32_param_GiB": 4 * P[target] / 2**30,
        "e61_observed_peak_GiB_fp32_7B": 33.0,
    }
    return result


RQ1_TMPL = "amortized_ue/results/rq1_latency_{target}.json"
SUMMARY_OUT = "amortized_ue/results/e61_efficiency_summary.json"


def stage_summary(targets):
    """Combine the (unmodified) rq1_latency.py blocks with this experiment's token/FLOPs/proxy
    numbers into one corrected efficiency table. Reads only; writes SUMMARY_OUT."""
    out = {"_note": ("Corrects the E61 write-up: the ~1100x/~1225x figure is Block-B (bs=1 sampler "
                     "+ clustering) divided by the BATCHED-proxy per-question latency — an "
                     "SE-ESTIMATION-STEP ratio that pairs an un-batched baseline with a batched "
                     "proxy. It is NOT an end-to-end speedup. True end-to-end with the batched "
                     "proxy is (A+B)/(A+C_batched), reported below."),
           "targets": {}}
    for t in targets:
        rq = json.load(open(RQ1_TMPL.format(target=t)))["blocks"]
        eff = json.load(open(OUT_TMPL.format(target=t)))
        A, B, C = rq["A"]["mean_s"], rq["B"]["mean_s"], rq["C"]["mean_s"]
        Cb = rq["C"]["batched_throughput"]["mean_s_per_question_amortized"]
        row = {
            "block_ms": {"A_canonical_gen": A * 1e3, "B_10samples_plus_clustering_bs1": B * 1e3,
                         "B_sampling": rq["B"]["sample_stage"]["mean_s"] * 1e3,
                         "B_deberta_clustering": rq["B"]["cluster_stage"]["mean_s"] * 1e3,
                         "C_proxy_fwd_bs1": C * 1e3, "C_proxy_fwd_batched_bs32": Cb * 1e3},
            "speedups": {
                "SE_step__B_over_C_bs1": B / C,
                "SE_step__B_over_C_batched": B / Cb,
                "end_to_end__A+B_over_A+C_bs1": (A + B) / (A + C),
                "end_to_end__A+B_over_A+C_batched": (A + B) / (A + Cb),
            },
            "tokens": eff["tokens"],
            "flops_estimated_per_question": eff["flops_estimated"]["per_question"],
            "flops_ratio_baseline_over_proposed": eff["flops_estimated"]["ratios"]["baseline_total / proposed_total"],
            "proxy_latency": eff.get("proxy_latency"),
            "target_generation_peak_mem": eff.get("target_generation_peak_mem"),
        }
        out["targets"][t] = row
    os.makedirs(os.path.dirname(SUMMARY_OUT), exist_ok=True)
    with open(SUMMARY_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  -> {SUMMARY_OUT}")

    print("\n" + "=" * 100)
    print("E61 CORRECTED efficiency table (per question, n=200, one L40)")
    print("=" * 100)
    hdr = f"{'':42s}" + "".join(f"{t.split('-')[0]+' '+t.split('-')[-1]:>26s}" for t in targets)
    for lbl, key, sub in [
        ("Block A  1 canonical generation (ms)", "block_ms", "A_canonical_gen"),
        ("Block B  10 samples + clustering, bs=1 (ms)", "block_ms", "B_10samples_plus_clustering_bs1"),
        ("Block C  proxy forward, bs=1 (ms)", "block_ms", "C_proxy_fwd_bs1"),
        ("Block C  proxy forward, batched bs=32 (ms/q)", "block_ms", "C_proxy_fwd_batched_bs32"),
        ("SE-step ratio   B / C(bs=1)", "speedups", "SE_step__B_over_C_bs1"),
        ("SE-step ratio   B / C(batched)  [NOT end-to-end]", "speedups", "SE_step__B_over_C_batched"),
        ("END-TO-END   (A+B)/(A+C, bs=1)", "speedups", "end_to_end__A+B_over_A+C_bs1"),
        ("END-TO-END   (A+B)/(A+C, batched)", "speedups", "end_to_end__A+B_over_A+C_batched"),
    ]:
        vals = "".join(f"{out['targets'][t][key][sub]:>26.1f}" for t in targets)
        print(f"{lbl:42s}{vals}")
    print("-" * 100)
    for lbl, sub in [("estimated FLOPs baseline pipeline", "baseline_pipeline_total"),
                     ("estimated FLOPs proposed pipeline", "proposed_pipeline_total"),
                     ("estimated FLOPs 10-sample generation", "ten_sample_generation"),
                     ("estimated FLOPs proxy forward", "proxy_forward")]:
        vals = "".join(f"{out['targets'][t]['flops_estimated_per_question'][sub]:>26.3e}" for t in targets)
        print(f"{lbl:42s}{vals}")
    vals = "".join(f"{out['targets'][t]['flops_ratio_baseline_over_proposed']:>26.1f}" for t in targets)
    print(f"{'estimated FLOPs ratio base/proposed':42s}{vals}")
    print("=" * 100)


def _merge_save(path, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing.update(result)
    existing.setdefault("meta", {})
    existing["meta"]["updated_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+",
                    default=["Llama-2-7b-chat", "Mistral-7B-Instruct-v0.2"])
    ap.add_argument("--dataset", default="trivia_qa")
    ap.add_argument("--num_samples", type=int, default=2000)
    ap.add_argument("--data_dir", default="/data2/mn1025/stage1")
    ap.add_argument("--stage", choices=["tokens", "proxy", "all", "summary"], default="tokens")
    ap.add_argument("--deploy_ckpt", default=DEPLOY_CKPT)
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()

    if a.stage == "summary":
        stage_summary(a.targets)
        return

    for target in a.targets:
        print(f"\n=== E61 efficiency — {target} (stage={a.stage}) ===")
        path = OUT_TMPL.format(target=target)
        result = {}
        result.setdefault("meta", {}).update(
            {"target": target, "dataset": a.dataset, "num_samples": a.num_samples,
             "experiment": "E61-efficiency (additive)"})
        if a.stage in ("tokens", "all"):
            stage_tokens(target, a.dataset, a.num_samples, a.data_dir, result)
        if a.stage in ("proxy", "all"):
            stage_proxy(target, a.dataset, a.num_samples, a.data_dir, a.deploy_ckpt, a.warmup, result)
        _merge_save(path, result)
        _print(target, path)


def _print(target, path):
    with open(path) as f:
        r = json.load(f)
    print(f"\n----- {target} -----")
    if "tokens" in r:
        t = r["tokens"]
        print(f"  input tok/q          {t['input_tokens_per_question']['mean']:.1f} "
              f"(med {t['input_tokens_per_question']['median']:.0f}, max {t['input_tokens_per_question']['max']:.0f})")
        print(f"  gen tok canonical    {t['generated_tokens_canonical']['mean']:.2f}")
        print(f"  gen tok / sample     {t['generated_tokens_per_sample']['mean']:.2f}  (x{t['num_generations']} samples)")
        print(f"  target gens/q        baseline {t['target_generations_per_question']['baseline']}  vs  proposed {t['target_generations_per_question']['proposed']}")
        print(f"  DeBERTa fwd/q        {t['deberta_forward_passes_per_question']['mean']:.1f}  "
              f"(pair len {t['deberta_pair_token_len']['mean']:.1f} tok)")
        print(f"  proxy input tok      {t['proxy_input_tokens']['mean']:.1f} (max {t['proxy_input_tokens']['max']:.0f})")
    if "flops_estimated" in r:
        pq = r["flops_estimated"]["per_question"]
        rr = r["flops_estimated"]["ratios"]
        print(f"  --- estimated FLOPs / question ---")
        for k, v in pq.items():
            print(f"    {k:28s} {v:.3e}")
        for k, v in rr.items():
            print(f"    ratio {k:34s} {v:.1f}x")
        print(f"    attention-term fraction of canonical gen: {r['flops_estimated']['attention_term_fraction_of_canonical_gen']*100:.2f}%")
    if "proxy_latency" in r:
        p = r["proxy_latency"]
        print(f"  proxy fwd-only       {p['forward_only_ms']['mean']:.2f} ms  "
              f"| tok+fwd {p['tokenize_plus_forward_ms']['mean']:.2f} ms  "
              f"| tok overhead {p['tokenizer_overhead_ms_mean']:.2f} ms")
        print(f"  proxy peak GPU mem   {p['peak_gpu_mem_forward_MiB']:.0f} MiB")


if __name__ == "__main__":
    main()
