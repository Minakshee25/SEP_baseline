"""RQ1 — inference-latency benchmark: one amortized proxy forward pass vs. the
N=10 sampling + DeBERTa-clustering pipeline it replaces.

For a target LLM's 200-question held-out test split (the Stage-2 test split of its
trivia_qa n2000 Stage-1 dataset: test_size=0.1, seed 42, over the id-sorted order —
exactly `Stage2Data`'s split), we time three blocks:

  Block A  (shared, report once) : one canonical low-temperature generation / question.
  Block B  (baseline being replaced) : `num_generations` high-temperature samples /
           question  +  DeBERTa entailment clustering -> cluster-assignment entropy.
           This reuses the Stage-1 code path verbatim (`HuggingfaceModel.predict`,
           `get_semantic_ids`, `cluster_assignment_entropy` — nothing modified).
  Block C  (proposed) : one `q_resp_only` proxy forward pass / question, using the
           deploy checkpoint (frozen Llama-3.2-3B + LoRA, pooled-Set-1 trained).

Timing protocol (per the RQ1 spec):
  * every model is loaded once;
  * `--warmup` (default 10) forward passes run before any timing;
  * `torch.cuda.synchronize()` immediately brackets every timed region;
  * batch size 1 per question across all 200 questions -> mean +/- std wall-clock
    per question is the primary number for B and C (and A);
  * one batched (batch=32) throughput number is reported as a secondary result for
    C (and, where the code path allows it, C only — see note in the results for B).

Blocks A + B need the *target LLM* env (`se_probes` for Llama-2; `se_probes_llama3`
for Mistral-v0.2). Block C needs the proxy env (`amortized_stage2` /
`amortized_stage2_v5`). Run the script once per env group via `--blocks`:

    # target-LLM env, on a free GPU:
    python -m amortized_ue.rq1_latency --target Llama-2-7b-chat --blocks A,B \
        --data_dir /data2/mn1025/stage1 --out amortized_ue/results/rq1_latency_Llama-2-7b-chat.json

    # proxy env, on a free GPU:
    python -m amortized_ue.rq1_latency --target Llama-2-7b-chat --blocks C \
        --data_dir /data2/mn1025/stage1 --out amortized_ue/results/rq1_latency_Llama-2-7b-chat.json

The two invocations merge into the same `--out` JSON (existing blocks are kept).

Additive only: imports Stage-1 / Stage-2 code read-only, modifies nothing.
"""
from __future__ import annotations

import os
import gc
import json
import time
import random
import argparse
import datetime
import statistics
from typing import Callable

import numpy as np
import torch
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------------------
# test-split resolution (identical to amortized_ue.stage2.data.Stage2Data)
# --------------------------------------------------------------------------------------
def resolve_test_ids(target: str, dataset: str, num_samples: int, data_dir: str | None,
                     test_size: float = 0.1, split_seed: int = 42):
    """Return (test_ids, records) — the held-out test split, id-keyed records."""
    from amortized_ue.config import Stage1Config
    from amortized_ue.loaders import load_records

    cfg = Stage1Config(model_name=target, dataset=dataset, num_samples=num_samples,
                       **({"output_dir": data_dir} if data_dir else {}))
    records = load_records(cfg)
    ids = sorted(records.keys())
    idx = np.arange(len(ids))
    tv_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=split_seed)
    test_idx = np.sort(test_idx)
    return [ids[i] for i in test_idx], records


# --------------------------------------------------------------------------------------
# prompt reconstruction (mirrors amortized_ue.stage1.build, verbatim)
# --------------------------------------------------------------------------------------
def build_prompts(target: str, dataset: str, num_samples: int, test_ids: list):
    """Rebuild the exact Stage-1 prompt (few-shot prefix + question) for each test id.

    Returns (list[(id, example, prompt)], sep_args) where `example` is the raw
    validation-set dict the SEP metric / entailment code expects.
    """
    from amortized_ue.config import Stage1Config
    from amortized_ue import sep_bridge

    cfg = Stage1Config(model_name=target, dataset=dataset, num_samples=num_samples)
    args = sep_bridge.build_sep_args(cfg)

    random.seed(args.random_seed)
    train_dataset, validation_dataset = sep_bridge.load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed)
    if args.dataset == "squad":
        val_answerable, _ = sep_bridge.sep_utils.split_dataset(validation_dataset)
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    answerable_indices, _ = sep_bridge.sep_utils.split_dataset(train_dataset)
    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    make_prompt = sep_bridge.sep_utils.get_make_prompt(args)
    BRIEF = sep_bridge.sep_utils.BRIEF_PROMPTS[args.brief_prompt]
    brief_always = args.brief_always if args.enable_brief else True
    fewshot_prompt = sep_bridge.sep_utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, brief_always, make_prompt)

    by_id = {ex["id"]: ex for ex in validation_dataset}
    missing = [i for i in test_ids if i not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} test ids not found in the validation set, e.g. {missing[:3]}")

    out = []
    for tid in test_ids:
        ex = by_id[tid]
        current_input = make_prompt(ex["context"], ex["question"], None, BRIEF, brief_always)
        out.append((tid, ex, fewshot_prompt + current_input))
    return out, args


# --------------------------------------------------------------------------------------
# stats helper
# --------------------------------------------------------------------------------------
def summarize(times_s: list, extra: dict | None = None) -> dict:
    a = np.asarray(times_s, dtype=float)
    d = {
        "n": int(a.size),
        "mean_s": float(a.mean()),
        "std_s": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median_s": float(np.median(a)),
        "p90_s": float(np.percentile(a, 90)),
        "min_s": float(a.min()),
        "max_s": float(a.max()),
        "total_s": float(a.sum()),
        "per_question_s": [round(float(x), 6) for x in a],
    }
    if extra:
        d.update(extra)
    return d


# --------------------------------------------------------------------------------------
# Block A — one canonical low-temperature generation per question
# --------------------------------------------------------------------------------------
def run_block_A(model, prompts, warmup: int, low_temperature: float) -> dict:
    for i in range(warmup):
        _, _, p = prompts[i % len(prompts)]
        model.predict(p, low_temperature, return_latent=True)
    torch.cuda.synchronize()

    per_q = []
    for _, _, p in prompts:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        model.predict(p, low_temperature, return_latent=True)
        torch.cuda.synchronize(); per_q.append(time.perf_counter() - t0)
    return summarize(per_q, {"block": "A", "description": "1x canonical low-temp generation (temp=%.2f)" % low_temperature})


# --------------------------------------------------------------------------------------
# Block B — N high-temp samples + DeBERTa clustering  (the pipeline being replaced)
# --------------------------------------------------------------------------------------
def _one_B(model, entailment_model, example, prompt, args, num_generations, high_temperature):
    """One question's sampling + clustering, exactly as amortized_ue.stage1.build does it.
    Returns (t_sample_s, t_cluster_s, cae, n_clusters)."""
    torch.cuda.synchronize(); t0 = time.perf_counter()
    sample_responses = []
    for _ in range(num_generations):
        answer, _tll, _lat = model.predict(prompt, high_temperature, return_latent=True)
        sample_responses.append(answer)
    torch.cuda.synchronize(); t1 = time.perf_counter()

    from amortized_ue import sep_bridge
    question = example["question"]
    if args.condition_on_question and args.entailment_model == "deberta":
        cluster_inputs = [f"{question} {r}" for r in sample_responses]
    else:
        cluster_inputs = sample_responses
    semantic_ids = sep_bridge.get_semantic_ids(
        cluster_inputs, model=entailment_model,
        strict_entailment=args.strict_entailment, example=example)
    cae = sep_bridge.cluster_assignment_entropy(semantic_ids)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    return (t1 - t0), (t2 - t1), float(cae), len(set(semantic_ids))


def run_block_B(model, entailment_model, prompts, warmup: int, args,
                num_generations: int, high_temperature: float) -> dict:
    wu = max(1, warmup // 5)  # each B iteration is ~num_generations generations; a few is plenty
    for i in range(wu):
        tid, ex, p = prompts[i % len(prompts)]
        _one_B(model, entailment_model, ex, p, args, num_generations, high_temperature)
    torch.cuda.synchronize()

    per_q, t_sample, t_cluster, caes, nclust = [], [], [], [], []
    for _, ex, p in prompts:
        ts, tc, cae, nc = _one_B(model, entailment_model, ex, p, args, num_generations, high_temperature)
        per_q.append(ts + tc); t_sample.append(ts); t_cluster.append(tc)
        caes.append(cae); nclust.append(nc)

    return summarize(per_q, {
        "block": "B",
        "description": f"{num_generations}x high-temp samples (temp={high_temperature:.2f}) + DeBERTa clustering",
        "num_generations": num_generations,
        "sample_stage": summarize(t_sample),
        "cluster_stage": summarize(t_cluster),
        "batched_throughput": {
            "note": "not measured — HuggingfaceModel.predict() generates one sequence per call; "
                    "batching the sampler would require modifying frozen Stage-1 generation code "
                    "(out of scope). bs=1 is the only mode this pipeline supports.",
        },
        "sanity_cae_mean": float(np.mean(caes)),
        "sanity_n_clusters_mean": float(np.mean(nclust)),
    })


# --------------------------------------------------------------------------------------
# Block C — one q_resp_only proxy forward pass per question
# --------------------------------------------------------------------------------------
def _qr_text(question: str, response: str) -> str:
    # verbatim from amortized_ue.stage2.train._arm_text("q_resp_only", ...)
    return f"Question: {question}\nAnswer: {response}"


def run_block_C(deploy_ckpt: str, records: dict, test_ids: list, warmup: int,
                device: str) -> dict:
    from amortized_ue.stage2.checkpoint import load_checkpoint

    model, meta, _transform = load_checkpoint(deploy_ckpt, device=device)
    model.eval()
    tok = model.tokenizer
    max_len = model.cfg.max_seq_len
    pad_id = tok.pad_token_id

    texts = [_qr_text(records[i]["question"], records[i]["canonical"]["response"]) for i in test_ids]
    tok_ids = [tok(t, add_special_tokens=False)["input_ids"][:max_len] for t in texts]

    def forward_bs1(ids):
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            return model(None, input_ids, attn)

    # ---- warm-up ----
    for i in range(warmup):
        forward_bs1(tok_ids[i % len(tok_ids)])
    torch.cuda.synchronize()

    # ---- primary: batch size 1 per question ----
    per_q = []
    for ids in tok_ids:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        forward_bs1(ids)
        torch.cuda.synchronize(); per_q.append(time.perf_counter() - t0)

    # ---- secondary: batched (batch=32) throughput ----
    bs = 32
    def forward_batch(batch_ids):
        T = max(len(s) for s in batch_ids)
        B = len(batch_ids)
        input_ids = torch.full((B, T), pad_id, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        for b, s in enumerate(batch_ids):
            if len(s):
                input_ids[b, T - len(s):] = torch.tensor(s, dtype=torch.long)  # left pad
                attn[b, T - len(s):] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)
        with torch.no_grad():
            return model(None, input_ids, attn)

    for _ in range(3):  # warm the batched path
        forward_batch(tok_ids[:bs])
    torch.cuda.synchronize()

    batch_times = []
    for i in range(0, len(tok_ids), bs):
        chunk = tok_ids[i:i + bs]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        forward_batch(chunk)
        torch.cuda.synchronize(); batch_times.append((time.perf_counter() - t0, len(chunk)))
    total_batch_s = sum(t for t, _ in batch_times)
    n_total = sum(c for _, c in batch_times)

    return summarize(per_q, {
        "block": "C",
        "description": "1x q_resp_only proxy forward pass (deploy checkpoint)",
        "deploy_ckpt": deploy_ckpt,
        "proxy_model": meta.get("proxy_model"),
        "arm": meta.get("arm"),
        "max_seq_len": max_len,
        "token_len_mean": float(np.mean([len(s) for s in tok_ids])),
        "token_len_max": int(max(len(s) for s in tok_ids)),
        "batched_throughput": {
            "batch_size": bs,
            "n_questions": int(n_total),
            "total_wall_s": float(total_batch_s),
            "questions_per_s": float(n_total / total_batch_s),
            "mean_s_per_question_amortized": float(total_batch_s / n_total),
        },
    })


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="Llama-2-7b-chat",
                   help="target LLM whose held-out split is benchmarked")
    p.add_argument("--dataset", default="trivia_qa")
    p.add_argument("--num_samples", type=int, default=2000,
                   help="Stage-1 dataset size whose Stage-2 split defines the 200 test ids")
    p.add_argument("--data_dir", default="/data2/mn1025/stage1",
                   help="Stage1Config.output_dir override (node-local copy)")
    p.add_argument("--blocks", default="A,B,C",
                   help="comma list of blocks to run this invocation (A,B need the target-LLM "
                        "env; C needs the proxy env)")
    p.add_argument("--deploy_ckpt",
                   default="amortized_ue/results/deploy_checkpoints/deploy_q_resp_only_seed0.pt")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help="cap #questions (debug only; 0 = all 200)")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    blocks = [b.strip().upper() for b in a.blocks.split(",") if b.strip()]
    device = "cuda"
    assert torch.cuda.is_available(), "this benchmark needs a GPU"
    out_path = a.out or f"amortized_ue/results/rq1_latency_{a.target}.json"

    test_ids, records = resolve_test_ids(a.target, a.dataset, a.num_samples, a.data_dir)
    if a.limit:
        test_ids = test_ids[:a.limit]
    print(f"[rq1] {a.target}: {len(test_ids)} held-out test questions "
          f"(split test_size=0.1 seed=42 over id-sorted n{a.num_samples})")

    # merge into an existing JSON if present (two-env workflow)
    result = {}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            result = json.load(fh)
    result.setdefault("meta", {})
    result["meta"].update({
        "target": a.target, "dataset": a.dataset, "num_samples": a.num_samples,
        "n_questions": len(test_ids), "warmup": a.warmup,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "updated_utc": datetime.datetime.utcnow().isoformat() + "Z",
    })
    result.setdefault("blocks", {})

    need_target_llm = any(b in blocks for b in ("A", "B"))
    if need_target_llm:
        from amortized_ue import sep_bridge
        prompts, sep_args = build_prompts(a.target, a.dataset, a.num_samples, test_ids)
        print(f"[rq1] loading target model {a.target} ...")
        model = sep_bridge.sep_utils.init_model(sep_args)
        entailment_model = None
        if "B" in blocks:
            print("[rq1] loading DeBERTa entailment model ...")
            entailment_model = sep_bridge.EntailmentDeberta()

        if "A" in blocks:
            print("[rq1] Block A ...")
            from amortized_ue.config import Stage1Config as _S1
            result["blocks"]["A"] = run_block_A(model, prompts, a.warmup, _S1.low_temperature)
            _save(result, out_path)
        if "B" in blocks:
            print("[rq1] Block B ...")
            result["blocks"]["B"] = run_block_B(
                model, entailment_model, prompts, a.warmup, sep_args,
                num_generations=sep_args.num_generations, high_temperature=sep_args.temperature)
            _save(result, out_path)

        del model, entailment_model
        gc.collect(); torch.cuda.empty_cache()

    if "C" in blocks:
        print("[rq1] Block C (proxy) ...")
        result["blocks"]["C"] = run_block_C(a.deploy_ckpt, records, test_ids, a.warmup, device)
        _save(result, out_path)

    print(f"[rq1] done -> {out_path}")
    _print_summary(result)


def _save(result: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)


def _print_summary(result: dict):
    print("\n==================== RQ1 latency summary ====================")
    m = result.get("meta", {})
    print(f"target={m.get('target')}  n={m.get('n_questions')}  gpu={m.get('gpu_name')}")
    for name in ("A", "B", "C"):
        b = result.get("blocks", {}).get(name)
        if not b:
            continue
        print(f"  Block {name}: {b['mean_s']*1000:8.1f} +/- {b['std_s']*1000:7.1f} ms/question   "
              f"(median {b['median_s']*1000:.1f})   {b.get('description','')}")
        if name == "B" and "sample_stage" in b:
            ss, cs = b["sample_stage"], b["cluster_stage"]
            print(f"           sampling {ss['mean_s']*1000:.1f} ms   clustering {cs['mean_s']*1000:.1f} ms")
        if name == "C" and "batched_throughput" in b:
            bt = b["batched_throughput"]
            print(f"           batched(bs={bt['batch_size']}): {bt['questions_per_s']:.1f} q/s "
                  f"({bt['mean_s_per_question_amortized']*1000:.2f} ms/q amortized)")
    A = result.get("blocks", {}).get("A")
    B = result.get("blocks", {}).get("B")
    C = result.get("blocks", {}).get("C")
    if B and C:
        print(f"\n  speedup  Block B / Block C  (bs=1) : {B['mean_s'] / C['mean_s']:.1f}x")
        if A:
            repl = A["mean_s"] + B["mean_s"]   # full replaced pipeline = canonical + samples/clustering
            prop = A["mean_s"] + C["mean_s"]   # proposed still needs the canonical answer for q_resp
            print(f"  end-to-end (A+B) / (A+C)          : {repl / prop:.1f}x")
    print("============================================================\n")


if __name__ == "__main__":
    main()
