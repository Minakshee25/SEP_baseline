"""Convert existing amortized_ue Stage-1 records into the calibration-tuning repo's
"offline" CSV dataset format, WITHOUT regenerating any answers.

Reuses (read-only): amortized_ue.loaders.load_records, amortized_ue.config.Stage1Config,
amortized_ue.linear_ceiling_probe.splits (the exact seeded train/val/test partition already
used by amortized_ue/correctness_eval.py for every other baseline in the comparison table).

Regime mirrors correctness_eval.py exactly:
    fit pool  = <model>_trivia_qa_n2000_full  -> splits() -> train (1440) / val (360) / (test,
                unused here -- correctness_eval's own held-out test is the fresh n1000 below)
    eval pool = <model>_trivia_qa_n1000_full  (verified 0 id-overlap with the n2000 fit pool)
                -> written as the "test" split (the ONLY split ever scored for AUROC)

Correctness label: identical binarisation to correctness_eval.py -- accuracy (SQuAD F1) >= 0.5
=> query_label=1 ("correct"/"yes"), else 0 ("incorrect"/"no"). No generation, no re-grading:
`canonical.response` becomes the CSV "output" column verbatim, and the calibration-tuning
trainer (llm/trainer/calibration_tune.py: compute_lm_loss) skips its own generation path
entirely whenever a "query_label" column is present -- it only trains the calibration-query
LoRA head on this precomputed (question, answer, label) triple.

Run under the `se_probes` conda env (needs amortized_ue's loaders), e.g.:
    cd /vol/bitbucket/mn1025/individual_project/semantic-entropy-probes
    python -m calibration_tuning_baseline.scripts.build_offline_dataset \
        --model_name Llama-2-7b-chat \
        --out_root /data2/mn1025/calibration_tuning_baseline/data \
        --dataset_name sep-Llama-2-7b-chat
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.linear_ceiling_probe import splits

FIT_N = 2000
EVAL_N = 1000
THRESH = 0.5
CSV_FIELDS = ["context", "target", "target_prompt", "prompt", "output", "query_label"]


def _row(rec: dict) -> dict:
    question = rec["question"]
    passage = rec.get("context") or ""
    ref = rec["reference"]
    # trivia_qa-style reference dict: {"answers": {"text": [...], ...}, "id": ...}
    if isinstance(ref, dict) and "answers" in ref:
        gold = ref["answers"]["text"][0] if ref["answers"]["text"] else ""
    elif isinstance(ref, dict) and "text" in ref:
        gold = ref["text"][0] if ref["text"] else ""
    else:
        gold = str(ref)

    context = f"Passage:\n{passage}\n\nQuestion:\n{question}" if passage else f"Question:\n{question}"
    accuracy = float(rec["canonical"]["accuracy"])
    query_label = int(accuracy >= THRESH)

    return {
        "context": context,
        "target": gold,
        "target_prompt": "\nAnswer:",
        "prompt": "",
        "output": rec["canonical"]["response"],
        "query_label": query_label,
    }


def _write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True, help="e.g. Llama-2-7b-chat or Mistral-7B-Instruct-v0.2")
    p.add_argument("--dataset", default="trivia_qa")
    p.add_argument("--out_root", required=True, help="calibration-tuning --data-dir root (e.g. /data2/.../data)")
    p.add_argument("--dataset_name", required=True, help="the <name> in offline:<name> (no colons)")
    p.add_argument("--prompt_style", default="oe")
    p.add_argument("--data_dir", default=None,
                    help="override Stage1Config.output_dir (e.g. /data2/mn1025/stage1 for speed; "
                         "falls back to the repo's NFS amortized_ue/data/stage1 if a subset is missing there)")
    args = p.parse_args()

    extra = {"output_dir": args.data_dir} if args.data_dir else {}
    fit_cfg = Stage1Config(model_name=args.model_name, dataset=args.dataset, num_samples=FIT_N, **extra)
    eval_cfg = Stage1Config(model_name=args.model_name, dataset=args.dataset, num_samples=EVAL_N, **extra)

    fit_records = load_records(fit_cfg)
    eval_records = load_records(eval_cfg)

    fit_ids = sorted(fit_records.keys())
    eval_ids = sorted(eval_records.keys())

    overlap = set(fit_ids) & set(eval_ids)
    assert not overlap, f"fit/eval id overlap ({len(overlap)} ids) -- would leak test rows into training"

    tr, va, te = splits(len(fit_ids))  # SEED=42, TEST_SIZE=0.1, VAL_SIZE=0.2 -- same as correctness_eval.py
    train_rows = [_row(fit_records[fit_ids[i]]) for i in tr]
    val_rows = [_row(fit_records[fit_ids[i]]) for i in va]
    # fit-pool `te` (the 200 held-out rows inside n2000) is intentionally NOT written anywhere:
    # it is not used by this baseline; the only test split scored for AUROC is the fresh n1000 below.
    test_rows = [_row(eval_records[i]) for i in eval_ids]

    root = f"{args.out_root}/offline/{args.dataset_name}-{args.prompt_style}"
    _write_csv(f"{root}/train/data.csv", train_rows)
    _write_csv(f"{root}/validation/data.csv", val_rows)
    _write_csv(f"{root}/test/data.csv", test_rows)

    n_incorrect_test = sum(1 - r["query_label"] for r in test_rows)
    print(f"[{args.model_name}] wrote {len(train_rows)} train / {len(val_rows)} val / {len(test_rows)} test rows -> {root}")
    print(f"  test incorrect-rate = {n_incorrect_test}/{len(test_rows)} = {n_incorrect_test/len(test_rows):.3f}")
    print(f"  test ids match correctness_eval.py's eval set: {eval_ids == sorted(eval_records.keys())}")


if __name__ == "__main__":
    main()
