# calibration-tuning baseline

Apples-to-apples comparison of the official LoRA+Prompt calibration-tuning method
([activatedgeek/calibration-tuning](https://github.com/activatedgeek/calibration-tuning),
arXiv:2406.08391) against this project's amortized-UE proxy, on the SAME data/labels/splits
already used by `amortized_ue/correctness_eval.py`.

## Layout

- **Git-tracked here** (`calibration_tuning_baseline/scripts/`): this README + the 3 glue scripts.
- **NOT git-tracked, on `/data2` for speed** (gitignored):
  - `/data2/mn1025/calibration_tuning_baseline/calibration-tuning/` — the cloned official repo,
    **unmodified except** `llm/models/llama2.py` (NousResearch redirect for the gated
    `meta-llama/Llama-2-7b-chat-hf`, same justification/pattern as this project's own
    `huggingface_models.py` — see that file's comment). Mistral needs no change (already ungated).
  - `/data2/mn1025/calibration_tuning_baseline/data/offline/sep-<model>-oe/{train,validation,test}/`
    — CSVs built by `build_offline_dataset.py`, no regeneration.
  - `/data2/mn1025/calibration_tuning_baseline/checkpoints/<dataset-name>/` — LoRA training output.
  - `/data2/mn1025/calibration_tuning_baseline/logs/eval_<dataset-name>/` — eval output
    (`.../metrics/offline:<dataset-name>/test/query_data.bin`).
- **Env**: conda env at `/data2/mn1025/conda_envs/calibration_tuning` (python 3.11, their
  `requirements*.txt` + `pip install -e .`) — fully separate from `se_probes`/`amortized_stage2`.

## Data / label / split provenance (verified, no leakage)

`build_offline_dataset.py` reads the SAME Stage-1 records `correctness_eval.py` uses:
- fit pool `<model>_trivia_qa_n2000_full` → `amortized_ue.linear_ceiling_probe.splits()`
  (SEED=42) → 1440 train / 360 val (the `te` 200 rows are NOT used by this baseline).
- eval pool `<model>_trivia_qa_n1000_full` → written as the CT "test" split. **Verified 0 id
  overlap with the n2000 fit pool** for both targets.
- label: `canonical.accuracy >= 0.5` → `query_label` (1=correct/"yes", 0=incorrect/"no") —
  IDENTICAL binarisation to `correctness_eval.py`'s `incorrect = (accuracy < 0.5)`.
- Sanity: converter-printed test incorrect-rates (Llama-2 0.391, Mistral 0.351) match
  `correctness_eval_<model>.json`'s `positive_rate_incorrect` exactly.
- `output` (the CSV's precomputed answer) = `canonical.response` verbatim — the calibration-tuning
  trainer/evaluator skip their own generation entirely whenever `query_label` is present in a row
  (`compute_lm_loss`), so no answers are ever regenerated.

## Pipeline

```bash
# 1. Convert Stage-1 records -> offline CSVs (run in se_probes; reads NFS or /data2 via --data_dir)
conda activate se_probes
cd /vol/bitbucket/mn1025/individual_project/semantic-entropy-probes
python -m calibration_tuning_baseline.scripts.build_offline_dataset \
    --model_name Llama-2-7b-chat --out_root /data2/mn1025/calibration_tuning_baseline/data \
    --dataset_name sep-Llama-2-7b-chat
python -m calibration_tuning_baseline.scripts.build_offline_dataset \
    --model_name Mistral-7B-Instruct-v0.2 --out_root /data2/mn1025/calibration_tuning_baseline/data \
    --dataset_name sep-Mistral-7B-Instruct-v0.2

# 2. Train the calibration-query LoRA (paper/README defaults: r8/alpha32/drop0.1, batch4,
#    kl_decay=1.0, max_steps=5000, lr=1e-4) -- wait for GPU headroom first, see below.
CUDA_VISIBLE_DEVICES=<gpu> bash calibration_tuning_baseline/scripts/run_calibration_tune.sh \
    llama2:7b-chat sep-Llama-2-7b-chat
CUDA_VISIBLE_DEVICES=<gpu> bash calibration_tuning_baseline/scripts/run_calibration_tune.sh \
    mistral:7b-instruct sep-Mistral-7B-Instruct-v0.2

# 3. Evaluate (their own experiments/evaluate.py --mode=query, unmodified)
CUDA_VISIBLE_DEVICES=<gpu> bash calibration_tuning_baseline/scripts/run_evaluate.sh \
    llama2:7b-chat sep-Llama-2-7b-chat
CUDA_VISIBLE_DEVICES=<gpu> bash calibration_tuning_baseline/scripts/run_evaluate.sh \
    mistral:7b-instruct sep-Mistral-7B-Instruct-v0.2

# 4. Merge into the existing comparison table (adds a "calibration_tuning" row, reuses the
#    same bootstrap-CI code as every other baseline; se_probes env)
conda activate se_probes
cd /vol/bitbucket/mn1025/individual_project/semantic-entropy-probes
python -m amortized_ue.correctness_eval --targets Llama-2-7b-chat Mistral-7B-Instruct-v0.2 \
    --data_dir /data2/mn1025/stage1 \
    --calibration_tuning_log_dir /data2/mn1025/calibration_tuning_baseline/logs
```

`uncertainty = 1 - P(correct)` where `P(correct) = softmax(q_logits)[:, 1]` — this IS the
repo's own native "yes"-token probability from `prepare_uncertainty_query` (`roman_choice`
format: token 0 = "i" = no, token 1 = "ii" = yes), no reinterpretation needed.

## GPU

Both GPUs are typically shared with this project's own live data-gen jobs (see root
`amortized_ue/CLAUDE.md` "Current state" for what's running). int8 LoRA training for a 7B model
needs ~10-12GB free — check `nvidia-smi` before launching, per this repo's GPU-headroom rule.
