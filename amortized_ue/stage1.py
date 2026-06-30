"""Stage 1 builder: generate, label, extract, and save one record per prompt.

For one target LLM and a QA dataset, for each validation prompt we:
  1. draw 1 low-temperature canonical answer + N high-temperature samples
     (reusing HuggingfaceModel.predict via SEP),
  2. cluster the high-temp samples with DeBERTa and compute the continuous
     cluster-assignment entropy label (reusing SEP's semantic_entropy code),
  3. extract TBG + SLT hidden states (all layers) for the canonical answer,
  4. write a self-contained, id-keyed record to local disk.

A later training stage consumes these records without ever re-running the LLM.
"""
from __future__ import annotations

import os
import time
import random
import logging
import datetime
import subprocess

import numpy as np
import torch

from amortized_ue.config import Stage1Config
from amortized_ue import sep_bridge
from amortized_ue import record as rec


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _build_meta(config: Stage1Config) -> dict:
    """Provenance shared by every record + the manifest."""
    return {
        "model_name": config.model_name,
        "dataset": config.dataset,
        "num_generations": config.num_generations,
        "high_temperature": config.temperature,
        "low_temperature": config.low_temperature,
        "model_max_new_tokens": config.model_max_new_tokens,
        "metric": config.metric,
        "entailment_model": config.entailment_model,
        "condition_on_question": config.condition_on_question,
        "strict_entailment": config.strict_entailment,
        "num_few_shot": config.num_few_shot,
        "random_seed": config.random_seed,
        "git_commit": _git_commit(),
        "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
        # explicit note so downstream never confuses positions with SEP's keys
        "hidden_state_positions": {
            "TBG": "hidden[0] (last input token before generation), all layers",
            "SLT": "hidden[n_generated-2] (second-last generated token), all layers",
        },
        "sep_key_mapping": {
            "TBG": "SEP key 'emb_tok_before_eos' / SEP probe slt_dataset",
            "SLT": "SEP key 'emb_last_tok_before_gen' / SEP probe tbg_dataset",
        },
    }


def build(config: Stage1Config) -> dict:
    sep_bridge.sep_utils.setup_logger()
    args = sep_bridge.build_sep_args(config)

    run_dir = config.run_dir()
    records_dir = config.records_dir()
    os.makedirs(records_dir, exist_ok=True)
    logging.info("Stage 1 run dir: %s", run_dir)

    # --- reproducible sampling (mirror SEP's RNG usage) ------------------------
    random.seed(args.random_seed)

    metric = sep_bridge.sep_utils.get_metric(args.metric)
    train_dataset, validation_dataset = sep_bridge.load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed)

    # squad forces answerable_only in SEP; replicate so the prompt set matches.
    if args.dataset == "squad":
        val_answerable, _ = sep_bridge.sep_utils.split_dataset(validation_dataset)
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    answerable_indices, _ = sep_bridge.sep_utils.split_dataset(train_dataset)

    # --- few-shot prompt prefix (from the train split) -------------------------
    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    make_prompt = sep_bridge.sep_utils.get_make_prompt(args)
    BRIEF = sep_bridge.sep_utils.BRIEF_PROMPTS[args.brief_prompt]
    brief_always = args.brief_always if args.enable_brief else True
    fewshot_prompt = sep_bridge.sep_utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, brief_always, make_prompt)
    logging.info("Few-shot prompt:\n%s", fewshot_prompt)

    # --- choose validation prompts --------------------------------------------
    possible_indices = list(range(len(validation_dataset)))
    n = min(args.num_samples, len(validation_dataset))
    indices = random.sample(possible_indices, n)

    # --- load models -----------------------------------------------------------
    logging.info("Loading target model %s ...", args.model_name)
    model = sep_bridge.sep_utils.init_model(args)
    logging.info("Loading entailment model (DeBERTa) ...")
    entailment_model = sep_bridge.EntailmentDeberta()

    meta = _build_meta(config)
    entries = []
    accuracies, entropies = [], []
    t0 = time.time()

    for it, index in enumerate(indices):
        example = validation_dataset[index]
        prompt_id = example["id"]
        question, context = example["question"], example["context"]

        record_filename = rec.safe_filename(prompt_id) + ".pt"
        record_path = os.path.join(records_dir, record_filename)
        if os.path.exists(record_path) and not config.overwrite:
            logging.info("[%d/%d] skip existing id=%s", it + 1, n, prompt_id)
            existing = rec.load_record(record_path)
            entries.append(rec.manifest_entry(existing, record_filename))
            accuracies.append(existing["canonical"]["accuracy"])
            entropies.append(existing["labels"]["cluster_assignment_entropy"])
            continue

        current_input = make_prompt(context, question, None, BRIEF, brief_always)
        local_prompt = fewshot_prompt + current_input

        # --- 1 low-temp canonical answer + N high-temp samples -----------------
        canonical = None
        sample_responses, sample_log_liks = [], []
        for i in range(config.num_generations + 1):
            temperature = config.low_temperature if i == 0 else config.temperature
            answer, token_log_likelihoods, (embedding, slt_emb, tbg_emb) = model.predict(
                local_prompt, temperature, return_latent=True)
            # predict() returns (last_token_scalar, sec_last=SLT, last_tok_before_gen=TBG)

            if i == 0:
                acc = metric(answer, example, model)
                canonical = dict(
                    response=answer,
                    accuracy=float(acc),
                    token_log_likelihoods=token_log_likelihoods,
                    tbg=tbg_emb.cpu() if tbg_emb is not None else None,  # native dtype
                    slt=slt_emb.cpu() if slt_emb is not None else None,
                )
            else:
                sample_responses.append(answer)
                sample_log_liks.append(token_log_likelihoods)

        # --- semantic clustering + continuous entropy label -------------------
        if args.condition_on_question and args.entailment_model == "deberta":
            cluster_inputs = [f"{question} {r}" for r in sample_responses]
        else:
            cluster_inputs = sample_responses
        semantic_ids = sep_bridge.get_semantic_ids(
            cluster_inputs, model=entailment_model,
            strict_entailment=args.strict_entailment, example=example)
        cae = sep_bridge.cluster_assignment_entropy(semantic_ids)

        record = rec.build_record(
            prompt_id=prompt_id,
            question=question,
            context=context,
            reference=sep_bridge.sep_utils.get_reference(example),
            canonical_response=canonical["response"],
            canonical_accuracy=canonical["accuracy"],
            canonical_token_log_likelihoods=canonical["token_log_likelihoods"],
            tbg_embedding=canonical["tbg"],
            slt_embedding=canonical["slt"],
            sample_responses=sample_responses,
            sample_token_log_likelihoods=sample_log_liks,
            semantic_ids=semantic_ids,
            cluster_assignment_entropy=cae,
            meta=meta,
        )
        filename = rec.save_record(record, records_dir)
        entries.append(rec.manifest_entry(record, filename))
        accuracies.append(canonical["accuracy"])
        entropies.append(cae)

        logging.info(
            "[%d/%d] id=%s acc=%.2f CAE=%.3f n_clusters=%d  | q=%s -> %s",
            it + 1, n, prompt_id, canonical["accuracy"], cae,
            record["labels"]["n_clusters"], question[:50],
            canonical["response"][:50])

    metrics = {
        "n_records": len(entries),
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "mean_cluster_assignment_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    rec.write_manifest(config.manifest_path(), config.as_dict(), {**meta, **metrics}, entries)
    logging.info("Wrote %d records -> %s", len(entries), run_dir)
    logging.info("Metrics: %s", metrics)

    del model, entailment_model
    torch.cuda.empty_cache()

    if config.push_to_wandb:
        from amortized_ue import wandb_io
        wandb_io.sync_to_wandb(config, metrics)

    return {"run_dir": run_dir, "metrics": metrics, "manifest": config.manifest_path()}


def run_smoke(config: Stage1Config) -> dict:
    """A handful of prompts end to end, then print one record's structure."""
    config.smoke = True
    result = build(config)

    manifest = rec.read_manifest(config.manifest_path())
    first_id = next(iter(manifest["records"]))
    first_file = manifest["records"][first_id]["file"]
    record = rec.load_record(os.path.join(config.records_dir(), first_file))

    print("\n" + "=" * 78)
    print("SMOKE TEST — saved record structure for one prompt")
    print("=" * 78)
    print(f"run_dir : {result['run_dir']}")
    print(f"metrics : {result['metrics']}")
    print(f"id      : {first_id}")
    print("-" * 78)
    print(rec.describe_record(record))
    print("=" * 78 + "\n")
    return result


def _config_from_cli() -> Stage1Config:
    import argparse
    p = argparse.ArgumentParser(description="Stage 1 dataset construction (amortized UE).")
    p.add_argument("--model_name", default=Stage1Config.model_name)
    p.add_argument("--dataset", default=Stage1Config.dataset)
    p.add_argument("--num_samples", type=int, default=Stage1Config.num_samples)
    p.add_argument("--num_generations", type=int, default=Stage1Config.num_generations)
    p.add_argument("--temperature", type=float, default=Stage1Config.temperature)
    p.add_argument("--num_few_shot", type=int, default=Stage1Config.num_few_shot)
    p.add_argument("--model_max_new_tokens", type=int, default=Stage1Config.model_max_new_tokens)
    p.add_argument("--output_dir", default=Stage1Config.output_dir)
    p.add_argument("--run_name", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--push_to_wandb", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_num_samples", type=int, default=Stage1Config.smoke_num_samples)
    a = p.parse_args()
    return Stage1Config(
        model_name=a.model_name, dataset=a.dataset, num_samples=a.num_samples,
        num_generations=a.num_generations, temperature=a.temperature,
        num_few_shot=a.num_few_shot, model_max_new_tokens=a.model_max_new_tokens,
        output_dir=a.output_dir, run_name=a.run_name, overwrite=a.overwrite,
        push_to_wandb=a.push_to_wandb, smoke=a.smoke, smoke_num_samples=a.smoke_num_samples)


if __name__ == "__main__":
    cfg = _config_from_cli()
    if cfg.smoke:
        run_smoke(cfg)
    else:
        build(cfg)
