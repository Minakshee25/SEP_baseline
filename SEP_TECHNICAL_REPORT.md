# Technical Report: Semantic Entropy Probes (SEP) Repository

Scope note: every claim below is from reading the source. The pipeline derives from
`jlko/semantic_uncertainty`; this fork adds the SEP training stage and machine-specific
patches. No files were modified in producing this report.

---

## 1. Repository layout

```
semantic-entropy-probes/
├── CLAUDE.md, README.md, LICENSE, .gitignore
├── sep_enviroment.yaml                         # conda env (note misspelling "enviroment")
├── slurm/
│   └── run.sh                                  # SLURM batch launcher
├── semantic_uncertainty/                       # Stages 1–3: SE generation pipeline
│   ├── generate_answers.py                     # ENTRY POINT — Stage 1: sample generation + hidden states
│   ├── compute_uncertainty_measures.py         # ENTRY POINT — Stage 2: semantic entropy + p_ik
│   ├── analyze_results.py                      # ENTRY POINT — Stage 3: AUROC/accuracy logging to wandb
│   ├── figures/                                # p_ik diagnostic plots (output)
│   └── uncertainty/
│       ├── data/data_utils.py                  # LIBRARY — dataset loaders (load_ds)
│       ├── models/
│       │   ├── base_model.py                   # LIBRARY — BaseModel ABC + STOP_SEQUENCES
│       │   └── huggingface_models.py           # LIBRARY — HuggingfaceModel: generate() + hidden-state extraction
│       ├── uncertainty_measures/
│       │   ├── semantic_entropy.py             # LIBRARY — entailment models, clustering, entropy
│       │   ├── p_ik.py                         # LIBRARY — logistic-regression correctness baseline
│       │   └── p_true.py                       # LIBRARY — p_true self-assessment baseline
│       └── utils/
│           ├── utils.py                        # LIBRARY — arg parser, prompts, metrics, save()
│           ├── openai.py                       # LIBRARY — OpenAI client (built at import)
│           └── eval_utils.py                   # LIBRARY — eval helpers
└── semantic_entropy_probes/                    # Stage 4: SEP training
    ├── train-latent-probe.ipynb                # NOTEBOOK — full 4-dataset paper experiment (ID + OOD)
    ├── run_falcon_probe.py                     # ENTRY POINT — standalone single-dataset probe (falcon sanity)
    ├── run_llama2_probe.py                     # ENTRY POINT — standalone single-dataset probe (Llama-2 baseline)
    └── models/                                 # trained probe pickles (output)
```

The two standalone `run_*_probe.py` scripts are near-identical; they copy the notebook's
probe functions verbatim and differ only in the `model_name`/`ds_paths` config block at the
top (`run_llama2_probe.py:42-47`).

---

## 2. Sample generation

**Script / function:** `semantic_uncertainty/generate_answers.py`, `main(args)`. The per-prompt
sampling loop is `generate_answers.py:186-235`.

**How N and temperature are set.** Per prompt the loop draws
`num_generations = args.num_generations + 1` samples (`generate_answers.py:184`). The first
(`i==0`) is the "most likely" low-temperature answer; the remaining `num_generations` are
high-temperature samples used for entropy. Temperature is fixed per iteration:

```python
# generate_answers.py:188-191
# Temperature for first generation is always `0.1`.
temperature = 0.1 if i == 0 else args.temperature
predicted_answer, token_log_likelihoods, (embedding, emb_last_before_gen, emb_before_eos) = model.predict(local_prompt, temperature, return_latent=True)
```

Defaults (`utils.py:68-73`): `--num_generations=10`, `--temperature=1.0`. So the default is
**1 low-temp (0.1) answer + 10 high-temp (1.0) answers = 11 generations per prompt**.
`--num_samples` (default 400, `utils.py:56`) controls how many prompts/questions are evaluated,
sampled randomly via `random.sample(possible_indices, min(args.num_samples, len(dataset)))`
(`generate_answers.py:150`). Decoding itself is `do_sample=True` with `temperature` passed to
`model.generate()` (`huggingface_models.py:264-274`).

**Prompt template.** Built in `utils.get_make_prompt` (`utils.py:286-303`), prompt_type `default`:

```python
# utils.py:288-299
prompt = ''
if brief_always: prompt += brief
if args.use_context and (context is not None): prompt += f"Context: {context}\n"
prompt += f"Question: {question}\n"
if answer: prompt += f"Answer: {answer}\n\n"
else: prompt += 'Answer:'
```

A few-shot prefix is assembled from `num_few_shot` (default 5) randomly chosen answerable
training examples by `construct_fewshot_prompt_from_indices` (`utils.py:160-176`), prepended to
the current question's input: `local_prompt = prompt + current_input` (`generate_answers.py:171`).
The `BRIEF` instruction is one of two strings in `BRIEF_PROMPTS` (`utils.py:14-16`): `default`
("Answer the following question as briefly as possible.") or `chat`.

**Datasets supported and loading.** `load_ds(dataset_name, seed, add_options)` in `data_utils.py:9`.
Parser `choices` (`utils.py:48-49`): `trivia_qa, squad, bioasq, nq, svamp` (the loader also
implements `med_qa` and `record`). Loading mechanisms:
- `trivia_qa`: `datasets.load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']`, then
  `train_test_split(test_size=0.2, seed=seed)` (`data_utils.py:49-54`).
- `squad`: `squad_v2` HF dataset (`data_utils.py:12-15`); `--answerable_only` forced true
  (`generate_answers.py:35-38`).
- `nq`: `nq_open`, reformatted; **id is an md5 hash of the question** (`data_utils.py:33-44`).
- `svamp`: `ChilleD/SVAMP`, forces `use_context=True` (`generate_answers.py:31-34`).
- `bioasq`: local JSON `~/uncertainty/data/bioasq/training11b.json`, `train_test_split(test_size=0.8)`
  (`data_utils.py:92-139`).

All loaders normalize each example to fields `question`, `context`, `answers` (dict with `text`
and optionally `answer_start`), and `id`.

**What is written and where.** Saving goes through `utils.save` (`utils.py:339-342`):

```python
def save(object, file):
    with open(f'{wandb.run.dir}/{file}', 'wb') as f:
        pickle.dump(object, f)
    wandb.save(f'{wandb.run.dir}/{file}')
```

So **everything is Python pickle** written into the active **wandb run directory**
(`{SCRATCH_DIR}/{USER}/uncertainty/wandb/run-<timestamp>-<id>/files/`). Stage 1 writes, per split
(`generate_answers.py:247, 260, 262`):
- `train_generations.pkl`, `validation_generations.pkl` — the `generations` dict.
- `uncertainty_measures.pkl` — `results_dict` (only p_true populated at this stage).
- `experiment_details.pkl` — args, prompt, indices.

`generations` is a **dict keyed by `example['id']`** (`generate_answers.py:166`). Each record's
keys (`generate_answers.py:166, 215-227, 235`):
- `question`, `context`
- `most_likely_answer` → dict with keys `response`, `token_log_likelihoods`, `embedding`,
  `accuracy`, `emb_last_tok_before_gen`, `emb_tok_before_eos` (`generate_answers.py:215-222`)
- `reference` (from `get_reference`, holds `answers` + `id`)
- `responses` — a **list of tuples** `(predicted_answer, token_log_likelihoods, embedding, acc)`,
  one per high-temp sample (`generate_answers.py:231-235`).

---

## 3. Semantic entropy labelling

**Script:** `semantic_uncertainty/compute_uncertainty_measures.py`, `main(args)`. Core math:
`uncertainty/uncertainty_measures/semantic_entropy.py`.

**NLI model and loading.** Default is DeBERTa (`utils.py:138`, `--entailment_model='deberta'`),
selected at `compute_uncertainty_measures.py:102-103`. `EntailmentDeberta` (`semantic_entropy.py:33-37`):

```python
self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge-mnli").to(DEVICE)
```

Alternatives exist (`EntailmentGPT4`, `EntailmentGPT35`, `EntailmentLlama`) with prompt-based
entailment and a wandb-restored prediction cache, but DeBERTa is the baseline.

**Directional entailment.** `check_implication` returns 0/1/2 = contradiction/neutral/entailment
via argmax of softmax logits (`semantic_entropy.py:39-53`):

```python
# semantic_entropy.py:44-48
outputs = self.model(**inputs)
logits = outputs.logits
# Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
largest_index = torch.argmax(F.softmax(logits, dim=1))
prediction = largest_index.cpu().item()
```

**Bidirectional equivalence + clustering.** `get_semantic_ids` (`semantic_entropy.py:175-211`).
Two responses are equivalent if entailment holds in *both* directions; the rule depends on
`strict_entailment` (default `True`, `utils.py:134-135`):

```python
# semantic_entropy.py:180-190
implication_1 = model.check_implication(text1, text2, example=example)
implication_2 = model.check_implication(text2, text1, example=example)
if strict_entailment:
    semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
else:
    implications = [implication_1, implication_2]
    semantically_equivalent = (0 not in implications) and ([1, 1] != implications)
```

The clustering is a greedy connected-components pass: each unlabeled string starts a new cluster
id, and every later equivalent string inherits it (`semantic_entropy.py:194-209`). Before
clustering, when DeBERTa is used and `condition_on_question` is true (default), each response is
prefixed with the question: `responses = [f'{question} {r}' for r in responses]`
(`compute_uncertainty_measures.py:239-240`).

**Cluster probabilities and final SE.** Per-cluster log-probability via `logsumexp_by_id`
(`semantic_entropy.py:214-237`); the default `sum` aggregation log-sum-exps sequence likelihoods
within a cluster (with a constant `- 5.0` offset, `semantic_entropy.py:227`). The entropy is a
Monte-Carlo estimate over clusters, `predictive_entropy` (`semantic_entropy.py:240-249`):
`entropy = -np.sum(log_probs) / len(log_probs)`. Driven at
`compute_uncertainty_measures.py:254-272`, sweeping token-aggregation (`mean`/`sum`) ×
cluster-aggregation variants, producing keys like `semantic_entropy`, `semantic_entropy_sum`,
`semantic_entropy_sum-normalized`, etc.

Also computed: **`cluster_assignment_entropy`** (`semantic_entropy.py:257-278`) — entropy of the
cluster-size distribution, ignoring token likelihoods:

```python
# semantic_entropy.py:273-277
counts = np.bincount(semantic_ids)
probabilities = counts/n_generations
entropy = - (probabilities * np.log(probabilities)).sum()
```

This `cluster_assignment_entropy` is the SE measure the **probe stage actually consumes**
(see §5/§7).

**Continuous or binarised?** Stage 2 stores SE **continuous** (raw floats). Binarisation happens
only later in the probe stage (§5).

**Save format/keys.** Pickle `uncertainty_measures.pkl` re-saved via `utils.save`
(`compute_uncertainty_measures.py:385`). `result_dict` gains:
- `semantic_ids` — list (per validation example) of cluster-id lists (`compute_uncertainty_measures.py:247`)
- `uncertainty_measures` — dict updated with all `entropies` defaultdict keys, incl.
  `cluster_assignment_entropy`, `regular_entropy*`, `semantic_entropy*` (`:340`), plus `p_ik` (`:368`)
- `validation_is_false`, `validation_unanswerable` (`:329, 332`)
- `alt_validation_accuracies_mean`, `alt_validation_is_false` (`:344-345`)

Each entropy value is a Python list, one entry per validation example, **in validation-dict
iteration order** (`compute_uncertainty_measures.py:194`).

---

## 4. Hidden-state extraction

**Function / file:** `HuggingfaceModel.predict` in
`semantic_uncertainty/uncertainty/models/huggingface_models.py:239-394`. Hidden states are captured
because `generate()` is called with `output_hidden_states=True` (`huggingface_models.py:269`).

HF's `decoder_hidden_states`/`hidden_states` is a tuple indexed by **generation step**; each step
is itself a tuple over **layers**, and layer 0 is the embedding output (HF convention), so a model
with L transformer blocks yields L+1 entries per step. The code picks the generation step
corresponding to the last generated content token: `n_generated = token_stop_index - n_input_token`
(`huggingface_models.py:313-315`), with guards for edge cases (`:326-348`).

**Three positions are saved:**

1. **Last-token scalar embedding** (`embedding`, used by p_ik). Last generation step → last layer →
   last token (`huggingface_models.py:350-353`):
   ```python
   last_layer = last_input[-1]                 # last layer only
   last_token_embedding = last_layer[:, -1, :].cpu()
   ```
   Shape `[1, hidden_dim]` (single layer).

2. **Second-to-last generated token, all layers** (`emb_tok_before_eos`, the SEP **SLT** position).
   `huggingface_models.py:356-363`:
   ```python
   # pick generation step n_generated - 2 (with guards)
   sec_last_input = hidden[n_generated - 2]
   sec_last_token_embedding = torch.stack([layer[:, -1, :] for layer in sec_last_input]).cpu()
   ```

3. **Last input token before generation, all layers** (`emb_last_tok_before_gen`, the SEP **TBG**
   = "token before generation" position). `huggingface_models.py:365-367`:
   ```python
   last_tok_bef_gen_input = hidden[0]          # step 0 = the prompt forward pass
   last_tok_bef_gen_embedding = torch.stack([layer[:, -1, :] for layer in last_tok_bef_gen_input]).cpu()
   ```

**Layers stored / shape.** For the two SEP embeddings, **all layers** are stacked (`torch.stack`
over the per-layer tuple). Each layer contributes `[:, -1, :]` → `[1, hidden_dim]`; stacking over
`L+1` layers gives shape **`[num_layers+1, 1, hidden_dim]`** (e.g. Llama-2-7b: `[33, 1, 4096]`).
The scalar p_ik embedding stores only the final layer. **Index 0 is the embedding-layer output**
(HF convention).

These are returned as a 3-tuple
`hidden_states = (last_token_embedding, sec_last_token_embedding, last_tok_bef_gen_embedding)`
(`huggingface_models.py:385-392`), moved to CPU at `generate_answers.py:194-196`, and pickled inside
each record's `most_likely_answer` dict under keys `embedding`, `emb_tok_before_eos`,
`emb_last_tok_before_gen` (`generate_answers.py:215-222`). **Save format/location:** pickle in the
wandb run dir, same as §2.

One important asymmetry: full per-layer SEP embeddings are saved **only for the most-likely (i==0)
answer**. The high-temp `responses` tuples store only the scalar `embedding`
(`generate_answers.py:232`). Probes therefore train on the most-likely answer's hidden states, with
the SE label derived from the spread of the high-temp samples.

---

## 5. Layer selection / probe training

Two routes: the standalone scripts (`run_llama2_probe.py` / `run_falcon_probe.py`,
in-distribution only) and the full notebook (`train-latent-probe.ipynb`, adds OOD + layer-band
concatenation). Both copy the same probe functions.

**Per-layer evaluation.** Hidden states are loaded transposed so axis 0 is the layer
(`run_llama2_probe.py:68-72`):
```python
tbg_dataset = torch.stack([record['most_likely_answer']['emb_last_tok_before_gen'] for record in generations.values()]).squeeze(-2).transpose(0, 1).to(torch.float32)
```
Resulting shape `[num_layers, N_samples, hidden_dim]`. `train_single_metric`
(`run_llama2_probe.py:179-198`) loops over the first axis, training one `LogisticRegression()` per
layer, recording per-layer test AUROC; the summary reports `mean AUROC` and
`best layer = argmax` (`run_llama2_probe.py:265-271`).

**Single layer vs band.** The standalone scripts evaluate each layer **independently** (no
concatenation). The notebook adds a contiguous-band selector, `decide_layer_range` (notebook
cell 37):
```python
def decide_layer_range(Ds, metric='entropy', limit=33):  # upper bound = num_layers+1
    ...
    for i in range(limit):
        for j in range(i+1, limit):
            if j - i < 5:        # band must span >5 layers
                continue
            if average(i, j) > best_mean:
                best_mean = average(i, j); best_range = [i, j]
    return best_mean, best_range
```
It picks the contiguous band (≥5 layers) with the best mean ID AUROC, separately for SEP and Acc.
probe. `limit` must be set to `num_layers+1` (33 for Llama-2-7b).

**Concatenation across layers.** Only in the notebook (cell 38), `concat_Xs_and_ys` concatenates
the selected band along the feature axis before fitting one probe:
```python
X_train_cc = np.concatenate(np.array(X_trains)[layer_range], axis=1)
```
So the concatenated-probe input dimensionality is `len(layer_range) * hidden_dim`; the per-layer
probes have input dim `hidden_dim` (4096 for Llama-2-7b).

**Probe type / input dim.** `sklearn.linear_model.LogisticRegression` (default L2, `lbfgs`).
Single-layer probe input = `hidden_dim`; concatenated-band input = `band_size × hidden_dim`.

**SE binarisation.** Two functions, notebook cell 22 / `run_llama2_probe.py:205-224`. `best_split`
chooses the threshold minimizing within-group squared error (a 1-D 2-means / minimum-reconstruction-
error split) over 100 candidate thresholds:
```python
splits = np.linspace(1e-10, ents.max(), 100)
mse = np.sum((ents[low_idxs]-low_mean)**2) + np.sum((ents[high_idxs]-high_mean)**2)
return splits[np.argmin(split_mses)]
```
A single **universal** threshold is computed across all datasets' entropies
(`run_llama2_probe.py:242-243`), then `binarize_entropy` applies it: `<thres → 0`, `>thres → 1`
(`run_llama2_probe.py:220-224`). The binarised SE is the SEP target; the **Acc. probe** uses raw
`accuracies` as its target (`run_llama2_probe.py:256-258`). The entropy source is
`measures['uncertainty_measures']['cluster_assignment_entropy']` (`run_llama2_probe.py:66`).

**Train/val/test split.** `create_Xs_and_ys` (`run_llama2_probe.py:76-101`): test 0.1, then val 0.2
of remainder, `random_state=42`. AUROC reported with a 1000-resample bootstrap CI (`bootstrap_func`,
`:105-146`).

---

## 6. Run mechanics and environment

**Conda env:** `sep_enviroment.yaml` (name `se_probes`, Python 3.11.5). Key pins:
- **PyTorch 2.1.1** (`py3.11_cuda11.8_cudnn8.7.0`), CUDA 11.8 (`sep_enviroment.yaml:138-139`)
- **transformers 4.35.2**, **tokenizers 0.15.0** (`:292, 287`) — CLAUDE.md flags these as a hard
  cap (do not upgrade)
- accelerate 0.25.0, bitsandbytes 0.41.2.post2, scikit-learn 1.3.2, scipy 1.11.4, datasets 2.12.0,
  wandb 0.16.0, openai 1.3.7, numpy 1.26.0

**Launch.** All via CLI argparse (`utils.get_parser`, `utils.py:19-148`); no config files.
Stage 1: `python generate_answers.py --model_name=... --dataset=... --num_samples=...`. By default
Stage 1 auto-chains into Stage 2 (`--compute_uncertainties` default True → `main_compute(args)`,
`generate_answers.py:283-287`), and Stage 2 auto-chains into Stage 3 (`--analyze_run` default True,
`compute_uncertainty_measures.py:391-394`). SLURM batch via `slurm/run.sh` (loops datasets,
`srun python ...`). **Must run from `semantic_uncertainty/`** — imports are relative (e.g.
`from compute_uncertainty_measures import main`).

**GPU selection.** No explicit `CUDA_VISIBLE_DEVICES` in code; models load with
`device_map="auto"` / `max_memory={0: '80GIB'}` (`huggingface_models.py:135-137`); tensors
`.to("cuda")` (`:245`). 70B/65B use `accelerate.infer_auto_device_map` across GPUs 0/1 (`:154-165`).
SLURM requests `--gres=gpu:a100:2` (commented `#SBATCH` template, `slurm/run.sh:1-5`).

**Weights & Biases — required.** `wandb.init` runs unconditionally in Stage 1
(`generate_answers.py:51-57`) and `utils.save` writes into `wandb.run.dir` then calls `wandb.save`
(`utils.py:339-342`) — there is no non-wandb save path, so wandb is structurally required. Stages
are linked by **run IDs**: Stage 2 takes `--eval_wandb_runid` (and optional `--train_wandb_runid`),
opens the old run via `wandb.Api()`, and downloads its pickles
(`compute_uncertainty_measures.py:47-89`):
```python
old_run = api.run(f'{args.restore_entity_eval}/{project}/{args.eval_wandb_runid}')
old_run.file(filename).download(replace=False, exist_ok=True, root=wandb.run.dir)
```
Requires env vars `USER`, `WANDB_ENT`; `--entity` defaults from `WANDB_SEM_UNC_ENTITY`
(`utils.py:20`). Offline mode would break the `wandb.Api()` artifact restore in Stage 2 when
`assign_new_wandb_id=True`. The auto-chain path avoids re-download by reusing the active run dir
(`restore()` no-op branch, `:70-73`) — when chained, `assign_new_wandb_id` is forced False
(`generate_answers.py:276-277`).

**Step-to-step handoff.** Within a single chained run: same `wandb.run.dir`, pickles read straight
back. Across separate invocations: the new run downloads the prior run's `*.pkl` by run id into its
own dir.

**Things that broke / workarounds (from CLAUDE.md, machine-specific):**
- Llama-2 Meta weights gated → redirected to `NousResearch/Llama-2-7b-chat-hf` ungated mirror via a
  branch in `huggingface_models.py:119-125` (original `meta-llama` mapping kept commented at
  `:109-115`).
- Home-dir 12 GB quota silently breaks downloads → `HF_HOME`, conda envs/pkgs, wandb dir all
  relocated to `/vol/bitbucket`.
- `~/.bashrc` `[ -z "$PS1" ] && return` guard hides exports from SLURM → exports must sit above it.
- `OPENAI_API_KEY` required even when unused (client built at import in
  `uncertainty/utils/openai.py`); placeholder accepted.
- wandb 0.16.0 CLI rejects the 86-char Imperial key, but `wandb.init()` accepts it (store in
  `WANDB_API_KEY`/`~/.netrc`).
- Mistral-7B / Phi-3 fail under pinned transformers/tokenizers; falcon-7b works.
- N too small breaks probes: `test_size` leaving a single-class test split makes
  `roc_auc_score`/`log_loss` raise (use N≥400).
- `*_UNANSWERABLE` AUROC is `nan` on trivia_qa (no unanswerable questions) — expected.

---

## 7. Reproducibility join keys

This is the critical and somewhat fragile part: **the pipeline joins records by ordering/position,
not by a propagated id.**

**Within Stage 1.** Each prompt's records live under one dict key, `example['id']`
(`generate_answers.py:166`). `most_likely_answer` (hidden states, accuracy) and `responses` (the N
samples) sit in the same record, so prompt ↔ response ↔ hidden state are co-located by id at write
time.

**Stage 1 → Stage 2.** Stage 2 iterates `for idx, tid in enumerate(validation_generations)`
(`compute_uncertainty_measures.py:194`), i.e. dict iteration order. For each it **appends** to
parallel Python lists: `result_dict['semantic_ids']` (`:247`), every `entropies[...]` list incl.
`cluster_assignment_entropy` (`:250`), `validation_is_true`/`validation_embeddings` (`:218-221`).
These lists carry **no id** — element *k* corresponds to the *k*-th validation example in dict order.
The join key between `uncertainty_measures.pkl` and `validation_generations.pkl` is therefore
**positional alignment under a stable dict iteration order** (Python 3.7+ insertion order; both
files derive from the same dict, so order matches as long as neither is rebuilt). `tid` (the id) is
used only for logging, not stored alongside the entropy arrays.

**Stage 2 → Stage 4 (probes).** `load_dataset` (`run_llama2_probe.py:60-72`) reads both pickles and
aligns them **purely by order**:
```python
entropy     = torch.tensor(measures['uncertainty_measures']['cluster_assignment_entropy'])   # list order
accuracies  = torch.tensor([record['most_likely_answer']['accuracy'] for record in generations.values()])
tbg_dataset = torch.stack([record['most_likely_answer']['emb_last_tok_before_gen'] for record in generations.values()])...
```
`entropy[i]`, `accuracies[i]`, and `tbg/slt_dataset[:, i, :]` are matched by index `i`. The entropy
list (from `uncertainty_measures.pkl`) and the embedding/accuracy arrays (from
`validation_generations.pkl` via `.values()`) are assumed to enumerate the **same examples in the
same order**. The id (dict key) is never used as a join key downstream.

**Practical implication for a port:** ordering stability is the entire join contract. It holds here
because (a) `generations` is an insertion-ordered dict written once, and (b) Stage 2 walks it in
order and appends without reordering or filtering ahead of the entropy arrays. It is fragile to:
subsetting one file but not the other, `--num_eval_samples` truncating Stage 2
(`compute_uncertainty_measures.py:322-325`) so entropy has fewer entries than `generations` has
records, or any re-serialization that changes dict order. Recommendation if adapting: carry
`example['id']` alongside every entropy/embedding array and join on it explicitly rather than
relying on positional alignment.
