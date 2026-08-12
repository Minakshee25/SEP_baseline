#!/bin/bash
# DeepSeek-LLM-7B-Chat Stage-1 SMOKE TEST -- 4th cross-LLM target (deepseek-ai/deepseek-llm-7b-chat,
# plain LlamaForCausalLM, 30 layers, 4096-dim). Mirrors smoke_llama3.sh.
#
# WHY: de-risks a fresh-family Stage-1 build. Verify (1) the sanctioned load branch added to
# huggingface_models.py loads the tokenizer+weights under transformers 4.44.2; (2) generations
# terminate cleanly (stop tokens / EOS); (3) answers look sane and SE labels are non-degenerate.
# Watch for tokenizer decode quirks like the Llama-3 " ?" issue (the generic offset-recovery in
# predict() at huggingface_models.py handles non-round-tripping tokenizers; confirm by reading answers).
#
# Env: se_probes_llama3 (transformers 4.44 -- also loads DeBERTa's .bin; see cross-llm-llama3-env).
# deepseek-ai/deepseek-llm-7b-chat is public (no gated access); ~13GB downloads on first run.

cd "$(dirname "$0")/.."                                   # repo root
source /data/sv/miniconda3/etc/profile.d/conda.sh
conda activate se_probes_llama3
export HF_HOME=/vol/bitbucket/mn1025/hf_cache OPENAI_API_KEY=placeholder
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONUNBUFFERED=1

echo ">>> DeepSeek-LLM-7B-Chat Stage-1 smoke: 3 prompts, trivia_qa, env=se_probes_llama3"
python -m amortized_ue.stage1 --smoke --smoke_num_samples 3 \
  --model_name deepseek-llm-7b-chat --dataset trivia_qa

echo
echo ">>> INSPECT the printed answers above: do they stop cleanly and look sane? SE labels non-degenerate?"
echo "    yes -> build the real dataset;  run-on/garbled -> fix deepseek stop-tokens/prompt first."
