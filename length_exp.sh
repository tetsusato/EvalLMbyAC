#!/bin/bash

# モデル名のリスト
models=(

#  "llm-jp/llm-jp-3-13b-instruct"
#  "llm-jp/llm-jp-3-3.7b-instruct"
#  "llm-jp/llm-jp-3-1.8b-instruct"

#  "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    #  "HuggingFaceTB/SmolLM2-135M-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

# 入力長のリスト
lengths=(
#  11288
#  7594
#  6080
#  4517
  3013
#  1527
)

# 実行ループ
for model in "${models[@]}"; do
  for len in "${lengths[@]}"; do
    echo "Running with model: $model, input length: $len"
    uv run python lmcr.py lmcr_exp1 \
      exp.llm="$model" \
      cache.enable=True \
      exp.input="JSAI-TEST-DATA/len_${len}.txt"
  done
done
