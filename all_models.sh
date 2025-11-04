#!/bin/bash

# モデル名のリスト
<<COMMENT
models=(
#    "Qwen/Qwen2.5-32B-Instruct"
#    )
    "google/gemma-1.1-7b-it"
    "google/gemma-7b-it"
    "Qwen/Qwen1.5-0.5B-Chat"
    "tiiuae/Falcon3-1B-Instruct"
    "google/gemma-2b-it"
    "llm-jp/llm-jp-3-440m-instruct2"
    "Qwen/Qwen2-0.5B-Instruct"
    "google/gemma-1.1-2b-it"
    "tiiuae/Falcon3-3B-Instruct"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    "llm-jp/llm-jp-3-980m-instruct2"
    "tiiuae/Falcon3-7B-Instruct"
    "weblab-GENIAC/Tanuki-8B-dpo-v1.0"
    "Qwen/Qwen1.5-4B-Chat"
    "llm-jp/llm-jp-3-1.8b-instruct"
    "meta-llama/Llama-2-7b-chat-hf"

    "Qwen/Qwen2-1.5B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "llm-jp/llm-jp-3-3.7b-instruct2"
    "Qwen/Qwen2.5-3B-Instruct"
    "llm-jp/llm-jp-3-3.7b-instruct"
    "Qwen/Qwen1.5-14B-Chat"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen1.5-32B-Chat"
    "llm-jp/llm-jp-3-13b-instruct"
    "google/gemma-2-27b-it"
    "llm-jp/llm-jp-3.1-1.8b-instruct4"
    "Qwen/Qwen2.5-14B-Instruct"
#    "Qwen/Qwen2.5-32B-Instruct"
    "llm-jp/llm-jp-3.1-13b-instruct4"
COMMENT
models=(
    "google/gemma-1.1-2b-it"
    "google/gemma-2b-it"
    "llm-jp/llm-jp-3-440m-instruct2"
    "Qwen/Qwen2-0.5B-Instruct"
    "tiiuae/Falcon3-3B-Instruct"
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
)
# 入力長のリスト
lengths=(
  11288
  7594
  6080
  4517
  3013
  1527
)

# 実行ループ
for model in "${models[@]}"; do
  for len in "${lengths[@]}"; do
    echo "Running with model: $model, input length: $len"
    uv run lmcr.py config/lmcr_exp_fit2025exp07.yaml \
       config.cache.enable=False \
       config.exp.hosting=huggingface \
       config.exp.llm="${model}" \
       config.exp.input="JSAI-TEST-DATA/len_${len}.txt"
  done
done
