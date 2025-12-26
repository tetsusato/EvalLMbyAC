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
    "Qwen/Qwen2.5-32B-Instruct"
    "llm-jp/llm-jp-3.1-13b-instruct4"
)
COMMENT

# 読み込みたい設定ファイル（引数で渡すか、ここで定義するか）
# 引数があればそれを使用、なければエラーにしたい
if [ -n "$1" ]; then
    CONFIG_FILE="$1"
else
    echo "Error: Config file not specified."
    exit 1
fi
# パス補完（configディレクトリにあると仮定）
if [ -f "config/$CONFIG_FILE" ]; then
    CONFIG_PATH="config/$CONFIG_FILE"
elif [ -f "$CONFIG_FILE" ]; then
    CONFIG_PATH="$CONFIG_FILE"
else
    echo "Error: Config file $CONFIG_FILE not found."
    exit 1
fi

# YAMLからモデルリストを抽出するPythonスクリプト
read_models_py=$(cat <<EOF
import sys
import yaml
try:
    with open("$CONFIG_PATH", "r") as f:
        conf = yaml.safe_load(f)
        models = conf.get("exp_models", {}).get("target_models", [])
        print(" ".join(models))
except Exception as e:
    print(f"e={e}", end="")
EOF
)

# モデルリストを取得
MODELS_STR=$(uv run python -c "$read_models_py")

if [ -z "$MODELS_STR" ]; then
    echo "Error: Could not read target_models from $CONFIG_PATH"
    exit 1
fi

# 配列に変換
IFS=' ' read -r -a models <<< "$MODELS_STR"

# 入力長のリスト
lengths=(
#  11288
#  7594
#  6080
#  4517
  3013
  1527
  
)

# 設定ファイルのパス
##CONFIG_FILE="config/lmcr_exp_ipsj2026_06.yaml"
#CONFIG_FILE="lmcr_exp_ipsj2026_paper_01.yaml"

# 設定ファイル名から実験名を抽出
# 例: config/lmcr_exp_ipsj2026_06.yaml → ipsj2026_06
CONFIG_BASENAME=$(basename "$CONFIG_FILE" .yaml)  # lmcr_exp_ipsj2026_06
EXP_TITLE=${CONFIG_BASENAME#lmcr_exp_}  # ipsj2026_06

echo "Config file: $CONFIG_FILE"
echo "Experiment title: $EXP_TITLE"

# 実験開始前に過去のrunをクリア（一度だけ実行）
# 実験開始前に過去のrunをクリア（設定ファイルで有効な場合のみ）
# YAMLからclear_previous_runsの値を取得
read_cleanup_flag_py=$(cat <<EOF
import sys
import yaml
try:
    with open("$CONFIG_PATH", "r") as f:
        conf = yaml.safe_load(f)
        #print(f"conf={conf}")
        flag = conf.get("exp", {}).get("clear_previous_runs", False)
        print(str(flag).lower())
except Exception as e:
    print("false")
EOF
)
SHOULD_CLEANUP=$(uv run python -c "$read_cleanup_flag_py")

if [ "$SHOULD_CLEANUP" = "true" ]; then
    echo "Clearing previous runs for $EXP_TITLE..."
    uv run python -c "
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri('http://localhost:8080')
runs = mlflow.search_runs(
    search_all_experiments=True,
    filter_string=\"tags.exp_title = '$EXP_TITLE'\",
    output_format='pandas'
)
if not runs.empty:
    client = MlflowClient()
    for run_id in runs['run_id']:
        client.delete_run(run_id)
    print(f'Deleted {len(runs)} runs')
else:
    print('No previous runs found')
"
else
    echo "Skipping cleanup (clear_previous_runs is not true in config)"
fi

# 実行ループ
echo "Starting experiments..."
START_TIME=$(date +%s)

for model in "${models[@]}"; do
  for len in "${lengths[@]}"; do
    echo "Running with model: $model, input length: $len"
    PYTHONPATH=. uv run src/lmcr/lmcr.py "$CONFIG_FILE" \
       exp.hosting=huggingface \
       exp.llm="${model}" \
       exp.input="JSAI-TEST-DATA/len_${len}.txt"
       #cache.l1_cache.enable=False \
       #cache.l2_cache.enable=True \       
  done
done

END_TIME=$(date +%s)
ELAPSED_SEC=$((END_TIME - START_TIME))
echo "Total execution time: $(date -u -d @${ELAPSED_SEC} +%H:%M:%S) (Total ${ELAPSED_SEC} seconds)"
