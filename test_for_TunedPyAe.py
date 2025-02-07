from accelerate import Accelerator
import csv
from executor import Executor
#from result import Result
from transformers import BertJapaneseTokenizer, BertModel
import argparse
import polars as pl
from print_diff_hl import print_diff_hl, get_diff_hl
import hydra
from hydra import compose, initialize
import io
import sys
import time
import torch


model_list = [

    "rinna/gemma-2-baku-2b-it", # OK
#    "2121-8/TinySlime-1.1B-v1.0", # NG
   "llm-jp/llm-jp-3-13b", # NG
    "llm-jp/llm-jp-3-3.7b", # NG
    "llm-jp/llm-jp-3-1.8b", # OK

#    "microsoft/Phi-3.5-mini-instruct", # NG
#    "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp", # NG
#    "gpt2",
#    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
#    "llm-jp/llm-jp-3-172b",
#    "meta-llama/Llama-3.1-405B",
#    "meta-llama/Llama-3.1-70B-Instruct",
#    "meta-llama/Llama-3.1-70B",
#    "meta-llama/Llama-3.1-8B",
#    "meta-llama/Llama-3.2-1B-Instruct",
#    "meta-llama/Llama-3.2-3B-Instruct",
    "./gemma-2-baku-2b-it_20241216",
#    "Qwen/Qwen2.5-3B",
#    "HuggingFaceTB/SmolLM2-135M-Instruct",
#    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
#    "SakanaAI/EvoLLM-JP-A-v1-7B",
#    "elyza/Llama-3-ELYZA-JP-8B",
#    "rinna/llama-3-youko-8b-instruct",
#    "Qwen/Qwen2.5-0.5B-Instruct",
#    "meta-llama/Llama-3.2-1B-Instruct",
#    "Qwen/Qwen2-0.5B",
#    "Qwen/Qwen1.5-0.5B",
#    "microsoft/phi-4"
    ]


#save_directory = f"{model}_model"

# スクリプトのディレクトリをPythonパスに追加
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logging import config
import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)
progress = logging.getLogger("progress")
summary = logging.getLogger("summary")

#input_dir = "../resume-data/TextOutputs"
input_dir = "../resume-data/doda-samples"


#accelerator = Accelerator()

if __name__ == "__main__":

    logger.info("START")
    logger.debug("START")

    with initialize(config_path="config", job_name=__file__):
        #cfg = compose(config_name="exp_test.yaml", return_hydra_config=True)
        cfg = compose(config_name=sys.argv[1], return_hydra_config=True)
    logger.debug(f"cfg={cfg}({cfg.__class__})")
    executor = Executor(
                        cfg,
                       )
    
    executor.llm_models_test(executor.execute_ppl)
    executor.llm_models_test(executor.execute_ae)
