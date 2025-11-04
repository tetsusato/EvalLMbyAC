from src.cache.cache import Cache
from print_diff_hl import print_diff_hl, get_diff_hl
#from gptzip import ArithmeticCoder
import omegaconf
import os
from hydra import compose, initialize
#import mlflow
#import openai
from pathlib import Path
import polars as pl
from perplexity import Perplexity
from result import Result
import sys
import time
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Callable
#from vllm import LLM, SamplingParams

from logging import config
import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)
progress = logging.getLogger("progress")
summary = logging.getLogger("summary")

if __name__ == "__main__":
    progress.info("START")
    config = sys.argv[1]
    if len(sys.argv) >= 3:
        override_options = sys.argv[2:]
        print(f"override_options={override_options}")
    else:
        override_options = None
    #with initialize(config_path="config", job_name=__file__):
    with initialize(config_path=".", job_name=__file__):
        if override_options:
            cfg = compose(config_name=sys.argv[1],
                          return_hydra_config=True,
                          overrides=override_options,
                          )
        else:
            cfg = compose(config_name=sys.argv[1],
                          return_hydra_config=True,
                          )
    progress.info(f"cfg={cfg.config}")
    # config_path="config"からconfig_path="."に変えてみた互換性のため
    cfg = cfg.config
    exp_title=cfg.exp.title
    exp_summary=cfg.exp.summary
    progress.info(f"config={config} title={exp_title} summary={exp_summary}")
    progress.info(f"cfg={cfg}")
    logger.info(f"cfg={cfg}")
#    exe = LMCR(cfg)
    model_name = cfg.exp.llm
    logger.info(f"model={model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input = "吾輩は猫である。名前はまだ無い。"
    input_ids_tensor = tokenizer(input, return_tensors='pt').input_ids.flatten()
    logger.info(f"input_ids_tensor={input_ids_tensor}")
    for token_id in input_ids_tensor:
        print(f"token_id={token_id}")
    decoded = [tokenizer.decode([token_id]) for token_id in input_ids_tensor]

    logger.info(f"decoded={decoded}")
    progress.info("END")
