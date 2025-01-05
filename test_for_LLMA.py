from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BertJapaneseTokenizer, BertModel
import argparse
from gptzip import ArithmeticCoder
import os
import sys
from pathlib import Path
import time
import torch

#model = "gpt2"
#model = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
#model = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
#model = "meta-llama/Llama-3.1-405B-Instruct"
#model = "llm-jp/llm-jp-3-172b"
#model = "llm-jp/llm-jp-3-13b"
#model = "llm-jp/llm-jp-3-3.7b"
#model = "llm-jp/llm-jp-3-1.8b"
#model = "meta-llama/Llama-3.1-70B"
#model = "meta-llama/Llama-3.1-8B"
#model = "rinna/gemma-2-baku-2b-it"
#model = "microsoft/Phi-3.5-mini-instruct"
#model = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"

model_list = [

    "rinna/gemma-2-baku-2b-it",
    "llm-jp/llm-jp-3-13b",
#    "2121-8/TinySlime-1.1B-v1.0",

    "llm-jp/llm-jp-3-3.7b",
    "llm-jp/llm-jp-3-1.8b",

    "microsoft/Phi-3.5-mini-instruct",
    "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp",
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

input_dir = "../resume-data/TextOutputs"

device="auto"
use_cache=True

#accelerator = Accelerator()

if __name__ == "__main__":
    logger.info("START")
    logger.debug("START")

    for model_name in model_list:
        
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                              device_map="auto",
                                              #device_map=device,

                                              #device_map="cpu",
                                              #trust_remote_code=True,
                                              #torch_dtype=torch.float64
                                              )
        model.eval()
        logger.info(f"model={model.__class__}({model_name})")
        progress.info(f"model={model.__class__}({model_name})")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        coder = ArithmeticCoder(lm=model,
                                tokenizer=tokenizer,
                                use_cache=False)
        file_list = os.listdir(input_dir)
        logger.debug(f"input files={file_list}")
        for text_path in file_list:
        #for text_path in ["resume_0102.txt"]:
            total_start = time.time()
            basic_info = f"device={device}, cache={use_cache}, model={model_name}, text={text_path}"
            is_success = True
            #msg = "This is a pen."
            ##msg = Path("../LLMA/alice.txt").read_text(encoding="utf-8")
            ##msg = Path("doda_cv.txt").read_text(encoding="utf-8")
            
            #msg = Path(f"{input_dir}/{text_path}").read_text(encoding="utf-8")
            msg = Path(f"{input_dir}/{text_path}").read_text(encoding="utf-8")[0:1200]
            
            msg_example = msg[0:40]
            logger.info(f"file={text_path}, contents={msg_example}")
            progress.info(f"file={text_path}, contents={msg_example}")
            text_limit = 300
            progress.info(f"[0] Encoding... `{msg[:text_limit]}`")
            start = time.time()
            code, num_padded_bits = coder.encode(
                msg, 
                return_num_padded_bits=True, 
            )
            end = time.time()
            encode_time = end-start
            logger.info(f"encode elapsed={encode_time}({basic_info})")
            progress.info(f"encode elapsed={encode_time}({basic_info})")
            print(f"[1] Code... `{code[:text_limit]}` ({len(code)} bytes, num_padded_bits={num_padded_bits})")
            print("\n" * 5)
            start = time.time()
            decoded_string, is_success = coder.decode(code, num_padded_bits=num_padded_bits)
            end = time.time()
            decode_time = end - start
            logger.info(f"decode elapsed={decode_time}({basic_info})")
            progress.info(f"decode elapsed={decode_time}({basic_info})")
            progress.info(f"[2] Decoded: {decoded_string[:text_limit]}")
            logger.debug(f"[2] Decoded: {decoded_string}")

            if msg != decoded_string:
                logger.info(f"!!!!!!!!!!!!!! The input string does ont match the output.")
                progress.info(f"!!!!!!!!!!!!!! The input string does ont match the output.")
                logger.info(f"input: {msg}")
                logger.info(f"output: {decoded_string}")
                is_success=False
            
            print(f"Compression {len(msg)} bytes to {len(code)} bytes.")
            logger.info(f"Compression {len(msg)} bytes to {len(code)} bytes.({basic_info})")
            progress.info(f"Compression {len(msg)} bytes to {len(code)} bytes.({basic_info})")
            data = f"data: {model_name}-{text_path}, "
            data += f"size: {len(msg)}-{len(code)}, "
            data += f"time: {encode_time}-{decode_time}"
            ratio = len(code)/len(msg)
            logger.info(f"{data}: ratio = {ratio}({basic_info})")
            progress.info(f"ratio={ratio}({basic_info})")

            total_end = time.time()
            summary.info(f"=========================== {text_path} ======================")
            summary.info(f"basic info={basic_info}")
            summary.info(f"success?={is_success}")
            summary.info(f"Compression {len(msg)} bytes to {len(code)} bytes")
            summary.info(f"{data}: ratio = {ratio}({basic_info})")
            summary.info(f"total time elapsed: {total_end-total_start}")
            
