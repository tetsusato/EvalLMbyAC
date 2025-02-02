from accelerate import Accelerator
import csv
from result import Result
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BertJapaneseTokenizer, BertModel
import argparse
import polars as pl
from print_diff_hl import print_diff_hl, get_diff_hl
from gptzip import ArithmeticCoder
import hydra
from hydra import compose, initialize
import io
import os
import sys
from pathlib import Path
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

device="auto"
use_cache=True

#accelerator = Accelerator()


def to_csv(*args):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(args)
    return output.getvalue().strip()  # 改行を削除

if __name__ == "__main__":
    logger.info("START")
    logger.debug("START")

    with initialize(config_path="config", job_name=__file__):
        #cfg = compose(config_name="exp_test.yaml", return_hydra_config=True)
        cfg = compose(config_name=sys.argv[1], return_hydra_config=True)
    exp_title = cfg.exp.title
    top_log_dir = cfg.exp.log_dir
    exp_log_dir = f"{top_log_dir}/{exp_title}"
    if not os.path.exists(top_log_dir):
        os.mkdir(top_log_dir)
    if not os.path.exists(exp_log_dir):
        os.mkdir(exp_log_dir)
    
    model_list = cfg.exp.llms
    input_dir = cfg.exp.inputs
    input_limit = cfg.exp.inputs_limit
    results_df = pl.DataFrame(schema=Result.__annotations__)
    for model_name in model_list:
        
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                              device_map="auto",
                                              #device_map=device,
                                              #       local_files_only=True,
                                              #device_map="cpu",
                                              #trust_remote_code=True,
                                              #torch_dtype=torch.float64
                                               torch_dtype="auto" # fujise method
                                              )
        model.eval()
        logger.info(f"model={model.__class__}({model_name})")
        progress.info(f"model={model.__class__}({model_name})")
        if model_name == "./gemma-2-baku-2b-it_20241216":
            tokenizer = AutoTokenizer.from_pretrained("rinna/gemma-2-baku-2b-it")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        file_list = os.listdir(input_dir)[0:input_limit]
        logger.debug(f"input files={file_list}")
        original_size_list = []
        compression_size_list = []
        compression_rate_list = []
        encode_time_list = []
        decode_time_list = []
        for text_path in file_list: # これがノーマル
        
        #for text_path in [file_list[0]]: # テストでファイル一個だけ試すやつ
        #for text_path in file_list[0:2]: # テストでファイル二個試すやつ
        #for text_path in ["resume_0102.txt"]:
            coder = ArithmeticCoder(lm=model,
                                tokenizer=tokenizer,
                                    use_cache=True)
            total_start = time.time()
            basic_info = f"device={device}, cache={use_cache}, model={model_name}, text={text_path}"
            is_success = True
            #msg = "This is a pen."
            ##msg = Path("../LLMA/alice.txt").read_text(encoding="utf-8")
            ##msg = Path("doda_cv.txt").read_text(encoding="utf-8")
            
            msg = Path(f"{input_dir}/{text_path}").read_text(encoding="utf-8")
            #msg = Path(f"{input_dir}/{text_path}").read_text(encoding="utf-8")[0:100] # for test
            #msg = Path(f"{input_dir}/{text_path}").read_text(encoding="utf-8")[0:1200]
            #msg = "This is a pen.\nThat is an apple."
            #msg = "This is a pen.\n"
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

            if msg.rstrip("\r\n") != decoded_string.rstrip("\r\n"):
                logger.info(f"!!!!!!!!!!!!!! The input string does ont match the output.")
                progress.info(f"!!!!!!!!!!!!!! The input string does ont match the output.")
                logger.info(f"input: {msg}")
                logger.info(f"output: {decoded_string}")
                logger.info(f"diff: {get_diff_hl(msg, decoded_string)}")
                is_success=False

            ratio = len(code)/len(msg)

            result = Result(
                            exp_title,
                            model_name,
                            text_path,
                            len(msg),
                            len(code),
                            len(decoded_string),
                            ratio,
                            encode_time,
                            decode_time,
                            basic_info,
                           )
            result_df = pl.DataFrame([result])
            print(result_df)
            logger.info(f"Compression {len(msg)} bytes to {len(code)} bytes.({basic_info})")
            progress.info(f"Compression {len(msg)} bytes to {len(code)} bytes.({basic_info})")
            logger.info(f"DeCompression {len(code)} bytes to {len(decoded_string)} bytes.({basic_info})")
            progress.info(f"DeCompression {len(code)} bytes to {len(decoded_string)} bytes.({basic_info})")            
            
            data = f"data: {model_name}-{text_path}, "
            data += f"size: {len(msg)}-{len(code)}, "
            data += f"time: {encode_time}-{decode_time}"
            logger.info(f"{data}: ratio = {ratio}({basic_info})")
            progress.info(f"ratio={ratio}({basic_info})")

            total_end = time.time()
            """
            summary.info(f"=========================== {text_path} ======================")
            summary.info(f"basic info={basic_info}")
            summary.info(f"success?={is_success}")
            summary.info(f"Compression {len(msg)} bytes to {len(code)} bytes")
            summary.info(f"Compression ratio {len(code)/len(msg)}")
            summary.info(f"DeCompression {len(code)} bytes to {len(decoded_string)} bytes")
            summary.info(f"{data}: ratio = {ratio}({basic_info})")
            summary.info(f"total time elapsed: {total_end-total_start}")
            """
            summary.info(result.summary_string())

            csv_header = to_csv("Model File Name",
                                "File Name",
	                        "Original Size (bytes)",
	                        "Compressed Size (bytes)",
	                        "Compression Ratio",
	                        "Compression Time (s)",
	                        "Decompression Time (s)"
                                )
            """
            csv_row = to_csv(model_name,
                             text_path,
                             len(msg),
                             len(code),
                             len(code)/len(msg),
                             encode_time,
                             decode_time
                             )
            """
            csv_row = result.to_csv()
            summary.info(f"to excel:\n{csv_row}")
            original_size_list.append(len(msg))
            compression_size_list.append(len(code))
            compression_rate_list.append(len(code)/len(msg))
            encode_time_list.append(encode_time)
            decode_time_list.append(decode_time)
            results_df = results_df.vstack(result_df)
            del coder
        from statistics import mean
        csv_row = to_csv(model_name,
                             f"{model_name}_average",
                             mean(original_size_list),
                             mean(compression_size_list),
                             mean(compression_rate_list),
                             mean(encode_time_list),
                             mean(decode_time_list),
                             )
        summary.info(f"{model_name} summary:\n{csv_row}")
        print(results_df)

    exp_save_path = f"summary/{exp_title}.parquet"
    results_df.write_parquet(exp_save_path)
