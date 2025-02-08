from cache.cache import Cache
from print_diff_hl import print_diff_hl, get_diff_hl
from gptzip import ArithmeticCoder
import omegaconf
import os
from hydra import compose, initialize
from pathlib import Path
import polars as pl
from perplexity import Perplexity
from result import Result
import sys
import time
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Callable

from logging import config
import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)
progress = logging.getLogger("progress")
summary = logging.getLogger("summary")


class Executor:
    def __init__(self,
                 cfg: omegaconf.dictconfig.DictConfig,
                 ):
        self.exp_title = cfg.exp.title
        self.tokenizer = None
        self.model = None
        self.top_log_dir = cfg.exp.log_dir
        self.exp_log_dir = f"{self.top_log_dir}/{self.exp_title}"
        self.model_list = cfg.exp.llms
        self.input_dir = cfg.exp.inputs
        self.input_limit = cfg.exp.inputs_limit
        self.device: str = cfg.exp.device
        self.use_cache: bool = cfg.exp.use_cache
        if not os.path.exists(self.top_log_dir):
            os.mkdir(self.top_log_dir)
        if not os.path.exists(self.exp_log_dir):
            os.mkdir(self.exp_log_dir)

        cache_filename = f"{self.exp_title}.db"
        self.cache = Cache(cfg=cfg,
                           cache_filename=cache_filename,
                           )

    # Done
    """
    def convert_with_func_name(self):
        keysvals = self.cache.listKeysVals()
        for key, val in keysvals:
            #print(f"key={key}, val={val}")
            print(f"key={key}, val={val.__class__}")
            if key.endswith("txt-ae"): # new cache
                print(f"{key} is already new one.")
            else:
                self.cache.delete(f"^{key}$")
                new_key = f"{key}-ae"
                self.cache.set(new_key, val)
    """     
        
    def llm_models_test(self,
                        func: Callable[
                                       [bool, # cache_val
                                        str,  # input_dir
                                        str,  # text_path
                                        str,  # basic_info
                                       ],
                                       pl.DataFrame # result
                                      ],
                        ):

        from result import Result
        func_name = func.__name__.split("_")[1] # assume "execute_hoge"
        for model_name in self.model_list:
            results_df = pl.DataFrame(schema=Result.__annotations__)

            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                  device_map="auto",
                                                  #device_map=device,
                                                  #       local_files_only=True,
                                                  #device_map="cpu",
                                                  #trust_remote_code=True,
                                                  #torch_dtype=torch.float64
                                                   torch_dtype="auto" # fujise method
                                                  )
            self.model = model
            model.eval()
            logger.info(f"model={model.__class__}({model_name})")
            progress.info(f"model={model.__class__}({model_name})")
            if model_name == "./gemma-2-baku-2b-it_20241216":
                tokenizer = AutoTokenizer.from_pretrained("rinna/gemma-2-baku-2b-it")
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer = tokenizer
            file_list = os.listdir(self.input_dir)[0:self.input_limit]
            logger.debug(f"input files={file_list}")

            for text_path in file_list: # これがノーマル
            #for text_path in [file_list[0]]: # テストでファイル一個だけ試すやつ
            #for text_path in file_list[0:2]: # テストでファイル二個試すやつ
            #for text_path in ["resume_0102.txt"]:

                basic_info = f"device={self.device}, cache={self.use_cache}, model={model_name}, text={text_path}"
                is_success = True
                cache_key = f"{self.exp_title}-{model_name}-{text_path}-{func_name}"
                cache_val = self.cache.get(cache_key)

                """
                if cache_val is None:
                    result_df = self.encode_decode_test(self.input_dir,
                                                        text_path,
                                                        basic_info,
                                                       )
                else:
                    
                    result_df = pl.DataFrame([cache_val])
                """
                result_df = func(cache_val,
                                 self.input_dir,
                                 text_path,
                                 basic_info,
                                 func_name,
                                )
                results_df = results_df.vstack(result_df)

                
            print(results_df)

            model_name_path = model_name.replace("/", "-")
            exp_snap_save_path = f"summary/{self.exp_title}_{model_name_path}-{func_name}.parquet"
            results_df.write_parquet(exp_snap_save_path)
        
    def execute_ae(self,
                cache_val: bool,
                input_dir: str,
                text_path: str,
                basic_info: str,
                func_name: str,
              ) -> pl.DataFrame:
        if cache_val is None:
            result_df = self.encode_decode_test(input_dir,
                                                text_path,
                                                basic_info,
                                                func_name=func_name,
                                               )
        else:

            result_df = pl.DataFrame([cache_val])
        return result_df    

    def execute_ppl(self,
                cache_val: bool,
                input_dir: str,
                text_path: str,
                    basic_info: str,
                    func_name: str,
              ) -> pl.DataFrame:
        if cache_val is None:
            result_df = self.perplexity_test(input_dir,
                                             text_path,
                                             basic_info,
                                             func_name=func_name
                                            )
        else:

            result_df = pl.DataFrame([cache_val])
        return result_df    


    def perplexity_test(self,
                           input_dir: str,
                           text_path: str,
                           basic_info: str,
                        func_name: str,
                           ):
        
        total_start = time.time()
        perplexity = Perplexity(
                                lm=self.model,
                                tokenizer=self.tokenizer
                               )
        msg = Path(f"{input_dir}/{text_path}").read_text(encoding="utf-8")
        msg_example = msg[0:40]
        logger.info(f"file={text_path}, contents={msg_example}")
        progress.info(f"file={text_path}, contents={msg_example}")
        text_limit = 300
        progress.info(f"[0] Encoding... `{msg[:text_limit]}`")
        start = time.time()
        score = perplexity.calculate(
                           msg
                          )
        end = time.time()
        encode_time = end-start
        model_name = self.model.name_or_path
        result = Result(
                        self.exp_title,
                        model_name,
                        text_path,
                        len(msg),
                        None,
                        None,
                        score,
                        encode_time,
                        None,
                        basic_info,
                       )

        cache_key = f"{self.exp_title}-{model_name}-{text_path}-{func_name}"
        self.cache.set(cache_key, result)

        result_df = pl.DataFrame([result])
        print(result_df)
        return result_df
        
    def encode_decode_test(self,
                           input_dir: str,
                           text_path: str,
                           basic_info: str,
                           func_name: str,
                           ):
        total_start = time.time()
        coder = ArithmeticCoder(lm=self.model,
                                tokenizer=self.tokenizer,
                                use_cache=True)
        msg = Path(f"{input_dir}/{text_path}").read_text(encoding="utf-8")
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

        model_name = self.model.name_or_path
        result = Result(
                        self.exp_title,
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

        cache_key = f"{self.exp_title}-{model_name}-{text_path}-{func_name}"
        self.cache.set(cache_key, result)

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
        summary.info(result.summary_string())

        csv_row = result.to_csv()
        summary.info(f"to excel:\n{csv_row}")
        del coder
        return result_df
    
if __name__ == "__main__":
    print("exe")
    with initialize(config_path="config", job_name=__file__):
        cfg = compose(config_name=sys.argv[1], return_hydra_config=True)
    exe = Executor(cfg)
    
    exe.convert_with_func_name()
