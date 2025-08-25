from src.cache.cache import Cache
from print_diff_hl import print_diff_hl, get_diff_hl
#from gptzip import ArithmeticCoder
import omegaconf
import os
from hydra import compose, initialize
import mlflow
import openai
from pathlib import Path
import polars as pl
from perplexity import Perplexity
from result import Result
import sys
import time
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Callable
from vllm import LLM, SamplingParams

from logging import config
import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)
progress = logging.getLogger("progress")
summary = logging.getLogger("summary")


class LMCR:
    def __init__(self,
                 cfg: omegaconf.dictconfig.DictConfig,
                 ):
        self.exp_title = cfg.exp.title
        self.tokenizer = None
        self.model = None
        self.top_log_dir = cfg.exp.log_dir
        self.exp_log_dir = f"{self.top_log_dir}/{self.exp_title}"
        self.model_huggingface_id = cfg.exp.llm
        if "gemma" in self.model_huggingface_id:
            self.is_gemma = True
        else:
            self.is_gemma = False
        self.input_dir = "." # 自明だけど互換性のために
        self.input = cfg.exp.input
        self.input_limit = cfg.exp.inputs_limit
        self.device: str = cfg.exp.device
        self.use_cache: bool = cfg.exp.use_cache # Use KV Cache
        #self.hosting = "huggingface"
        #self.hosting = "vllm"
        #self.hosting = "openai" # vllmサーバを使うとき
        self.hosting: str = cfg.exp.hosting

        if not os.path.exists(self.top_log_dir):
            os.mkdir(self.top_log_dir)
        if not os.path.exists(self.exp_log_dir):
            os.mkdir(self.exp_log_dir)

        if cfg.cache.enable:
            cache_filename = f"{self.exp_title}.db"
            self.cache = Cache(cfg=cfg,
                               cache_filename=cache_filename,
                               )
        else:
            self.cache = None
        mlflow.log_params(cfg)

    def get_cache_key(self,
                      func, # assume "execute_hoge"
                      ):
        func_name = func.__name__.split("_")[1] # assume "execute_hoge"
        if self.hosting == "huggingface":
            model_name = self.model.name_or_path
        elif self.hosting == "vllm":
            model_name = self.model.llm_engine.model_config.model
        elif self.hosting == "openai":
            model_name = self.model_huggingface_id
        cache_key = f"{self.exp_title}"\
                   +f"-{model_name}"\
                   +f"-{self.input}"\
                   +f"-{func_name}"
        return cache_key
    
    def record_to_mlflow(self,
                         results_df: pl.DataFrame,
                         ):
        #logger = logging.getLogger("httpx")          
        #logger.setLevel(logging.ERROR)      
        for row in results_df.iter_rows(named=True):
            logger.info(f"row={row}")
            for key, value in row.items():
                logger.info(f"key={key}, value={value}")
                import numbers
                if isinstance(value, numbers.Number):
                    mlflow.log_metric(key, value)
                else:
                    mlflow.set_tag(key, value)

    def calculate_token_efficiency(self,
                                   ) -> float:
        input_dir = self.input_dir
        text_path = self.input
        print(f"dir={input_dir}")
        print(f"text_path={text_path}")
        msg = Path(f"{input_dir}/{text_path}").read_text(encoding="utf-8")
        print(f"msg len={len(msg)}")
        print(f"msg examples={msg[:40]}")
        print(f"msg examples len={len(msg[:40])}")        
        input_ids_tensor = self.tokenizer(msg, return_tensors='pt').input_ids
        print(f"input_ids_tensor examples={input_ids_tensor[-1, :40]}")
        decoded = self.tokenizer.decode(input_ids_tensor[-1, :40],
                                        clean_up_tokenization_spaces=False,
                                        )
        progress.info(f"decoded examples={decoded}(tokenizer.decode)")
        decoded_token_list = []
        #for i in input_ids_tensor[-1, :40]:
        for i in input_ids_tensor[-1, :]:
            #print(f"token={i}")
            decoded = self.tokenizer.decode(i,
                                        clean_up_tokenization_spaces=False,
                                        )
            #print(f"decoded={decoded}")
            decoded_token_list.append(decoded)
            

        print(f"decoded_token_list={decoded_token_list}")
        decoded = self.tokenizer.convert_ids_to_tokens(input_ids_tensor[-1, :40],
                                        )
        #mlflow.set_tag("token_separated_response
        progress.info(f"decoded examples2={decoded}(tokenizer.convert_ids_tokens)")
        tokenized_size = len(input_ids_tensor[-1])
        print(f"tokenized msg len={tokenized_size}")
        mlflow.log_metric("tokenized_size", tokenized_size)
        input_len = len(msg)
        tokens_len = len(input_ids_tensor[-1])
        ratio = tokens_len / input_len
        print(f"ratio={ratio}")
        mlflow.log_metric("token_efficiency", ratio)
        return ratio
        
    def generating_test(self,
                        ):
        if self.is_gemma:
            messages = [
                {"role": "user", "content": "あなたは日本文化に詳しいAIアシスタントです．吾輩は，に続く文章を生成して下さい"},
              ]
        else:
            messages = [
                {"role": "system", "content": "あなたは日本文化に詳しいAIアシスタントです"},
                {"role": "user", "content": "吾輩は，に続く文章を生成して下さい"},
              ]
        prompt = self.tokenizer.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True,
                                                    )
        print(f"prompt={prompt}")
        if self.hosting == "huggingface":
            chat_input = self.tokenizer(prompt,
                                        return_tensors="pt",
                                        return_token_type_ids=False,
                                        ).to(self.model.device)
            print(f"chat_input={chat_input}")
            chat_outputs = self.model.generate(**chat_input, max_new_tokens=50)
            output_text = chat_outputs[0][chat_input["input_ids"].shape[-1]:]
            response = self.tokenizer.decode(output_text, skip_special_tokens=True)
        elif self.hosting == "vllm":
            sampling_params = SamplingParams(
                                             max_tokens=50,
                                             temperature=0.7,  # 必要に応じて調整
                                             skip_special_tokens=True  # 特殊トークンをスキップ
                                         )
            outputs = self.model.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text
        elif self.hosting == "openai":
            response = self.model.completions.create(
                                          model=self.model_huggingface_id,
                                          prompt=prompt,
                                          max_tokens=50,
                                          temperature=0.7,
                                          top_p=0.9,
                                          stop=None,
                                          stream=False
                                          )
            response = response.choices[0].text
    
        print("\nAssistant Response:", response)
        mlflow.set_tag("response", response)


        
    def prepare_llm(self,
                    model_name: str,
                    ):
        if self.hosting == "huggingface":
            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                  device_map="auto",
                                                  #device_map="cuda:1",
                                                  #device_map=device,
                                                  local_files_only=True,
                                                  #device_map="cpu",
                                                  trust_remote_code=True,
                                                  #torch_dtype=torch.float64
                                                   torch_dtype="auto" # fujise method
                                                  )
            model.eval()
        elif self.hosting == "vllm":
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
            model = LLM(model=model_name, task="generate")
        elif self.hosting == "openai":
            model = openai.OpenAI(
                                   base_url="http://localhost:8000/v1",
                                   api_key="dummy"  # vLLMでは認証不要だが必須パラメータ
                                   )
        
        
        self.model = model
        logger.info(f"model={model.__class__}({model_name})")
        progress.info(f"model={model.__class__}({model_name})")
        if model_name == "./gemma-2-baku-2b-it_20241216":
            tokenizer = AutoTokenizer.from_pretrained("rinna/gemma-2-baku-2b-it")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer
        
    def input_analysis(self,
                        func: Callable[
                                       [bool, # cache_val
                                        str,  # input_dir
                                        str,  # text_path
                                        str,  # basic_info
                                        str,  # func_name
                                       ],
                                       pl.DataFrame # result
                                      ],
                        ):

        start = time.time()
        from result import Result
        func_name = func.__name__.split("_")[1] # assume "execute_hoge"
        model_name = self.model_huggingface_id
        logger.info(f"model={model_name}")

        self.prepare_llm(model_name)
        
        results_df = pl.DataFrame(schema=Result.__annotations__)


        text_path = self.input

        basic_info = f"device={self.device}, cache={self.use_cache}, model={model_name}, text={text_path}"
        is_success = True
        logger.debug(f"cache={self.cache}")
        logger.debug(f"basic_info={basic_info}")
        if self.cache is not None:
            cache_key = self.get_cache_key(func)
            logger.debug(f"cache key={cache_key}")
            if cache_key is not None:
                logger.debug("cache key found")
                cache_val = self.cache.get(cache_key)
            else:
                # 通条，ここは通らないはず
                logger.debug("cache key not found")
                cache_val = None
        else:
            cache_val = None
        if cache_val is None:
            logger.debug("cache val not found. execute func..")
            result_df = func(cache_val,
                             self.input_dir,
                             text_path,
                             basic_info,
                             func_name,
                            )
            logger.debug(f"result_df={result_df}")
            if self.cache is not None:
                self.cache.set(cache_key,
                               result_df,
                               )
        else:
            logger.debug("cache val found.")
            result_df = cache_val
        results_df = results_df.vstack(result_df)

        self.record_to_mlflow(results_df)            
        progress.info(results_df)
        model_name_path = model_name.replace("/", "-")
        exp_snap_save_path = f"summary/{self.exp_title}_{model_name_path}-{func_name}.parquet"
        results_df.write_parquet(exp_snap_save_path)
        end = time.time()
        input_analysis_time = end - start
        logger.info(f"total time={input_analysis_time}")
        progress.info(f"total time={input_analysis_time}")

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
    
    def encode_decode_test(self,
                           input_dir: str,
                           text_path: str,
                           basic_info: str,
                           func_name: str,
                           ):
        if self.hosting == "huggingface":
            from gptzip.gptzip import ArithmeticCoder
        elif self.hosting == "openai":
            from gptzip.gptzip_online import ArithmeticCoder
        
        coder = ArithmeticCoder(lm=self.model,
                                tokenizer=self.tokenizer,
                                use_cache=self.use_cache,
                                cache=self.cache,
                                )
        msg = Path(f"{input_dir}/{text_path}").read_text(encoding="utf-8")
        msg_example = msg[0:40]
        logger.info(f"file={text_path}, contents={msg_example}")
        progress.info(f"file={text_path}, contents={msg_example}")
        start = time.time()        
        coder, code, num_padded_bits, text_limit = self.encoding(input_dir,
                                                     text_path,
                                                     basic_info,
                                                     func_name,
                                                     )
        end = time.time()
        encode_time = end-start
        start = time.time()
        decoded_string, is_success = self.decoding(coder,
                                                   code,
                                                   num_padded_bits,
                                                   text_limit,
                                                   input_dir,
                                                   text_path,
                                                   basic_info,
                                                   func_name,
                                                   )
        end = time.time()
        decode_time = end - start
        
        if msg.rstrip("\r\n") != decoded_string.rstrip("\r\n"):
            logger.info(f"!!!!!!!!!!!!!! The input string does ont match the output.")
            progress.info(f"!!!!!!!!!!!!!! The input string does ont match the output.")
            logger.info(f"input: {msg}")
            logger.info(f"output: {decoded_string}")
            logger.info(f"diff: {get_diff_hl(msg, decoded_string)}")
            is_success=False

        ratio = len(code)/len(msg)
        logger.info(f"LMCR: {len(code)}/{len(msg)}={ratio}")

        model_name = self.model.name_or_path
        title = f"{self.exp_title}(ae)"
        hosting = self.hosting
        result = Result(
                        title,
                        hosting,
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

        if self.cache is not None:
            cache_key = f"{self.exp_title}-{model_name}-{text_path}-{func_name}"
            self.cache.set(cache_key, result)

        result_df = pl.DataFrame([result])
        logger.info(f"result_df={result_df}")
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
        title = f"{self.exp_title}(ppl)"
        hosting = self.hosting
        result = Result(
                        title,
                        hosting,
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

        if self.cache is not None:
            cache_key = f"{self.exp_title}-{model_name}-{text_path}-{func_name}"
            self.cache.set(cache_key, result)

        result_df = pl.DataFrame([result])
        print(result_df)
        return result_df

    def encoding(self,
                 input_dir: str,
                 text_path: str,
                 basic_info: str,
                 func_name: str,
                 ):
        total_start = time.time()
        if self.hosting == "huggingface":
            from gptzip.gptzip import ArithmeticCoder
        elif self.hosting == "openai":
            from gptzip.gptzip_online import ArithmeticCoder
        coder = ArithmeticCoder(lm=self.model,
                                tokenizer=self.tokenizer,
                                #use_cache=self.use_cache,
                                cache=self.cache,
                                model_huggingface_id = self.model_huggingface_id,
                                )
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
        progress.info(f"[1] Code... `{code[:text_limit]}` ({len(code)} bytes, num_padded_bits={num_padded_bits})")
        print("\n" * 5)

        return coder, code, num_padded_bits, text_limit
    
    def decoding(self,
                 coder,
                 code,
                 num_padded_bits,
                 text_limit,
                 input_dir: str,
                 text_path: str,
                 basic_info: str,
                 func_name: str,
                 ):
        start = time.time()
        decoded_string, is_success = coder.decode(code, num_padded_bits=num_padded_bits)
        end = time.time()
        decode_time = end - start
        logger.info(f"decode elapsed={decode_time}({basic_info})")
        progress.info(f"decode elapsed={decode_time}({basic_info})")
        progress.info(f"[2] Decoded: {decoded_string[:text_limit]}")
        logger.debug(f"[2] Decoded: {decoded_string}")

        return decoded_string, is_success
    

if __name__ == "__main__":
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
    print(f"cfg={cfg.config}")
    # config_path="config"からconfig_path="."に変えてみた互換性のため
    cfg = cfg.config
    exp_title=cfg.exp.title
    exp_summary=cfg.exp.summary
    progress.info(f"config={config} title={exp_title} summary={exp_summary}")
    progress.info(f"cfg={cfg}")
    exe = LMCR(cfg)
    run_name = f"{exp_title}(ae)"
    mlflow.end_run()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tracking_uri(uri="http://localhost:8080")
        mlflow.set_tag("algorithm", "ae")
        exe.input_analysis(exe.execute_ae)
        exe.calculate_token_efficiency()    
        #exe.generating_test()
    mlflow.end_run()
    """
    run_name = f"{exp_title}(ppl)"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tracking_uri(uri="http://localhost:8080")
        mlflow.set_tag("algorithm", "ppl")
        exe.input_analysis(exe.execute_ppl)
        exe.calculate_token_efficiency()    
        exe.generating_test()
    mlflow.end_run()
    """
