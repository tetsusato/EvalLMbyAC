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
#from vllm import LLM, SamplingParams

from logging import config
import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)
progress = logging.getLogger("progress")
summary = logging.getLogger("summary")


class LMCR:
    """
    Language Model Compression Ratio (LMCR) 実験クラス
    
    言語モデルを用いた算術符号化による圧縮・解凍実験を実行し、
    圧縮率、パープレキシティ、トークン効率などを評価する。
    
    Args:
        cfg: Hydraによる設定ファイル（omegaconf.DictConfig）
    """
    def __init__(self,
                 cfg: omegaconf.dictconfig.DictConfig,
                 ):
        # 実験設定の初期化
        self.exp_title = cfg.exp.title  # 実験タイトル
        self.encoding_algorithm = cfg.exp.encoding_algorithm  # 符号化アルゴリズム
        self.tokenizer = None  # トークナイザー（後で初期化）
        self.tokenizer_name = None  # トークナイザー名
        self.model = None  # 言語モデル（後で初期化）
        self.compression_algorithm = None  # 圧縮アルゴリズム名
        self.top_log_dir = cfg.exp.log_dir  # ログディレクトリのトップ
        self.exp_log_dir = f"{self.top_log_dir}/{self.exp_title}"  # 実験ごとのログディレクトリ
        self.model_huggingface_id = cfg.exp.llm  # HuggingFaceモデルID
        # Gemmaモデルの場合は特別な処理が必要
        if "gemma" in self.model_huggingface_id:
            self.is_gemma = True
        else:
            self.is_gemma = False
        # 入力データの設定
        self.input_dir = "."  # 入力ファイルのディレクトリ（自明だけど互換性のために）
        self.input = cfg.exp.input  # 入力ファイル名
        self.input_limit = cfg.exp.inputs_limit  # 入力データの制限数
        self.device: str = cfg.exp.device  # 使用デバイス（cuda:0, cpu等）
        self.use_cache: bool = cfg.exp.use_cache  # KV Cacheを使用するか
        # モデルのホスティング方式（huggingface, vllm, openai）
        # huggingface: HuggingFace Transformersで直接実行
        # vllm: vLLMライブラリで実行（高速推論）
        # openai: OpenAI互換APIサーバ経由で実行（vllmサーバ等）
        self.hosting: str = cfg.exp.hosting
        
        # VR-LMCR計算用パラメータ（IPSJ2026実験用）
        self.max_vocab_size: int | None = cfg.exp.get('max_vocab_size', None)
        self.gamma: float = cfg.exp.get('gamma', 1.0)

        # ログディレクトリの作成
        if not os.path.exists(self.top_log_dir):
            os.mkdir(self.top_log_dir)
        if not os.path.exists(self.exp_log_dir):
            os.mkdir(self.exp_log_dir)

        # キャッシュの初期化（実験結果の再利用のため）
        logger.info(f"Debug Cache Config: l1_enable={cfg.cache.l1_cache.enable}, l2_enable={cfg.cache.l2_cache.enable}")
        
        # L1 CACHE: Research Results (tied to exp_title)
        if cfg.cache.l1_cache.enable:
            self.L1_CACHE = Cache(cfg=cfg.cache.l1_cache,
                                  cache_filename=f"{self.exp_title}.db",
                                  )
        else:
            self.L1_CACHE = None
        
        # L2 CACHE: LLM Outputs (tied to model_name + input file)
        # Avoid collisions/redundancy by using model+input as key for separation
        self.model_name_safe = self.model_huggingface_id.replace("/", "-")
        input_basename = Path(self.input).stem
        input_basename_safe = input_basename.replace("/", "-").replace(".", "-")
        l2_filename = f"{self.model_name_safe}-{input_basename_safe}.db"
    
        if cfg.cache.l2_cache.enable:
            self.L2_CACHE = Cache(cfg=cfg.cache.l2_cache,
                                  cache_filename=l2_filename,
                                  )
        else:
            self.L2_CACHE = None
        
        # 古いコードがあったらエラーで検知する
        self.cache = None
        # MLflowに設定パラメータを記録
        # Note: mlflow.log_params()はmlflow.start_run()のコンテキスト内で呼ぶ必要がある
        # そうしないと自動的に新しいrunが作成されてしまう
        # パラメータの記録はmain関数内のmlflow.start_run()の後で行う

    def get_cache_key(self,
                      func,  # 実行する関数（例: execute_ae, execute_ppl）
                      ):
        """
        キャッシュキーを生成する
        
        Args:
            func: 実行する関数（execute_ae, execute_ppl等）
            
        Returns:
            str: キャッシュキー（実験タイトル-モデル名-入力-関数名）
        """
        func_name = func.__name__.split("_")[1]  # "execute_hoge" -> "hoge"
        # ホスティング方式に応じてモデル名を取得
        if self.hosting == "huggingface":
            model_name = self.model.name_or_path
        elif self.hosting == "vllm":
            model_name = self.model.llm_engine.model_config.model
        elif self.hosting == "openai":
            model_name = self.model_huggingface_id
        # キャッシュキーの生成: 実験タイトル-モデル名-入力ファイル-関数名
        cache_key = f"{self.exp_title}"\
                   +f"-{model_name}"\
                   +f"-{self.input}"\
                   +f"-{func_name}"
        return cache_key
    
    def record_to_mlflow(self,
                         results_df: pl.DataFrame,
                         ):
        """
        実験結果をMLflowに記録する
        
        Args:
            results_df: 実験結果のDataFrame
        """
        #logger = logging.getLogger("httpx")          
        #logger.setLevel(logging.ERROR)      
        # DataFrameの各行を処理
        for row in results_df.iter_rows(named=True):
            logger.info(f"row={row}")
            for key, value in row.items():
                logger.info(f"key={key}, value={value}")
                import numbers
                # 数値はメトリックとして、それ以外はタグとして記録
                if isinstance(value, numbers.Number):
                    mlflow.log_metric(key, value)
                    logger.info(f"Logged metric: {key}={value}")
                else:
                    mlflow.set_tag(key, value)
                    logger.info(f"Logged tag: {key}={value}")

    def calculate_token_efficiency(self,
                                   ) -> float:
        """
        トークン効率を計算する
        
        入力テキストの文字数に対するトークン数の比率を計算し、
        トークナイザーの効率性を評価する。
        
        Returns:
            float: トークン効率（トークン数/文字数）
        """
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
        
        # tokenized_sizeは入力テキスト長に依存して変化する（同じモデルでも入力が異なれば変わる）
        # モデルごとの語彙数（vocab_size）とは異なるので注意
        mlflow.log_metric("tokenized_size", tokenized_size)
        input_len = len(msg)
        tokens_len = len(input_ids_tensor[-1])
        ratio = tokens_len / input_len
        print(f"ratio={ratio}")
        mlflow.log_metric("token_efficiency", ratio)
        mlflow.log_metric("vocab_size", self.tokenizer_vocab_size)
        return ratio
        
    def generating_test(self,
                        ):
        """
        言語モデルの生成テスト
        
        「吾輩は、に続く文章を生成して下さい」というプロンプトで
        モデルの文章生成能力をテストする。
        """
        # Gemmaモデルはsystemロールをサポートしないため分岐
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
        """
        言語モデルとトークナイザーを準備する
        
        ホスティング方式（huggingface/vllm/openai）に応じて
        適切な方法でモデルをロードする。
        
        Args:
            model_name: HuggingFaceモデルID
        """
        if self.hosting == "huggingface":
            if self.device == "auto" or self.device == "cpu":
                device_map = self.device
            else:
                 # "cuda:0"のような指定の場合、そのデバイスにマッピング
                 # device_map={"": "cuda:0"} のような形式が必要な場合もあるが
                 # 一般的には "cuda:0" や "auto" で動作する。
                 # ここではconfigの値をそのまま使う簡単な実装にするか、
                 # 元のコードにあった {"": "cuda:0"} の形式を尊重して分岐するか検討が必要。
                 # 元のコードが device_map={"": "cuda:0"} だったので、
                 # もし self.device が "cuda:0" なら {"": self.device} にする処理を入れる。
                 if "cuda" in self.device:
                     device_map = {"": self.device}
                 else:
                     device_map = self.device

            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                  device_map=device_map,
                                                  #device_map="cuda:0",
                                                  #device_map="auto",
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
            self.tokenizer_name = "rinna/gemma-2-baku-2b-it"
            self.tokenizer_vocab_size = tokenizer.vocab_size
            
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer_name = model_name
            self.tokenizer_vocab_size = tokenizer.vocab_size
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
        """
        入力データに対して分析を実行する（メインの実行フロー）
        
        キャッシュの確認、モデルの準備、分析の実行、結果の記録を行う。
        
        Args:
            func: 実行する分析関数（execute_ae または execute_ppl）
        """

        start = time.time()
        from result import Result
        func_name = func.__name__.split("_")[1] # assume "execute_hoge"
        model_name = self.model_huggingface_id
        logger.info(f"model={model_name}")

        self.prepare_llm(model_name)
        

        text_path = self.input

        basic_info = f"device={self.device}, cache={self.use_cache}, model={model_name}, text={text_path}"
        is_success = True
        logger.debug(f"cache={self.cache}")
        logger.debug(f"basic_info={basic_info}")
        # キャッシュが有効な場合、過去の実験結果を確認
        # overwrite が True の場合はキャッシュを無視する
        if self.L1_CACHE is not None and not self.L1_CACHE.overwrite:
            cache_key = self.get_cache_key(func)
            logger.debug(f"cache key={cache_key}")
            if cache_key is not None:
                logger.debug("cache key found")
                cache_val = self.L1_CACHE.get(cache_key)
            else:
                # 通常、ここは通らないはず
                logger.debug("cache key not found")
                cache_val = None
        else:
            if self.L1_CACHE is not None and self.L1_CACHE.overwrite:
                logger.info("L1_CACHE overwrite is True. Skipping cache lookup.")
            cache_val = None
        # キャッシュがない場合は実際に実験を実行
        if cache_val is None:
            logger.info("cache val not found. execute func..")
            result_df = func(cache_val,
                             self.input_dir,
                             text_path,
                             basic_info,
                             func_name,
                            )
            logger.debug(f"result_df={result_df}")
            # 実行結果をキャッシュに保存
            if self.L1_CACHE is not None:
                self.L1_CACHE.set(cache_key,
                               result_df,
                               )
        else:
            # キャッシュがある場合はそれを使用
            logger.debug("cache val found.")
            result_df = cache_val
        results_df = result_df  # 空のDataFrameとのvstackを削除

        # 結果をMLflowに記録
        self.record_to_mlflow(results_df)            
        progress.info(results_df)
        # 結果をParquetファイルとして保存
        model_name_path = model_name.replace("/", "-")
        exp_snap_save_path = f"summary/{self.exp_title}_{model_name_path}-{func_name}.parquet"
        results_df.write_parquet(exp_snap_save_path)
        # 実行時間の記録
        end = time.time()
        input_analysis_time = end - start
        logger.info(f"total time={input_analysis_time}")
        progress.info(f"total time={input_analysis_time}")

    def execute_ae(self,
                l1_cache_val: bool,
                input_dir: str,
                text_path: str,
                basic_info: str,
                func_name: str,
              ) -> pl.DataFrame:
        """
        算術符号化（Arithmetic Encoding）による圧縮・解凍実験を実行
        
        Args:
            cache_val: キャッシュされた値（あればそれを使用）
            input_dir: 入力ファイルのディレクトリ
            text_path: 入力ファイルのパス
            basic_info: 実験の基本情報
            func_name: 関数名
            
        Returns:
            pl.DataFrame: 実験結果
        """
        if l1_cache_val is None:
            result_df = self.encode_decode_test(input_dir,
                                                text_path,
                                                basic_info,
                                                func_name=func_name,
                                               )
        else:
            result_df = pl.DataFrame([l1_cache_val])
        return result_df    
    def execute_ppl(self,
                l1_cache_val: bool,
                input_dir: str,
                text_path: str,
                    basic_info: str,
                    func_name: str,
              ) -> pl.DataFrame:
        """
        パープレキシティ（Perplexity）を計算する実験を実行
        
        Args:
            cache_val: キャッシュされた値（あればそれを使用）
            input_dir: 入力ファイルのディレクトリ
            text_path: 入力ファイルのパス
            basic_info: 実験の基本情報
            func_name: 関数名
            
        Returns:
            pl.DataFrame: 実験結果
        """
        if l1_cache_val is None:
            result_df = self.perplexity_test(input_dir,
                                             text_path,
                                             basic_info,
                                             func_name=func_name
                                            )
        else:

            result_df = pl.DataFrame([l1_cache_val])
        return result_df    
    
    def encode_decode_test(self,
                           input_dir: str,
                           text_path: str,
                           basic_info: str,
                           func_name: str = "ae",
                           ) -> pl.DataFrame:
        logger.info("Starting encode_decode_test...")
        start = time.time()
        """
        エンコード・デコードのテストを実行
        
        入力テキストを算術符号化で圧縮し、その後解凍して
        元のテキストと一致するかを確認する。
        
        Args:
            input_dir: 入力ファイルのディレクトリ
            text_path: 入力ファイルのパス
            basic_info: 実験の基本情報
            func_name: 関数名
            
        Returns:
            pl.DataFrame: 実験結果（圧縮率、実行時間等）
        """
        # ホスティング方式に応じて適切なArithmeticCoderをインポート
        if self.hosting == "huggingface":
            from gptzip.gptzip import ArithmeticCoder
            self.compression_algorithm = "gptzip"
        elif self.hosting == "openai":
            from gptzip.gptzip_online import ArithmeticCoder
            self.compression_algorithm = "gptzip_online"
        
        if self.encoding_algorithm == "ae":
            # 算術符号化器の初期化
            coder = ArithmeticCoder(lm=self.model,
                                    tokenizer=self.tokenizer,
                                    use_cache=self.use_cache,
                                    cache=self.cache,
                                    )
        msg = Path(f"{input_dir}/{text_path}").read_text(encoding="utf-8")
        msg_example = msg[0:40]
        logger.info(f"file={text_path}, contents={msg_example}")
        progress.info(f"file={text_path}, contents={msg_example}")
        # エンコード（圧縮）の実行

        l2_cache_key = f"{self.model_name_safe}-{input_dir}-{text_path}-{func_name}"
        code = None
        if self.L2_CACHE is not None and not self.L2_CACHE.overwrite:

            code = self.L2_CACHE.get(l2_cache_key)
            if code is not None:
                logger.info(f"Cache hit for {l2_cache_key}")
                progress.info(f"Cache hit for {l2_cache_key}")
                # キャッシュを使った場合，msg == decoded_stringは保証されている
                decoded_string = msg
                encode_time = 0
                decode_time = 0

        if self.L2_CACHE is None or code is None:
            # キャッシュを使わない設定か，キャッシュがない場合
            start = time.time()
            coder, code, num_padded_bits, text_limit = self.encoding(input_dir,
                                                     text_path,
                                                     basic_info,
                                                     func_name,
                                                     )
            end = time.time()
            encode_time = end-start
            # デコード（解凍）の実行
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
        
            # 入力と出力が一致するか検証（ロスレス圧縮の確認）
            if msg.rstrip("\r\n") != decoded_string.rstrip("\r\n"):
                logger.info(f"!!!!!!!!!!!!!! The input string does ont match the output.")
                progress.info(f"!!!!!!!!!!!!!! The input string does ont match the output.")
                logger.info(f"input: {msg}")
                logger.info(f"output: {decoded_string}")
                logger.info(f"diff: {get_diff_hl(msg, decoded_string)}")
                is_success=False

            # 検証完了ならキャッシュに保存
            if self.L2_CACHE is not None:
                self.L2_CACHE.set(l2_cache_key, code)

        # 圧縮率の計算
        ratio = len(code)/len(msg)
        logger.info(f"LMCR: {len(code)}/{len(msg)}={ratio}")

        # VR-LMCR（ナイーブ実装）の計算
        vr_lmcr_naive = None
        logger.info(f"Debug VR-LMCR: max_vocab_size={self.max_vocab_size}, tokenizer_vocab_size={self.tokenizer_vocab_size}")
        if self.max_vocab_size is not None and self.tokenizer_vocab_size is not None:
            vocab_ratio = self.tokenizer_vocab_size / self.max_vocab_size
            vr_lmcr_naive = ratio * (vocab_ratio ** self.gamma)
            logger.info(f"VR-LMCR_naive: {ratio} * ({self.tokenizer_vocab_size}/{self.max_vocab_size})^{self.gamma} = {vr_lmcr_naive}")
            progress.info(f"VR-LMCR_naive={vr_lmcr_naive}")
        else:
            logger.info("Skipped VR-LMCR calculation due to missing vocab size info")

        model_name = self.model.name_or_path
        #title = f"{self.exp_title}(ae)"
        title = self.exp_title
        hosting = self.hosting
        result = Result(
                        title,
                        self.encoding_algorithm,
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
                        vocab_size=self.tokenizer_vocab_size,
                        vr_lmcr_naive=vr_lmcr_naive,
                        vr_lmcr_realistic=None,  # 後で実装
                       )

        # if self.L2_CACHE is not None:
        #     cache_key = f"{self.exp_title}-{model_name}-{text_path}-{func_name}"
        #     self.L2_CACHE.set(cache_key, result)

        result_df = pl.DataFrame([result])
        logger.info(f"result_df={result_df}")
        logger.info(f"Compression {len(msg)} chars to {len(code)} bytes.({basic_info})")
        progress.info(f"Compression {len(msg)} chars to {len(code)} bytes.({basic_info})")
        logger.info(f"DeCompression {len(code)} bytes to {len(decoded_string)} chars.({basic_info})")
        progress.info(f"DeCompression {len(code)} bytes to {len(decoded_string)} chars.({basic_info})")            

        data = f"data: {model_name}-{text_path}, "
        data += f"size: {len(msg)}-{len(code)}, "
        data += f"time: {encode_time}-{decode_time}"
        logger.info(f"{data}: ratio = {ratio}({basic_info})")
        progress.info(f"ratio={ratio}({basic_info})")

        total_end = time.time()
        summary.info(result.summary_string())

        csv_row = result.to_csv()
        summary.info(f"to excel:\n{csv_row}")
        
        # Log algorithm metadata to MLflow
        mlflow.set_tag("tokenizer_name", self.tokenizer_name)
        mlflow.set_tag("compression_algorithm", self.compression_algorithm)
        mlflow.set_tag("llm_model", self.model_huggingface_id)
        
        del coder
        return result_df

    def perplexity_test(self,
                        input_dir: str,
                        text_path: str,
                        basic_info: str,
                        func_name: str,
                        ) -> pl.DataFrame:
        """
        パープレキシティ（困惑度）を計算する
        """
        l2_cache_key = f"{self.model_name_safe}-{text_path}-{func_name}"
        if self.L2_CACHE is not None and not self.L2_CACHE.overwrite:
            cache_val = self.L2_CACHE.get(l2_cache_key)
            if cache_val is not None:
                logger.info(f"L2 Cache hit for Perplexity: {l2_cache_key}")
                # cache_val は Result オブジェクトの想定
                return pl.DataFrame([cache_val])
        elif self.L2_CACHE is not None and self.L2_CACHE.overwrite:
            logger.info(f"L2_CACHE overwrite is True. Skipping cache lookup for {l2_cache_key}.")

        # キャッシュがない場合は計算
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
        score = perplexity.calculate(msg)
        end = time.time()
        encode_time = end - start
        
        model_name = self.model.name_or_path
        title = self.exp_title
        hosting = self.hosting
        
        result = Result(
                        title,
                        self.encoding_algorithm,
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
                        vocab_size=self.tokenizer_vocab_size,
                       )

        if self.L2_CACHE is not None:
            self.L2_CACHE.set(l2_cache_key, result)

        result_df = pl.DataFrame([result])
        print(result_df)
        return result_df

    def encoding(self,
                 input_dir: str,
                 text_path: str,
                 basic_info: str,
                 func_name: str,
                 ):
        """
        テキストを算術符号化で圧縮する
        
        Args:
            input_dir: 入力ファイルのディレクトリ
            text_path: 入力ファイルのパス
            basic_info: 実験の基本情報
            func_name: 関数名
            
        Returns:
            tuple: (coder, code, num_padded_bits, text_limit)
        """
        total_start = time.time()
        if self.hosting == "huggingface":
            from gptzip.gptzip import ArithmeticCoder
        elif self.hosting == "openai":
            from gptzip.gptzip_online import ArithmeticCoder
        coder = ArithmeticCoder(lm=self.model,
                                tokenizer=self.tokenizer,
                                #use_cache=self.use_cache,
                                cache=self.L2_CACHE, # Use L2 Cache for LLM outputs
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
        """
        圧縮されたコードを解凍して元のテキストに戻す
        
        Args:
            coder: ArithmeticCoderインスタンス
            code: 圧縮されたコード
            num_padded_bits: パディングされたビット数
            text_limit: テキスト表示の制限文字数
            input_dir: 入力ファイルのディレクトリ
            text_path: 入力ファイルのパス
            basic_info: 実験の基本情報
            func_name: 関数名
            
        Returns:
            tuple: (decoded_string, is_success)
        """
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
    # コマンドライン引数から設定ファイル名を取得
    #config = sys.argv[1]
    # オーバーライドオプションがあれば取得（例: exp.llm=model_name）
    if len(sys.argv) >= 3:
        override_options = sys.argv[2:]
        print(f"override_options={override_options}")
    else:
        override_options = None
    
    # 設定ファイル名を取得（sys.argv[1]から）
    config_name = sys.argv[1] if len(sys.argv) > 1 else "config"
    print(f"config_name={config_name}")
    # パスを除去してファイル名のみ取得
    config_name = os.path.basename(config_name).replace('.yaml', '')
    config_name = os.path.basename(config_name)
    # Hydraで設定ファイルを読み込み
    #with initialize(config_path=".", job_name=__file__):
    with initialize(config_path="../../config", job_name=__file__):
    #with initialize(config_path="../../", job_name=__file__):
        if override_options:
            cfg = compose(#config_name=sys.argv[1],
                          config_name=config_name,
                          return_hydra_config=True,
                          overrides=override_options,
                          )
        else:
            cfg = compose(#config_name=sys.argv[1],
                          config_name=config_name,
                          return_hydra_config=True,
                          )
    #print(f"cfg={cfg.config}")
    import pprint
    pprint.pprint(cfg, indent=4, sort_dicts=False)
    #pprint.pprint(cfg[1], indent=4, sort_dicts=False)
    # config_path="config"からconfig_path="."に変えた（互換性のため）
    #cfg = cfg.config
    
    # 設定ファイル名から実験名を自動抽出
    # 命名規則: lmcr_exp_{実験名}.yaml → {実験名}
    # ファイル名から抽出できた場合は常に優先する
    import re
    # 元の引数からファイル名を取得
    raw_config_path = sys.argv[1] if len(sys.argv) > 1 else ""
    basename = os.path.basename(raw_config_path)
    match = re.match(r'lmcr_exp_([^.]+)\.yaml$', basename)
    
    if match:
        auto_title = match.group(1)
        logger.info(f"Auto-extracted exp_title from config filename: {auto_title}")
        # cfg.exp.titleをセット/上書き
        from omegaconf import OmegaConf
        try:
             OmegaConf.set_struct(cfg, False)  # 構造の変更を許可
             if 'exp' not in cfg:
                 cfg.exp = {}
             cfg.exp.title = auto_title
             OmegaConf.set_struct(cfg, True)  # 構造を再度固定
        except Exception as e:
             logger.warning(f"Failed to set exp_title: {e}")
             if isinstance(cfg, dict) or hasattr(cfg, '__setitem__'):
                 if 'exp' not in cfg: cfg['exp'] = {}
                 cfg['exp']['title'] = auto_title
    else:
        # ファイル名が命名規則に従っていない場合は、config内の記述を確認
        current_title = cfg.exp.get('title')

        logger.info(f"Using exp_title from config: {current_title}")


    exp_title=cfg.exp.title
    exp_summary=cfg.exp.summary
    progress.info(f"config={config_name} title={exp_title} summary={exp_summary}")
    progress.info(f"cfg={cfg}")
    # MLflowの設定とLMCRインスタンスの作成
    mlflow.set_tracking_uri(uri="http://localhost:8080")
    
    
    exe = LMCR(cfg)
    # 算術符号化（Arithmetic Encoding）実験の実行
    run_suffix = cfg.exp.get('run_suffix', "")
    if run_suffix:
        run_name = f"{exp_title}({run_suffix})"
    else:
        run_name = exp_title
    
    logger.info(f"Run Name set to: {run_name} (exp_title={exp_title}, run_suffix={run_suffix})")
    
    mlflow.end_run()  # 既存のrunがあれば終了
    with mlflow.start_run(run_name=run_name) as run:
        # 実行開始日時を記録（loggerのログと突き合わせるため）
        from datetime import datetime
        start_datetime = datetime.now()
        start_time_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
        mlflow.set_tag("start_time", start_time_str)
        # 時間・分・秒を数値として取得し、辞書形式でまとめる
        start_time_metrics = {
            "start_hour": start_datetime.hour,
            "start_minute": start_datetime.minute,
            "start_second": start_datetime.second
        }
        mlflow.log_metrics(start_time_metrics)
        
        # 設定パラメータを記録
        mlflow.log_params(cfg)
        
        # VR-LMCR計算用パラメータを記録
        mlflow.log_param("max_vocab_size", exe.max_vocab_size)
        mlflow.log_param("gamma", exe.gamma)
        
        # デバイス設定を記録
        mlflow.set_tag("device", exe.device)

        mlflow.set_tag("algorithm", "ae")
        exe.input_analysis(exe.execute_ae)  # 圧縮・解凍実験
        exe.calculate_token_efficiency()  # トークン効率の計算
        #exe.generating_test()  # 生成テスト（コメントアウト中）
        end_datetime = datetime.now()
        end_time_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
        mlflow.set_tag("end_time", end_time_str)
        # 差分（duration）の計算
        duration = end_datetime - start_datetime
        total_seconds = duration.total_seconds()

        # 各単位への換算 
        duration_metrics = {
            "duration_hours": total_seconds / 3600,          # 時間単位
            "duration_minutes": total_seconds / 60,          # 分単位
            "duration_seconds": total_seconds,               # 秒単位
            "duration_ms": total_seconds * 1000             # ミリ秒単位
        }

        # MLflowに記録
        mlflow.log_metrics(duration_metrics)

        print(f"経過時間（秒）: {total_seconds}")
    # パープレキシティ実験（現在はコメントアウト中）
    """
    run_name = f"{exp_title}(ppl)"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tracking_uri(uri="http://localhost:8080")
        mlflow.set_tag("algorithm", "ppl")
        exe.input_analysis(exe.execute_ppl)  # パープレキシティ計算
        exe.calculate_token_efficiency()  # トークン効率の計算
        exe.generating_test()  # 生成テスト
    mlflow.end_run()
    """
