from src.cache.cache import Cache
from hydra import initialize, compose, initialize_config_dir
import mlflow
import numpy as np
import os
from pathlib import Path
import pandas as pd
from gradio_client import Client
import polars as pl


class Analyzer:

    # モデル名のリスト
    models = [
        "google/gemma-1.1-7b-it",
        "google/gemma-7b-it",
        "Qwen/Qwen1.5-0.5B-Chat",
        "tiiuae/Falcon3-1B-Instruct",
        "google/gemma-2b-it",
        "llm-jp/llm-jp-3-440m-instruct2",
        "Qwen/Qwen2-0.5B-Instruct",
        "google/gemma-1.1-2b-it",
        "tiiuae/Falcon3-3B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "llm-jp/llm-jp-3-980m-instruct2",
        "tiiuae/Falcon3-7B-Instruct",
        "weblab-GENIAC/Tanuki-8B-dpo-v1.0",
        "Qwen/Qwen1.5-4B-Chat",
        "llm-jp/llm-jp-3-1.8b-instruct",
        "meta-llama/Llama-2-7b-chat-hf",
        "Qwen/Qwen2-1.5B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "llm-jp/llm-jp-3-3.7b-instruct2",
        "Qwen/Qwen2.5-3B-Instruct",
        "llm-jp/llm-jp-3-3.7b-instruct",
        "Qwen/Qwen1.5-14B-Chat",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen1.5-32B-Chat",
        "llm-jp/llm-jp-3-13b-instruct",
        "google/gemma-2-27b-it",
        "llm-jp/llm-jp-3.1-1.8b-instruct4",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "llm-jp/llm-jp-3.1-13b-instruct4",
    ]
    
    def __init__(self,
                 project_root: str,
                 config_name: str = "lmcr_exp_fit2025exp07",
                 ):
        """
        Analyzerクラスの初期化
        
        Args:
            project_root (str): プロジェクトのルートディレクトリのパス. uvのディレクトリ．
            config_name (str): 読み込む設定ファイルの名前（拡張子なし）. デフォルトは "lmcr_exp_fit2025exp07".
            
        Note:
            指定されたHydraの設定をロードし、キャッシュとMLflowのURIを設定します。
        """
        
        project_root_path = Path(project_root)
        self.config_name = config_name
        self.config_dir = project_root_path / "config"
        with initialize_config_dir(version_base=None,
                                      config_dir=str(self.config_dir),
                                   ):
            cfg = compose(config_name=self.config_name)
        self.cfg = cfg
        self.cache = Cache(cfg=cfg,
              cache_filename="cache_test",
              prefix="test",
              )
        mlflow.set_tracking_uri("http://localhost:8080")

    def reload_config(self,
                     ):
        with initialize_config_dir(version_base=None,
                                      config_dir=str(self.config_dir),
                                   ):
            cfg = compose(config_name=self.config_name)
        self.cfg = cfg  # 更新
        self.cache = Cache(cfg=cfg,
              cache_filename="cache_test",
              prefix="test",
              )

        
    def get_leaderboard_result(self,
                               query,
                               cache_base_name="leaderboard_result",
                               cache_tag_name="0814",
                               ):
        
        key_base_name = cache_base_name
        key_tag_name = cache_tag_name
        enable_cache = None
        if key_base_name is key_tag_name is None:
            enable_cache = False
        if key_base_name is not None and key_tag_name is not None:
            enable_cache = True
        if enable_cache is None:
            raise ValueError("cache_base_name and cache_tag_name must be both None or both not None")
        if enable_cache:
            # キャッシュを使う
            key = f"{key_base_name}_{key_tag_name}"
            cache_result = self.cache.get(key)
            print(f"cache={cache_result}")
            if cache_result is not None:
                # キャッシュが存在したらそれを返す
                return cache_result
            # 存在しなければ仕方がないので取得してくる
            #result = self._get_leaderboard_result(query)
            result = self._get_leaderboard_result_via_space(query)
            # キャッシュに保存する
            self.cache.set(key, result)
        else:
            # キャッシュを使わない
            result = self._get_leaderboard_result(query)
        return result

    def _get_leaderboard_result_via_api(self,
                               query,
                               ):
        #client = Client("llm-jp/open-japanese-llm-leaderboard")
        client = Client.duplicate("llm-jp/open-japanese-llm-leaderboard",
                                  os.getenv("HF_TOKEN"))
        result = client.predict(
                                #type_query=["🟢 pretrained","🔶 fine-tuned","⭕ instruction-tuned","🟦 RL-tuned (Preference optimization)","🌸 multimodal","🤝 base merges and moerges"],
                                type_query=["⭕ instruction-tuned"],
                		precision_query=["float16","bfloat16","float32"],
                                #size_query=["0~3B","3~7B","7~13B","13~35B","35~60B","60B+","?"],
                                size_query=["0~3B","3~7B","7~13B","13~35B"],
                                add_special_tokens_query=["True","False"],
                                num_few_shots_query=[4],
                                version_query=["v1.4.1"],
                                vllm_query=["v0.6.3.post1"],
                                query=query, #"llm-jp; HuggingFaceTB",
                                param_8=["AVG"],
                                param_9=["AVG (NLI)"], # 自然言語推論
                                #param_9=[],
                                param_10=[],
                                param_11=["AVG (RC)"], # 読解力
                                #param_11=[],
                                param_12=["AVG (MC)"], # 多肢選択式質問応答
                                #param_12=[],
                                param_13=["AVG (EL)"], # エンティティリンキング
                                #param_13=[],
                                param_14=["AVG (FA)"], # 基礎分析
                                #param_14=[],
                                param_15=["AVG (MR)"], # 数学的推論
                                #param_15=[],
                                param_16=["AVG (MT)"], # 機械翻訳
                                #param_16=[],
                                #param_17=[],
                                param_18=["AVG (HE)"], # 試験問題
                                #param_18=[],
                                param_19=["AVG (CG)"], # コード生成
                                #param_19=[],
                                param_20=["AVG (SUM)"], 
                                #param_21=[],
                                api_name="/update_table"
        )
        return result
    def _get_leaderboard_result_via_space(self,
                                          query,
                                          ):
        import datasets
        from src.llmjp.about import Tasks
        from src.llmjp.populate import get_leaderboard_df
        from src.llmjp.utils import (AutoEvalColumn,
                                    COLS,
                                    BENCHMARK_COLS,
                                    )
        df = get_leaderboard_df("llm-jp/leaderboard-contents", 
                                COLS, 
                                BENCHMARK_COLS)
        return df
        
        
    def _leaderboard_data_to_polar_dataframe(self,
                                            result):
        schema_names = result["headers"]
        schema_names.append("average_type")
        df_all = None
        for data in result["data"]:
            #print(schema_names)
            # HTML処理
            #print(data[1])
            html_string = data[1]
            import re
            match = re.search(r'>([^<]+)</a>', html_string)
            if match:
                full_text = match.group(1)  # "SakanaAI/EvoLLM-JP-A-v1-7B (4-shot)"
                # 括弧内と前方を分離
                split_match = re.match(r'(.+?)\s*\(([^)]+)\)', full_text)
                if split_match:
                    result = [split_match.group(1), split_match.group(2)]
                    #print(result)
                else:
                    print("括弧が見つかりません")
            else:
                print("HTML中のテキストが見つかりません")
            data[1] = result[0]
            data.append(result[1])
            #schema_names.append("average_type")
            df_item = pl.DataFrame([data], schema=schema_names)
            #print(df_item)

            if df_all is None:
                df_all = df_item
            else:
                df_all = df_all.vstack(df_item)
        return df_all
    def get_leaderboard_df(self):
        key_base_name = "leaderboard_result"
        key_tag_name = "0801"
        key = f"{key_base_name}_{key_tag_name}"
        cache_result = self.cache.get(key)
        print(f"cache={cache_result}")
        if cache_result is None:
            result = self._get_leaderboard_result("")
            cache.set(key, result)
        else:
            result = cache_result
        print(f"leaderboard result={result}")
        print(f"leaderboard result class={result.__class__}")
        leaderboard_df = self._leaderboard_data_to_polar_dataframe(result)

        print(f"lbdf={leaderboard_df}")
        return leaderboard_df

    def search_model_from_mlflow(self,
                                 queries=None,
                                 size_filter=None,
                                 run_name=None,
                                 ):
        """
        MLflowからモデルの実験結果を検索し、各モデルの最新の実行結果を取得する。

        指定されたモデル名（queries）と実験タイトル（run_name）に一致する実行（run）を検索し、
        モデル名、アルゴリズム、入力テキスト長ごとに最新の1件を抽出して結合したデータを返します。

        Args:
            queries (list[str], optional): 検索対象のモデル名のリスト。
                Noneの場合はHydra設定（self.cfg.exp_models.target_models）を使用します。
            size_filter (int, optional): 入力テキスト長（metrics.input_text_length）でフィルタリングする場合の数値。
            run_name (str, optional): 検索対象の実験タイトル（tags.exp_title）。

        Returns:
            pl.DataFrame: 検索結果を格納したPolars DataFrame。
                各モデル名、アルゴリズム、入力長に対して最新の1件のみが含まれます。
        """
        # 条件をフィルタリングして検索（DataFrameで返る）
        
        if queries is None:
            # configからtarget_modelsを取得
            try:
                # OmegaConfのListConfigはイテラブルなのでそのまま使える
                queries = self.cfg.exp_models.target_models
                print(f"Using target_models from config: {queries}")
            except Exception as e:
                print(f"Error retrieving target_models from config: {e}")
                queries = []

        # experiment_idを指定（もしくは名前で取得）
        experiment_id = "Default"
        runs_df = None
        for query in queries:
            #print(f"query={query}")
            query_string = f"tags.model_name like '{query}'"
            #print(f"query_string={query_string}")
            if size_filter is not None:
                query_string = f"{query_string} and metrics.input_text_length = {size_filter}"
            #print(f"query string={query_string}")
            filter_string = f"tags.model_name like '{query}' and "\
                            + f"tags.exp_title like '{run_name}'"
            #print(f"filter string={filter_string}")
            df = mlflow.search_runs(
                search_all_experiments=True,
                #experiment_names=[experiment_id],
                filter_string=filter_string,
                #filter_string=f"tags.model_name like '{query}%'",
                #filter_string=f"tags.model_name like '{query}'",
                
                order_by=["start_time DESC"],
                output_format="pandas"
            )
            #print(f"df from mlflow.search_runs={df}")
            #print(f"include tags.model_name?={df}")
            if df.empty is False:
                df = df.sort_values("start_time", ascending=False)\
                       .groupby(["tags.model_name", "tags.algorithm", "metrics.input_text_length"])\
                       .first()\
                       .reset_index()
                #print(f"first=>{df}")
                if runs_df is None:
                    runs_df = df
                else:
                    runs_df = pd.concat([runs_df, df])
        runs_df = pl.from_pandas(runs_df)
        return runs_df

    def mlflow_results_df_to_analytics_df(self,
                                          df):
        """
        MLflowから取得した実験結果DataFrameを分析用に整形する

        MLflowのtagsやmetricsプレフィックス付きのカラムから、分析に必要な以下のカラムを選択します：
        - tags.model_name
        - tags.input_file_name
        - tags.algorithm
        - metrics.input_text_length
        - metrics.tokenized_size
        - metrics.compressed_size
        - metrics.token_efficiency
        - metrics.llm_score
        - metrics.vr_lmcr_naive

        また、データを `tags.algorithm` および `tags.model_name` でグルーピングして再構築することで、
        データの並び順を整えています。

        Args:
            df (pl.DataFrame): MLflowの検索結果（Polars DataFrame）

        Returns:
            pl.DataFrame: 分析に必要なカラムを抽出し、整列させたDataFrame
        """
        from pprint import pprint
        # グラフを描くために必要な項目の取り出しと確認
        #pprint(df.schema)
        df_exp = df.select(#pl.col('tags.mlflow.source.name'),
                  pl.col('tags.model_name'),
                  pl.col('tags.input_file_name'),
                  pl.col('tags.algorithm'),
                  pl.col('metrics.input_text_length'),
                  pl.col('metrics.tokenized_size'),
                  pl.col('metrics.compressed_size'),
                  pl.col('metrics.token_efficiency'),
                  pl.col('metrics.llm_score'),
                  pl.col('metrics.vr_lmcr_naive')  # VR-LMCRメトリックを追加
                 ).group_by("tags.algorithm",maintain_order=True)
        total_exp_df = pl.DataFrame()
        for algorithm, data_df in df_exp:
            #print(algorithm)
            for model_name, data_df in data_df.group_by('tags.model_name'):
                #print(model_name)
                #print(data_df)
                #data_df = data_df.with_columns(pl.col("tags.model_name").alias("model_name"))
                total_exp_df = total_exp_df.vstack(data_df)
        return total_exp_df

    def shrink_model_name(self,
                          model_names):
        names = [x.replace('jsai-dev-01_', '')\
                  .replace('deepseek-ai-', '')\
                  .replace('meta-llama/', '')\
                  .replace('Qwen/', '')\
                  .replace('llm-jp/', '')\
                  .replace('google/', '')\
                  .replace('tiiuae/', '')\
                  .replace('HuggingFaceTB/', '')\
                  .replace('weblab-GENIAC/', '')
                  .replace('Instruct', "Inst")\
                  .replace("instruct", "inst")\
                   for x in model_names]
        return names

    def get_llm_score(self,
                      df):
        return df["metrics.llm_score"].item()

    def get_modified_llm_score(self,
                               df):
        # これがLate-stage LMCRのはず
        original_size = df["metrics.input_text_length"].item()
        tokenized_size = df["metrics.tokenized_size"].item()
        encoded_size = df["metrics.compressed_size"].item()
        score = encoded_size/tokenized_size
        return score

    def get_complex_llm_score(self,
                               df):
        original_size = df["metrics.input_text_length"].item()
        tokenized_size = df["metrics.tokenized_size"].item()
        encoded_size = df["metrics.compressed_size"].item()
        token_efficiency = tokenized_size/original_size

        # metrics.compressed_sizeはplot_leaderboard_comprex_lmcrで改ざんされてる
        score = encoded_size/original_size
        return score
        
    def get_poly_interpolated_llm_score(self,
                               df):

        # 2次回帰予測値
        score = df["interpolated_llm_score"].item()

        return score


    def get_token_efficiency(self,
                             df):
        return df["metrics.token_efficiency"].item()

    def get_leaderboard_score(self,
                              df):
        return df["AVG"].item()

    def plot_leaderboard_score_score_by_model(self,
                                              length_df, 
                                              algorithm, 
                                              option,
                                              x_func,
                                              y_func,
                                              x_title,
                                              y_title,
                                              title,
                                             ):
        import matplotlib.pyplot as plt
        plt.figure()
        print(f"option={option}")

        # 回帰直線を計算するために，入力長ごとにデータを集める
        x_arr = []
        y_arr = []
        corr_list = []
        # LLMのモデルごとにプロットするので，補間アルゴリズムを使う場合は
        # 別途仕組みが必要
        for model, length_model_df in length_df.group_by("Model"):
            #print(f"model={model}")
            #with pl.Config(tbl_cols=-1):
            #    print(f"plot target length_model_df={length_model_df}")
            # 変なデータがリーダーボードに登録されている
            if model[0] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
                length_model_df = length_model_df.limit(1)
            #x = length_model_df["AVG"].item()
            x = x_func(length_model_df)
            #y = length_model_df["metrics.llm_score"].item()
            y = y_func(length_model_df)
            x_arr.append(x)
            y_arr.append(y)
            #print(f"(x,y)=({x}, {y})")
            plt.scatter(x, 
                        y,
                        label=f"{model[0]}-{algorithm[0]}",  # ← ラベルを設定！
                        marker="o",
                       )
            model_name = self.shrink_model_name(model)
            if option == "token_efficiency":
                token_efficiency = length_model_df["metrics.token efficiency"].item()
                plt.text(x, y, f"{model_name[0]}({token_efficiency:.2f})")
            elif option is None:
                plt.text(x, y, f"{model_name[0]}")
            else:
                plt.text(x, y, f"{model_name[0]}({option})")
        x = np.array(x_arr)
        y = np.array(y_arr)
        slope, intercept = np.polyfit(x, y, deg=1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = slope * x_fit + intercept
        y_pred = slope * x + intercept
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        try:
            mse = mean_squared_error(y, y_pred)
        except ValueError as e:
            mse = -1
        try:
            r2 = r2_score(y, y_pred)
        except ValueError as e:
            r2 = -1
        corr = np.corrcoef(x, y)[0, 1]
        corr_list.append(corr)
        plt.plot(x_fit, y_fit, color="red", label="MSE={mse:.2f}")
        plt.plot(x_fit, y_fit, color="red", label=f"MSE={mse:.2f}")
        print(f"corr={corr}")
        plt.text(0.95, 
                 0.99, 
                 f"MSE={mse:.2f}, r2={r2:.2f}, corr={corr:.2f}",
                 transform=plt.gca().transAxes,
                 ha="right",
                 va="top"
                )

        # 凡例とラベルの表示
        #plt.xlabel("Learderboard Score")
        plt.xlabel(x_title)
        if algorithm[0] == "ae":
            plt.ylabel("LMCR Index")
        else:
            plt.ylabel("Perplexity")
        plt.ylabel(y_title)
        #plt.title("Score vs Original Size by Algorithm and Input Size")
        plt.title(title)
        #plt.legend()  # ← 凡例を表示する
        plt.grid(True)
        plt.show()  
        return corr_list

    def leaderboard_df_to_report_df(self,
                                    leaderboard_df,
                                    total_exp_df,
                                    ):
        """
        リーダーボードのデータと実験結果のデータを結合し、レポート用のDataFrameを作成する。

        処理内容:
        1. リーダーボードデータと実験結果データをモデル名をキーに結合する。
        2. Few-shot数が4のデータのみを抽出する。
        3. 不要なカラム（ID, Few-shot）を削除する。

        Args:
            leaderboard_df (pl.DataFrame): リーダーボードのデータ
            total_exp_df (pl.DataFrame): 実験結果のデータ（mlflow_results_df_to_analytics_dfの出力）

        Returns:
            pl.DataFrame: 結合・フィルタリング済みのレポート用DataFrame
        """
          #.filter(pl.col("average_type") == "4-shot")\
                  #.drop(pl.col("average_type"))
        df_report = leaderboard_df.join(total_exp_df.rename({"tags.model_name": "Model"}), on="Model")\
        .filter(pl.col("Few-shot") == 4)\
        .drop(pl.col("ID"))\
        .drop(pl.col("Few-shot"))
        return df_report
    
    def plot_size_score_by_model(self,
                                 df,
                                 algorithm,
                                 lang,
                                 ):
        import matplotlib.pyplot as plt
        plt.figure()

        # 入力長を変化させたときのスコアをプロット
        for model, model_df in df.group_by("Model"):
            #print(f"model={model}")
            #print(f"model_df={model_df}")
            plt.plot(model_df["metrics.input_text_length"], 
                     model_df["metrics.llm_score"],
                     label=f"{model[0]}-{algorithm[0]}",  # ← ラベルを設定！
                     marker="o",
                    )
        # 凡例とラベルの表示
        plt.xlabel("Input Text Length")
        if algorithm[0] == "ae":
            plt.ylabel("LMCR Index")
        else:
            plt.ylabel("Perplexity")

        plt.title(f"Score({lang}) vs Input Text Length by Algorithm and Model")
        #plt.legend()  # ← 凡例を表示する
        plt.grid(True)
        plt.show()
        
    def plot_leaderboard_lmcr(self,
                              df,
                              algorithm,
                              lang):
        #plt.figure()
        # 入力長別に正解とスコアをプロット
        # 入力長ごとに，モデルごとの相関係数
        corr_list = []
        for input_length, length_df in df.group_by("metrics.input_text_length"):
            print(f"input_length={input_length}")
            #print(f"length_df={length_df}")

            # input_lengthはtupleなのでスカラー値として取り出す
            y_title = f"LMCR Index(input={int(input_length[0])} characters)"
            corr_list_result = self.plot_leaderboard_score_score_by_model(
                                        length_df, 
                                        algorithm, 
                                        #option=input_length,
                                        option=None,
                                        x_func=self.get_leaderboard_score,
                                        y_func=self.get_llm_score,
                                        x_title="Leaderboard Score",
                                        y_title=y_title,
                                        title=f"Leaderboard Score vs LMCR Index({lang})"
                                      )
            corr_list.append(corr_list_result)

        print(f"corr_list={corr_list}, avg={np.average(corr_list)}")

    def plot_leaderboard_modified_lmcr(self,
                                       df,
                                       algorithm,
                                       lang,
                                       ):
        corr_list = []
        for input_length, length_df in df.group_by("metrics.input_text_length"):
            print(f"input_length={input_length}")
            #print(f"length_df={length_df}")
            # input_lengthはtupleなのでスカラー値として取り出す
            y_title = f"Late-stage LMCR Index(input={int(input_length[0])} characters)"
            corr_list_result = self.plot_leaderboard_score_score_by_model(
                                  length_df, 
                                  algorithm, 
                                  #option=input_length,
                                  option=None,
                                  x_func=self.get_leaderboard_score,
                                  y_func=self.get_modified_llm_score,
                                  x_title="Leaderboard Score",
                                  y_title=y_title,
                                  title=f"Leaderboard Score vs Late-stage LMCR Index({lang})"
                                                                    )
            corr_list.append(corr_list_result)  
    def plot_leaderboard_complex_lmcr(self,
                                       df,
                                       algorithm,
                                       lang,
                                       ):
        cs_col = pl.col("metrics.compressed_size")
        ts_col = pl.col("metrics.tokenized_size")
        size_col = pl.col("metrics.input_text_length")
        X = df.select((cs_col/size_col).alias("compression_ratio"),
                      (ts_col/size_col).alias("tokenized_ratio"),
                      ) # shape(N, 2)

        from scipy.stats import mode
        #mode_val = mode(X["tokenized_ratio"], keepdims=False).mode
        #shift = 1 - mode_val
        mean_val = X.select(pl.mean("tokenized_ratio")).item()

        # 平均値・最頻値の値からのずれ
        shift = X["tokenized_ratio"] - mean_val
        #weight = np.exp(-2.0*shift) # corr=-0.74
        weight = np.exp(-3.0*shift) # corr=-0.75
        #weight = np.exp(-4.0*shift) # corr=-0.75
        #weight = np.exp(-8.0*shift) # corr=-0.58

        # 平均値・最頻値の値を1.0とする
        centered = shift * weight + 1.0
        # トークン効率の悪いFalconなんかは0.8とかになって，
        # トークン効率の良いllm-jpは1.2とかになるはず

        transformed = 2 - centered

        # 改ざん
        df = df.with_columns([
               (cs_col * transformed).alias("metrics.compressed_size")
            ])

        corr_list = []
        for input_length, length_df in df.group_by("metrics.input_text_length"):
            print(f"input_length={input_length}")
            #print(f"length_df={length_df}")
            # input_lengthはtupleなのでスカラー値として取り出す
            y_title = f"Complex LMCR Index(input={int(input_length[0])} characters)"
            corr_list_result = self.plot_leaderboard_score_score_by_model(
                                  length_df, 
                                  algorithm, 
                                  #option=input_length,
                                  option=None,
                                  x_func=self.get_leaderboard_score,
                                  y_func=self.get_complex_llm_score,
                                  x_title="Leaderboard Score",
                                  y_title=y_title,
                                  title=f"Leaderboard Score vs Complex LMCR Index({lang})"
                                                                    )
            corr_list.append(corr_list_result)  

    def plot_leaderboard_token_efficiency(self,
                                          df,
                                          algorithm,
                                          lang,
                                          ):
        corr_list = []
        for input_length, length_df in df.group_by("metrics.input_text_length"):
            print(f"input_length={input_length}")
            #print(f"length_df={length_df}")
            y_title = f"Token Efficiency(input={int(input_length[0])} characters)"
            corr_list_result = self.plot_leaderboard_score_score_by_model(
                                      length_df, 
                                      algorithm, 
                                      #option=input_length,
                                      option=None,
                                      x_func=self.get_leaderboard_score,
                                      y_func=self.get_token_efficiency,
                                      x_title="Leaderboard Score",
                                      y_title=y_title,
                                      title=f"Leaderboard Score vs Token Efficiency({lang})"
                                                                    )
            corr_list.append(corr_list_result)

    def plot_lmcr_token_efficiency(self,
                                   df,
                                   algorithm,
                                   lang,
                                   ):
        corr_list = []
        for input_length, length_df in df.group_by("metrics.input_text_length"):
            print(f"input_length={input_length}")
            #print(f"length_df={length_df}")
            y_title = f"Token Efficiency(input={int(input_length[0])} characters)"
            corr_list_result = self.plot_leaderboard_score_score_by_model(
                                      length_df, 
                                      algorithm, 
                                      #option=input_length,
                                      option=None,
                                      x_func=self.get_llm_score,
                                      y_func=self.get_token_efficiency,
                                      x_title="LMCR Index",
                                      y_title=y_title,
                                      title=f"LMCR Index vs Token Efficiency({lang})"
                                                                    )
            corr_list.append(corr_list_result)

    def plot_leaderboard_2vars_linearregression(self,
                                   df,
                                   algorithm,
                                   lang,
                                   ):
        # 入力が２変数で線形回帰するやつ
        # 入力は圧縮率とトーン効率
        cs_col = pl.col("metrics.compressed_size")
        ts_col = pl.col("metrics.tokenized_size")
        size_col = pl.col("metrics.input_text_length")
        X = df.select((cs_col/size_col).alias("compression_ratio"),
                      (ts_col/size_col).alias("tokenized_ratio"),
                      ) # shape(N, 2)
        #from sklearn.preprocessing import PolynomialFeatures
        #poly_extractor = PolynomialFeatures(degree=2, include_bias=True)
        #X_poly = poly_extractor.fit_transform(X) # shape(N, 6)

        score_col = pl.col("AVG")
        Y = df.select(score_col)
        from sklearn.linear_model import LinearRegression
        linear_model = LinearRegression()
        #linear_model.fit(X_poly, Y)
        linear_model.fit(X, Y) # 回帰モデル生成


        corr_list = []
        for input_length, length_df in df.group_by("metrics.input_text_length"):
            print(f"input_length={input_length}")
            # Xと違ってxは１行の想定
            x = df.select((cs_col/size_col).alias("compression_ratio"),
                      (ts_col/size_col).alias("tokenized_ratio"),
                      ) # shape(1, 2)
            assert x.shape[0] == 1 # assertなど入れてみる
            # nはモデル数
            y_pred = linear_model.predict(x) # (n, 1)
            length_df = length_df.with_columns([
                    pl.Series("interpolated_llm_score", y.reshape(-1))
                ]
                )
            
            #print(f"length_df={length_df}")
            y_title = f"Linear Regression AVG Score(input={int(input_length[0])} characters)"
            corr_list_result = self.plot_leaderboard_score_score_by_model(
                                      length_df, 
                                      algorithm, 
                                      #option=input_length,
                                      option=None,
                                      x_func=self.get_leaderboard_score,
                                      y_func=self.get_poly_interpolated_llm_score,
                                      x_title="Leaderboard AVG Score",
                                      y_title=y_title,
                                      title=f"Linear Regression AVG Score vs AVG Score({lang})"
                                                                    )
            corr_list.append(corr_list_result)

    def plot_leaderboard_poly_interpolated_lmcr_score(self,
                                   df,
                                   algorithm,
                                   lang,
                                   ):
        # 入力は圧縮率とトーン効率
        cs_col = pl.col("metrics.compressed_size")
        ts_col = pl.col("metrics.tokenized_size")
        size_col = pl.col("metrics.input_text_length")
        X = df.select((cs_col/size_col).alias("compression_ratio"),
                      (ts_col/size_col).alias("tokenized_ratio"),
                      ) # shape(N, 2)
        from sklearn.preprocessing import PolynomialFeatures
        poly_extractor = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly_extractor.fit_transform(X) # shape(N, 6)

        score_col = pl.col("AVG")
        Y = df.select(score_col)
        from sklearn.linear_model import LinearRegression
        linear_model = LinearRegression()
        linear_model.fit(X_poly, Y)


        corr_list = []
        for input_length, length_df in df.group_by("metrics.input_text_length"):
            print(f"input_length={input_length}")
            # nはモデル数
            x = length_df.select((cs_col/size_col).alias("compression_ratio"),
                                 (ts_col/size_col).alias("tokenized_ratio"),
                                 ) # shape(n, 2)
            x_poly = poly_extractor.fit_transform(x) # shape(n, 6)
            y = linear_model.predict(x_poly) # (n, 1)
            length_df = length_df.with_columns([
                    pl.Series("interpolated_llm_score", y.reshape(-1))
                ]
                )
            
            #print(f"length_df={length_df}")
            y_title = f"Interpolated LMCR Index(input={int(input_length[0])} characters)"
            corr_list_result = self.plot_leaderboard_score_score_by_model(
                                      length_df, 
                                      algorithm, 
                                      #option=input_length,
                                      option=None,
                                      x_func=self.get_leaderboard_score,
                                      y_func=self.get_poly_interpolated_llm_score,
                                      x_title="Leaderboard Score",
                                      y_title=y_title,
                                      title=f"Leaderboard Score vs Interpolated LMCR Index({lang})"
                                                                    )
            corr_list.append(corr_list_result)

    def plot_graph(self,
                   df_report: pl.DataFrame,
                   lang: str,
                   ) -> None:
        """
        グラフをプロットする
        Args:
            df_report (pl.DataFrame): レポート用のDataFrame
            lang (str): 言語
        """
        # リーダーボードの個別のタスクスコアの列名リスト
        # _get_leaderboard_result メソッドの param_9 から param_19, param_20 に対応するタスク名を想定
        task_columns = [
            "RC", "NLI", "MC", "EL", "FA",
            "MR", "MT", "HE", "CG", "SUM"
        ]

        for algorithm,df in df_report.group_by("tags.algorithm"):
        #for algorithm,df in df_report.filter(pl.col("Model").str.starts_with("llm-jp")).group_by("tags.algorithm"):
            print(f"algorithm={algorithm}")
            if algorithm[0] == "ppl":
                continue
            #df = df.drop("tags.algorithm")
            #print(f"df={df}")
            self.plot_size_score_by_model(df, algorithm, lang)

            self.plot_leaderboard_lmcr(df, algorithm, lang)

            self.plot_leaderboard_modified_lmcr(df, algorithm, lang)

            self.plot_leaderboard_complex_lmcr(df, algorithm, lang)            

            self.plot_leaderboard_poly_interpolated_lmcr_score(df, algorithm, lang)

            self.plot_leaderboard_token_efficiency(df, algorithm, lang)

            self.plot_lmcr_token_efficiency(df, algorithm, lang)

        for algorithm,df in df_report.group_by("tags.algorithm"):
            print(f"algorithm={algorithm}")
            #print(f"df={df}")

            # 個別のタスクスコアとLMCRの比較グラフを生成
            for task_col in task_columns:
                # DataFrameにタスク列が存在するか確認
                if f"AVG ({task_col})" in df.columns:
                    print(f"creating graph for {task_col}")
                    for input_length, length_df in df.group_by("metrics.input_text_length"):
                        y_title = f"LMCR Index(input={int(input_length[0])} characters)"
                        self.plot_leaderboard_score_score_by_model(
                            #df,
                            length_df,
                            algorithm,
                            option=None,
                            # x軸の値を指定されたタスク列から取得するラムダ関数
                            x_func=lambda df_item, col=f"AVG ({task_col})": df_item[col].item(),
                            y_func=self.get_llm_score,
                            x_title=f"{task_col} Score",
                            y_title=y_title,
                            title=f"{task_col} Score vs LMCR Index({lang})"
                        )


    def plot_token_efficiency_by_model(self,
                              df,
                              df_e,
                              ):
        # データ取得
        models = self.shrink_model_name(df["Model"].to_list())
        efficiency = df["metrics.token_efficiency"].to_list()
        print(list(zip(models, efficiency)))
        print(sorted(zip(models, efficiency)))
        models, efficiency = zip(*sorted(zip(models, efficiency)))
        models = self.shrink_model_name(df_e["Model"].to_list())
        efficiency_e = df_e["metrics.token_efficiency"].to_list()

        models, efficiency_e = zip(*sorted(zip(models, efficiency_e)))    

        # 棒グラフを描画
        x = np.arange(len(models))
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        width = 0.4  # バーの幅
        plt.bar(x-width/2, efficiency, width, color="skyblue", edgecolor="black", label="Japanese")
        plt.bar(x+width/2, efficiency_e, width, color="green", edgecolor="black", label="English")
        plt.xlabel("Model")
        plt.ylabel("Token Efficiency(token length/input bytes)")
        plt.title(f"Token Efficiency by Model")
        plt.xticks(x-width/2-2, models, rotation=45)  # X軸のラベルを回転して見やすくする
        plt.legend()
        plt.show()

    def plot_token_efficiency_by_avg_score(self,
                              df,
                              df_e,
                              ):
        # データ取得
        models = self.shrink_model_name(df["Model"].to_list())
        efficiency = df["metrics.token_efficiency"].to_list()
        scores = df["AVG"].to_list()
        tup = zip(models, efficiency, scores)
        #print(list(tup))
        sorted_tup = sorted(tup, key=lambda llm: llm[2])
        print(sorted_tup)
        models, efficiency, scores = zip(*sorted_tup)

        #models = self.shrink_model_name(df_e["Model"].to_list())
        #efficiency_e = df_e["metrics.token_efficiency"].to_list()

        #models, efficiency_e = zip(*sorted(zip(models, efficiency_e)))    

        # 棒グラフを描画
        x = np.arange(len(models))
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        width = 0.4  # バーの幅
        plt.scatter(scores, efficiency)
        for x, y, label in zip(scores, efficiency, models):
            plt.text(x + 0.01, y + 0.01, label, fontsize=10, color='darkslategray')
    
        plt.xlabel("AVG Scores")
        plt.ylabel("Token Efficiency(token length/input bytes)")
        plt.title(f"Token Efficiency by Score")
        #plt.xticks(x-width/2-2, models, rotation=45)  # X軸のラベルを回転して見やすくする
        plt.legend()
        plt.show()
