from src.cache.cache import Cache
from hydra import initialize, compose, initialize_config_dir
import mlflow
import numpy as np
import os
from pathlib import Path
import pandas as pd
from gradio_client import Client
import polars as pl
from typing import List, Optional, Callable


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
    def __init__(
                 self,
                 project_root: str,
                 ):
        """
        コンストラクタ
        Args:
            project_root (str): プロジェクトルートのパス
        """
        project_root_path = Path(project_root)
        config = "lmcr_exp_fit2025exp07"
        config_dir = project_root_path / "config"
        with initialize_config_dir(version_base=None,
                                   config_dir=str(config_dir),
                                   ):
            cfg = compose(config_name=config)
        self.cache = Cache(cfg=cfg,
              cache_filename="cache_test",
              prefix="test",
              )


    def get_leaderboard_result(
                               self,
                               query: str,
                               ) -> dict:
        """
        リーダーボードの結果を取得する
        キャッシュがあればキャッシュから、なければAPIから取得する
        Args:
            query (str): 検索クエリ
        Returns:
            dict: リーダーボードの結果
        """
        key_base_name = "leaderboard_result"
        key_tag_name = "0801"
        key = f"{key_base_name}_{key_tag_name}"
        cache_result = self.cache.get(key)
        print(f"cache={cache_result}")
        if cache_result is None:
            result = self._get_leaderboard_result("")
            self.cache.set(key, result)
        else:
            result = cache_result
        return result
    def _get_leaderboard_result(
                               self,
                               query: str,
                               ) -> dict:
        """
        リーダーボードの結果をAPIから取得する
        Args:
            query (str): 検索クエリ
        Returns:
            dict: リーダーボードの結果
        """
        client = Client("llm-jp/open-japanese-llm-leaderboard")
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
    		#param_9=["AVG (NLI)"],
        param_9=[],
        param_10=[],
    		#param_11=["AVG (RC)"],
        param_11=[],
    		#param_12=["AVG (MC)"],
        param_12=[],
    		#param_13=["AVG (EL)"],
        param_13=[],
    		#param_14=["AVG (FA)"],
        param_14=[],
    		#param_15=["AVG (MR)"],
        param_15=[],
    		#param_16=["AVG (MT)"],
        param_16=[],
    		#param_17=[],
    		#param_18=["AVG (HE)"],
        param_18=[],
    		#param_19=["AVG (CG)"],
        param_19=[],
    	param_20=["AVG (SUM)"],
    		#param_21=[],
        	api_name="/update_table"
        )
        return result
    
    def _leaderboard_data_to_polar_dataframe(
                                            self,
                                            result: dict) -> pl.DataFrame:
        """
        リーダーボードのデータをpolarsのDataFrameに変換する
        Args:
            result (dict): リーダーボードのデータ
        Returns:
            pl.DataFrame: polarsのDataFrame
        """
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
    def get_leaderboard_df(self) -> pl.DataFrame:
        """
        リーダーボードのDataFrameを取得する
        Returns:
            pl.DataFrame: polarsのDataFrame
        """
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

    def search_model_from_mlflow(
                                 self,
                                 queries: List[str],
                                 size_filter: Optional[int] = None,
                                 run_name: Optional[str] = None,
                                 ) -> pl.DataFrame:
        """
        MLflowからモデルを検索する
        Args:
            queries (List[str]): 検索クエリのリスト
            size_filter (Optional[int]): サイズのフィルタ
            run_name (Optional[str]): 実行名
        Returns:
            pl.DataFrame: polarsのDataFrame
        """
        # 条件をフィルタリングして検索（DataFrameで返る）

        # experiment_idを指定（もしくは名前で取得）
        experiment_id = "Default"
        runs_df = None
        for query in queries:
            #print(f"query={query}")
            query_string = f"tags.model_name like \'{query}\'"
            #print(f"query_string={query_string}")
            if size_filter is not None:
                query_string = f"{query_string} and metrics.original_size = {size_filter}"
            #print(f"query string={query_string}")
            df = mlflow.search_runs(
                experiment_names=[experiment_id],
                filter_string=f"tags.model_name like \'{query}\' and tags.exp_title like \'{run_name}\'",
                #filter_string=f"tags.model_name like \'{query}%\'",
                order_by=["start_time DESC"],
                output_format="pandas"
            )
            #print(f"df={df}")
            #print(f"include tags.model_name?={df}")
            df = df.sort_values("start_time", ascending=False)\
                   .groupby(["tags.model_name", "tags.algorithm", "metrics.original_size"])\
                   .first()\
                   .reset_index()
            #print(f"first=>{df}")
            if runs_df is None:
                runs_df = df
            else:
                runs_df = pd.concat([runs_df, df])
        runs_df = pl.from_pandas(runs_df)
        return runs_df

    def mlflow_results_df_to_analytics_df(
                                          self,
                                          df: pd.DataFrame) -> pl.DataFrame:
        """
        MLflowの結果のDataFrameを解析用のDataFrameに変換する
        Args:
            df (pd.DataFrame): MLflowの結果のDataFrame
        Returns:
            pl.DataFrame: 解析用のDataFrame
        """
        from pprint import pprint
        # グラフを描くために必要な項目の取り出しと確認
        #pprint(df.schema)
        df_exp = df.select(#pl.col(\'tags.mlflow.source.name\'),
                  pl.col(\'tags.model_name\\'),
                  pl.col(\'tags.input_file_name\'),
                  pl.col(\'tags.algorithm\'),
                  pl.col(\'metrics.original_size\'),
                  pl.col(\'metrics.tokenized_size\'),
                  pl.col(\'metrics.compressed_size\'),
                  pl.col(\'metrics.token_efficiency\'),
                  pl.col(\'metrics.llm_score\')
                 ).group_by("tags.algorithm",maintain_order=True)
        total_exp_df = pl.DataFrame()
        for algorithm, data_df in df_exp:
            #print(algorithm)
            for model_name, data_df in data_df.group_by(\'tags.model_name\'):
                #print(model_name)
                #print(data_df)
                #data_df = data_df.with_columns(pl.col("tags.model_name").alias("model_name"))
                total_exp_df = total_exp_df.vstack(data_df)
        return total_exp_df

    def shrink_model_name(
                          self,
                          model_names: List[str]) -> List[str]:
        """
        モデル名を短縮する
        Args:
            model_names (List[str]): モデル名のリスト
        Returns:
            List[str]: 短縮されたモデル名のリスト
        """
        names = [x.replace(\'jsai-dev-01_\', \'\\')
                  .replace(\'deepseek-ai-\\' , \'
')
                  .replace(\'meta-llama/\', \'
')
                  .replace(\'Qwen/\', \'
')
                  .replace(\'llm-jp/\', \'
')
                  .replace(\'google/\', \'
')
                  .replace(\'tiiuae/\', \'
')
                  .replace(\'HuggingFaceTB/\', \'
')
                  .replace(\'weblab-GENIAC/\', \'
')
                  .replace(\'Instruct\', \"Inst\")
                  .replace(\"instruct\", \"inst\")
                   for x in model_names]
        return names

    def get_llm_score(
                      self,
                      df: pl.DataFrame) -> float:
        """
        LLMのスコアを取得する
        Args:
            df (pl.DataFrame): DataFrame
        Returns:
            float: LLMのスコア
        """
        return df["metrics.llm_score"].item()

    def get_modified_llm_score(
                               self,
                               df: pl.DataFrame) -> float:
        """
        修正されたLLMのスコアを取得する
        Args:
            df (pl.DataFrame): DataFrame
        Returns:
            float: 修正されたLLMのスコア
        """
        original_size = df["metrics.original_size"].item()
        tokenized_size = df["metrics.tokenized_size"].item()
        encoded_size = df["metrics.compressed_size"].item()
        score = encoded_size/tokenized_size
        return score

    def get_token_efficiency(
                             self,
                             df: pl.DataFrame) -> float:
        """
        トークンの効率を取得する
        Args:
            df (pl.DataFrame): DataFrame
        Returns:
            float: トークンの効率
        """
        return df["metrics.token_efficiency"].item()

    def get_leaderboard_score(
                              self,
                              df: pl.DataFrame) -> float:
        """
        リーダーボードのスコアを取得する
        Args:
            df (pl.DataFrame): DataFrame
        Returns:
            float: リーダーボードのスコア
        """
        return df["AVG"].item()

    def plot_leaderboard_score_score_by_model(
                                              self,
                                              length_df: pl.DataFrame, 
                                              algorithm: List[str], 
                                              option: Optional[str],
                                              x_func: Callable[[pl.DataFrame], float],
                                              y_func: Callable[[pl.DataFrame], float],
                                              x_title: str,
                                              y_title: str,
                                              title: str,
                                             ) -> List[float]:
        """
        モデルごとのリーダーボードのスコアとスコアをプロットする
        Args:
            length_df (pl.DataFrame): 長さごとのDataFrame
            algorithm (List[str]): アルゴリズム
            option (Optional[str]): オプション
            x_func (Callable[[pl.DataFrame], float]): x軸の関数
            y_func (Callable[[pl.DataFrame], float]): y軸の関数
            x_title (str): x軸のタイトル
            y_title (str): y軸のタイトル
            title (str): グラフのタイトル
        Returns:
            List[float]: 相関係数のリスト
        """
        import matplotlib.pyplot as plt
        plt.figure()
        print(f"option={option}")

        # 回帰直線を計算するために，入力長ごとにデータを集める
        x_arr = []
        y_arr = []
        corr_list = []
        for model, length_model_df in length_df.group_by("Model"):
            print(f"model={model}")
            #with pl.Config(tbl_cols=-1):
            #    print(f"length_model_df={length_model_df}")
            # 変なデータがリーダーボードに登録されている
            if model[0] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
                length_model_df = length_model_df.limit(1)
            #x = length_model_df["AVG"].item()
            x = x_func(length_model_df)
            #y = length_model_df["metrics.llm_score"].item()
            y = y_func(length_model_df)
            x_arr.append(x)
            y_arr.append(y)
            print(f"(x,y)=({x}, {y})")
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
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
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

    def leaderboard_df_to_report_df(
                                    self,
                                    leaderboard_df: pl.DataFrame,
                                    total_exp_df: pl.DataFrame,
                                    ) -> pl.DataFrame:
        """
        リーダーボードのDataFrameをレポート用のDataFrameに変換する
        Args:
            leaderboard_df (pl.DataFrame): リーダーボードのDataFrame
            total_exp_df (pl.DataFrame): 実験結果のDataFrame
        Returns:
            pl.DataFrame: レポート用のDataFrame
        """
        df_report = leaderboard_df.join(total_exp_df.rename({"tags.model_name": "Model"}), on="Model")\
        .filter(pl.col("average_type") == "4-shot")\
        .drop(pl.col("ID"))\
        .drop(pl.col("average_type"))
        return df_report
    
    def plot_size_score_by_model(
                                 self,
                                 df: pl.DataFrame,
                                 algorithm: List[str],
                                 lang: str,
                                 ) -> None:
        """
        モデルごとのサイズとスコアをプロットする
        Args:
            df (pl.DataFrame): DataFrame
            algorithm (List[str]): アルゴリズム
            lang (str): 言語
        """
        import matplotlib.pyplot as plt
        plt.figure()

        # 入力長を変化させたときのスコアをプロット
        for model, model_df in df.group_by("Model"):
            #print(f"model={model}")
            #print(f"model_df={model_df}")
            plt.plot(model_df["metrics.original_size"], 
                     model_df["metrics.llm_score"],
                     label=f"{model[0]}-{algorithm[0]}",  # ← ラベルを設定！
                     marker="o",
                    )
        # 凡例とラベルの表示
        plt.xlabel("Original Size")
        if algorithm[0] == "ae":
            plt.ylabel("LMCR Index")
        else:
            plt.ylabel("Perplexity")

        plt.title(f"Score({lang}) vs Original Size by Algorithm and Model")
        #plt.legend()  # ← 凡例を表示する
        plt.grid(True)
        plt.show()
        
    def plot_leaderboard_lmcr(
                              self,
                              df: pl.DataFrame,
                              algorithm: List[str],
                              lang: str) -> None:
        """
        リーダーボードのスコアとLMCRをプロットする
        Args:
            df (pl.DataFrame): DataFrame
            algorithm (List[str]): アルゴリズム
            lang (str): 言語
        """
        #plt.figure()
        # 入力長別に正解とスコアをプロット
        # 入力長ごとに，モデルごとの相関係数
        corr_list = []
        for input_length, length_df in df.group_by("metrics.original_size"):
            print(f"input_length={input_length}")
            #print(f"length_df={length_df}")
            corr_list_result = self.plot_leaderboard_score_score_by_model(
                                        length_df, 
                                        algorithm, 
                                        #option=input_length,
                                        option=None,
                                        x_func=self.get_leaderboard_score,
                                        y_func=self.get_llm_score,
                                        x_title="Leaderboard Score",
                                        y_title="LMCR Index",
                                        title=f"Leaderboard Score vs LMCR Index({lang})"
                                      )
            corr_list.append(corr_list_result)

        print(f"corr_list={corr_list}, avg={np.average(corr_list)}")

    def plot_leaderboard_modified_lmcr(
                                       self,
                                       df: pl.DataFrame,
                                       algorithm: List[str],
                                       lang: str,
                                       ) -> None:
        """
        リーダーボードのスコアと修正されたLMCRをプロットする
        Args:
            df (pl.DataFrame): DataFrame
            algorithm (List[str]): アルゴリズム
            lang (str): 言語
        """
        corr_list = []
        for input_length, length_df in df.group_by("metrics.original_size"):
            print(f"input_length={input_length}")
            #print(f"length_df={length_df}")
            corr_list_result = self.plot_leaderboard_score_score_by_model(length_df, 
                                                                     algorithm, 
                                                                     #option=input_length,
                                                                     option=None,
                                                                     x_func=self.get_leaderboard_score,
                                                                     y_func=self.get_modified_llm_score,
                                                                     x_title="Leaderboard Score",
                                                                     y_title="Late-stage LMCR Index",
                                                                     title=f"Leaderboard Score vs Late-stage LMCR Index({lang})"
                                                                    )
            corr_list.append(corr_list_result)  

    def plot_leaderboard_token_efficiency(
                                          self,
                                          df: pl.DataFrame,
                                          algorithm: List[str],
                                          lang: str,
                                          ) -> None:
        """
        リーダーボードのスコアとトークンの効率をプロットする
        Args:
            df (pl.DataFrame): DataFrame
            algorithm (List[str]): アルゴリズム
            lang (str): 言語
        """
        corr_list = []
        for input_length, length_df in df.group_by("metrics.original_size"):
            print(f"input_length={input_length}")
            #print(f"length_df={length_df}")
            corr_list_result = self.plot_leaderboard_score_score_by_model(length_df, 
                                                                     algorithm, 
                                                                     #option=input_length,
                                                                     option=None,
                                                                     x_func=self.get_leaderboard_score,
                                                                     y_func=self.get_token_efficiency,
                                                                     x_title="Leaderboard Score",
                                                                     y_title="Token Efficiency",
                                                                     title=f"Leaderboard Score vs Token Efficiency({lang})"
                                                                    )
            corr_list.append(corr_list_result)

    def plot_lmcr_token_efficiency(
                                   self,
                                   df: pl.DataFrame,
                                   algorithm: List[str],
                                   lang: str,
                                   ) -> None:
        """
        LMCRとトークンの効率をプロットする
        Args:
            df (pl.DataFrame): DataFrame
            algorithm (List[str]): アルゴリズム
            lang (str): 言語
        """
        corr_list = []
        for input_length, length_df in df.group_by("metrics.original_size"):
            print(f"input_length={input_length}")
            #print(f"length_df={length_df}")
            corr_list_result = self.plot_leaderboard_score_score_by_model(length_df, 
                                                                     algorithm, 
                                                                     #option=input_length,
                                                                     option=None,
                                                                     x_func=self.get_llm_score,
                                                                     y_func=self.get_token_efficiency,
                                                                     x_title="LMCR Index",
                                                                     y_title="Token Efficiency",
                                                                     title=f"LMCR Index vs Token Efficiency({lang})"
                                                                    )
            corr_list.append(corr_list_result)

    def plot_graph(
                   self,
                   df_report: pl.DataFrame,
                   lang: str,
                   ) -> None:
        """
        グラフをプロットする
        Args:
            df_report (pl.DataFrame): レポート用のDataFrame
            lang (str): 言語
        """
        for algorithm,df in df_report.group_by("tags.algorithm"):
        #for algorithm,df in df_report.filter(pl.col("Model").str.starts_with("llm-jp")).group_by("tags.algorithm"):
            print(f"algorithm={algorithm}")
            if algorithm[0] == "ppl":
                continue
            #df = df.drop("tags.algorithm")
            print(f"df={df}")
            self.plot_size_score_by_model(df, algorithm, lang)

            self.plot_leaderboard_lmcr(df, algorithm, lang)

            self.plot_leaderboard_modified_lmcr(df, algorithm, lang)

            self.plot_leaderboard_token_efficiency(df, algorithm, lang)

            self.plot_lmcr_token_efficiency(df, algorithm, lang)
    def plot_token_efficiency(
                              self,
                              df: pl.DataFrame,
                              df_e: pl.DataFrame,
                              ) -> None:
        """
        トークンの効率をプロットする
        Args:
            df (pl.DataFrame): DataFrame
            df_e (pl.DataFrame): 英語のDataFrame
        """
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
        plt.xticks(x, models, rotation=90)  # X軸のラベルを回転して見やすくする
        plt.legend()
        plt.show()