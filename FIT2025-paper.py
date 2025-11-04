import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from cache.cache import Cache
    from hydra import initialize, compose, initialize_config_dir
    import os
    from pathlib import Path

    # 実験ごとに異なる設定で良い．
    # ただ，キャッシュはグローバル的じゃないと意味がないので，configのcacheセクションは
    # これくらいで固定がいいんじゃないか
    # cache:
    #    enable: True
    #    top_dir: "CacheStorage"
    #    root: "fit2025"
    import os

    path = os.getcwd()

    print(path)
    #config = "config/lmcr_exp_fit2025ex07.yaml"
    config = "lmcr_exp_fit2025exp07"
    # プロジェクトディレクトリに移動
    os.chdir("/home/tetsu.sato/FIT2025-paper")
    print(f"Changed to: {os.getcwd()}")
    project_root = Path("/home/tetsu.sato/FIT2025-paper")
    source_config_dir = project_root / "config"

    #with initialize(version_base=None, config_path="config"):
    with initialize_config_dir(version_base=None, config_dir=str(source_config_dir) ):
            #cfg = compose(config_name=config)
            cfg = compose(config_name=config)
            cache = Cache(cfg=cfg,
                          cache_filename="cache_test",
                          prefix="test",
                          )
    cache.delete("test-key")
    print(cache.get("test-key"))
    cache.set("test-key", "test-val")
    print(cache.get("test-key"))
    return (cache,)


@app.cell
def _(cache):
    from gradio_client import Client
    import polars as pl

    def get_leaderboard_result(query):
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

    #print(result["headers"])
    def leaderboard_data_to_polar_dataframe(result):
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

    key_base_name = "leaderboard_result"
    key_tag_name = "0801"
    key = f"{key_base_name}_{key_tag_name}"
    cache_result = cache.get(key)
    print(f"cache={cache_result}")
    if cache_result is None:
        result = get_leaderboard_result("")
        cache.set(key, result)
    else:
        result = cache_result
    print(f"leaderboard result={result}")
    print(f"leaderboard result class={result.__class__}")
    leaderboard_df = leaderboard_data_to_polar_dataframe(result)
    print(leaderboard_df)

    import matplotlib.pyplot as plt
    import japanize_matplotlib 
    avg_values = leaderboard_df["AVG"].to_numpy()
    #print(f"avg_values={avg_values}")
    plt.hist(avg_values, bins=20, edgecolor="black")
    plt.xlabel("スコア（平均値）")
    plt.ylabel("モデル数")
    plt.title("スコア（平均値）の分布")
    plt.show()


    return leaderboard_df, pl, plt


@app.cell
def _(leaderboard_df, pl):
    import numpy as np
    # bins の定義
    bins = 21
    labels = [f'Bin_{i+1}' for i in range(bins)]
    print(f"labels={labels}")
    all_range=[min(leaderboard_df["AVG"]), max(leaderboard_df["AVG"])]
    each_range_length=(all_range[1]-all_range[0])/(bins-1)
    print(f"each range={each_range_length}")
    range_list=[[all_range[0]+i*each_range_length, all_range[0]+(i+1)*each_range_length] for i in range(bins)]
    print(f"range_list={range_list}")

    # bins の範囲に従ってカテゴリを付与
    df = leaderboard_df.with_columns(
        #pl.col("AVG").cut(each_range_length).alias('bin_group')
        pl.col("AVG").cut([i[0] for i in range_list]).alias('bin_group')
    )

    print(f"df={df}")
    # 各 bins に属する Model の値を出力
    for bin_label in df["bin_group"].unique():
        models_in_bin = df.filter(df["bin_group"] == bin_label).select("Model")
        with pl.Config(tbl_rows=-1, fmt_str_lengths=50):
            print(f"{bin_label}: {models_in_bin}")
    return np, range_list


@app.cell
def _(range_list):
    [i[0] for i in range_list]
    return


@app.cell
def _(mo):
    mo.md(r"""## all_models.shの結果を可視化したい""")
    return


@app.cell
def _():
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
    return (models,)


@app.cell
def _(pl):
    import mlflow
    import pandas as pd

    # experiment_idを指定（もしくは名前で取得）
    experiment_id = "Default"

    # 条件をフィルタリングして検索（DataFrameで返る）

    def search_model_from_mlflow(queries, size_filter=None, run_name=None):
        runs_df = None
        for query in queries:
            print(f"query={query}")
            query_string = f"tags.model_name like '{query}'"
            print(f"query_string={query_string}")
            if size_filter is not None:
                query_string = f"{query_string} and metrics.original_size = {size_filter}"
            #print(f"query string={query_string}")
            df = mlflow.search_runs(
                experiment_names=[experiment_id],
                filter_string=f"tags.model_name like '{query}' and tags.exp_title like '{run_name}'",
                #filter_string=f"tags.model_name like '{query}%'",
                order_by=["start_time DESC"],
                output_format="pandas"
            )
            #print(f"df={df}")
            print(f"include tags.model_name?={df}")
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
    return (search_model_from_mlflow,)


@app.cell
def _(models):
    print(";".join(models[0:3]))
    return


@app.cell
def _(models, pl, search_model_from_mlflow):
    #runs_df = search_model_from_mlflow(models[0:3], run_name="fit2025-paper-04%")
    runs_df = search_model_from_mlflow(models[0:-2], run_name="fit2025-paper-05%")
    print(runs_df)
    with pl.Config(tbl_cols=-1):
        print(runs_df.select(["tags.model_name", "tags.algorithm", "metrics.token_efficiency", "metrics.llm_score"]))
    return (runs_df,)


@app.cell
def _(runs_df):
    runs_df.schema
    return


@app.cell
def _(runs_df):
    runs_df.select("metrics.llm_score")
    return


@app.cell
def _(pl, runs_df):
    def mlflow_results_df_to_analytics_df(df):
        from pprint import pprint
        # グラフを描くために必要な項目の取り出しと確認
        pprint(df.schema)
        df_exp = df.select(#pl.col('tags.mlflow.source.name'),
                  pl.col('tags.model_name'),
                  pl.col('tags.input_file_name'),
                  pl.col('tags.algorithm'),
                  pl.col('metrics.original_size'),
                  pl.col('metrics.tokenized_size'),
                  pl.col('metrics.compressed_size'),
                  pl.col('metrics.token_efficiency'),
                  pl.col('metrics.llm_score')
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
    total_exp_df = mlflow_results_df_to_analytics_df(runs_df)
    print(total_exp_df)
    return mlflow_results_df_to_analytics_df, total_exp_df


@app.cell
def _(pl, total_exp_df):
    total_exp_df.group_by(pl.col("tags.model_name")).all()
    return


@app.cell
def _(leaderboard_df):
    # llm-jpリーダーボードの，ある程度条件を絞った全てのデータが入っている
    print(leaderboard_df)
    return


@app.cell
def _(models):
    # FIT2025ペーパー用に実験したリスト
    models
    # runs_df→total_exp_dfは，modelsの中から選ばれている
    return


@app.cell
def _(leaderboard_df, total_exp_df):
    # 積集合を取りたい
    target_leaderboard_df = leaderboard_df.join(total_exp_df, left_on="Model", right_on="tags.model_name", how="inner")
    print(target_leaderboard_df)
    return


@app.function
def plot_size_score_by_model(df, algorithm, lang):
    import matplotlib.pyplot as plt
    plt.figure()

    # 入力長を変化させたときのスコアをプロット
    for model, model_df in df.group_by("Model"):
        print(f"model={model}")
        print(f"model_df={model_df}")
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
    plt.legend()  # ← 凡例を表示する
    plt.grid(True)
    plt.show()


@app.function
def shrink_model_name(model_names):
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


@app.cell
def _(np):
    def get_llm_score(df):
        return df["metrics.llm_score"].item()
    def get_modified_llm_score(df):
        original_size = df["metrics.original_size"].item()
        tokenized_size = df["metrics.tokenized_size"].item()
        encoded_size = df["metrics.compressed_size"].item()
        score = encoded_size/tokenized_size
        return score
    def get_token_efficiency(df):
        return df["metrics.token_efficiency"].item()
    def get_leaderboard_score(df):
        return df["AVG"].item()

    def plot_leaderboard_score_score_by_model(length_df, 
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
            model_name = shrink_model_name(model)
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

    return (
        get_leaderboard_score,
        get_llm_score,
        get_modified_llm_score,
        get_token_efficiency,
        plot_leaderboard_score_score_by_model,
    )


@app.cell
def _(leaderboard_df, pl, total_exp_df):
    def leaderboard_df_to_report_df(leaderboard_df, total_exp_df):
        df_report = leaderboard_df.join(total_exp_df.rename({"tags.model_name": "Model"}), on="Model")\
        .filter(pl.col("average_type") == "4-shot")\
        .drop(pl.col("ID"))\
        .drop(pl.col("average_type"))
        return df_report
    df_report = leaderboard_df_to_report_df(leaderboard_df, total_exp_df)
    with pl.Config(tbl_cols=-1, fmt_str_lengths=50):
        print(df_report)
    return df_report, leaderboard_df_to_report_df


@app.cell
def _(
    df_report,
    get_leaderboard_score,
    get_llm_score,
    get_modified_llm_score,
    get_token_efficiency,
    np,
    plot_leaderboard_score_score_by_model,
):
    def plot_leaderboard_lmcr(df, algorithm, lang):
        #plt.figure()
        # 入力長別に正解とスコアをプロット
        # 入力長ごとに，モデルごとの相関係数
        corr_list = []
        for input_length, length_df in df.group_by("metrics.original_size"):
            print(f"input_length={input_length}")
            #print(f"length_df={length_df}")
            corr_list_result = plot_leaderboard_score_score_by_model(length_df, 
                                                                     algorithm, 
                                                                     #option=input_length,
                                                                     option=None,
                                                                     x_func=get_leaderboard_score,
                                                                     y_func=get_llm_score,
                                                                     x_title="Leaderboard Score",
                                                                     y_title="LMCR Index",
                                                                     title=f"Leaderboard Score vs LMCR Index({lang})"
                                                                    )
            corr_list.append(corr_list_result)

        print(f"corr_list={corr_list}, avg={np.average(corr_list)}")

    def plot_leaderboard_modified_lmcr(df, algorithm, lang):
        corr_list = []
        for input_length, length_df in df.group_by("metrics.original_size"):
            print(f"input_length={input_length}")
            #print(f"length_df={length_df}")
            corr_list_result = plot_leaderboard_score_score_by_model(length_df, 
                                                                     algorithm, 
                                                                     #option=input_length,
                                                                     option=None,
                                                                     x_func=get_leaderboard_score,
                                                                     y_func=get_modified_llm_score,
                                                                     x_title="Leaderboard Score",
                                                                     y_title="Late-stage LMCR Index",
                                                                     title=f"Leaderboard Score vs Late-stage LMCR Index({lang})"
                                                                    )
            corr_list.append(corr_list_result)  

    def plot_leaderboard_token_efficiency(df, algorithm, lang):
        corr_list = []
        for input_length, length_df in df.group_by("metrics.original_size"):
            print(f"input_length={input_length}")
            #print(f"length_df={length_df}")
            corr_list_result = plot_leaderboard_score_score_by_model(length_df, 
                                                                     algorithm, 
                                                                     #option=input_length,
                                                                     option=None,
                                                                     x_func=get_leaderboard_score,
                                                                     y_func=get_token_efficiency,
                                                                     x_title="Leaderboard Score",
                                                                     y_title="Token Efficiency",
                                                                     title=f"Leaderboard Score vs Token Efficiency({lang})"
                                                                    )
            corr_list.append(corr_list_result)

    def plot_lmcr_token_efficiency(df, algorithm, lang):
        corr_list = []
        for input_length, length_df in df.group_by("metrics.original_size"):
            print(f"input_length={input_length}")
            #print(f"length_df={length_df}")
            corr_list_result = plot_leaderboard_score_score_by_model(length_df, 
                                                                     algorithm, 
                                                                     #option=input_length,
                                                                     option=None,
                                                                     x_func=get_llm_score,
                                                                     y_func=get_token_efficiency,
                                                                     x_title="LMCR Index",
                                                                     y_title="Token Efficiency",
                                                                     title=f"LMCR Index vs Token Efficiency({lang})"
                                                                    )
            corr_list.append(corr_list_result)

    def plot_graph(df_report, lang):
        for algorithm,df in df_report.group_by("tags.algorithm"):
        #for algorithm,df in df_report.filter(pl.col("Model").str.starts_with("llm-jp")).group_by("tags.algorithm"):
            print(f"algorithm={algorithm}")
            if algorithm[0] == "ppl":
                continue
            #df = df.drop("tags.algorithm")
            print(f"df={df}")
            plot_size_score_by_model(df, algorithm, lang)

            plot_leaderboard_lmcr(df, algorithm, lang)

            plot_leaderboard_modified_lmcr(df, algorithm, lang)

            plot_leaderboard_token_efficiency(df, algorithm, lang)

            plot_lmcr_token_efficiency(df, algorithm, lang)

    plot_graph(df_report, "j")
    return (plot_graph,)


@app.cell
def _(df_report):
    df_report
    return


@app.cell
def _(df_report, pl):
    df_report.filter(pl.col("Model").str.starts_with("llm-jp"))
    return


@app.cell
def _(df_report, pl):
    df_report.filter(pl.col("Model").str.starts_with("tiiuae"))
    return


@app.cell
def _(df_report, pl):
    df_report.filter(pl.col("Model").str.starts_with("google"))
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 従来研究で，異常値は過学習じゃないかとあったが，職務経歴書って学習されてなさそう
    特に日本語だとトークナイザの影響が大きいんじゃないか
    ということで，英語テキストでの実験もやっておきたい
    """
    )
    return


@app.cell
def _(pl):
    kaggle_resume_path = "./kaggle-resume/Resume.csv"
    kaggle_resume_df = pl.read_csv(kaggle_resume_path)
    kaggle_resume_df = kaggle_resume_df.select("ID", "Resume_str", "Category")\
                                       .with_columns(pl.col("Resume_str").str.len_chars().alias("Resume_str_length"))
    kaggle_resume_df.select(pl.col("Category").unique())
    return (kaggle_resume_df,)


@app.cell
def _(kaggle_resume_df, pl):
    kaggle_resume_df.filter(pl.col("Category") == "INFORMATION-TECHNOLOGY").sort(pl.col("Resume_str_length"))
    return


@app.cell
def _(kaggle_resume_df, pl):
    kaggle_resume_df.filter(pl.col("ID") == 12635195)\
                    .select(pl.col("Resume_str"))\
                    .write_csv("./kaggle-resume/12635195.txt",
                              include_header=False,
                              quote_style="never")
    return


@app.cell
def _(
    leaderboard_df,
    leaderboard_df_to_report_df,
    mlflow_results_df_to_analytics_df,
    models,
    plot_graph,
    search_model_from_mlflow,
):
    runs_df_e = search_model_from_mlflow(models[0:-2], run_name="fit2025-paper-06%")
    total_exp_df_e = mlflow_results_df_to_analytics_df(runs_df_e)
    target_leaderboard_df_e = leaderboard_df.join(total_exp_df_e, left_on="Model", right_on="tags.model_name", how="inner")
    df_report_e = leaderboard_df_to_report_df(leaderboard_df, total_exp_df_e)
    plot_graph(df_report_e, "e")
    return (df_report_e,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    原稿はベンチマークありきなので，ベンチマークが信頼できることを説明したい
    例えば，パラメータ数とベンチマークスコアが線形の関係にある図が欲しい
    """
    )
    return


@app.cell
def _():
    # 分かりやすい説明にするために，いくつかのモデルにしぼりたい
    # s=small_examples
    #runs_df_s = search_model_from_mlflow(["llm-jp/%", "Qwen/%"], run_name="fit2025-paper-05%")
    #total_exp_df_s = mlflow_results_df_to_analytics_df(runs_df_s)
    #target_leaderboard_df_s = leaderboard_df.join(total_exp_df_s, left_on="Model", right_on="tags.model_name", how="inner")
    #df_report_s = leaderboard_df_to_report_df(leaderboard_df, total_exp_df_s)
    #plot_graph(df_report_s, "j")
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    論文書いてると，訓練データに対する過学習以外にも，トークナイザの影響がある．と書きたくなってきたので，「英語データだとトークン効率のLLMによる差はこうだが，日本語データだとこうである」と行きたい
    LLM vs トークン効率？
    縦軸がトークンとして横軸なんだ
    適当にLLMの名前で，棒グラフでいいのか
    """
    )
    return


@app.cell
def _(df_report):
    df_report
    return


@app.cell
def _(df_report_e):
    df_report_e
    return


@app.cell
def _(df_report, pl):
    df_report.select(pl.col("Model"), pl.col("metrics.token_efficiency"))
    return


@app.cell
def _(df_report, df_report_e, np, pl, plt):
    def plot_token_efficiency(df, df_e):
        # データ取得
        models = shrink_model_name(df["Model"].to_list())
        efficiency = df["metrics.token_efficiency"].to_list()
        print(list(zip(models, efficiency)))
        print(sorted(zip(models, efficiency)))
        models, efficiency = zip(*sorted(zip(models, efficiency)))
        models = shrink_model_name(df_e["Model"].to_list())
        efficiency_e = df_e["metrics.token_efficiency"].to_list()

        models, efficiency_e = zip(*sorted(zip(models, efficiency_e)))    

        # 棒グラフを描画
        x = np.arange(len(models))
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

    plot_token_efficiency(df_report.filter(pl.col("tags.algorithm")=="ae"),
                          df_report_e.filter(pl.col("tags.algorithm")=="ae"),
                         )
    #plot_token_efficiency(df_report_e, "English")

    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    トークナイザーの影響の除いても，gemma-8b-it, gemma-1.1-2b-it, Falcon3-1b-Inst, Falcon3-3b-Inst, SmolLM2-1.7B-Instが大きく乖離しているのかが分からない
    リーダーボードスコアが近くて，上に乖離・乖離が少ない・下に乖離，を探すと，
    - gemma-1.1-2b-it, gemma-2b-it
    - llm-jp-3-440m-inst2, Qwen2-0.5B-Inst
    - Falcon3-3B-Inst, SmolLm2-1.7B-Inst
    とかか
    """
    )
    return


@app.cell
def _():
    return


@app.cell
def _(
    leaderboard_df,
    leaderboard_df_to_report_df,
    mlflow_results_df_to_analytics_df,
    pl,
    plot_graph,
    runs_df,
    search_model_from_mlflow,
):
    runs_df_sample = search_model_from_mlflow(["%"], run_name="fit2025-paper-07%")
    print(f"sample={runs_df_sample}")
    print(f"original={runs_df}")
    # "tags.model_name" をキーに結合
    runs_df_sample = runs_df_sample.join(
        runs_df, 
        left_on="tags.model_name", 
        right_on="tags.model_name", 
        how="left"
    ).filter(pl.col("tags.algorithm")=="ae")
    total_exp_df_sample = mlflow_results_df_to_analytics_df(runs_df_sample)
    target_leaderboard_df_sample = leaderboard_df.join(total_exp_df_sample, left_on="Model", right_on="tags.model_name", how="inner")
    df_report_sample = leaderboard_df_to_report_df(leaderboard_df, total_exp_df_sample)\
                       .filter(pl.col("tags.algorithm")=="ae")\
                       .unique()
    with pl.Config(tbl_cols=-1, fmt_str_lengths=50):
        print(f"df_report_sample={df_report_sample}")
    plot_graph(df_report_sample, "j")

    return (df_report_sample,)


@app.cell
def _(kairidf_report):
    kairidf_report
    return


@app.cell
def _(df_report_sample, plot_graph):
    plot_graph(df_report_sample, "j")
    return


@app.cell
def _(df_report_sample):
    plot_size_score_by_model(df_report_sample, "ae", "Japanese")
    return


@app.cell
def _(models):
    models[0:-2]
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""ちょっとバイト数なのか文字数なのか心配になったので確認""")
    return


app._unparsable_cell(
    r"""
    !pwd
    """,
    name="_"
)


@app.cell
def _():
    from pathlib import Path
    msg = Path("small_text.txt").read_text(encoding="utf-8")
    print(f"msg={msg}")
    print(f"msg len={len(msg)}")
    return


@app.cell
def _(o):
    o
    return


if __name__ == "__main__":
    app.run()
