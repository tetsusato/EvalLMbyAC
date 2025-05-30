import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import mlflow
    import pandas as pd
    import polars as pl

    # experiment_idを指定（もしくは名前で取得）
    experiment_id = "Default"

    # 条件をフィルタリングして検索（DataFrameで返る）

    def search_model_from_mlflow(queries, size_filter=None):
        runs_df = None
        for query in queries:
            #print(f"query={query}")
            query_string = f"tags.model_name like '{query}%'"
            if size_filter is not None:
                query_string = f"{query_string} and metrics.original_size = {size_filter}"
            #print(f"query string={query_string}")
            df = mlflow.search_runs(
                experiment_names=[experiment_id],
                filter_string=f"tags.model_name like '{query}%'",
                order_by=["start_time DESC"],
                output_format="pandas"
            )
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
        return runs_df

    runs_df = search_model_from_mlflow(["llm-jp", "HuggingFaceTB"])
    # CSVに保存
    #runs_df.to_csv("filtered_runs.csv", index=False)
    #print(runs_df)

    runs_df = pl.from_pandas(runs_df)
    print(runs_df.shape)
    print(runs_df.limit(2))
    return pl, runs_df, search_model_from_mlflow


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
                  #pl.col('metrics.compressed_size'),
                  pl.col('metrics.token efficiency'),
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
def _(pl):
    from gradio_client import Client

    def get_leaderboard_result(query):
        client = Client("llm-jp/open-japanese-llm-leaderboard")
        result = client.predict(
        		type_query=["🟢 pretrained","🔶 fine-tuned","⭕ instruction-tuned","🟦 RL-tuned (Preference optimization)","🌸 multimodal","🤝 base merges and moerges"],
        		precision_query=["float16","bfloat16","float32"],
        		size_query=["0~3B","3~7B","7~13B","13~35B","35~60B","60B+","?"],
        		add_special_tokens_query=["True","False"],
        		num_few_shots_query=[0,4],
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
    #print(result)
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
    result = get_leaderboard_result("llm-jp; HuggingFaceTB")
    leaderboard_df = leaderboard_data_to_polar_dataframe(result)
    print(leaderboard_df)
    return (
        get_leaderboard_result,
        leaderboard_data_to_polar_dataframe,
        leaderboard_df,
    )


@app.cell
def _(leaderboard_df):
    leaderboard_df
    return


@app.cell
def _(leaderboard_df, pl, total_exp_df):
    def leaderboard_df_to_report_df(leaderboard_df, total_exp_df):
        df_report = leaderboard_df.join(total_exp_df.rename({"tags.model_name": "Model"}), on="Model")\
        .filter(pl.col("average_type") == "4-shot")\
        .drop(pl.col("ID"))\
        .drop(pl.col("average_type"))
        return df_report
    df_report = leaderboard_df_to_report_df(leaderboard_df, total_exp_df)
    with pl.Config(tbl_cols=-1):
        print(df_report)
    return df_report, leaderboard_df_to_report_df


@app.cell
def _():
    return


@app.cell
def _(df_report, pl):
    #import matplotlib.pyplot as plt
    import numpy as np
    reshape_df = pl.DataFrame()

    def plot_size_score_by_model(df, algorithm):
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

        plt.title("Score vs Original Size by Algorithm and Model")
        plt.legend()  # ← 凡例を表示する
        plt.grid(True)
        plt.show()
    def shrink_model_name(model_names):
        names = [x.replace('jsai-dev-01_', '')\
                  .replace('deepseek-ai-', '')\
                  .replace('meta-llama-', '')\
                  .replace('Qwen-', '')\
                  .replace('llm-jp-l', 'l')\
                  .replace('google-', '')\
                  .replace('Instruct', "Inst")\
                  .replace("instruct", "inst")\
                   for x in model_names]
        return names
    def plot_leaderboard_score_score_by_model(length_df, algorithm, option):
        import matplotlib.pyplot as plt
        plt.figure()
        print(f"option={option}")

        # 回帰直線を計算するために，入力長ごとにデータを集める
        x_arr = []
        y_arr = []

        for model, length_model_df in length_df.group_by("Model"):
            print(f"model={model}")
            #with pl.Config(tbl_cols=-1):
            #    print(f"length_model_df={length_model_df}")
            # 変なデータがリーダーボードに登録されている
            if model[0] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
                length_model_df = length_model_df.limit(1)
            x = length_model_df["AVG"].item()
            y = length_model_df["metrics.llm_score"].item()
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
        plt.text(x[0], y[0], f"MSE={mse:.2f}, r2={r2:.2f}, corr={corr:.2f}")

        # 凡例とラベルの表示
        plt.xlabel("Learderboard Score")
        if algorithm[0] == "ae":
            plt.ylabel("LMCR Index")
        else:
            plt.ylabel("Perplexity")
        plt.title("Score vs Original Size by Algorithm and Input Size")
        #plt.legend()  # ← 凡例を表示する
        plt.grid(True)
        plt.show()  
    for algorithm,df in df_report.group_by("tags.algorithm"):
    #for algorithm,df in df_report.filter(pl.col("Model").str.starts_with("llm-jp")).group_by("tags.algorithm"):
        print(f"algorithm={algorithm}")
        #df = df.drop("tags.algorithm")
        print(f"df={df}")
        plot_size_score_by_model(df, algorithm)

        #plt.figure()
        # 入力長別に正解とスコアをプロット
        # 入力長ごとに，モデルごとの相関係数
        corr_list = []
        for input_length, length_df in df.group_by("metrics.original_size"):
            plot_leaderboard_score_score_by_model(length_df, algorithm, input_length)
        print(f"corr_list={corr_list}, avg={np.average(corr_list)}")
    return (plot_leaderboard_score_score_by_model,)


@app.cell
def _(
    get_leaderboard_result,
    leaderboard_data_to_polar_dataframe,
    leaderboard_df_to_report_df,
    mlflow_results_df_to_analytics_df,
    pl,
    plot_leaderboard_score_score_by_model,
    search_model_from_mlflow,
):

    def token_research():
        models = ["llm-jp",
                  "HuggingFaceTB",
                  "meta-llama/Llama-3.2",
                  "deepseek-ai/DeepSeek-R1-Distill",
                  "Qwen/Qwen2.5",
            
                 ]
        runs_df = search_model_from_mlflow(models, size_filter=1121)
        runs_df = pl.from_pandas(runs_df)
        #print(f"runs_df={runs_df}")
        runs_df = runs_df.filter(pl.col("metrics.original_size") == 1121)
        total_exp_df = mlflow_results_df_to_analytics_df(runs_df)
        models_str = ";".join(models)
        #print(f"models_str={models_str}")
        result = get_leaderboard_result(models_str)
        leaderboard_df = leaderboard_data_to_polar_dataframe(result)
        df_report = leaderboard_df_to_report_df(leaderboard_df, total_exp_df)
        #print(df_report)
        return df_report

    # 入力長は固定（順次入力でいい）で，LMCR IndexとPerplexityの関係
    def analyze_token_prop_score():
        df = token_research()
        #print(df)
        for length,length_df in df.group_by("metrics.original_size"): # 入力長ごとに
            #print(f"length={length}")
            #print(f"length df={length_df}")

            for algorithm, algorithm_length_df in length_df.group_by("tags.algorithm"): # LMCRとperplexityは分けて
                print(f"algorithm_length_df={algorithm_length_df.sort(["AVG", "metrics.token efficiency"])}")
                #plot_size_score_by_model(algorithm_length_df, algorithm)
                plot_leaderboard_score_score_by_model(algorithm_length_df, algorithm, "token_efficiency")

    analyze_token_prop_score()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
