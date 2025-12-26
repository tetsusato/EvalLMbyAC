import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    dir = os.getcwd()
    print(dir)
    return (os,)


@app.cell
def _(os):
    os.chdir("/home/tetsu.sato/IPSJ2026-paper")
    print(os.getcwd())
    return


@app.cell
def _():
    from src.analytics.analytics import Analyzer
    print(Analyzer)
    return (Analyzer,)


@app.cell
def _(Analyzer):
    an = Analyzer("/home/tetsu.sato/IPSJ2026-paper/",
                  config_name="lmcr_exp_ipsj2026_paper_03")
    print(an)
    return (an,)


@app.cell
def _(mo):
    mo.md(r"""
    ### HuggingFaceのSpaceに保存されているベンチマーク結果をdatasetsライブラリ経由で取る実験
    """)
    return


@app.cell
def _():
    import pandas as pd
    import datasets
    from src.llmjp.about import Tasks
    from src.llmjp.utils import AutoEvalColumn
    from src.llmjp.populate import get_leaderboard_df

    import pandas as pd
    from src.llmjp.utils import (COLS, BENCHMARK_COLS)

    ldf = get_leaderboard_df("llm-jp/leaderboard-contents", 
                      COLS,
                      BENCHMARK_COLS)
    ldf
    return (ldf,)


@app.cell
def _(an):
    result = an.get_leaderboard_result("",
                                      cache_base_name="leaderboard_result",
                                      cache_tag_name="ipsj2026_paper")
    print(f"leaderboard result={result}")
    print(f"leaderboard result class={result.__class__}")
    #leaderboard_df = an._leaderboard_data_to_polar_dataframe(result)
    import polars as pl
    leaderboard_df = pl.from_pandas(result)
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
    return leaderboard_df, pl


@app.cell
def _(ldf):
    ldf.columns
    return


@app.cell
def _(leaderboard_df, pl):
    leaderboard_df.filter(pl.col("Model") == "google/gemma-7b")\
                  .select(pl.col(["T", "Model", "AVG", "#Params (B)", "Few-shot"]))
    return


@app.cell
def _(os):
    mlflow_type="local" # "azure" or "local"
    if mlflow_type == "azure":
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    elif mlflow_type == "local":
        mlflow_tracking_uri = "http://localhost:8080"
    print(f"mlflow uri={mlflow_tracking_uri}")
    import mlflow         
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    return


@app.cell
def _(an, leaderboard_df):
    #runs_df_e = an.search_model_from_mlflow(an.models[0:-2], run_name="ipsj2026_paper_02%")
    runs_df = an.search_model_from_mlflow(run_name="ipsj2026_paper_03%")
    print(f"runs_df_e={runs_df}")
    total_exp_df = an.mlflow_results_df_to_analytics_df(runs_df)
    target_leaderboard_df = leaderboard_df.join(total_exp_df, left_on="Model", right_on="tags.model_name", how="inner")
    df_report = an.leaderboard_df_to_report_df(leaderboard_df, total_exp_df)
    an.plot_graph(df_report, "j")
    return (df_report,)


@app.cell
def _(an, df_report, pl):
    an.plot_token_efficiency_by_model(
                          df_report.filter(pl.col("tags.algorithm")=="ae"),
                         )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
