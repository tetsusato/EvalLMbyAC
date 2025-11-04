import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    dir = os.getcwd()
    print(dir)
    return (os,)


@app.cell
def _(os):
    os.chdir("FIT2025-presen")
    os.getcwd()
    return


@app.cell
def _():
    from src.analytics.analytics import Analyzer
    print(Analyzer)
    return (Analyzer,)


@app.cell
def _(Analyzer):
    an = Analyzer("/home/tetsu.sato//FIT2025-presen/")
    print(an)
    return (an,)


@app.cell
def _(an):

    result = an.get_leaderboard_result("")
    print(f"leaderboard result={result}")
    print(f"leaderboard result class={result.__class__}")
    leaderboard_df = an._leaderboard_data_to_polar_dataframe(result)
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
    return (leaderboard_df,)


@app.cell
def _(an):
    import polars as pl
    runs_df = an.search_model_from_mlflow(an.models[0:-2], run_name="fit2025-paper-05%")
    print(runs_df)
    with pl.Config(tbl_cols=-1):
        print(runs_df.select(["tags.model_name", "tags.algorithm", "metrics.token_efficiency", "metrics.llm_score"]))
    return pl, runs_df


@app.cell
def _(an, runs_df):
    total_exp_df = an.mlflow_results_df_to_analytics_df(runs_df)
    print(total_exp_df)
    return (total_exp_df,)


@app.cell
def _(leaderboard_df, total_exp_df):
    # 積集合を取りたい
    target_leaderboard_df = leaderboard_df.join(total_exp_df, left_on="Model", right_on="tags.model_name", how="inner")
    print(target_leaderboard_df)
    return


@app.cell
def _(an, leaderboard_df, pl, total_exp_df):
    df_report = an.leaderboard_df_to_report_df(leaderboard_df, total_exp_df)
    with pl.Config(tbl_cols=-1, fmt_str_lengths=50):
        print(df_report)
    return (df_report,)


@app.cell
def _(an, df_report):
    an.plot_graph(df_report, "j")
    return


@app.cell
def _(an, leaderboard_df):
    runs_df_e = an.search_model_from_mlflow(an.models[0:-2], run_name="fit2025-paper-06%")
    total_exp_df_e = an.mlflow_results_df_to_analytics_df(runs_df_e)
    target_leaderboard_df_e = leaderboard_df.join(total_exp_df_e, left_on="Model", right_on="tags.model_name", how="inner")
    df_report_e = an.leaderboard_df_to_report_df(leaderboard_df, total_exp_df_e)
    an.plot_graph(df_report_e, "e")
    return (df_report_e,)


@app.cell
def _(an, df_report, df_report_e, pl):
    an.plot_token_efficiency(df_report.filter(pl.col("tags.algorithm")=="ae"),
                          df_report_e.filter(pl.col("tags.algorithm")=="ae"),
                         )
    return


@app.cell
def _(an, leaderboard_df, pl, runs_df):
    runs_df_sample = an.search_model_from_mlflow(["%"], run_name="fit2025-paper-07%")
    print(f"sample={runs_df_sample}")
    print(f"original={runs_df}")
    # "tags.model_name" をキーに結合
    runs_df_sample = runs_df_sample.join(
        runs_df, 
        left_on="tags.model_name", 
        right_on="tags.model_name", 
        how="left"
    ).filter(pl.col("tags.algorithm")=="ae")
    total_exp_df_sample = an.mlflow_results_df_to_analytics_df(runs_df_sample)
    target_leaderboard_df_sample = leaderboard_df.join(total_exp_df_sample, left_on="Model", right_on="tags.model_name", how="inner")
    df_report_sample = an.leaderboard_df_to_report_df(leaderboard_df, total_exp_df_sample)\
                       .filter(pl.col("tags.algorithm")=="ae")\
                       .unique()
    with pl.Config(tbl_cols=-1, fmt_str_lengths=50):
        print(f"df_report_sample={df_report_sample}")
    an.plot_graph(df_report_sample, "j")

    return (df_report_sample,)


@app.cell
def _(an, df_report_sample):
    an.plot_size_score_by_model(df_report_sample, "ae", "Japanese")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
