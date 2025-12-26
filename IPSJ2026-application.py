import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    dir = os.getcwd()
    print(dir)
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
    return leaderboard_df, plt


@app.cell
def _(an, df, leaderboard_df):
    def cell4():
        import polars as pl
        #runs_df = an.search_model_from_mlflow(an.models[0:-2], run_name="fit2025-paper-05%")
        runs_df = an.search_model_from_mlflow(an.models[0:2], run_name="fit2025-paper-05%")
        runs_df = runs_df[0:10] # デバグ用に量を減らす
        print("runs_df=", runs_df)
        print([tag for tag in df.columns if tag.startswith("tags.")])
        with pl.Config(tbl_cols=-1):
            print(runs_df.select(["tags.model_name", "tags.algorithm", "metrics.token_efficiency", "metrics.llm_score"]))
        total_exp_df = an.mlflow_results_df_to_analytics_df(runs_df)
        print(total_exp_df)
        # 積集合を取りたい
        target_leaderboard_df = leaderboard_df.join(total_exp_df, left_on="Model", right_on="tags.model_name", how="inner")
        print(target_leaderboard_df)
        df_report = an.leaderboard_df_to_report_df(leaderboard_df, total_exp_df)
        with pl.Config(tbl_cols=-1, fmt_str_lengths=50):
            print(df_report)
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
    an.plot_token_efficiency_by_model(df_report.filter(pl.col("tags.algorithm")=="ae"),
                          df_report_e.filter(pl.col("tags.algorithm")=="ae"),
                         )
    return


@app.cell
def _(an, df_report, df_report_e, pl):
    an.plot_token_efficiency_by_avg_score(df_report.filter(pl.col("tags.algorithm")=="ae"),
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
def _(an, df_report):
    # プレゼンのため，falcon/llm-jpを除いたグラフを描いて，外れ値（乖離）が無ければ良い感じなことを示したい→トークン効率ごとのグラフにしたい→トークン効率とLMCRの２軸とリーダーボードスコアのグラフって実現できないか
    def plot_graph_without_outlier(df_report, lang):
        # plot_graphをシミュレート
        for algorithm,df in df_report.group_by("tags.algorithm"):
            if algorithm[0] == "ppl":
                continue             
            print(f"df[Model]={df.select("Model")}")
            #df = df.filter(~pl.col("Model").str.contains("Falcon") )
            #df = df.filter(~pl.col("Model").str.contains("llm-jp") )
            print(f"filtered={df.select("Model")}")
            an.plot_leaderboard_lmcr(df, algorithm, lang)

    plot_graph_without_outlier(df_report, "j")
    return


@app.cell
def _(df_report):
    df_report
    return


@app.cell
def _(mo):
    mo.md(r"""# リーダーボードスコア，圧縮率，トークン効率→""")
    return


@app.cell
def _(df_report):
    # この材料で予測モデルを作りたい
    df_report["Model", "AVG", "tags.algorithm", "metrics.original_size", "metrics.tokenized_size", "metrics.compressed_size"]
    return


@app.cell
def _(df_report, pl):
    df_report.select(pl.col("metrics.compressed_size"), pl.col("metrics.original_size"),
                     (pl.col("metrics.compressed_size")/pl.col("metrics.original_size")).alias("compression_ratio")
                    )
    return


@app.cell
def _(df_report, pl):
    # まずは変数選択と変数変換？
    regression_df = df_report.filter(pl.col("tags.algorithm") == "ae")\
             .select(pl.col("Model"), pl.col("AVG"), pl.col("metrics.tokenized_size").alias("tokenized_efficiency")/pl.col("metrics.original_size"),
       pl.col("metrics.compressed_size").alias("compressed_ratio") /pl.col("metrics.original_size")             )
    regression_df
    return (regression_df,)


@app.cell
def _(pl, regression_df):
    X = regression_df.select(pl.col("tokenized_efficiency"), pl.col("compressed_ratio")).to_numpy()
    X
    return (X,)


@app.cell
def _(pl, regression_df):
    Y = regression_df.select(pl.col("AVG")).to_numpy()
    Y
    return (Y,)


@app.cell
def _():
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    return LinearRegression, r2_score


@app.cell
def _(LinearRegression, X, Y, plt, r2_score):


    def score_linear_regression(x, y):
        model = LinearRegression()
        model.fit(x, y)

        print(f"R²: {r2_score(y, model.predict(x))}")

        plt.scatter(x, y, color="blue", label="Data Points (X[:, 0])")  # 元データ点
        plt.plot(x, model.predict(x), color="red", label="Regression Line")  # 回帰直線
        plt.xlabel("X[:, 0]")
        plt.ylabel("Y")
        plt.legend()
        plt.title("2D Input to 1D Output")
        plt.show()

    score_linear_regression(X[:, 0].reshape(-1, 1), Y)
    score_linear_regression(X[:, 1].reshape(-1, 1), Y)
    return


app._unparsable_cell(
    r"""
    def plot_reg_line(ax, x, y):
        #print(f\"x={x}\")
        #print(f\"x.shape={x.shape}\")
        print(x.__class__)
        if x.ndim == 2 and x.shape[1]==1: # (:, 1)なら(:)にする．reshape(-1)と同じ
            x = x.flatten()
        if y.ndim == 2 and y.shape[1]==1: # (:, 1)なら(:)にする．reshape(-1)と同じ
            y = y.flatten()
        import numpy as np
        corr_list = []
        # (x, y)という点群に対し線形回帰直線を描く
        x = np.array(x)
        y = np.array(y)
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
        #print(f\"x={x}\")
        #print(f\"y={y}\")
        corr = np.corrcoef(x, y)[0, 1]
        #corr_list.append(corr)
        ax.plot(x_fit, y_fit, color=\"red\", label=f\"MSE={mse:.2e}, r2={r2:.2f}, corr={corr:.2f}\")
        \"\"\"
        ax.text(0.95,
             0.99,
             f\"MSE={mse:.2f}, r2={r2:.2f}, corr={corr:.2f}\",
             #transform=plt.gca().transAxes,
             ha=\"right\",
             va=\"top\"
            )
        \"\"\"

    def AVG_predict_plot(x, # 教師データであるリーダーボードAVGスコア
                         y, # なんらかの予測結果
                         t, # 点の色付けに使う第３の値
                         model_name,
                         label, # 点の説明文字列
                         x_label,
                         ax, # 描画オブジェクト
                        ):

        scatter = ax.scatter(x, 
                    y,
                    #color=\"green\",
                    c=t,
                    cmap=\"viridis\",
                    label=label)
        colorbar = fig.colorbar(scatter, ax=ax)

        plot_reg_line(ax, x, y)

        #plt.scatter(X[:, 0], model_poly.predict(X_poly))
        #plt.scatter(X[:, 1], model_poly.predict(X_poly))
        #plt.scatter(X[:, 1], model_poly.predict(X_poly))
        tup = zip(x, y, model_name.reshape(-1))
        for xi, yi, name in tup:
            ax.text(xi, yi, name)

        ax.set_xlabel(\"AVG Score\")
        ax.set_ylabel(x_label)
        ax.legend()

    def plot_linear(fig,
                    pos,
                    y_title,
                    x, # 入力教師データ
                    y, # 出力教師データ
                    t
                   ):

        linear_model = LinearRegression(positive=False)

        linear_model.fit(x, y)
        ax = fig.add_subplot(pos)
        linear_model_predict = linear_model.predict(x)
        #linear_model_predict = linear_model.predict(x_centered)
        AVG_predict_plot(y,
                         linear_model_predict,
                         t,
                         model_name,
                         y_title,
                         \"AVG Score\",
                         ax
                        )



    #from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR

    model_name = regression_df.select(pl.col(\"Model\")).to_numpy()
    #print(model_name)

    # 2次の多項式特徴量
    poly_extractor = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly_extractor.fit_transform(X)

    #svr = SVR(kernel=\"rbf\", C=100, gamma=0.1, epsilon=0.1)
    #svr = SVR(kernel=\"rbf\", C=1000, gamma=\"auto\", epsilon=0.1)
    #svr = SVR(kernel=\"linear\", C=100, gamma=\"auto\")
    svr = SVR(kernel=\"poly\", C=1, gamma=\"auto\", degree=5, epsilon=0.1, coef0=10)
    fig = plt.figure(figsize=(24, 12))

    plot_linear(fig, 221, \"2-vars linear regression\", X, Y, X[:, 0])

    plot_linear(fig, 222, \"1d linear regression\", X[:, 1].reshape(-1, 1), Y, X[:, 0])

    plot_linear(fig, 223, \"polynomial linear regression\", X_poly, Y, X[:, 0])

    new_lmcr = 

    plt.show()
    """,
    name="_"
)


@app.cell
def _():
    """
    #from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR

    model_name = regression_df.select(pl.col("Model")).to_numpy()
    #print(model_name)

    # 2次の多項式特徴量
    poly_extractor = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly_extractor.fit_transform(X)

    #svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    #svr = SVR(kernel="rbf", C=1000, gamma="auto", epsilon=0.1)
    #svr = SVR(kernel="linear", C=100, gamma="auto")
    svr = SVR(kernel="poly", C=1, gamma="auto", degree=5, epsilon=0.1, coef0=10)


    from scipy.stats import mode
    center = mode(X[:, 0].reshape(-1, 1), axis=0, keepdims=False)
    print(f"center={center}")
    print(f"original={X[:, 0].reshape(-1)}")
    print(f"centerize={X[:, 0].reshape(-1)-center.mode}")
    x_centered = X.copy()
    x_centered[:, 0] = X[:, 0].reshape(-1)-center.mode

    linear_model = LinearRegression(positive=False)

    #linear_model.fit(X, Y)
    linear_model.fit(x_centered, Y)

    fig = plt.figure(figsize=(24, 12))
    ax1 = fig.add_subplot(221)
    #ax1.scatter(Y, X[:, 0], color="blue", label="tokenized_effeciency")
    #ax1.scatter(Y, X[:, 1], color="red", label="compression_ratio")
    #linear_model_predict = linear_model.predict(X)
    linear_model_predict = linear_model.predict(x_centered)
    AVG_predict_plot(Y,
                    linear_model_predict,
                    X[:, 0],
                    model_name,
                     "2-vars linear regression",
                     "AVG Score",
                     ax1
                    )



    linear_model = LinearRegression(positive=False)
    linear_model.fit(X[:, 1].reshape(-1,1), Y)

    ax2 = fig.add_subplot(222)
    #ax1.scatter(Y, X[:, 0], color="blue", label="tokenized_effeciency")
    #ax1.scatter(Y, X[:, 1], color="red", label="compression_ratio")


    linear_model_predict = linear_model.predict(X[:, 1].reshape(-1, 1))

    AVG_predict_plot(Y,
                    linear_model_predict,
                    X[:, 0],
                    model_name,
                     "linear regression",
                     "AVG Score",
                     ax2
                    )


    linear_model.fit(X_poly, Y)

    ax3 = fig.add_subplot(223)
    model_poly_predict = linear_model.predict(X_poly)

    AVG_predict_plot(Y,
                    model_poly_predict,
                    X[:, 0],
                    model_name,
                     "linear regression",
                     "AVG Score",
                     ax3
                    )






    import lightgbm as lgb
    lightgbm = lgb.LGBMRegressor()
    #print(f"X={X}")
    #print(f"Y={Y}")
    #lightgbm_predict = lightgbm.fit(X, Y).predict(X)

    ax4 = fig.add_subplot(224)
    svr_predict = svr.fit(X, Y).predict(X)
    #print(f"svr coef={svr.coef_}")
    print(f"svr coef={svr.coef0}")
    predict = svr_predict

    poly_extractor = PolynomialFeatures(degree=3, include_bias=True)
    X_poly = poly_extractor.fit_transform(X)
    linear_model.fit(X_poly, Y)
    predict = linear_model.predict(X_poly)

    AVG_predict_plot(Y,
                    predict,
                    X[:, 0],
                    model_name,
                     "linear regression",
                     "AVG Score",
                     ax4
                    )


    plt.show()
    """
    return


@app.cell
def _(X, Y, model_name, pl):
    pl.DataFrame(model_name).with_columns([pl.Series("X", X), pl.Series("Y", Y)])
    return


@app.cell
def _(df_report, mo, pl):
    pl.Config(tbl_rows=-1, fmt_str_lengths=50)
    mo.plain(
        df_report.filter(pl.col("tags.algorithm") == "ae").select(pl.col("Model"), pl.col("metrics.tokenized_size"), pl.col("metrics.compressed_size"), pl.col("AVG"))
    )

    return


@app.cell
def _(X):
    X
    return


@app.cell
def _(SVR, X, Y):
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    scaler.fit(X)
    X_std=scaler.transform(X)
    print(f"X_std={X_std}")
    def objective(trial):
        kernel = trial.suggest_categorical('kernel', ['linear','rbf','poly','sigmoid'])
        gamma = trial.suggest_float('gamma',1e-5,1e5, log=True)
        C = trial.suggest_float('C',1e-5,1e5, log=True)
        epsilon = trial.suggest_float('epsilon',1e-5,1e5, log=True)
        regr = SVR(kernel = kernel, gamma = gamma, C = C ,epsilon = epsilon)
        from sklearn.model_selection import cross_val_score

        score = cross_val_score(regr, X_std, Y, cv=3, scoring="r2")
        r2_mean = score.mean()
        return r2_mean

    import optuna
    study = optuna.create_study(direction='maximize')
    #study.optimize(objective, n_trials=100)
    return


@app.cell
def _():
    # 圧縮率にしてるのがいまいちかなあ．オリジナルサイズ，圧縮サイズ，トークナイズサイズの３パラにしてみるとか
    # いや，オリジナルサイズは全部同じだから，変わらんか

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 違うな，(compression ratio, token efficiency)-> scoreじゃなくて
    ```
    (index(compression ratio, token efficiency))-> score
    ```
    となる関数indexを求めたい
    """
    )
    return


@app.cell
def _(df_report, linear_model, poly_extractor):
    import polars as pl
    cs_col = pl.col("metrics.compressed_size")
    ts_col = pl.col("metrics.tokenized_size")
    size_col = pl.col("metrics.original_size")
    for algorithm,df in df_report.group_by("tags.algorithm"):    
        if algorithm[0] == "ppl":  
            continue               
        for input_length, length_df in df.group_by("metrics.original_size"):
            print(input_length)
            print(length_df)
            x = length_df.select((cs_col/size_col).alias("compression_ratio"),
                      (ts_col/size_col).alias("tokenized_ratio"),
                      ) # shape(1, 2)
            print(f"x={x}")
            x_poly = poly_extractor.fit_transform(x)
            print(f"x_poly={x_poly}")
            y = linear_model.predict(x_poly) # (1, 1)
            length_df = length_df.with_columns([
                    pl.Series("interpolated_llm_score", y.reshape(-1))
                ]
                )
        print(f"length_df={length_df}")
    return df, pl


@app.cell
def _(Y):
    Y[0:3]
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
    mo.md(r"""# VR-LMCR実験（ipsj2026-exp04→ipsj2026-exp05（exp04はmlflowサーバ立ち上げディレクトリミスがあったので））""")
    return


@app.cell
def _(an, leaderboard_df, pl):
    # VR-LMCR実験データの取得
    runs_df_vr = an.search_model_from_mlflow(["%"], run_name="ipsj2026-exp06%")
    # タグ列だけ抽出
    print([tag for tag in runs_df_vr.columns if tag.startswith("tags.")]) 
    print(f"VR-LMCR runs: {runs_df_vr[["tags.exp_title"]]}")

    # aeアルゴリズムのみフィルタ
    runs_df_vr_filtered = runs_df_vr.filter(pl.col("tags.algorithm")=="ae")

    # DataFrameに変換
    total_exp_df_vr = an.mlflow_results_df_to_analytics_df(runs_df_vr_filtered)

    # Leaderboardと結合
    df_report_vr = an.leaderboard_df_to_report_df(leaderboard_df, total_exp_df_vr)\
                       .filter(pl.col("tags.algorithm")=="ae")\
                       .unique()

    with pl.Config(tbl_cols=-1, fmt_str_lengths=50):
        print(f"df_report_vr={df_report_vr}")

    return (df_report_vr,)


@app.cell
def _(df_report_vr, pl, plt):
    # VR-LMCRスコアとLeaderboard Scoreのグラフ
    #import japanize_matplotlib

    # データの抽出
    vr_lmcr_scores = df_report_vr.select(pl.col("metrics.vr_lmcr_naive")).to_numpy().flatten()
    leaderboard_scores = df_report_vr.select(pl.col("AVG")).to_numpy().flatten()
    model_names = df_report_vr.select(pl.col("Model")).to_numpy().flatten()

    # グラフ描画
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(leaderboard_scores, vr_lmcr_scores, c=leaderboard_scores, 
                        cmap="viridis", s=100, alpha=0.6, edgecolors="black")

    # モデル名をラベル表示
    for i, name in enumerate(model_names):
        ax.annotate(name, (leaderboard_scores[i], vr_lmcr_scores[i]), 
                   fontsize=8, alpha=0.7)

    # 軸ラベルとタイトル
    ax.set_xlabel("Leaderboard Score (AVG)", fontsize=12)
    ax.set_ylabel("VR-LMCR Score (naive)", fontsize=12)
    ax.set_title("VR-LMCR vs Leaderboard Score", fontsize=14)
    ax.grid(True, alpha=0.3)

    # カラーバー
    plt.colorbar(scatter, ax=ax, label="Leaderboard Score")

    plt.tight_layout()
    plt.show()

    return


@app.cell
def _():
    # デバグコード
    def debug_search_runs():
        import mlflow
        import polars as pl
        from pprint import pprint
        df = mlflow.search_runs(
            search_all_experiments=True,
            experiment_ids = ["0"],
            max_results=10000,
             run_view_type=mlflow.entities.ViewType.ALL ,
            #experiment_names=["Default"],
            #filter_string = "tags.mlflow.runName like 'ipsj2026-exp05(ae)'"
        )
        print(df.__class__)
        df = pl.from_pandas(df)
        print(df.__class__)
        pprint(df.schema)
        print("describe=", df.describe())
        print(df.select(#pl.col("params.exp"),
                        pl.col("tags.mlflow.runName"),
                        pl.col("start_time"),
                        pl.col("tags.model_name"))\
                .sort([pl.col("start_time"),pl.col("tags.mlflow.runName")], descending=True)
             )
    debug_search_runs()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
