import marimo

__generated_with = "0.13.4"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import os
    from glob import glob
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 読み込み対象のディレクトリ
    directory = "/home/tetsu.sato/gptzip/summary"
    # 指定条件のファイルを取得 (-ppl.parquet, -ae.parquet)
    ppl_files = glob(os.path.join(directory, "*-ppl.parquet"))
    ae_files = glob(os.path.join(directory, "*-ae.parquet"))

    # データを保存するリスト
    ppl_data = []
    ae_data = []

    # ppl ファイルの読み込み
    for file in ppl_files:
        filename = os.path.basename(file)
        try:
            df = pl.read_parquet(file)
            if "compression_rate" in df.columns:
                df = df.with_columns(pl.lit(filename).alias("filename"))
                ppl_data.append(df.select(["filename", "compression_rate"]))
                print(f"Loaded (ppl): {filename} (rows: {df.shape[0]})")
            else:
                print(f"Skipped (ppl) {filename}: 'compression_rate' column not found.")
        except Exception as e:
            print(f"Error loading (ppl) {filename}: {e}")

    # ae ファイルの読み込み
    for file in ae_files:
        filename = os.path.basename(file)
        try:
            df = pl.read_parquet(file)
            if "compression_rate" in df.columns:
                df = df.with_columns(pl.lit(filename).alias("filename"))
                ae_data.append(df.select(["filename", "compression_rate"]))
                print(f"Loaded (ae): {filename} (rows: {df.shape[0]})")
            else:
                print(f"Skipped (ae) {filename}: 'compression_rate' column not found.")
        except Exception as e:
            print(f"Error loading (ae) {filename}: {e}")

    # データの統合
    if ppl_data:
        ppl_df = pl.concat(ppl_data).to_pandas()
    else:
        ppl_df = None

    if ae_data:
        ae_df = pl.concat(ae_data).to_pandas()
    else:
        ae_df = None

    # 可視化
    plt.figure(figsize=(12, 6))

    # ppl のグラフ
    if ppl_df is not None:
        plt.subplot(1, 2, 1)
        sns.boxplot(x="filename", y="compression_rate", data=ppl_df)
        plt.xticks(rotation=90)
        plt.title("Compression Rate (ppl)")
        plt.xlabel("File Name")
        plt.ylabel("Compression Rate")
        plt.grid(True)

    # ae のグラフ
    if ae_df is not None:
        plt.subplot(1, 2, 2)
        sns.boxplot(x="filename", y="compression_rate", data=ae_df)
        plt.xticks(rotation=90)
        plt.title("Compression Rate (ae)")
        plt.xlabel("File Name")
        plt.ylabel("Compression Rate")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
    return glob, os, pl, plt, sns


@app.cell
def _(glob, os, pl, plt, sns):
    directory_1 = '/home/tetsu.sato/gptzip/summary'
    ppl_files_1 = sorted(glob(os.path.join(directory_1, '*-ppl.parquet')))
    ae_files_1 = sorted(glob(os.path.join(directory_1, '*-ae.parquet')))
    ppl_data_1 = []
    ae_data_1 = []
    for file_1 in ppl_files_1:
        filename_1 = os.path.basename(file_1)
        try:
            df_1 = pl.read_parquet(file_1)
            if 'compression_rate' in df_1.columns:
                df_1 = df_1.with_columns(pl.lit(filename_1).alias('filename'))
                ppl_data_1.append(df_1.select(['filename', 'compression_rate']))
                print(f'Loaded (ppl): {filename_1} (rows: {df_1.shape[0]})')
            else:
                print(f"Skipped (ppl) {filename_1}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ppl) {filename_1}: {e}')
    for file_1 in ae_files_1:
        filename_1 = os.path.basename(file_1)
        try:
            df_1 = pl.read_parquet(file_1)
            if 'compression_rate' in df_1.columns:
                df_1 = df_1.with_columns(pl.lit(filename_1).alias('filename'))
                ae_data_1.append(df_1.select(['filename', 'compression_rate']))
                print(f'Loaded (ae): {filename_1} (rows: {df_1.shape[0]})')
            else:
                print(f"Skipped (ae) {filename_1}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ae) {filename_1}: {e}')
    ppl_df_1 = pl.concat(ppl_data_1).to_pandas() if ppl_data_1 else None
    ae_df_1 = pl.concat(ae_data_1).to_pandas() if ae_data_1 else None
    plt.figure(figsize=(12, 6))
    if ppl_df_1 is not None:
        plt.subplot(1, 2, 1)
        sns.boxplot(x='filename', y='compression_rate', data=ppl_df_1, order=sorted(ppl_df_1['filename'].unique()))
        plt.xticks(rotation=90)
        plt.title('Compression Rate (ppl)')
        plt.xlabel('File Name (sorted)')
        plt.ylabel('Compression Rate')
        plt.grid(True)
    if ae_df_1 is not None:
        plt.subplot(1, 2, 2)
        sns.boxplot(x='filename', y='compression_rate', data=ae_df_1, order=sorted(ae_df_1['filename'].unique()))
        plt.xticks(rotation=90)
        plt.title('Compression Rate (ae)')
        plt.xlabel('File Name (sorted)')
        plt.ylabel('Compression Rate')
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(glob, os, pl, plt, sns):
    directory_2 = '/home/tetsu.sato/gptzip/summary'
    ppl_files_2 = sorted(glob(os.path.join(directory_2, '*-ppl.parquet')))
    ae_files_2 = sorted(glob(os.path.join(directory_2, '*-ae.parquet')))
    ppl_data_2 = []
    ae_data_2 = []
    for file_2 in ppl_files_2:
        filename_2 = os.path.basename(file_2)
        try:
            df_2 = pl.read_parquet(file_2)
            if 'compression_rate' in df_2.columns:
                df_2 = df_2.with_columns(pl.lit(filename_2).alias('filename'))
                ppl_data_2.append(df_2.select(['filename', 'compression_rate']))
                print(f'Loaded (ppl): {filename_2} (rows: {df_2.shape[0]})')
            else:
                print(f"Skipped (ppl) {filename_2}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ppl) {filename_2}: {e}')
    for file_2 in ae_files_2:
        filename_2 = os.path.basename(file_2)
        try:
            df_2 = pl.read_parquet(file_2)
            if 'compression_rate' in df_2.columns:
                df_2 = df_2.with_columns(pl.lit(filename_2).alias('filename'))
                ae_data_2.append(df_2.select(['filename', 'compression_rate']))
                print(f'Loaded (ae): {filename_2} (rows: {df_2.shape[0]})')
            else:
                print(f"Skipped (ae) {filename_2}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ae) {filename_2}: {e}')
    ppl_df_2 = pl.concat(ppl_data_2).to_pandas() if ppl_data_2 else None
    ae_df_2 = pl.concat(ae_data_2).to_pandas() if ae_data_2 else None
    plt.figure(figsize=(12, 6))
    if ppl_df_2 is not None:
        plt.subplot(1, 2, 1)
        sns.boxplot(x='filename', y='compression_rate', data=ppl_df_2, order=sorted(ppl_df_2['filename'].unique()))
        plt.xticks(rotation=45, ha='right')
        plt.title('Compression Rate (ppl)')
        plt.xlabel('File Name (sorted)')
        plt.ylabel('Compression Rate')
        plt.grid(True)
    if ae_df_2 is not None:
        plt.subplot(1, 2, 2)
        sns.boxplot(x='filename', y='compression_rate', data=ae_df_2, order=sorted(ae_df_2['filename'].unique()))
        plt.xticks(rotation=45, ha='right')
        plt.title('Compression Rate (ae)')
        plt.xlabel('File Name (sorted)')
        plt.ylabel('Compression Rate')
        plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    return (ppl_df_2,)


@app.cell
def _(ppl_df_2):
    ppl_df_2
    return


@app.cell
def _(glob, os, pl, plt, sns):
    directory_3 = '/home/tetsu.sato/gptzip/summary'
    ppl_files_3 = sorted(glob(os.path.join(directory_3, '*-ppl.parquet')))
    ae_files_3 = sorted(glob(os.path.join(directory_3, '*-ae.parquet')))
    ppl_data_3 = []
    ae_data_3 = []
    for file_3 in ppl_files_3:
        filename_3 = os.path.basename(file_3)
        try:
            df_3 = pl.read_parquet(file_3)
            if 'compression_rate' in df_3.columns:
                df_3 = df_3.with_columns(pl.lit(filename_3).alias('filename'))
                ppl_data_3.append(df_3.select(['filename', 'compression_rate']))
                print(f'Loaded (ppl): {filename_3} (rows: {df_3.shape[0]})')
            else:
                print(f"Skipped (ppl) {filename_3}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ppl) {filename_3}: {e}')
    for file_3 in ae_files_3:
        filename_3 = os.path.basename(file_3)
        try:
            df_3 = pl.read_parquet(file_3)
            if 'compression_rate' in df_3.columns:
                df_3 = df_3.with_columns(pl.lit(filename_3).alias('filename'))
                ae_data_3.append(df_3.select(['filename', 'compression_rate']))
                print(f'Loaded (ae): {filename_3} (rows: {df_3.shape[0]})')
            else:
                print(f"Skipped (ae) {filename_3}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ae) {filename_3}: {e}')
    ppl_df_3 = pl.concat(ppl_data_3).to_pandas() if ppl_data_3 else None
    ae_df_3 = pl.concat(ae_data_3).to_pandas() if ae_data_3 else None
    plt.figure(figsize=(12, 6))
    if ppl_df_3 is not None:
        plt.subplot(1, 2, 1)
        sns.boxplot(x='filename', y='compression_rate', data=ppl_df_3, order=sorted(ppl_df_3['filename'].unique()))
        plt.xticks(rotation=45, ha='right')
        plt.title('Compression Rate (ppl)')
        plt.xlabel('File Name (sorted)')
        plt.ylabel('Compression Rate')
        plt.grid(True)
    if ae_df_3 is not None:
        plt.subplot(1, 2, 2)
        sns.boxplot(x='filename', y='compression_rate', data=ae_df_3, order=sorted(ae_df_3['filename'].unique()))
        plt.xticks(rotation=45, ha='right')
        plt.title('Compression Rate (ae)')
        plt.xlabel('File Name (sorted)')
        plt.ylabel('Compression Rate')
        plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    return


@app.cell
def _(glob, os, pl, plt, sns):
    directory_4 = '/home/tetsu.sato/gptzip/summary'
    ppl_files_4 = sorted(glob(os.path.join(directory_4, '*-ppl.parquet')))
    ae_files_4 = sorted(glob(os.path.join(directory_4, '*-ae.parquet')))
    ppl_data_4 = []
    ae_data_4 = []
    for file_4 in ppl_files_4:
        filename_4 = os.path.basename(file_4)
        try:
            df_4 = pl.read_parquet(file_4)
            if 'compression_rate' in df_4.columns:
                df_4 = df_4.with_columns(pl.lit(filename_4).alias('filename'))
                ppl_data_4.append(df_4.select(['filename', 'compression_rate']))
                print(f'Loaded (ppl): {filename_4} (rows: {df_4.shape[0]})')
            else:
                print(f"Skipped (ppl) {filename_4}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ppl) {filename_4}: {e}')
    for file_4 in ae_files_4:
        filename_4 = os.path.basename(file_4)
        try:
            df_4 = pl.read_parquet(file_4)
            if 'compression_rate' in df_4.columns:
                df_4 = df_4.with_columns(pl.lit(filename_4).alias('filename'))
                ae_data_4.append(df_4.select(['filename', 'compression_rate']))
                print(f'Loaded (ae): {filename_4} (rows: {df_4.shape[0]})')
            else:
                print(f"Skipped (ae) {filename_4}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ae) {filename_4}: {e}')
    ppl_df_4 = pl.concat(ppl_data_4).to_pandas() if ppl_data_4 else None
    ae_df_4 = pl.concat(ae_data_4).to_pandas() if ae_data_4 else None
    plt.figure(figsize=(12, 6))
    if ppl_df_4 is not None:
        ppl_avg = ppl_df_4.groupby('filename')['compression_rate'].mean().reset_index()
        plt.subplot(1, 2, 1)
        sns.barplot(x='filename', y='compression_rate', data=ppl_avg, order=sorted(ppl_avg['filename']))
        plt.xticks(rotation=45, ha='right')
        plt.title('Average Compression Rate (ppl)')
        plt.xlabel('File Name (sorted)')
        plt.ylabel('Average Compression Rate')
        plt.grid(axis='y')
    if ae_df_4 is not None:
        ae_avg = ae_df_4.groupby('filename')['compression_rate'].mean().reset_index()
        plt.subplot(1, 2, 2)
        sns.barplot(x='filename', y='compression_rate', data=ae_avg, order=sorted(ae_avg['filename']))
        plt.xticks(rotation=45, ha='right')
        plt.title('Average Compression Rate (ae)')
        plt.xlabel('File Name (sorted)')
        plt.ylabel('Average Compression Rate')
        plt.grid(axis='y')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    return


@app.cell
def _(glob, os, pl, plt, sns):
    directory_5 = '/home/tetsu.sato/gptzip/summary'
    ppl_files_5 = sorted(glob(os.path.join(directory_5, '*-ppl.parquet')))
    ae_files_5 = sorted(glob(os.path.join(directory_5, '*-ae.parquet')))
    ppl_data_5 = []
    ae_data_5 = []
    for file_5 in ppl_files_5:
        filename_5 = os.path.basename(file_5)
        try:
            df_5 = pl.read_parquet(file_5)
            if 'compression_rate' in df_5.columns:
                df_5 = df_5.with_columns(pl.lit(filename_5).alias('filename'))
                ppl_data_5.append(df_5.select(['filename', 'compression_rate']))
                print(f'Loaded (ppl): {filename_5} (rows: {df_5.shape[0]})')
            else:
                print(f"Skipped (ppl) {filename_5}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ppl) {filename_5}: {e}')
    for file_5 in ae_files_5:
        filename_5 = os.path.basename(file_5)
        try:
            df_5 = pl.read_parquet(file_5)
            if 'compression_rate' in df_5.columns:
                df_5 = df_5.with_columns(pl.lit(filename_5).alias('filename'))
                ae_data_5.append(df_5.select(['filename', 'compression_rate']))
                print(f'Loaded (ae): {filename_5} (rows: {df_5.shape[0]})')
            else:
                print(f"Skipped (ae) {filename_5}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ae) {filename_5}: {e}')
    ppl_df_5 = pl.concat(ppl_data_5).to_pandas() if ppl_data_5 else None
    ae_df_5 = pl.concat(ae_data_5).to_pandas() if ae_data_5 else None
    plt.figure(figsize=(12, 6))
    if ppl_df_5 is not None:
        ppl_max = ppl_df_5.groupby('filename')['compression_rate'].max().reset_index()
        plt.subplot(1, 2, 1)
        sns.barplot(x='filename', y='compression_rate', data=ppl_max, order=sorted(ppl_max['filename']))
        plt.xticks(rotation=45, ha='right')
        plt.title('Max Compression Rate (ppl)')
        plt.xlabel('File Name (sorted)')
        plt.ylabel('Max Compression Rate')
        plt.grid(axis='y')
    if ae_df_5 is not None:
        ae_max = ae_df_5.groupby('filename')['compression_rate'].max().reset_index()
        plt.subplot(1, 2, 2)
        sns.barplot(x='filename', y='compression_rate', data=ae_max, order=sorted(ae_max['filename']))
        plt.xticks(rotation=45, ha='right')
        plt.title('Max Compression Rate (ae)')
        plt.xlabel('File Name (sorted)')
        plt.ylabel('Max Compression Rate')
        plt.grid(axis='y')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    return


@app.cell
def _(glob, os, pl, plt, sns):
    directory_6 = '/home/tetsu.sato/gptzip/summary'
    ppl_files_6 = sorted(glob(os.path.join(directory_6, '*-ppl.parquet')))
    ae_files_6 = sorted(glob(os.path.join(directory_6, '*-ae.parquet')))
    ppl_data_6 = []
    ae_data_6 = []
    for file_6 in ppl_files_6:
        filename_6 = os.path.basename(file_6)
        try:
            df_6 = pl.read_parquet(file_6)
            if 'compression_rate' in df_6.columns:
                df_6 = df_6.with_columns(pl.lit(filename_6).alias('filename'))
                ppl_data_6.append(df_6.select(['filename', 'compression_rate']))
                print(f'Loaded (ppl): {filename_6} (rows: {df_6.shape[0]})')
            else:
                print(f"Skipped (ppl) {filename_6}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ppl) {filename_6}: {e}')
    for file_6 in ae_files_6:
        filename_6 = os.path.basename(file_6)
        try:
            df_6 = pl.read_parquet(file_6)
            if 'compression_rate' in df_6.columns:
                df_6 = df_6.with_columns(pl.lit(filename_6).alias('filename'))
                ae_data_6.append(df_6.select(['filename', 'compression_rate']))
                print(f'Loaded (ae): {filename_6} (rows: {df_6.shape[0]})')
            else:
                print(f"Skipped (ae) {filename_6}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ae) {filename_6}: {e}')
    ppl_df_6 = pl.concat(ppl_data_6).to_pandas() if ppl_data_6 else None
    ae_df_6 = pl.concat(ae_data_6).to_pandas() if ae_data_6 else None
    plt.figure(figsize=(15, 6))
    if ppl_df_6 is not None:
        ppl_max_1 = ppl_df_6.groupby('filename')['compression_rate'].max().reset_index()
        ppl_min = ppl_df_6.groupby('filename')['compression_rate'].min().reset_index()
        ppl_avg_1 = ppl_df_6.groupby('filename')['compression_rate'].mean().reset_index()
        plt.subplot(1, 3, 1)
        sns.barplot(x='filename', y='compression_rate', data=ppl_max_1, order=sorted(ppl_max_1['filename']))
        plt.xticks(rotation=45, ha='right')
        plt.title('Max Compression Rate (ppl)')
        plt.xlabel('File Name (sorted)')
        plt.ylabel('Max Compression Rate')
        plt.grid(axis='y')
        plt.subplot(1, 3, 2)
        sns.barplot(x='filename', y='compression_rate', data=ppl_min, order=sorted(ppl_min['filename']))
        plt.xticks(rotation=45, ha='right')
        plt.title('Min Compression Rate (ppl)')
        plt.xlabel('File Name (sorted)')
        plt.ylabel('Min Compression Rate')
        plt.grid(axis='y')
        plt.subplot(1, 3, 3)
        sns.barplot(x='filename', y='compression_rate', data=ppl_avg_1, order=sorted(ppl_avg_1['filename']))
        plt.xticks(rotation=45, ha='right')
        plt.title('Avg Compression Rate (ppl)')
        plt.xlabel('File Name (sorted)')
        plt.ylabel('Avg Compression Rate')
        plt.grid(axis='y')
    if ae_df_6 is not None:
        ae_max_1 = ae_df_6.groupby('filename')['compression_rate'].max().reset_index()
        ae_min = ae_df_6.groupby('filename')['compression_rate'].min().reset_index()
        ae_avg_1 = ae_df_6.groupby('filename')['compression_rate'].mean().reset_index()
        plt.subplot(1, 3, 1)
        sns.barplot(x='filename', y='compression_rate', data=ae_max_1, order=sorted(ae_max_1['filename']))
        plt.xticks(rotation=45, ha='right')
        plt.title('Max Compression Rate (ae)')
        plt.xlabel('File Name (sorted)')
        plt.ylabel('Max Compression Rate')
        plt.grid(axis='y')
        plt.subplot(1, 3, 2)
        sns.barplot(x='filename', y='compression_rate', data=ae_min, order=sorted(ae_min['filename']))
        plt.xticks(rotation=45, ha='right')
        plt.title('Min Compression Rate (ae)')
        plt.xlabel('File Name (sorted)')
        plt.ylabel('Min Compression Rate')
        plt.grid(axis='y')
        plt.subplot(1, 3, 3)
        sns.barplot(x='filename', y='compression_rate', data=ae_avg_1, order=sorted(ae_avg_1['filename']))
        plt.xticks(rotation=45, ha='right')
        plt.title('Avg Compression Rate (ae)')
        plt.xlabel('File Name (sorted)')
        plt.ylabel('Avg Compression Rate')
        plt.grid(axis='y')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    return


@app.cell
def _(glob, os, pl, plt, sns):
    directory_7 = '/home/tetsu.sato/gptzip/summary'
    ppl_files_7 = sorted(glob(os.path.join(directory_7, '*-ppl.parquet')))
    ae_files_7 = sorted(glob(os.path.join(directory_7, '*-ae.parquet')))
    ppl_data_7 = []
    ae_data_7 = []
    for file_7 in ppl_files_7:
        filename_7 = os.path.basename(file_7)
        try:
            df_7 = pl.read_parquet(file_7)
            if 'compression_rate' in df_7.columns:
                df_7 = df_7.with_columns(pl.lit(filename_7).alias('filename'))
                ppl_data_7.append(df_7.select(['filename', 'compression_rate']))
                print(f'Loaded (ppl): {filename_7} (rows: {df_7.shape[0]})')
            else:
                print(f"Skipped (ppl) {filename_7}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ppl) {filename_7}: {e}')
    for file_7 in ae_files_7:
        filename_7 = os.path.basename(file_7)
        try:
            df_7 = pl.read_parquet(file_7)
            if 'compression_rate' in df_7.columns:
                df_7 = df_7.with_columns(pl.lit(filename_7).alias('filename'))
                ae_data_7.append(df_7.select(['filename', 'compression_rate']))
                print(f'Loaded (ae): {filename_7} (rows: {df_7.shape[0]})')
            else:
                print(f"Skipped (ae) {filename_7}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ae) {filename_7}: {e}')
    ppl_df_7 = pl.concat(ppl_data_7).to_pandas() if ppl_data_7 else None
    ae_df_7 = pl.concat(ae_data_7).to_pandas() if ae_data_7 else None
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    if ppl_df_7 is not None:
        ppl_max_2 = ppl_df_7.groupby('filename')['compression_rate'].max().reset_index()
        ppl_min_1 = ppl_df_7.groupby('filename')['compression_rate'].min().reset_index()
        ppl_avg_2 = ppl_df_7.groupby('filename')['compression_rate'].mean().reset_index()
        sns.barplot(x='filename', y='compression_rate', data=ppl_max_2, order=sorted(ppl_max_2['filename']), ax=axes[0, 0])
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
        axes[0, 0].set_title('Max Compression Rate (ppl)')
        axes[0, 0].set_xlabel('File Name (sorted)')
        axes[0, 0].set_ylabel('Max Compression Rate')
        axes[0, 0].grid(axis='y')
        sns.barplot(x='filename', y='compression_rate', data=ppl_min_1, order=sorted(ppl_min_1['filename']), ax=axes[0, 1])
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
        axes[0, 1].set_title('Min Compression Rate (ppl)')
        axes[0, 1].set_xlabel('File Name (sorted)')
        axes[0, 1].set_ylabel('Min Compression Rate')
        axes[0, 1].grid(axis='y')
        sns.barplot(x='filename', y='compression_rate', data=ppl_avg_2, order=sorted(ppl_avg_2['filename']), ax=axes[0, 2])
        axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), rotation=45, ha='right')
        axes[0, 2].set_title('Avg Compression Rate (ppl)')
        axes[0, 2].set_xlabel('File Name (sorted)')
        axes[0, 2].set_ylabel('Avg Compression Rate')
        axes[0, 2].grid(axis='y')
    if ae_df_7 is not None:
        ae_max_2 = ae_df_7.groupby('filename')['compression_rate'].max().reset_index()
        ae_min_1 = ae_df_7.groupby('filename')['compression_rate'].min().reset_index()
        ae_avg_2 = ae_df_7.groupby('filename')['compression_rate'].mean().reset_index()
        sns.barplot(x='filename', y='compression_rate', data=ae_max_2, order=sorted(ae_max_2['filename']), ax=axes[1, 0])
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
        axes[1, 0].set_title('Max Compression Rate (ae)')
        axes[1, 0].set_xlabel('File Name (sorted)')
        axes[1, 0].set_ylabel('Max Compression Rate')
        axes[1, 0].grid(axis='y')
        sns.barplot(x='filename', y='compression_rate', data=ae_min_1, order=sorted(ae_min_1['filename']), ax=axes[1, 1])
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
        axes[1, 1].set_title('Min Compression Rate (ae)')
        axes[1, 1].set_xlabel('File Name (sorted)')
        axes[1, 1].set_ylabel('Min Compression Rate')
        axes[1, 1].grid(axis='y')
        sns.barplot(x='filename', y='compression_rate', data=ae_avg_2, order=sorted(ae_avg_2['filename']), ax=axes[1, 2])
        axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=45, ha='right')
        axes[1, 2].set_title('Avg Compression Rate (ae)')
        axes[1, 2].set_xlabel('File Name (sorted)')
        axes[1, 2].set_ylabel('Avg Compression Rate')
        axes[1, 2].grid(axis='y')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(glob, os, pl, plt, sns):
    directory_8 = '/home/tetsu.sato/gptzip/summary'
    ppl_files_8 = sorted(glob(os.path.join(directory_8, '*-ppl.parquet')))
    ae_files_8 = sorted(glob(os.path.join(directory_8, '*-ae.parquet')))
    ppl_data_8 = []
    ae_data_8 = []

    def shorten_filename(filename):
        return filename.replace('jsai-dev-01_', '').replace('.parquet', '')
    for file_8 in ppl_files_8:
        filename_8 = shorten_filename(os.path.basename(file_8))
        try:
            df_8 = pl.read_parquet(file_8)
            if 'compression_rate' in df_8.columns:
                df_8 = df_8.with_columns(pl.lit(filename_8).alias('filename'))
                ppl_data_8.append(df_8.select(['filename', 'compression_rate']))
                print(f'Loaded (ppl): {filename_8} (rows: {df_8.shape[0]})')
            else:
                print(f"Skipped (ppl) {filename_8}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ppl) {filename_8}: {e}')
    for file_8 in ae_files_8:
        filename_8 = shorten_filename(os.path.basename(file_8))
        try:
            df_8 = pl.read_parquet(file_8)
            if 'compression_rate' in df_8.columns:
                df_8 = df_8.with_columns(pl.lit(filename_8).alias('filename'))
                ae_data_8.append(df_8.select(['filename', 'compression_rate']))
                print(f'Loaded (ae): {filename_8} (rows: {df_8.shape[0]})')
            else:
                print(f"Skipped (ae) {filename_8}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ae) {filename_8}: {e}')
    ppl_df_8 = pl.concat(ppl_data_8).to_pandas() if ppl_data_8 else None
    ae_df_8 = pl.concat(ae_data_8).to_pandas() if ae_data_8 else None
    fig_1, axes_1 = plt.subplots(2, 3, figsize=(18, 10))
    if ppl_df_8 is not None:
        ppl_max_3 = ppl_df_8.groupby('filename')['compression_rate'].max().reset_index()
        ppl_min_2 = ppl_df_8.groupby('filename')['compression_rate'].min().reset_index()
        ppl_avg_3 = ppl_df_8.groupby('filename')['compression_rate'].mean().reset_index()
        sns.barplot(x='filename', y='compression_rate', data=ppl_max_3, order=sorted(ppl_max_3['filename']), ax=axes_1[0, 0])
        axes_1[0, 0].set_xticklabels(axes_1[0, 0].get_xticklabels(), rotation=45, ha='right')
        axes_1[0, 0].set_title('Max Compression Rate (ppl)')
        axes_1[0, 0].set_xlabel('File Name (sorted)')
        axes_1[0, 0].set_ylabel('Max Compression Rate')
        axes_1[0, 0].grid(axis='y')
        sns.barplot(x='filename', y='compression_rate', data=ppl_min_2, order=sorted(ppl_min_2['filename']), ax=axes_1[0, 1])
        axes_1[0, 1].set_xticklabels(axes_1[0, 1].get_xticklabels(), rotation=45, ha='right')
        axes_1[0, 1].set_title('Min Compression Rate (ppl)')
        axes_1[0, 1].set_xlabel('File Name (sorted)')
        axes_1[0, 1].set_ylabel('Min Compression Rate')
        axes_1[0, 1].grid(axis='y')
        sns.barplot(x='filename', y='compression_rate', data=ppl_avg_3, order=sorted(ppl_avg_3['filename']), ax=axes_1[0, 2])
        axes_1[0, 2].set_xticklabels(axes_1[0, 2].get_xticklabels(), rotation=45, ha='right')
        axes_1[0, 2].set_title('Avg Compression Rate (ppl)')
        axes_1[0, 2].set_xlabel('File Name (sorted)')
        axes_1[0, 2].set_ylabel('Avg Compression Rate')
        axes_1[0, 2].grid(axis='y')
    if ae_df_8 is not None:
        ae_max_3 = ae_df_8.groupby('filename')['compression_rate'].max().reset_index()
        ae_min_2 = ae_df_8.groupby('filename')['compression_rate'].min().reset_index()
        ae_avg_3 = ae_df_8.groupby('filename')['compression_rate'].mean().reset_index()
        sns.barplot(x='filename', y='compression_rate', data=ae_max_3, order=sorted(ae_max_3['filename']), ax=axes_1[1, 0])
        axes_1[1, 0].set_xticklabels(axes_1[1, 0].get_xticklabels(), rotation=45, ha='right')
        axes_1[1, 0].set_title('Max Compression Rate (ae)')
        axes_1[1, 0].set_xlabel('File Name (sorted)')
        axes_1[1, 0].set_ylabel('Max Compression Rate')
        axes_1[1, 0].grid(axis='y')
        sns.barplot(x='filename', y='compression_rate', data=ae_min_2, order=sorted(ae_min_2['filename']), ax=axes_1[1, 1])
        axes_1[1, 1].set_xticklabels(axes_1[1, 1].get_xticklabels(), rotation=45, ha='right')
        axes_1[1, 1].set_title('Min Compression Rate (ae)')
        axes_1[1, 1].set_xlabel('File Name (sorted)')
        axes_1[1, 1].set_ylabel('Min Compression Rate')
        axes_1[1, 1].grid(axis='y')
        sns.barplot(x='filename', y='compression_rate', data=ae_avg_3, order=sorted(ae_avg_3['filename']), ax=axes_1[1, 2])
        axes_1[1, 2].set_xticklabels(axes_1[1, 2].get_xticklabels(), rotation=45, ha='right')
        axes_1[1, 2].set_title('Avg Compression Rate (ae)')
        axes_1[1, 2].set_xlabel('File Name (sorted)')
        axes_1[1, 2].set_ylabel('Avg Compression Rate')
        axes_1[1, 2].grid(axis='y')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(glob, os, pl, plt, sns):
    directory_9 = '/home/tetsu.sato/gptzip/summary'
    ppl_files_9 = sorted(glob(os.path.join(directory_9, '*-ppl.parquet')))
    ae_files_9 = sorted(glob(os.path.join(directory_9, '*-ae.parquet')))
    ppl_data_9 = []
    ae_data_9 = []

    def shorten_filename_1(filename):
        return filename.replace('jsai-dev-01_', '').replace('.parquet', '')
    for file_9 in ppl_files_9:
        filename_9 = shorten_filename_1(os.path.basename(file_9))
        try:
            df_9 = pl.read_parquet(file_9)
            if 'compression_rate' in df_9.columns:
                df_9 = df_9.with_columns(pl.lit(filename_9).alias('filename'))
                ppl_data_9.append(df_9.select(['filename', 'compression_rate']))
                print(f'Loaded (ppl): {filename_9} (rows: {df_9.shape[0]})')
            else:
                print(f"Skipped (ppl) {filename_9}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ppl) {filename_9}: {e}')
    for file_9 in ae_files_9:
        filename_9 = shorten_filename_1(os.path.basename(file_9))
        try:
            df_9 = pl.read_parquet(file_9)
            if 'compression_rate' in df_9.columns:
                df_9 = df_9.with_columns(pl.lit(filename_9).alias('filename'))
                ae_data_9.append(df_9.select(['filename', 'compression_rate']))
                print(f'Loaded (ae): {filename_9} (rows: {df_9.shape[0]})')
            else:
                print(f"Skipped (ae) {filename_9}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ae) {filename_9}: {e}')
    ppl_df_9 = pl.concat(ppl_data_9).to_pandas() if ppl_data_9 else None
    ae_df_9 = pl.concat(ae_data_9).to_pandas() if ae_data_9 else None
    fig_2, axes_2 = plt.subplots(2, 1, figsize=(18, 10))
    print(f'axes={axes_2}')
    if ppl_df_9 is not None:
        ppl_avg_4 = ppl_df_9.groupby('filename')['compression_rate'].mean().reset_index()
        sns.barplot(x='filename', y='compression_rate', data=ppl_avg_4, order=sorted(ppl_avg_4['filename']), ax=axes_2[0])
        axes_2[0].set_xticklabels(axes_2[0].get_xticklabels(), rotation=45, ha='right')
        axes_2[0].set_title('Avg Compression Rate (ppl)')
        axes_2[0].set_xlabel('File Name (sorted)')
        axes_2[0].set_ylabel('Avg Compression Rate')
        axes_2[0].grid(axis='y')
    if ae_df_9 is not None:
        ae_avg_4 = ae_df_9.groupby('filename')['compression_rate'].mean().reset_index()
        sns.barplot(x='filename', y='compression_rate', data=ae_avg_4, order=sorted(ae_avg_4['filename']), ax=axes_2[1])
        axes_2[1].set_xticklabels(axes_2[1].get_xticklabels(), rotation=45, ha='right')
        axes_2[1].set_title('Avg Compression Rate (ae)')
        axes_2[1].set_xlabel('File Name (sorted)')
        axes_2[1].set_ylabel('Avg Compression Rate')
        axes_2[1].grid(axis='y')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(glob, os, pl, plt, sns):
    directory_10 = '/home/tetsu.sato/gptzip/summary'
    ppl_files_10 = sorted(glob(os.path.join(directory_10, '*-ppl.parquet')))
    ae_files_10 = sorted(glob(os.path.join(directory_10, '*-ae.parquet')))
    ppl_data_10 = []
    ae_data_10 = []

    def shorten_filename_2(filename):
        return filename.replace('jsai-dev-01_', '').replace('.parquet', '')
    for file_10 in ppl_files_10:
        filename_10 = shorten_filename_2(os.path.basename(file_10))
        try:
            df_10 = pl.read_parquet(file_10)
            if 'compression_rate' in df_10.columns:
                df_10 = df_10.with_columns(pl.lit(filename_10).alias('filename'))
                ppl_data_10.append(df_10.select(['filename', 'compression_rate']))
                print(f'Loaded (ppl): {filename_10} (rows: {df_10.shape[0]})')
            else:
                print(f"Skipped (ppl) {filename_10}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ppl) {filename_10}: {e}')
    for file_10 in ae_files_10:
        filename_10 = shorten_filename_2(os.path.basename(file_10))
        try:
            df_10 = pl.read_parquet(file_10)
            if 'compression_rate' in df_10.columns:
                df_10 = df_10.with_columns(pl.lit(filename_10).alias('filename'))
                ae_data_10.append(df_10.select(['filename', 'compression_rate']))
                print(f'Loaded (ae): {filename_10} (rows: {df_10.shape[0]})')
            else:
                print(f"Skipped (ae) {filename_10}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ae) {filename_10}: {e}')
    ppl_df_10 = pl.concat(ppl_data_10).to_pandas() if ppl_data_10 else None
    ae_df_10 = pl.concat(ae_data_10).to_pandas() if ae_data_10 else None
    fig_3, axes_3 = plt.subplots(2, 3, figsize=(18, 10))
    if ppl_df_10 is not None:
        ppl_max_4 = ppl_df_10.groupby('filename')['compression_rate'].max().reset_index()
        ppl_min_3 = ppl_df_10.groupby('filename')['compression_rate'].min().reset_index()
        ppl_avg_5 = ppl_df_10.groupby('filename')['compression_rate'].mean().reset_index()
        sns.barplot(x='filename', y='compression_rate', data=ppl_max_4, order=sorted(ppl_max_4['filename']), ax=axes_3[0, 0])
        axes_3[0, 0].set_xticklabels(axes_3[0, 0].get_xticklabels(), rotation=45, ha='right')
        axes_3[0, 0].set_title('Max Compression Rate (ppl)')
        axes_3[0, 0].set_xlabel('File Name (sorted)')
        axes_3[0, 0].set_ylabel('Max Compression Rate')
        axes_3[0, 0].grid(axis='y')
        sns.barplot(x='filename', y='compression_rate', data=ppl_min_3, order=sorted(ppl_min_3['filename']), ax=axes_3[0, 1])
        axes_3[0, 1].set_xticklabels(axes_3[0, 1].get_xticklabels(), rotation=45, ha='right')
        axes_3[0, 1].set_title('Min Compression Rate (ppl)')
        axes_3[0, 1].set_xlabel('File Name (sorted)')
        axes_3[0, 1].set_ylabel('Min Compression Rate')
        axes_3[0, 1].grid(axis='y')
        sns.barplot(x='filename', y='compression_rate', data=ppl_avg_5, order=sorted(ppl_avg_5['filename']), ax=axes_3[0, 2])
        axes_3[0, 2].set_xticklabels(axes_3[0, 2].get_xticklabels(), rotation=45, ha='right')
        axes_3[0, 2].set_title('Avg Compression Rate (ppl)')
        axes_3[0, 2].set_xlabel('File Name (sorted)')
        axes_3[0, 2].set_ylabel('Avg Compression Rate')
        axes_3[0, 2].grid(axis='y')
    if ae_df_10 is not None:
        ae_max_4 = ae_df_10.groupby('filename')['compression_rate'].max().reset_index()
        ae_min_3 = ae_df_10.groupby('filename')['compression_rate'].min().reset_index()
        ae_avg_5 = ae_df_10.groupby('filename')['compression_rate'].mean().reset_index()
        sns.barplot(x='filename', y='compression_rate', data=ae_max_4, order=sorted(ae_max_4['filename']), ax=axes_3[1, 0])
        axes_3[1, 0].set_xticklabels(axes_3[1, 0].get_xticklabels(), rotation=45, ha='right')
        axes_3[1, 0].set_title('Max Compression Rate (ae)')
        axes_3[1, 0].set_xlabel('File Name (sorted)')
        axes_3[1, 0].set_ylabel('Max Compression Rate')
        axes_3[1, 0].grid(axis='y')
        sns.barplot(x='filename', y='compression_rate', data=ae_min_3, order=sorted(ae_min_3['filename']), ax=axes_3[1, 1])
        axes_3[1, 1].set_xticklabels(axes_3[1, 1].get_xticklabels(), rotation=45, ha='right')
        axes_3[1, 1].set_title('Min Compression Rate (ae)')
        axes_3[1, 1].set_xlabel('File Name (sorted)')
        axes_3[1, 1].set_ylabel('Min Compression Rate')
        axes_3[1, 1].grid(axis='y')
        sns.barplot(x='filename', y='compression_rate', data=ae_avg_5, order=sorted(ae_avg_5['filename']), ax=axes_3[1, 2])
        axes_3[1, 2].set_xticklabels(axes_3[1, 2].get_xticklabels(), rotation=45, ha='right')
        axes_3[1, 2].set_title('Avg Compression Rate (ae)')
        axes_3[1, 2].set_xlabel('File Name (sorted)')
        axes_3[1, 2].set_ylabel('Avg Compression Rate')
        axes_3[1, 2].grid(axis='y')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(glob, os, pl, plt, sns):
    directory_11 = '/home/tetsu.sato/gptzip/summary'
    ppl_files_11 = sorted(glob(os.path.join(directory_11, '*-ppl.parquet')))
    ae_files_11 = sorted(glob(os.path.join(directory_11, '*-ae.parquet')))

    def shorten_filename_3(filename):
        return filename.replace('jsai-dev-01_', '').replace('.parquet', '').replace('-ppl', '').replace('-ae', '')
    data = {}
    for file_11 in ppl_files_11:
        base_filename = shorten_filename_3(os.path.basename(file_11))
        try:
            df_11 = pl.read_parquet(file_11)
            if 'compression_rate' in df_11.columns:
                data.setdefault(base_filename, {})['ppl_max'] = df_11['compression_rate'].max()
                data.setdefault(base_filename, {})['ppl_min'] = df_11['compression_rate'].min()
            else:
                print(f"Skipped (ppl) {base_filename}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ppl) {base_filename}: {e}')
    for file_11 in ae_files_11:
        base_filename = shorten_filename_3(os.path.basename(file_11))
        try:
            df_11 = pl.read_parquet(file_11)
            if 'compression_rate' in df_11.columns:
                data.setdefault(base_filename, {})['ae_max'] = df_11['compression_rate'].max()
                data.setdefault(base_filename, {})['ae_min'] = df_11['compression_rate'].min()
            else:
                print(f"Skipped (ae) {base_filename}: 'compression_rate' column not found.")
        except Exception as e:
            print(f'Error loading (ae) {base_filename}: {e}')
    import pandas as pd
    df_11 = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df_11 = df_11.rename(columns={'index': 'Model'})
    df_11 = df_11.fillna(0)
    df_11 = df_11.sort_values(by='Model')
    fig_4, axes_4 = plt.subplots(2, 1, figsize=(14, 10))
    df_melt_max = df_11.melt(id_vars=['Model'], value_vars=['ppl_max', 'ae_max'], var_name='Type', value_name='Compression Rate')
    sns.barplot(x='Model', y='Compression Rate', hue='Type', data=df_melt_max, ax=axes_4[0])
    axes_4[0].set_xticklabels(axes_4[0].get_xticklabels(), rotation=45, ha='right')
    axes_4[0].set_title('Max Compression Rate (ppl vs ae)')
    axes_4[0].set_xlabel('Model')
    axes_4[0].set_ylabel('Max Compression Rate')
    axes_4[0].grid(axis='y')
    df_melt_min = df_11.melt(id_vars=['Model'], value_vars=['ppl_min', 'ae_min'], var_name='Type', value_name='Compression Rate')
    sns.barplot(x='Model', y='Compression Rate', hue='Type', data=df_melt_min, ax=axes_4[1])
    axes_4[1].set_xticklabels(axes_4[1].get_xticklabels(), rotation=45, ha='right')
    axes_4[1].set_title('Min Compression Rate (ppl vs ae)')
    axes_4[1].set_xlabel('Model')
    axes_4[1].set_ylabel('Min Compression Rate')
    axes_4[1].grid(axis='y')
    plt.tight_layout()
    plt.show()
    return (data,)


@app.cell
def _(data):
    data
    return


@app.cell
def _(pl):
    df_rinna_ae = pl.read_parquet("summary/jsai-dev-01_rinna-gemma-2-baku-2b-it-ae.parquet")
    return (df_rinna_ae,)


@app.cell
def _(df_rinna_ae):
    df_rinna_ae
    return


@app.cell
def _(pl):
    df_rinna_ppl = pl.read_parquet("summary/jsai-dev-01_rinna-gemma-2-baku-2b-it-ppl.parquet")
    return (df_rinna_ppl,)


@app.cell
def _(df_rinna_ppl):
    df_rinna_ppl
    return


@app.cell
def _(df_rinna_ae, df_rinna_ppl):
    df_12 = df_rinna_ae['compression_rate'].to_frame().rename({'compression_rate': 'ae'}).with_columns(df_rinna_ppl['compression_rate'].to_frame().rename({'compression_rate': 'ppl'}))
    df_12
    return (df_12,)


@app.cell
def _(df_12, pl):
    df_12.select(pl.corr('ae', 'ppl'))
    return


@app.cell
def _(df_12, plt):
    plt.scatter(df_12['ae'], df_12['ppl'])
    return


@app.cell
def _():
    llmjp_score = {
    "jsai-dev-01_rinna-gemma-2-baku-2b-it" : 0.4477 ,                          
    "jsai-dev-01_llm-jp-llm-jp-3-1.8b-instruct": 0.3923 ,                      
    "jsai-dev-01_llm-jp-llm-jp-3-13b-instruct": 0.5462 ,                       
    "jsai-dev-01_llm-jp-llm-jp-3-3.7b-instruct": 0.4597 ,                      
    "jsai-dev-01_meta-llama-Llama-3.2-1B-Instruct": 0.3059 ,                   
    "jsai-dev-01_meta-llama-Llama-3.2-3B-Instruct": 0.4111 ,                   
    "jsai-dev-01_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B": 0.2584 ,          
    "jsai-dev-01_deepseek-ai-DeepSeek-R1-Distill-Qwen-7B": 0.3919 ,            
    "jsai-dev-01_deepseek-ai-DeepSeek-R1-Distill-Llama-8B": 0.414 ,            
    "jsai-dev-01_Qwen-Qwen2.5-7B-Instruct": 0.5304 ,                           
    "jsai-dev-01_Qwen-Qwen2.5-3B-Instruct": 0.4801 ,                           
    "jsai-dev-01_Qwen-Qwen2.5-1.5B-Instruct": 0.4431 ,                         
    "jsai-dev-01_Qwen-Qwen2.5-0.5B-Instruct": 0.3112 ,                         
    "jsai-dev-01_google-gemma-2-9b-it": 0.5206 ,                               
    "jsai-dev-01_google-gemma-2-2b-it": 0.4051 ,                               
    "jsai-dev-01_HuggingFaceTB-SmolLM2-135M-Instruct": 0.1564 ,                
    "jsai-dev-01_HuggingFaceTB-SmolLM2-1.7B-Instruct": 0.3132 ,                
    "jsai-dev-01_ibm-granite-granite-3.1-8b-instruct": 0.4897 ,                
    "jsai-dev-01_ibm-granite-granite-3.1-3b-a800m-instruct": 0.3829 ,          
    "jsai-dev-01_ibm-granite-granite-3.1-2b-instruct": 0.4346 ,                
    "jsai-dev-01_ibm-granite-granite-3.1-1b-a400m-instruct": 0.2913 ,    
    }
    return (llmjp_score,)


@app.cell
def _(llmjp_score):
    llmjp_score
    return


@app.cell
def _(llmjp_score, os, pl):
    directory_12 = '/home/tetsu.sato/gptzip/summary'
    data_1 = []
    print(f"llmjp_score={llmjp_score}")
    for lm in llmjp_score.keys():
        filename_11 = f'{directory_12}/{lm}-ae.parquet'
        print(f"filename={filename_11}")
        if os.path.isfile(filename_11):
            df_ae = pl.read_parquet(filename_11)
            filename_11 = f'{directory_12}/{lm}-ppl.parquet'
            df_ppl = pl.read_parquet(filename_11)
            for input_file_name, ae_rate, ppl_rate in zip(df_ae['input_file_name'], df_ae['compression_rate'], df_ppl['compression_rate']):
                row_name = f'{lm}-{input_file_name}'
                data_1.append({'index': lm, 'input_file_name': input_file_name, 'ae': ae_rate, 'ppl': ppl_rate, 'llmjp-score': llmjp_score[lm]})
    print(f"data_1={data_1}")
    df_final = pl.DataFrame(data_1).sort('index')
    print(df_final)
    return (df_final,)


@app.cell
def _(df_final, pl):
    df_grouped = df_final.group_by("index").agg(
        pl.col("ae").mean().alias("ae_mean"),
        pl.col("ppl").mean().alias("ppl_mean"),
        pl.col("llmjp-score").first().alias("llmjp-score")
    )
    df_grouped
    return (df_grouped,)


@app.cell
def _(df_grouped, pl):
    df_grouped.select(pl.corr("ae_mean", "llmjp-score"))
    return


@app.cell
def _(df_grouped, pl):
    df_grouped.select(pl.corr("ppl_mean", "llmjp-score"))
    return


@app.cell
def _(df_grouped, plt):
    plt.scatter(df_grouped["ae_mean"], df_grouped["llmjp-score"])
    return


@app.cell
def _(df_grouped, plt):
    plt.scatter(df_grouped["ppl_mean"], df_grouped["llmjp-score"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## これは前の実験結果と思われるので，キャッシュから消さなければ
    ```
    jsai-dev-01_Qwen-Qwen2.5-3B-Instruct.parquet
    jsai-dev-01_Qwen-Qwen2.5-1.5B-Instruct.parquet
    jsai-dev-01_Qwen-Qwen2.5-0.5B-Instruct.parquet
    jsai-dev-01_google-gemma-2-9b-it.parquet
    jsai-dev-01_google-gemma-2-2b-jpn-it.parquet
    ```
    """
    )
    return


@app.cell
def _(mo):
    from cache.cache import Cache
    from hydra import compose, initialize_config_dir

    directory_15 = '/home/tetsu.sato/gptzip/'

    with initialize_config_dir(config_dir=f"{mo.notebook_dir()}/config"):
        cfg = compose(config_name="exp_dev.yaml", return_hydra_config=True)

    cache_filename = f"{cfg.exp.title}.db"
    cache = Cache(cfg=cfg,                      
                  cache_filename=cache_filename,
                  )                    
    cache
    return Cache, cache, compose


@app.cell
def _(cache):
    qwen3blist = cache.listKeys("jsai-dev-01-Qwen/Qwen2.5-3B.*")
    qwen7blist = cache.listKeys("jsai-dev-01-Qwen/Qwen2.5-7B.*")
    return qwen3blist, qwen7blist


@app.cell
def _(cache):
    cache.get('jsai-dev-01-Qwen/Qwen2.5-3B-Instruct-職務経歴書サンプル_金融_資産運用.txt-ae')
    return


@app.cell
def _(cache):
    cache.get('jsai-dev-01-Qwen/Qwen2.5-7B-Instruct-職務経歴書サンプル_金融_資産運用.txt-ae')
    return


@app.cell
def _(qwen3blist, qwen7blist):
    print(len(qwen3blist))
    print(len(qwen7blist))
    return


@app.cell
def _(df_grouped, pl):
    with pl.Config() as cfg_1:
        cfg_1.set_tbl_cols(30)
        cfg_1.set_tbl_rows(30)
        cfg_1.set_fmt_str_lengths(80)
        print(df_grouped['index', 'ae_mean', 'llmjp-score'].sort('index'))
    return


@app.cell
def _(df_final, pl):
    with pl.Config() as cfg_2:
        cfg_2.set_tbl_cols(30)
        cfg_2.set_tbl_rows(30)
        cfg_2.set_fmt_str_lengths(80)
        df_count = df_final.group_by('index').agg(pl.len().alias('count'), pl.col('llmjp-score').first().alias('llmjp-score'))
        print(df_count.sort('index'))
    return


@app.cell
def _(df_final, pl):
    with pl.Config() as cfg_3:
        cfg_3.set_tbl_cols(30)
        cfg_3.set_tbl_rows(30)
        cfg_3.set_fmt_str_lengths(80)
        df_select = df_final.filter(pl.col('index') == 'jsai-dev-01_Qwen-Qwen2.5-0.5B-Instruct')
        print(df_select.select(pl.count()))
        df_select = df_select.filter(pl.col('input_file_name') == '職務経歴書サンプル_営業_IT法人営業.txt')
        print(df_select)
    return


@app.cell
def _(cache):
    cache.get('jsai-dev-01-Qwen/Qwen2.5-0.5B-Instruct-職務経歴書サンプル_営業_IT法人営業.txt-ae')
    return


@app.cell
def _(os, pl):
    directory_13 = '/home/tetsu.sato/gptzip/summary'
    data_2 = []
    lm_1 = 'jsai-dev-01_Qwen-Qwen2.5-0.5B-Instruct'
    filename_12 = f'{directory_13}/{lm_1}-ae.parquet'
    if os.path.isfile(filename_12):
        df_ae_1 = pl.read_parquet(filename_12)
        print(df_ae_1)
        filename_12 = f'{directory_13}/{lm_1}-ppl.parquet'
        df_ppl_1 = pl.read_parquet(filename_12)
        sample_list = list(zip(df_ae_1['input_file_name'], df_ae_1['compression_rate'], df_ppl_1['compression_rate']))
        print(len(df_ae_1['input_file_name']))
        print(len(sample_list))
    else:
        print('not exist')
    return


@app.cell
def _():
    list
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# バグ取ったと思うので改めて""")
    return


@app.cell
def _(llmjp_score, os, pl):
    directory_14 = '/home/tetsu.sato/gptzip/summary'
    data_3 = []
    for lm_2 in llmjp_score.keys():
        filename_13 = f'{directory_14}/{lm_2}-ae.parquet'
        if os.path.isfile(filename_13):
            df_ae_2 = pl.read_parquet(filename_13)
            filename_13 = f'{directory_14}/{lm_2}-ppl.parquet'
            df_ppl_2 = pl.read_parquet(filename_13)
            for input_file_name_1, ae_rate_1, ppl_rate_1 in zip(df_ae_2['input_file_name'], df_ae_2['compression_rate'], df_ppl_2['compression_rate']):
                row_name_1 = f'{lm_2}-{input_file_name_1}'
                original_filename = f'/home/tetsu.sato/resume-data/doda-samples/{input_file_name_1}'
                with open(original_filename, 'r', encoding='utf-8') as fp:
                    content = fp.read()
                data_3.append({'index': lm_2, 'input_file_name': input_file_name_1, 'ae': ae_rate_1, 'ppl': ppl_rate_1, 'llmjp-score': llmjp_score[lm_2], 'content_length': len(content), 'content': content})
    df_final_1 = pl.DataFrame(data_3).sort('index')
    with pl.Config() as cfg_4:
        cfg_4.set_tbl_cols(30)
        cfg_4.set_tbl_rows(10)
        cfg_4.set_fmt_str_lengths(40)
        print(df_final_1)
    return (df_final_1,)


@app.cell
def _(df_final_1, pl):
    df_grouped_1 = df_final_1.group_by('index').agg(pl.col('ae').mean().alias('ae_mean'), pl.col('ppl').mean().alias('ppl_mean'), pl.col('llmjp-score').first().alias('llmjp-score'))
    with pl.Config() as cfg_5:
        cfg_5.set_tbl_cols(30)
        cfg_5.set_tbl_rows(30)
        cfg_5.set_fmt_str_lengths(80)
        print(df_grouped_1.sort('index'))
    return (df_grouped_1,)


@app.cell
def _(df_grouped_1, pl):
    df_grouped_1.select(pl.corr('ae_mean', 'llmjp-score'))
    return


@app.cell
def _(df_grouped_1, pl):
    df_grouped_1.select(pl.corr(pl.col('ppl_mean'), 'llmjp-score'))
    return


@app.cell
def _(df_grouped_1, pl):
    df_grouped_1.select(pl.corr(pl.col('ppl_mean'), pl.col('llmjp-score') * 0.1))
    return


@app.cell
def _(df_grouped_1, pl, plt):
    import numpy as np
    from sklearn.linear_model import LinearRegression

    print(df_grouped_1)
    print(f"相関係数={df_grouped_1.select(pl.corr('ae_mean','llmjp-score'))}")

    fig12, ax12 = plt.subplots()

    x = np.array(df_grouped_1['llmjp-score'])
    print(x)
    y = np.array(df_grouped_1['ae_mean'].to_list())
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    print(model.coef_)
    plt.scatter(x, y, label=df_grouped_1['index'].to_list())
    X = np.arange(min(x), max(x), 0.01)
    print(X)
    Y = model.coef_[0, 0] * X + model.intercept_[0]
    plt.plot(X, Y, color='red')
    labels = [x.replace('jsai-dev-01_', '').replace('deepseek-ai-', '').replace('meta-llama-', '').replace('Qwen-', '').replace('llm-jp-l', 'l').replace('google-', '') for x in df_grouped_1['index']]
    for i, label in enumerate(labels):
        plt.text(x[i], y[i], label)

    ax12.set_xlabel("llmjp-score")
    ax12.set_ylabel("圧縮率")
    plt.grid(which="major")
    plt.show()

    df_grouped_2 = df_grouped_1.filter(~pl.col("index").str.contains("llm-jp"))
    df_grouped_2 = df_grouped_2.filter(~pl.col("index").str.contains("DeepSeek"))
    print(df_grouped_2)
    print(f"相関係数={df_grouped_2.select(pl.corr('ae_mean','llmjp-score'))}")
    return LinearRegression, np


@app.cell
def _(LinearRegression, df_grouped_1, np, plt):
    x_1 = np.array(df_grouped_1['llmjp-score'])
    print(x_1)
    y_1 = np.array(df_grouped_1['ppl_mean'].to_list())
    model_1 = LinearRegression()
    model_1.fit(x_1.reshape(-1, 1), y_1.reshape(-1, 1))
    print(model_1.coef_)
    plt.scatter(x_1, y_1)
    X_1 = np.arange(min(x_1), max(x_1), 0.01)
    print(X_1)
    Y_1 = model_1.coef_[0, 0] * X_1 + model_1.intercept_[0]
    plt.plot(X_1, Y_1, color='red')
    labels_1 = [x.replace('jsai-dev-01_', '').replace('deepseek-ai-', '').replace('meta-llama-', '').replace('Qwen-', '').replace('llm-jp-l', 'l').replace('google-', '') for x in df_grouped_1['index']]
    for i_1, label_1 in enumerate(labels_1):
        plt.text(x_1[i_1], y_1[i_1], label_1)

    plt.show()
    return


@app.cell
def _(df_grouped_1, pl):
    with pl.Config() as cfg_6:
        cfg_6.set_tbl_cols(30)
        cfg_6.set_tbl_rows(30)
        cfg_6.set_fmt_str_lengths(80)
        print(df_grouped_1.sort('index'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## キャッシュコンテンツ移動""")
    return


@app.cell
def _(Cache, compose, initialize):
    with initialize(config_path='config'):
        cfg_7 = compose(config_name='exp_dev_l4.yaml', return_hydra_config=True)
    cache_filename_1 = f'{cfg_7.exp.title}.db'
    cachel4 = Cache(cfg=cfg_7, cache_filename=cache_filename_1)
    cachel4
    return (cachel4,)


@app.cell
def _(cachel4):
    len(cachel4.listKeys("jsai-dev-01-HuggingFaceTB/SmolLM2-1.7B.*"))
    return


@app.cell
def _(cache):
    len(cache.listKeys("jsai-dev-01-HuggingFaceTB/SmolLM2-1.7B.*"))
    return


@app.cell
def _(cachel4):
    src_keys = cachel4.listKeys("jsai-dev-01-HuggingFaceTB/SmolLM2-1.7B.*")
    return (src_keys,)


@app.cell
def _(cache, cachel4, src_keys):
    for key in src_keys:
        val = cachel4.get(key)
        cache.set(key, val)
    return


@app.cell
def _(cache):
    len(cache.listKeys("jsai-dev-01-HuggingFaceTB/SmolLM2-1.7B.*"))
    return


@app.cell
def _(cache):
    cache.listKeys("jsai-dev-01-HuggingFaceTB/SmolLM2-1.7B.*")
    return


@app.cell
def _(cache):
    cache.listKeys("jsai-dev-01-HuggingFaceTB/SmolLM2-1.7B-Instruct-職務経歴書サンプル_企画・管理_リサーチ.*")
    return


@app.cell
def _(cachel4):
    cachel4.listKeys("jsai-dev-01-HuggingFaceTB/SmolLM2-1.7B-Instruct-職務経歴書サンプル_企画・管理_リサーチ.*")
    return


@app.cell
def _(cachel4):
    len(cachel4.listKeys("jsai-dev.*/SmolLM2-1.7B.*ppl"))
    return


@app.cell
def _(cache):
    len(cache.listKeys("jsai-dev.*/SmolLM2-1.7B.*ppl"))
    return


@app.cell
def _(cache):
    len(cache.listKeys("jsai-dev.*/SmolLM2-1.7B.*ae"))
    return


@app.cell
def _(cache):
    len(cache.listKeys("jsai-dev-01-\\..*"))
    return


@app.cell
def _(df_final_1, pl):
    df_sample = df_final_1.filter(pl.col('index') == 'jsai-dev-01_HuggingFaceTB-SmolLM2-135M-Instruct')
    return (df_sample,)


@app.cell
def _(df_sample, pl):
    with pl.Config() as cfg_8:
        cfg_8.set_tbl_cols(40)
        cfg_8.set_tbl_rows(10)
        cfg_8.set_fmt_str_lengths(40)
        print(df_sample)
        print(df_sample.describe())
    return


@app.cell
def _(df_sample):
    df_sample_1 = df_sample.with_columns(df_sample['ppl'] / df_sample['ppl'].max())
    df_sample_1
    return (df_sample_1,)


@app.cell
def _(df_sample_1, plt, sns):
    ppl_max_5 = df_sample_1['ppl'].max()
    df_sample_1
    df_sample_melted = df_sample_1.melt(id_vars=['input_file_name'], value_vars=['ae', 'ppl'], variable_name='Type', value_name='Value')
    plt.figure(figsize=(8, 5))
    sns.barplot(x='input_file_name', y='Value', hue='Type', data=df_sample_melted, palette={'ae': 'blue', 'ppl': 'red'})
    plt.xlabel('Input File Name')
    plt.ylabel('Value')
    plt.title('AE vs PPL per Input File')
    plt.xticks(rotation=45)
    plt.legend(title='Type')
    plt.show()
    return


@app.cell
def _(LinearRegression, df_sample_1, np, pl, plt):
    import japanize_matplotlib
    x_2 = np.array(df_sample_1['ae'])
    y_2 = np.array(df_sample_1['ppl'].to_list())
    model_2 = LinearRegression()
    model_2.fit(x_2.reshape(-1, 1), y_2.reshape(-1, 1))
    print(model_2.coef_)
    plt.scatter(x_2, y_2)
    X_2 = np.arange(min(x_2), max(x_2), 0.01)
    print(X_2)
    Y_2 = model_2.coef_[0, 0] * X_2 + model_2.intercept_[0]
    plt.plot(X_2, Y_2, color='red')
    plt.scatter(df_sample_1['ae'], df_sample_1['ppl'])
    plt.title('圧縮率とPerplexity')
    plt.xlabel('圧縮率')
    plt.ylabel('Perplexity')
    plt.grid(axis='y')
    df_sample_1.select(pl.corr('ae', 'ppl'))
    return


@app.cell
def _(df_sample_1, sns):
    sns.scatterplot(x=df_sample_1['llmjp-score'], y=df_sample_1['ae'], color='blue', label='AE', s=100)
    sns.scatterplot(x=df_sample_1['llmjp-score'], y=df_sample_1['ppl'], color='red', label='PPL', s=100)
    return


@app.cell
def _(df_sample_1, pl):
    with pl.Config() as cfg_9:
        cfg_9.set_tbl_cols(30)
        cfg_9.set_tbl_rows(5)
        cfg_9.set_fmt_str_lengths(20)
        df_sample_with_diff = df_sample_1.with_columns((df_sample_1['ae'] - df_sample_1['ppl'] / df_sample_1['ppl'].max()).alias('diff')).sort('diff')
        print(df_sample_with_diff)
        print(df_sample_with_diff.sort('diff', descending=True))
    return (df_sample_with_diff,)


@app.cell
def _(df_sample_with_diff, pl):
    df_sample_with_diff.select(pl.corr("content_length", "diff"))
    return


@app.cell
def _(df_sample_with_diff, pl):
    df_sample_with_diff.select(pl.corr("ppl", "content_length"))
    return


@app.cell
def _(df_sample_with_diff, pl):
    df_sample_with_diff.select(pl.corr("ae", "content_length"))
    return


@app.cell
def _(df_sample_with_diff, plt):
    plt.scatter(df_sample_with_diff["content_length"], df_sample_with_diff["ppl"])
    return


@app.cell
def _(df_sample_with_diff, plt):
    plt.scatter(df_sample_with_diff["content_length"], df_sample_with_diff["ae"])
    return


@app.cell
def _(LinearRegression, df_sample_1, df_sample_with_diff, np, pl, plt):
    fig_5, axes_5 = plt.subplots(1, 2)
    x_3 = np.array(df_sample_1['ae'])
    y_3 = np.array(df_sample_1['ppl'].to_list())
    model_3 = LinearRegression()
    model_3.fit(x_3.reshape(-1, 1), y_3.reshape(-1, 1))
    print(model_3.coef_)
    axes_5[0].scatter(x_3, y_3)
    X_3 = np.arange(min(x_3), max(x_3), 0.01)
    print(X_3)
    Y_3 = model_3.coef_[0, 0] * X_3 + model_3.intercept_[0]
    axes_5[0].plot(X_3, Y_3, color='red')
    axes_5[0].scatter(df_sample_1['ae'], df_sample_1['ppl'])
    axes_5[0].set_title('圧縮率とPerplexity')
    axes_5[0].set_xlabel('圧縮率')
    axes_5[0].set_ylabel('Perplexity')
    axes_5[1].scatter(df_sample_with_diff['content_length'], df_sample_with_diff['diff'].abs())
    x_3 = np.array(df_sample_with_diff['content_length'])
    y_3 = np.array(df_sample_with_diff['diff'].abs())
    model_3 = LinearRegression()
    model_3.fit(x_3.reshape(-1, 1), y_3.reshape(-1, 1))
    print(model_3.coef_)
    axes_5[1].scatter(x_3, y_3)
    X_3 = np.arange(min(x_3), max(x_3), 0.01)
    print(X_3)
    Y_3 = model_3.coef_[0, 0] * X_3 + model_3.intercept_[0]
    axes_5[1].plot(X_3, Y_3, color='red')
    axes_5[1].scatter(x_3, y_3)
    axes_5[1].set_title('入力データ長と(圧縮率-Perplexity)')
    axes_5[1].set_xlabel('入力データ長')
    axes_5[1].set_ylabel('圧縮率-Perplexity')
    axes_5[1].grid(axis='y')
    df_sample_with_diff.select(pl.corr('content_length', 'diff'))
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(LinearRegression, df_sample_1, np, plt):
    x_4 = np.array(df_sample_1['ae'])
    y_4 = np.array(df_sample_1['ppl'].to_list())
    model_4 = LinearRegression()
    model_4.fit(x_4.reshape(-1, 1), y_4.reshape(-1, 1))
    print(model_4.coef_)
    plt.subplot(1, 2, 1)
    plt.scatter(x_4, y_4)
    X_4 = np.arange(min(x_4), max(x_4), 0.01)
    print(X_4)
    Y_4 = model_4.coef_[0, 0] * X_4 + model_4.intercept_[0]
    plt.plot(X_4, Y_4, color='red')
    plt.scatter(df_sample_1['ae'], df_sample_1['ppl'])
    plt.title('圧縮率とPerplexity')
    plt.xlabel('圧縮率')
    plt.ylabel('Perplexity')
    return


@app.cell
def _(df_sample_with_diff, pl, sns):
    n = 3
    df_regularized = df_sample_with_diff.with_columns(pl.col('ppl') / pl.col('ppl').max())
    with pl.Config() as cfg_10:
        cfg_10.set_tbl_cols(30)
        cfg_10.set_tbl_rows(10)
        cfg_10.set_fmt_str_lengths(20)
        print(df_regularized)
    df_melt = df_regularized.melt(id_vars=['File'], value_vars=['ppl', 'ae'], variable_name='Type', value_name='Score')
    sns.barplot(x=df_sample_with_diff['input_file_name'][0:n], y=df_sample_with_diff['ae'][0:n], color='blue', label='AE')
    sns.barplot(x=df_sample_with_diff['input_file_name'][0:n], y=df_sample_with_diff['ppl'][0:n], color='blue', label='AE')
    return df_regularized, n


@app.cell
def _(df_regularized, n, pl, sns):
    df_melt_1 = df_regularized[0:n].melt(id_vars=['input_file_name'], value_vars=['ppl', 'ae'], variable_name='Type', value_name='Score')
    with pl.Config() as cfg_11:
        cfg_11.set_tbl_cols(30)
        cfg_11.set_tbl_rows(10)
        cfg_11.set_fmt_str_lengths(20)
        print(df_melt_1)
    sns.barplot(x='input_file_name', y='Score', hue='Type', data=df_melt_1)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
