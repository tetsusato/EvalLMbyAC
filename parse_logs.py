import re
import csv
import sys
import io

# ログから必要な情報を抽出する正規表現
log_pattern = re.compile(r"""
    data:\s+(?P<model>[\w/.-]+)-(?P<file_name>[\w_.-]+),\s+  # モデル名とファイル名
    size:\s+(?P<original_size>\d+)-(?P<compressed_size>\d+),\s+  # 圧縮前後のバイト数
    time:\s+(?P<compression_time>[\d.]+)-(?P<decompression_time>[\d.]+):\s+  # 圧縮/展開処理時間
    ratio\s+=\s+(?P<compression_ratio>[\d.]+)  # 圧縮率
""", re.VERBOSE)

# データを保存するリスト
parsed_data = []

# 標準入力からログを読み込み
for line in sys.stdin:
    match = log_pattern.search(line)
    if match:
        parsed_data.append({
            "Model": match.group("model"),
            "File Name": match.group("file_name"),
            "Original Size (bytes)": int(match.group("original_size")),
            "Compressed Size (bytes)": int(match.group("compressed_size")),
            "Compression Ratio": float(match.group("compression_ratio")),
            "Compression Time (s)": float(match.group("compression_time")),
            "Decompression Time (s)": float(match.group("decompression_time")),
        })

# CSVに書き出し
output = io.StringIO()
fieldnames = [
    "Model",
    "File Name",
    "Original Size (bytes)",
    "Compressed Size (bytes)",
    "Compression Ratio",
    "Compression Time (s)",
    "Decompression Time (s)"
]
writer = csv.DictWriter(output, fieldnames=fieldnames)

# ヘッダー行を書き出し
writer.writeheader()

# データ行を書き出し
writer.writerows(parsed_data)

# 標準出力に出力
print(output.getvalue())
