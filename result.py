import csv
from dataclasses import dataclass
import pandera.polars as pa
from pydantic import BaseModel
import polars as pl
import io

@dataclass
class Test():
    model_name: str

@dataclass
class Result():
    exp_title: str
    encoding_algorithm: str
    hosting: str
    model_name: str
    input_file_name: str
    input_text_length: int
    compressed_size: int
    decoded_size: int
    llm_score: float
    encode_time: int
    decode_time: int
    experiment_environment: str
    vocab_size: int | None = None  # トークナイザーの語彙数
    vr_lmcr_naive: float | None = None  # VR-LMCR（ナイーブ実装）
    vr_lmcr_realistic: float | None = None  # VR-LMCR（現実的実装）※後で実装

    def to_csv(self):
        output = io.StringIO()
        writer = csv.writer(output)
        row = [
            self.exp_title,
            self.encoding_algorithm,
            self.hosting,
            self.model_name,
            self.input_file_name,
            self.input_text_length,
            self.compressed_size,
            self.decoded_size,
            self.llm_score,
            self.encode_time,
            self.decode_time,
            ]
        writer.writerow(row)
        return output.getvalue().strip()  # 改行を削除

    #def to_df(self):
        
    def summary_string(self):
        crlf = "\n"
        summary = ""
        summary += f"=========================== {self.model_name} ======================" + crlf
        #summary += f"basic info={basic_info}" + crlf
        #summary += f"success?={is_success}" + crlf
        summary += f"Experiment: {self.exp_title}" + crlf
        summary += f"Encoding algorithm: {self.encoding_algorithm}" + crlf
        summary += f"Hosting: {self.hosting}" + crlf
        summary += f"Compression {self.input_text_length} chars to {self.compressed_size} bytes" + crlf
        summary += f"Compression ratio {self.llm_score}" + crlf
        summary += f"DeCompression {self.compressed_size} bytes to {self.decoded_size} bytes" + crlf
        summary += f"env: {self.experiment_environment}" + crlf
        summary += f"total time elapsed: {self.encode_time+self.decode_time}" + crlf
        
        return summary
