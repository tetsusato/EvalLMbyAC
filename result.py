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
    model_name: str
    input_file_name: str
    original_size: int
    compressed_size: int
    decoded_size: int
    compression_rate: float
    encode_time: int
    decode_time: int
    experiment_environment: str

    def to_csv(self):
        output = io.StringIO()
        writer = csv.writer(output)
        row = [
            self.exp_title,
            self.model_name,
            self.input_file_name,
            self.original_size,
            self.compressed_size,
            self.decoded_size,
            self.compression_rate,
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
        summary += f"Compression {self.original_size} bytes to {self.compressed_size} bytes" + crlf
        summary += f"Compression ratio {self.compression_rate}" + crlf
        summary += f"DeCompression {self.compressed_size} bytes to {self.decoded_size} bytes" + crlf
        summary += f"env: {self.experiment_environment}" + crlf
        summary += f"total time elapsed: {self.encode_time+self.decode_time}" + crlf
        
        return summary
