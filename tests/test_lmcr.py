import pytest
from unittest.mock import MagicMock, patch, mock_open
import omegaconf
from pathlib import Path
import polars as pl
import os
import sys
import importlib.machinery

# src をパスに追加
sys.path.append(os.path.abspath("src"))
# ルートもパスに追加（result.py などのため）
sys.path.append(os.path.abspath("."))

# LMCRが依存しているがテスト環境にない可能性があるライブラリをモック化
def mock_package(name):
    m = MagicMock()
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules[name] = m

mock_package("vllm")
mock_package("openai")
mock_package("gptzip")
mock_package("gptzip.gptzip")
mock_package("gptzip.gptzip_online")
mock_package("mlflow")
mock_package("mlflow.tracking")

from lmcr.lmcr import LMCR

@pytest.fixture
def mock_cfg():
    cfg = omegaconf.DictConfig({
        "exp": {
            "title": "test_exp",
            "encoding_algorithm": "ae",
            "log_dir": "test_logs",
            "llm": "test-model",
            "input": "test_input.txt",
            "inputs_limit": 1,
            "device": "cpu",
            "use_cache": True,
            "hosting": "huggingface",
            "max_vocab_size": 32000,
            "gamma": 1.0
        },
        "cache": {
            "l1_cache": {
                "enable": True,
                "top_dir": "test_cache",
                "root": "l1"
            },
            "l2_cache": {
                "enable": True,
                "top_dir": "test_cache",
                "root": "l2"
            }
        }
    })
    return cfg

def test_lmcr_init(mock_cfg):
    with patch("lmcr.lmcr.Cache") as MockCache:
        with patch("os.path.exists", return_value=True):
            exe = LMCR(mock_cfg)
            assert exe.exp_title == "test_exp"
            assert exe.model_huggingface_id == "test-model"
            assert MockCache.call_count == 2

def test_lmcr_init_without_title(mock_cfg):
    # title を削除
    del mock_cfg.exp.title
    # コマンドライン引数を模倣
    with patch("sys.argv", ["lmcr.py", "config/lmcr_exp_test_auto.yaml"]):
        with patch("lmcr.lmcr.Cache"):
            with patch("os.path.exists", return_value=True):
                # main ブロック相当の処理 (実際には lmcr.py の末尾のロジックを模倣)
                import os
                import re
                from omegaconf import OmegaConf
                
                # lmcr.py の main ブロックのロジックを一部再現して cfg を加工
                raw_config_path = "config/lmcr_exp_test_auto.yaml"
                basename = os.path.basename(raw_config_path)
                match = re.match(r'lmcr_exp_([^.]+)\.yaml$', basename)
                if match:
                    auto_title = match.group(1)
                    OmegaConf.set_struct(mock_cfg, False)
                    mock_cfg.exp.title = auto_title
                    OmegaConf.set_struct(mock_cfg, True)
                
                exe = LMCR(mock_cfg)
                assert exe.exp_title == "test_auto"

def test_get_cache_key(mock_cfg):
    with patch("lmcr.lmcr.Cache"):
        with patch("os.path.exists", return_value=True):
            exe = LMCR(mock_cfg)
            exe.model = MagicMock()
            exe.model.name_or_path = "test-model-path"
            
            mock_func = MagicMock()
            mock_func.__name__ = "execute_ae"
            
            key = exe.get_cache_key(mock_func)
            assert "test_exp" in key
            assert "test-model-path" in key
            assert "test_input.txt" in key
            assert "ae" in key

def test_input_analysis_l1_hit(mock_cfg):
    with patch("lmcr.lmcr.Cache") as MockCache:
        with patch("os.path.exists", return_value=True):
            exe = LMCR(mock_cfg)
            # L1 キャッシュヒットをシミュレート
            mock_df = pl.DataFrame({"result": [1.0]})
            exe.L1_CACHE.get.return_value = mock_df
            
            mock_func = MagicMock()
            mock_func.__name__ = "execute_ae"
            
            # 外部副作用をモック化
            exe.prepare_llm = MagicMock()
            exe.record_to_mlflow = MagicMock()
            
            exe.input_dir = "dummy_dir"
            exe.input = "test_input.txt"
            
            # polars の df.write_parquet をパッチ
            with patch.object(pl.DataFrame, "write_parquet"):
                with patch("lmcr.lmcr.Path.read_text", return_value="dummy content"):
                    with patch.object(LMCR, "get_cache_key", return_value="dummy_key"):
                        # input_analysis は値を返さないため、副作用を検証する
                        exe.input_analysis(mock_func)
                        
                        exe.L1_CACHE.get.assert_called_with("dummy_key")
                        exe.record_to_mlflow.assert_called_once()
                        mock_func.assert_not_called()

def test_encode_decode_test_l2_hit(mock_cfg):
    with patch("lmcr.lmcr.Cache"):
        with patch("os.path.exists", return_value=True):
            exe = LMCR(mock_cfg)
            exe.model_name_safe = "test-model"
            exe.L2_CACHE = MagicMock()
            # L2 キャッシュヒット (codeデータを返す)
            exe.L2_CACHE.get.return_value = b"compressed_data"
            
            # 必要な属性のセット
            exe.hosting = "huggingface"
            exe.encoding_algorithm = "ae"
            exe.tokenizer = MagicMock()
            exe.model = MagicMock()
            exe.model.name_or_path = "test-model"
            exe.max_vocab_size = 32000
            exe.tokenizer_vocab_size = 32000
            exe.gamma = 1.0
            
            # lmcr.lmcr.Path.read_text() をモック
            with patch("lmcr.lmcr.Path.read_text", return_value="original text"):
                # ArithmeticCoder は import 時にモック化されているためパッチ不要
                result_df = exe.encode_decode_test("dir", "file.txt", "info", "ae")
                
                assert result_df["compressed_size"][0] == len(b"compressed_data")
                # キャッシュヒット時は encode_time=0, decode_time=0
                assert result_df["encode_time"][0] == 0
                assert result_df["decode_time"][0] == 0
                exe.L2_CACHE.get.assert_called()
