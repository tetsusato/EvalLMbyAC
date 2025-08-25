import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import os
import sys
from pathlib import Path
import torch # Import torch
import polars as pl # Import polars for type hinting in mocks

# Add the project root to the sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.lmcr.lmcr import LMCR

class TestLMCR(unittest.TestCase):

    def setUp(self):
        # Mock the omegaconf.dictconfig.DictConfig object
        self.mock_cfg = MagicMock()
        self.mock_cfg.exp.title = "test_exp"
        self.mock_cfg.exp.log_dir = "/tmp/test_log"
        self.mock_cfg.exp.llm = "test_model"
        self.mock_cfg.exp.input = "test_input.txt"
        self.mock_cfg.exp.inputs_limit = 100
        self.mock_cfg.exp.device = "cpu"
        self.mock_cfg.exp.use_cache = True
        self.mock_cfg.exp.hosting = "huggingface"
        self.mock_cfg.cache.enable = True

        # Mock mlflow
        self.mock_mlflow_patch = patch('src.lmcr.lmcr.mlflow')
        self.mock_mlflow = self.mock_mlflow_patch.start()

        # Mock Cache
        self.mock_cache_patch = patch('src.lmcr.lmcr.Cache')
        self.mock_cache_cls = self.mock_cache_patch.start()
        self.mock_cache_instance = MagicMock()
        self.mock_cache_cls.return_value = self.mock_cache_instance

        # Mock Path.mkdir
        self.mock_os_mkdir_patch = patch('src.lmcr.lmcr.os.mkdir')
        self.mock_os_mkdir = self.mock_os_mkdir_patch.start()

        # Mock Path.exists
        self.mock_os_path_exists_patch = patch('src.lmcr.lmcr.os.path.exists')
        self.mock_os_path_exists = self.mock_os_path_exists_patch.start()
        self.mock_os_path_exists.return_value = True # Assume directories exist for simplicity

        # Mock logging
        self.mock_logging_patch = patch('src.lmcr.lmcr.logging')
        self.mock_logging = self.mock_logging_patch.start()
        self.mock_logging.getLogger.return_value = MagicMock()


    def tearDown(self):
        self.mock_mlflow_patch.stop()
        self.mock_cache_patch.stop()
        self.mock_os_mkdir_patch.stop()
        self.mock_os_path_exists_patch.stop()
        self.mock_logging_patch.stop()

    def test_init(self):
        lmcr_instance = LMCR(self.mock_cfg)

        self.assertEqual(lmcr_instance.exp_title, "test_exp")
        self.assertEqual(lmcr_instance.model_huggingface_id, "test_model")
        self.assertEqual(lmcr_instance.input, "test_input.txt")
        self.assertEqual(lmcr_instance.device, "cpu")
        self.assertEqual(lmcr_instance.use_cache, True)
        self.assertEqual(lmcr_instance.hosting, "huggingface")
        self.assertTrue(lmcr_instance.cache is not None)
        self.mock_mlflow.log_params.assert_called_once_with(self.mock_cfg)
        self.mock_cache_cls.assert_called_once() # Cache should be instantiated

    def test_get_cache_key_huggingface(self):
        lmcr_instance = LMCR(self.mock_cfg)
        lmcr_instance.model = MagicMock()
        lmcr_instance.model.name_or_path = "mock_hf_model"
        lmcr_instance.hosting = "huggingface"

        mock_func = MagicMock()
        mock_func.__name__ = "execute_ae"

        key = lmcr_instance.get_cache_key(mock_func)
        self.assertEqual(key, "test_exp-mock_hf_model-test_input.txt-ae")

    def test_get_cache_key_vllm(self):
        self.mock_cfg.exp.hosting = "vllm"
        lmcr_instance = LMCR(self.mock_cfg)
        lmcr_instance.model = MagicMock()
        lmcr_instance.model.llm_engine.model_config.model = "mock_vllm_model"
        lmcr_instance.hosting = "vllm"

        mock_func = MagicMock()
        mock_func.__name__ = "execute_ppl"

        key = lmcr_instance.get_cache_key(mock_func)
        self.assertEqual(key, "test_exp-mock_vllm_model-test_input.txt-ppl")

    def test_get_cache_key_openai(self):
        self.mock_cfg.exp.hosting = "openai"
        lmcr_instance = LMCR(self.mock_cfg)
        lmcr_instance.model_huggingface_id = "mock_openai_model"
        lmcr_instance.hosting = "openai"

        mock_func = MagicMock()
        mock_func.__name__ = "execute_ae"

        key = lmcr_instance.get_cache_key(mock_func)
        self.assertEqual(key, "test_exp-mock_openai_model-test_input.txt-ae")

    @patch('src.lmcr.lmcr.Path')
    @patch('src.lmcr.lmcr.AutoTokenizer.from_pretrained')
    def test_calculate_token_efficiency(self, mock_from_pretrained, mock_Path):
        lmcr_instance = LMCR(self.mock_cfg)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer_output = MagicMock()

        # Use torch.tensor directly for input_ids
        mock_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        mock_input_ids_tensor = torch.tensor([mock_tokens]) # Create a 2D tensor

        mock_tokenizer_output.input_ids = mock_input_ids_tensor
        mock_tokenizer.return_value = mock_tokenizer_output
        lmcr_instance.tokenizer = mock_tokenizer

        # Mock Path.read_text
        mock_file_content = "This is a test string for token efficiency."
        mock_Path.return_value.read_text.return_value = mock_file_content

        # Call the method
        ratio = lmcr_instance.calculate_token_efficiency()

        # Assertions
        expected_tokenized_size = len(mock_tokens)
        expected_input_len = len(mock_file_content)
        expected_ratio = expected_tokenized_size / expected_input_len

        self.assertEqual(ratio, expected_ratio)
        self.mock_mlflow.log_metric.assert_any_call("tokenized_size", expected_tokenized_size)
        self.mock_mlflow.log_metric.assert_any_call("token_efficiency", expected_ratio)
        mock_Path.assert_called_with(f"{lmcr_instance.input_dir}/{lmcr_instance.input}")
        mock_Path.return_value.read_text.assert_called_once()

    @patch('src.lmcr.lmcr.LMCR.encode_decode_test')
    def test_execute_ae_cache_miss(self, mock_encode_decode_test):
        lmcr_instance = LMCR(self.mock_cfg)
        # Ensure cache.get returns None for cache miss
        lmcr_instance.cache.get.return_value = None

        # Mock the return value of encode_decode_test
        mock_result_df = MagicMock(spec=pl.DataFrame)
        mock_encode_decode_test.return_value = mock_result_df

        # Call execute_ae
        result_df = lmcr_instance.execute_ae(
            cache_val=None,
            input_dir="mock_dir",
            text_path="mock_path",
            basic_info="mock_info",
            func_name="ae"
        )

        # Assertions
        mock_encode_decode_test.assert_called_once_with(
            "mock_dir", "mock_path", "mock_info", func_name="ae"
        )
        self.assertEqual(result_df, mock_result_df)

    @patch('src.lmcr.lmcr.LMCR.encode_decode_test')
    def test_execute_ae_cache_hit(self, mock_encode_decode_test):
        lmcr_instance = LMCR(self.mock_cfg)
        # Ensure cache.get returns a value for cache hit
        mock_cached_val = MagicMock()
        lmcr_instance.cache.get.return_value = mock_cached_val

        # Call execute_ae
        result_df = lmcr_instance.execute_ae(
            cache_val=mock_cached_val, # Pass the cached value
            input_dir="mock_dir",
            text_path="mock_path",
            basic_info="mock_info",
            func_name="ae"
        )

        # Assertions
        mock_encode_decode_test.assert_not_called() # Should not call the underlying method
        lmcr_instance.cache.set.assert_not_called() # Should not set cache
        # The result_df should be a polars DataFrame created from the cached value
        # Since the original code does pl.DataFrame([cache_val]), we need to mock pl.DataFrame
        with patch('src.lmcr.lmcr.pl.DataFrame') as mock_pl_dataframe:
            # Fix: Remove spec=pl.DataFrame as it's already a mock
            mock_pl_dataframe.return_value = MagicMock()
            result_df_expected = mock_pl_dataframe([mock_cached_val])
            # Re-run the method to get the mocked pl.DataFrame
            result_df = lmcr_instance.execute_ae(
                cache_val=mock_cached_val,
                input_dir="mock_dir",
                text_path="mock_path",
                basic_info="mock_info",
                func_name="ae"
            )
            # Compare content by converting to list, as direct mock comparison might fail due to object identity
            self.assertEqual(result_df.to_series().to_list(), result_df_expected.to_series().to_list())

    @patch('src.lmcr.lmcr.LMCR.perplexity_test')
    def test_execute_ppl_cache_miss(self, mock_perplexity_test):
        lmcr_instance = LMCR(self.mock_cfg)
        lmcr_instance.cache.get.return_value = None

        mock_result_df = MagicMock(spec=pl.DataFrame)
        mock_perplexity_test.return_value = mock_result_df

        result_df = lmcr_instance.execute_ppl(
            cache_val=None,
            input_dir="mock_dir",
            text_path="mock_path",
            basic_info="mock_info",
            func_name="ppl"
        )

        mock_perplexity_test.assert_called_once_with(
            "mock_dir", "mock_path", "mock_info", func_name="ppl"
        )
        self.assertEqual(result_df, mock_result_df)

    @patch('src.lmcr.lmcr.LMCR.perplexity_test')
    def test_execute_ppl_cache_hit(self, mock_perplexity_test):
        lmcr_instance = LMCR(self.mock_cfg)
        mock_cached_val = MagicMock()
        lmcr_instance.cache.get.return_value = mock_cached_val

        result_df = lmcr_instance.execute_ppl(
            cache_val=mock_cached_val,
            input_dir="mock_dir",
            text_path="mock_path",
            basic_info="mock_info",
            func_name="ppl"
        )

        mock_perplexity_test.assert_not_called()
        lmcr_instance.cache.set.assert_not_called()
        with patch('src.lmcr.lmcr.pl.DataFrame') as mock_pl_dataframe:
            # Fix: Remove spec=pl.DataFrame as it's already a mock
            mock_pl_dataframe.return_value = MagicMock()
            result_df_expected = mock_pl_dataframe([mock_cached_val])
            result_df = lmcr_instance.execute_ppl(
                cache_val=mock_cached_val,
                input_dir="mock_dir",
                text_path="mock_path",
                basic_info="mock_info",
                func_name="ppl"
            )
            # Compare content by converting to list, as direct mock comparison might fail due to object identity
            self.assertEqual(result_df.to_series().to_list(), result_df_expected.to_series().to_list())