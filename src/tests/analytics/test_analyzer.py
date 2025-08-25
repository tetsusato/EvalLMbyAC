import polars as pl
import pytest
from src.analytics.analytics import Analyzer

class TestAnalyzer:
    # あとで設定ファイル導入する
    root = "/home/tetsu.sato/FIT2025-presen"
    def test_create_object(self):
        analyzer = Analyzer(self.root)
        assert analyzer is not None
        assert analyzer.cache is not None

        analyzer.cache.delete("test-key")
        assert analyzer.cache.get("test-key") is None
        analyzer.cache.set("test-key", "test-val")
        assert analyzer.cache.get("test-key") == "test-val"

    def test_get_leaderboard_result(self):
        analyzer = Analyzer(self.root)
        assert analyzer is not None
        assert analyzer.cache is not None

        result: dict = analyzer._get_leaderboard_result("")
        assert result.__class__ is dict
        assert result["headers"] is not None
        assert result["data"] is not None

    def test_leaderboard_data_to_polar_dataframe(self):
        analyzer = Analyzer(self.root)
        assert analyzer is not None
        assert analyzer.cache is not None

        result: dict = analyzer._get_leaderboard_result("")
        assert result.__class__ is dict

        leaderboard_df = analyzer._leaderboard_data_to_polar_dataframe(result)

        import polars as pl
        assert leaderboard_df.__class__ is pl.dataframe.frame.DataFrame

    def test_get_leaderboard_df(self):
        analyzer = Analyzer(self.root)
        assert analyzer is not None
        assert analyzer.cache is not None

        leaderboard_df = analyzer.get_leaderboard_df()
        assert leaderboard_df.__class__ is pl.dataframe.frame.DataFrame

    def test_search_model_from_mlflow(self):
        analyzer = Analyzer(self.root)
        assert analyzer is not None
        assert analyzer.cache is not None
        runs_df = analyzer.search_model_from_mlflow(analyzer.models[0:-2],
                                                    run_name="fit2025-paper-05%")
        #print(runs_df)
        assert runs_df.__class__ is pl.dataframe.frame.DataFrame
        assert runs_df.shape == (62, 29)
        #print(runs_df.shape)
        #print(runs_df.schema)
        #print(runs_df.columns)
        assert runs_df.columns[0] == "tags.model_name"
    def test_mlflow_results_df_to_analytics_df(self):
        analyzer = Analyzer(self.root)
        assert analyzer is not None
        assert analyzer.cache is not None
        runs_df = analyzer.search_model_from_mlflow(analyzer.models[0:-2],
                                                    run_name="fit2025-paper-05%")
        total_exp_df = analyzer.mlflow_results_df_to_analytics_df(runs_df)
        #print(f"exp_df={total_exp_df}")
        assert total_exp_df.shape == (62, 8)
        assert total_exp_df.columns[0] == "tags.model_name"
    def test_shrink_model_name(self):
        analyzer = Analyzer(self.root)
        assert analyzer is not None
        assert analyzer.cache is not None

        model_names_examples = ["llm-jp/llm-jp-3-980m-instruct2",]
        model_names_shrinked = ["llm-jp-3-980m-inst2",]
        print(f"examples={model_names_examples}")
        shrinked = analyzer.shrink_model_name(model_names_examples)
        assert shrinked == model_names_shrinked
    def test_leaderboard_df_to_report_df(self):
        analyzer = Analyzer(self.root)
        assert analyzer is not None
        assert analyzer.cache is not None
        result: dict = analyzer._get_leaderboard_result("")
        assert result.__class__ is dict
        
        leaderboard_df = analyzer._leaderboard_data_to_polar_dataframe(result)
        runs_df = analyzer.search_model_from_mlflow(analyzer.models[0:-2],
                                                    run_name="fit2025-paper-05%")
        total_exp_df = analyzer.mlflow_results_df_to_analytics_df(runs_df)
        
        report_df = analyzer.leaderboard_df_to_report_df(leaderboard_df, total_exp_df)
        #print(f"report_df={report_df}")
        assert report_df.shape == (62, 11)
        assert report_df.columns[0] == "T"

    def test_search_model_from_mlflow(self):
        analyzer = Analyzer(self.root)
        assert analyzer is not None
        assert analyzer.cache is not None
        runs_df = analyzer.search_model_from_mlflow(analyzer.models[0:2],
                                           run_name="fit2025-paper-05%")
        assert runs_df[0]["tags.model_name"][0] == "google/gemma-1.1-7b-it"
        

    def test_mlflow_results_df_to_analytics_df(self):
        analyzer = Analyzer(self.root)
        assert analyzer is not None
        assert analyzer.cache is not None
        runs_df = analyzer.search_model_from_mlflow(analyzer.models[0:2],
                                           run_name="fit2025-paper-05%")
        assert runs_df[0]["tags.model_name"][0] == "google/gemma-1.1-7b-it"
        
    
