import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        f"""
    ## 週末のうちに一時申し込みしたい→2025/05/05 2:54 申し込んでた
    ```
     ■ 講演情報
    20. 講演形式: 選奨論文
    21. 講演応募分野(研究会): E-1　自然言語処理(NL)
    22. 使用言語: 日本語
    23. 講演題目(和文): 言語モデルにおける符号化能力とベンチマーク性能との乖離現象
    24. 講演題目(英文): Divergence Phenomenon between Encoding Capability and Benchmark Performance in Language Models
    25. 原稿ページ数:4
    26. 抄録:
    ニューラルネットワークを基盤とする大規模言語モデル（Large Language Model; LLM）の発展に伴い，その知識を活用した情報圧縮手法が注目されている．圧縮性能はLMCR（Language Model-based Compression Rate）で評価され．一般にLMCRは各種ベンチマークにおけるLLMの性能と高い相関を示す．しかし一部のLLMでは，この傾向から逸脱したLMCRが観測されており，符号化能力とタスク性能の関係には未解明の側面が残されている．本研究では，この乖離現象の原因を分析・考察する．
    ```
    """
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    import hydra
    from hydra import compose, initialize, initialize_config_dir
    from lmcr import LMCR
    import os
    import sys

    from logging import config                                              
    import logging                                                          
    config.fileConfig("logging.conf", disable_existing_loggers = False)     

    logger = logging.getLogger(__name__)                                    
    progress = logging.getLogger("progress")                                
    summary = logging.getLogger("summary")                                  
    return LMCR, compose, config, initialize_config_dir, os, progress


@app.cell
def _():
    print("hello")
    return


@app.cell
def _(LMCR, compose, config, initialize_config_dir, os, progress):
    config_name = "lmcr_exp1"
    override_options = ""       

    def execute(config_name: str,
               override_options: str,
               ):
        #with initialize(config_path="config", job_name=__file__):    
        with initialize_config_dir(config_dir=f"{os.getcwd()}/config", job_name=__file__):    
            print(f"Working directory : {os.getcwd()}")
            print(f"")
            if override_options:                                                 
                cfg = compose(config_name=config_name,                           
                              return_hydra_config=True,                          
                              overrides=[override_options],                      
                              )                                                  
            else:                                                                
                cfg = compose(config_name=config_name,                           
                              return_hydra_config=True,                          
                              )    
            #print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")     
        exp_title=cfg.exp.title                                                  
        exp_summary=cfg.exp.summary                                              
        progress.info(f"config={config} title={exp_title} summary={exp_summary}")
        exe = LMCR(cfg)                                                          

        exe.input_analysis(exe.execute_ae)     

    execute(config_name, override_options)

    return config_name, execute, override_options


@app.cell
def _(config_name, execute):
    # キャッシュオフにしてみるか
    execute(config_name, "exp.use_cache=False")
    return


@app.cell
def _(config_name, execute, override_options):
    execute(config_name, override_options)
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""なんとなくできてるから，DeepSeekが悪くてLLM-jpが良いのを確認するか""")
    return


@app.cell
def _():
    # deepseekのconfigをlmcr_exp_deepseek1で．
    #execute("lmcr_exp_deepseek1", "")
    return


@app.cell
def _(execute, os):
    os.environ["HF_HOME"] =  "/cache/" # marimo立ち上げる前に設定が必要かも→そうだった
    execute("lmcr_exp_qwen3", "")
    return


@app.cell
def _():
    # 再チャレンジ
    #execute("lmcr_exp_qwen3", "")
    return


@app.cell
def _(execute):
    execute("lmcr_exp1", "") # llm: "rinna/gemma-2-baku-2b-it"
    return


@app.cell
def _(execute):
    execute("lmcr_exp2", "") # llm: "llm-jp/llm-jp-3-3.7b-instruct"
    return


@app.cell
def _(execute):

    execute("lmcr_exp3", "") # llm: "llm-jp/llm-jp-3-13b-instruct"
    return


@app.cell
def _(execute):
    execute("lmcr_exp_deepseek-r1-distill-qwen-1.5b", "") # llm: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 4-shot 0.2584
    return


@app.cell
def _(execute):
    execute("lmcr_exp_deepseek-r1-distill-qwen-7b", "") # llm: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 4-shot 0.3928
    return


@app.cell
def _(execute):
    execute("lmcr_exp_llama-3.2-1b-inst", "") # llm: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 4-shot 0.3059

    return


@app.cell
def _(execute):
    execute("lmcr_exp_llama-3.2-3b-inst", "") # llm: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 4-shot 0.3059

    return


@app.cell
def _(execute):
    execute("lmcr_exp_qwen3_30b", "")
    return


@app.cell
def _(execute):
    execute("lmcr_exp_qwen3_14b", "")
    return


@app.cell
def _(execute):
    execute("lmcr_exp_qwen3_8b", "")
    return


@app.cell
def _(execute):
    execute("lmcr_exp_qwen3_4b", "")
    return


@app.cell
def _(execute):
    execute("lmcr_exp_qwen3_1.7b", "")
    return


@app.cell
def _(execute):
    execute("lmcr_exp_qwen3_0.6b", "")
    return


@app.cell
def _(execute):
    execute("lmcr_exp_qwen3", "")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
