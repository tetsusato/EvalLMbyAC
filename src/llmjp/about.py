from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    AVG = "Average - 平均"
    NLI = "NLI - 自然言語推論"
    QA = "QA - 質問応答"
    RC = "RC - 読解力"
    MC = "MC - 多肢選択式質問応答"
    EL = "EL - エンティティリンキング"
    FA = "FA - 基礎分析"
    MR = "MR - 数学的推論"
    MT = "MT - 機械翻訳"
    STS = "STS - 意味的類似度"
    HE = "HE - 試験問題"
    CG = "CG - コード生成"
    SUM = "SUM - 要約"
    NotTask = "?"


@dataclass
class Task:
    benchmark: str
    metric: str
    col_name: str
    task_type: TaskType
    average: bool = False


# Select your tasks here
# ---------------------------------------------------
class Tasks(Enum):
    AVG = Task("scores", "AVG", "AVG", TaskType.AVG, True)
    NLI = Task("scores", "NLI", "AVG (NLI)", TaskType.NLI, True)  # Natural Language Inference - 自然言語推論
    QA = Task("scores", "QA", "AVG (QA)", TaskType.QA, True)  # Question Answering - 質問応答
    RC = Task("scores", "RC", "AVG (RC)", TaskType.RC, True)  # Reading Comprehension - 文章読解
    MC = Task("scores", "MC", "AVG (MC)", TaskType.MC, True)  # Multiple Choice question answering - 多肢選択式問題
    EL = Task("scores", "EL", "AVG (EL)", TaskType.EL, True)  # Entity Linking - エンティティリンキング
    FA = Task("scores", "FA", "AVG (FA)", TaskType.FA, True)  # Fundamental Analysis - 基礎解析
    MR = Task("scores", "MR", "AVG (MR)", TaskType.MR, True)  # Mathematical Reasoning - 数学的推論
    MT = Task("scores", "MT", "AVG (MT)", TaskType.MT, True)  # Machine Translation - 機械翻訳
    HE = Task("scores", "HE", "AVG (HE)", TaskType.HE, True)  # Human Examination - 試験問題
    CG = Task("scores", "CG", "AVG (CG)", TaskType.CG, True)  # Code Generation - コード生成
    SUM = Task("scores", "SUM", "AVG (SUM)", TaskType.SUM, True)  # Summarization - 要約
    alt_e_to_j_bert_score_ja_f1 = Task("scores", "alt-e-to-j_bert_score_ja_f1", "ALT E to J BERT Score", TaskType.MT)
    alt_e_to_j_bleu_ja = Task("scores", "alt-e-to-j_bleu_ja", "ALT E to J BLEU", TaskType.MT)
    alt_e_to_j_comet_wmt22 = Task("scores", "alt-e-to-j_comet_wmt22", "ALT E to J COMET WMT22 ⭐", TaskType.MT)
    alt_j_to_e_bert_score_en_f1 = Task("scores", "alt-j-to-e_bert_score_en_f1", "ALT J to E BERT Score", TaskType.MT)
    alt_j_to_e_bleu_en = Task("scores", "alt-j-to-e_bleu_en", "ALT J to E BLEU", TaskType.MT)
    alt_j_to_e_comet_wmt22 = Task("scores", "alt-j-to-e_comet_wmt22", "ALT J to E COMET WMT22 ⭐", TaskType.MT)
    chabsa_set_f1 = Task("scores", "chabsa_set_f1", "ChABSA ⭐", TaskType.EL)
    commonsensemoralja_exact_match = Task(
        "scores", "commonsensemoralja_exact_match", "CommonSenseMoralJA ⭐", TaskType.MC
    )
    jamp_exact_match = Task("scores", "jamp_exact_match", "JAMP ⭐", TaskType.NLI)
    janli_exact_match = Task("scores", "janli_exact_match", "JANLI ⭐", TaskType.NLI)
    jcommonsenseqa_exact_match = Task("scores", "jcommonsenseqa_exact_match", "JCommonSenseQA ⭐", TaskType.MC)
    jemhopqa_char_f1 = Task("scores", "jemhopqa_char_f1", "JEMHopQA ⭐", TaskType.QA)
    jmmlu_exact_match = Task("scores", "jmmlu_exact_match", "JMMLU ⭐", TaskType.HE)
    jnli_exact_match = Task("scores", "jnli_exact_match", "JNLI ⭐", TaskType.NLI)
    jsem_exact_match = Task("scores", "jsem_exact_match", "JSEM ⭐", TaskType.NLI)
    jsick_exact_match = Task("scores", "jsick_exact_match", "JSICK ⭐", TaskType.NLI)
    jsquad_char_f1 = Task("scores", "jsquad_char_f1", "JSquad ⭐", TaskType.RC)
    jsts_pearson = Task(
        "scores", "jsts_pearson", "JSTS (Pearson)", TaskType.STS
    )  # Semantic Textual Similarity - 意味的類似度
    jsts_spearman = Task(
        "scores", "jsts_spearman", "JSTS (Spearman)", TaskType.STS
    )  # Semantic Textual Similarity - 意味的類似度
    kuci_exact_match = Task("scores", "kuci_exact_match", "KUCI ⭐", TaskType.MC)
    mawps_exact_match = Task("scores", "mawps_exact_match", "MAWPS ⭐", TaskType.MR)
    mbpp_code_exec = Task("scores", "mbpp_code_exec", "MBPP (exec) (0 shots only) ⭐", TaskType.CG)
    mbpp_pylint_check = Task("scores", "mbpp_pylint_check", "MBPP (pylint) (0 shots only)", TaskType.CG)
    mmlu_en_exact_match = Task("scores", "mmlu_en_exact_match", "MMLU ⭐", TaskType.HE)
    niilc_char_f1 = Task("scores", "niilc_char_f1", "NIILC ⭐", TaskType.QA)
    aio_char_f1 = Task("scores", "aio_char_f1", "JAQKET ⭐", TaskType.QA)
    wiki_coreference_set_f1 = Task("scores", "wiki_coreference_set_f1", "Wiki Coreference ⭐", TaskType.FA)
    wiki_dependency_set_f1 = Task("scores", "wiki_dependency_set_f1", "Wiki Dependency ⭐", TaskType.FA)
    wiki_ner_set_f1 = Task("scores", "wiki_ner_set_f1", "Wiki NER ⭐", TaskType.FA)
    wiki_pas_set_f1 = Task("scores", "wiki_pas_set_f1", "Wiki PAS ⭐", TaskType.FA)
    wiki_reading_char_f1 = Task("scores", "wiki_reading_char_f1", "Wiki Reading ⭐", TaskType.FA)
    wikicorpus_e_to_j_bert_score_ja_f1 = Task(
        "scores", "wikicorpus-e-to-j_bert_score_ja_f1", "WikiCorpus E to J BERT Score", TaskType.MT
    )
    wikicorpus_e_to_j_bleu_ja = Task("scores", "wikicorpus-e-to-j_bleu_ja", "WikiCorpus E to J BLEU", TaskType.MT)
    wikicorpus_e_to_j_comet_wmt22 = Task(
        "scores", "wikicorpus-e-to-j_comet_wmt22", "WikiCorpus E to J COMET WMT22 ⭐", TaskType.MT
    )
    wikicorpus_j_to_e_bert_score_en_f1 = Task(
        "scores", "wikicorpus-j-to-e_bert_score_en_f1", "WikiCorpus J to E BERT Score", TaskType.MT
    )
    wikicorpus_j_to_e_bleu_en = Task("scores", "wikicorpus-j-to-e_bleu_en", "WikiCorpus J to E BLEU", TaskType.MT)
    wikicorpus_j_to_e_comet_wmt22 = Task(
        "scores", "wikicorpus-j-to-e_comet_wmt22", "WikiCorpus J to E COMET WMT22 ⭐", TaskType.MT
    )
    xlsum_ja_bert_score_ja_f1 = Task(
        "scores", "xlsum_ja_bert_score_ja_f1", "XL-Sum JA BERT Score (0 shots only)", TaskType.SUM
    )
    xlsum_ja_bleu_ja = Task("scores", "xlsum_ja_bleu_ja", "XL-Sum JA BLEU (0 shots only)", TaskType.SUM)
    xlsum_ja_rouge1 = Task("scores", "xlsum_ja_rouge1", "XL-Sum ROUGE1 (0 shots only)", TaskType.SUM)
    xlsum_ja_rouge2 = Task("scores", "xlsum_ja_rouge2", "XL-Sum ROUGE2 (0 shots only) ⭐", TaskType.SUM)
    # xlsum_ja_rouge2_scaling = Task("scores", "xlsum_ja_rouge2_scaling", "XL-Sum JA ROUGE2 Scaling")
    xlsum_ja_rougeLsum = Task("scores", "xlsum_ja_rougeLsum", "XL-Sum ROUGE-Lsum (0 shots only)", TaskType.SUM)


NUM_FEWSHOT = 0  # Change with your few shot
# ---------------------------------------------------

# Your leaderboard name
TITLE = """<h1 align="center" id="space-title">🇯🇵 Open Japanese LLM Leaderboard 🌸<br>オープン日本語LLMリーダーボード</h1>"""

# What does your leaderboard evaluate?
INTRODUCTION_TEXT = """
The __Open Japanese LLM Leaderboard__ by __[LLM-jp](https://llm-jp.nii.ac.jp/en/)__ evaluates
the performance of Japanese Large Language Models (LLMs) with more than 16 tasks from
classical to modern NLP tasks. The __Open Japanese LLM Leaderboard__ was built by open-source
contributors of __[LLM-jp](https://llm-jp.nii.ac.jp/en/)__, a cross-organizational project
for the research and development of Japanese LLMs supported by the _National Institute of
Informatics_ in Tokyo, Japan.

On the __"LLM Benchmark"__ page, the question mark **"?"** refers to the parameters that
are unknown in the model card on Hugging Face. For more information about datasets,
please consult the __"About"__ page or refer to the website of
__[LLM-jp](https://llm-jp.nii.ac.jp/en/)__. And on the __"Submit here!"__ page, you can
evaluate the performance of your model, and be part of the leaderboard.
"""
INTRODUCTION_TEXT_JA = """\
__[LLM-jp](https://llm-jp.nii.ac.jp/)__ による __オープン日本語LLMリーダーボード__ は、\
古典的なものから最新のものまで16種類以上のNLPタスクを用いて日本語大規模言語モデル(LLM)の\
性能を評価します。__オープン日本語LLMリーダーボード__ は、日本の国立情報学研究所を中心に\
日本語LLMの研究開発を行う組織横断プロジェクト __[LLM-jp](https://llm-jp.nii.ac.jp/)__ \
のオープンソース貢献者によって構築されました。

__「LLM Benchmark」__ ページでは、疑問符 **「?」** はHugging Faceのモデルカードで不明な\
パラメータを示しています。データセットに関する詳細情報については、__「About」__ ページを\
参照するか、__[LLM-jp](https://llm-jp.nii.ac.jp/)__ のウェブサイトをご覧ください。\
また、__「Submit here!」__ ページでは、あなたのモデルの性能を評価し、リーダーボードに\
参加することができます。
"""

# Which evaluations are you running? how can people reproduce what you have?
LLM_BENCHMARKS_TEXT = """
## How it works
📈 We evaluate Japanese Large Language Models on 16 tasks leveraging our evaluation tool [llm-jp-eval](https://github.com/llm-jp/llm-jp-eval), a unified framework to evaluate Japanese LLMs on various evaluation tasks.

**NLI (Natural Language Inference)**

* `Jamp`, a Japanese NLI benchmark focused on temporal inference [Source](https://github.com/tomo-ut/temporalNLI_dataset) (License CC BY-SA 4.0)

* `JaNLI`, Japanese Adversarial Natural Language Inference [Source](https://github.com/verypluming/JaNLI) (License CC BY-SA 4.0)

* `JNLI`, Japanese Natural Language Inference (part of JGLUE) [Source](https://github.com/yahoojapan/JGLUE) (License CC BY-SA 4.0)

* `JSeM`, Japanese semantic test suite [Source](https://github.com/DaisukeBekki/JSeM) (License BSD 3-Clause)

* `JSICK`, Japanese Sentences Involving Compositional Knowledge [Source](https://github.com/verypluming/JSICK) (License CC BY-SA 4.0)

**QA (Question Answering)**

* `JEMHopQA`, Japanese Explainable Multi-hop Question Answering [Source](https://github.com/aiishii/JEMHopQA) (License CC BY-SA 4.0)

* `NIILC`, NIILC Question Answering Dataset [Source](https://github.com/mynlp/niilc-qa) (License CC BY-SA 4.0)

* `JAQKET`, Japanese QA dataset on the subject of quizzes [Source](https://www.nlp.ecei.tohoku.ac.jp/projects/jaqket/) (License CC BY-SA 4.0 - Other licenses are required for corporate usage)

**RC (Reading Comprehension)**

* `JSQuAD`, Japanese version of SQuAD (part of JGLUE) [Source](https://github.com/yahoojapan/JGLUE) (License CC BY-SA 4.0)

**MC (Multiple Choice question answering)**

* `JCommonsenseMorality`, Japanese dataset for evaluating commonsense morality understanding [Source](https://github.com/Language-Media-Lab/commonsense-moral-ja) (License MIT License)

* `JCommonsenseQA`, Japanese version of CommonsenseQA [Source](https://github.com/yahoojapan/JGLUE) (License CC BY-SA 4.0)

* `KUCI`, Kyoto University Commonsense Inference dataset [Source](https://github.com/ku-nlp/KUCI (License CC BY-SA 4.0)

**EL (Entity Linking)**

* `chABSA`, Aspect-Based Sentiment Analysis dataset [Source](https://github.com/chakki-works/chABSA-dataset) (License CC BY-SA 4.0)

**FA (Fundamental Analysis)**

* `Wikipedia Annotated Corpus`, [Source](https://github.com/ku-nlp/WikipediaAnnotatedCorpus) (License CC BY-SA 4.0)

List of tasks: (Reading Prediction, Named-entity recognition (NER), Dependency Parsing, Predicate-argument structure analysis (PAS), Coreference Resolution)

**MR (Mathematical Reasoning)**

* `MAWPS`, Japanese version of MAWPS (A Math Word Problem Repository) [Source](https://github.com/nlp-waseda/chain-of-thought-ja-dataset) (License Apache-2.0)

* `MGSM`, Japanese part of MGSM (Multilingual Grade School Math Benchmark) [Source](https://huggingface.co/datasets/juletxara/mgsm) (License MIT License)

**MT (Machine Translation)**

* `ALT`, Asian Language Treebank (ALT) - Parallel Corpus [Source](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/index.html) (License CC BY-SA 4.0)

* `WikiCorpus`, Japanese-English Bilingual Corpus of Wikipedia's articles about the city of Kyoto [Source](https://alaginrc.nict.go.jp/WikiCorpus/) (License CC BY-SA 3.0)

**STS (Semantic Textual Similarity)**

This task is supported by llm-jp-eval, but it is not included in the evaluation score average.

* `JSTS`,  Japanese version of the STS (Semantic Textual Similarity) (part of JGLUE) [Source](https://github.com/yahoojapan/JGLUE) (License CC BY-SA 4.0)

**HE (Human Examination)**

* `MMLU`, Measuring Massive Multitask Language Understanding [Source](https://github.com/hendrycks/test) (License MIT License)

* `JMMLU`, Japanese Massive Multitask Language Understanding Benchmark [Source](https://github.com/nlp-waseda/JMMLU) (License CC BY-SA 4.0 (3 tasks under the CC BY-NC-ND 4.0 license)

**CG (Code Generation)**

* `MBPP`, Japanese version of Mostly Basic Python Problems (MBPP) [Source](https://huggingface.co/datasets/llm-jp/mbpp-ja) (License CC BY-SA 4.0)

**SUM (Summarization)**

* `XL-Sum`, XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages [Source](https://github.com/csebuetnlp/xl-sum) (License CC BY-NC-SA 4.0, due to the non-commercial license, this dataset will not be used, unless you specifically agree to the license and terms of use)


## Reproducibility
To reproduce our results, please follow the instructions of the evalution tool, **llm-jp-eval** available in [Japanese](https://github.com/llm-jp/llm-jp-eval/blob/main/README.md) and in [English](https://github.com/llm-jp/llm-jp-eval/blob/main/README_en.md).

## Average Score Calculation
The calculation of the average score (AVG) includes only the scores of datasets marked with a ⭐.

"""

LLM_BENCHMARKS_TEXT_JA = """
## 仕組み
📈 評価ツール [llm-jp-eval](https://github.com/llm-jp/llm-jp-eval) を活用し、16種類のタスクで日本語の大規模言語モデルを評価します。このツールは、様々な評価タスクで日本語LLMを評価するための統一的なフレームワークです。

**NLI（自然言語推論）**

* `Jamp`、時間推論に焦点を当てた日本語NLIベンチマーク [ソース](https://github.com/tomo-ut/temporalNLI_dataset)（ライセンス CC BY-SA 4.0）

* `JaNLI`、日本語の敵対的推論データセット [ソース](https://github.com/verypluming/JaNLI)（ライセンス CC BY-SA 4.0）

* `JNLI`、日本語自然言語推論（JGLUEの一部）[ソース](https://github.com/yahoojapan/JGLUE)（ライセンス CC BY-SA 4.0）

* `JSeM`、日本語意味論テストセット [ソース](https://github.com/DaisukeBekki/JSeM)（ライセンス BSD 3-Clause）

* `JSICK`、構成的知識を含む日本語文データセット [ソース](https://github.com/verypluming/JSICK)（ライセンス CC BY-SA 4.0）

**QA（質問応答）**

* `JEMHopQA`、日本語の説明可能なマルチホップ質問応答 [ソース](https://github.com/aiishii/JEMHopQA)（ライセンス CC BY-SA 4.0）

* `NIILC`、NIILC質問応答データセット [ソース](https://github.com/mynlp/niilc-qa)（ライセンス CC BY-SA 4.0）

* `JAQKET`、クイズを題材とした日本語QAデータセット [ソース](https://www.nlp.ecei.tohoku.ac.jp/projects/jaqket/)（ライセンス CC BY-SA 4.0 - 企業利用には別途ライセンスが必要）

**RC（読解）**

* `JSQuAD`、SQuADの日本語版（JGLUEの一部）[ソース](https://github.com/yahoojapan/JGLUE)（ライセンス CC BY-SA 4.0）

**MC（選択式質問応答）**

* `JCommonsenseMorality`、常識的な道徳理解を評価する日本語データセット [ソース](https://github.com/Language-Media-Lab/commonsense-moral-ja)（ライセンス MIT License）

* `JCommonsenseQA`、CommonsenseQAの日本語版 [ソース](https://github.com/yahoojapan/JGLUE)（ライセンス CC BY-SA 4.0）

* `KUCI`、京都大学常識推論データセット [ソース](https://github.com/ku-nlp/KUCI)（ライセンス CC BY-SA 4.0）

**EL（エンティティリンキング）**

* `chABSA`、アスペクトベースの感情分析データセット [ソース](https://github.com/chakki-works/chABSA-dataset)（ライセンス CC BY-SA 4.0）

**FA（基礎解析）**

* `Wikipedia Annotated Corpus`、[ソース](https://github.com/ku-nlp/WikipediaAnnotatedCorpus)（ライセンス CC BY-SA 4.0）

タスク一覧：（読解予測、固有表現認識（NER）、依存構造解析、述語項構造解析（PAS）、共参照解析）

**MR（数学的推論）**

* `MAWPS`、MAWPS（A Math Word Problem Repository）の日本語版 [ソース](https://github.com/nlp-waseda/chain-of-thought-ja-dataset)（ライセンス Apache-2.0）

* `MGSM`、MGSM（Multilingual Grade School Math Benchmark）の日本語部分 [ソース](https://huggingface.co/datasets/juletxara/mgsm)（ライセンス MIT License）

**MT（機械翻訳）**

* `ALT`、アジア言語ツリーバンク（ALT） - 並行コーパス [ソース](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/index.html)（ライセンス CC BY-SA 4.0）

* `WikiCorpus`、京都市に関するWikipedia記事の日本語-英語対訳コーパス [ソース](https://alaginrc.nict.go.jp/WikiCorpus/)（ライセンス CC BY-SA 3.0）

**STS（意味的テキスト類似度）**

このタスクはllm-jp-evalでサポートされていますが、平均スコア (AVG) の計算には含まれていません。

* `JSTS`、STS（Semantic Textual Similarity）の日本語版（JGLUEの一部）[ソース](https://github.com/yahoojapan/JGLUE)（ライセンス CC BY-SA 4.0）

**HE（試験問題）**

* `MMLU`、大規模マルチタスク言語理解ベンチマーク（英語） [ソース](https://github.com/hendrycks/test)（ライセンス MIT License）

* `JMMLU`、日本語大規模マルチタスク言語理解ベンチマーク [ソース](https://github.com/nlp-waseda/JMMLU)（ライセンス CC BY-SA 4.0（3つのタスクはCC BY-NC-ND 4.0ライセンス）

**CG（コード生成）**

* `MBPP`、Mostly Basic Python Problems（MBPP）の日本語版 [ソース](https://huggingface.co/datasets/llm-jp/mbpp-ja)（ライセンス CC BY-SA 4.0）

**SUM（要約）**

* `XL-Sum`、44言語の大規模多言語抽象型要約データセットの日本語部分 [ソース](https://github.com/csebuetnlp/xl-sum)（ライセンス CC BY-NC-SA 4.0、非商用ライセンスのため、このデータセットは使用しません。ライセンスと利用規約に明確に同意した場合を除きます）

## 再現性
結果を再現するには、評価ツール **llm-jp-eval** の指示に従ってください。詳細は [日本語](https://github.com/llm-jp/llm-jp-eval/blob/main/README.md) と [英語](https://github.com/llm-jp/llm-jp-eval/blob/main/README_en.md) でご覧いただけます。

## 平均スコアの計算について
平均スコア (AVG) の計算には、⭐マークのついたスコアのみが含まれます

"""


EVALUATION_QUEUE_TEXT = """
## First Steps Before Submitting a Model
### 1. Ensure Your Model Loads with AutoClasses
Verify that you can load your model and tokenizer using AutoClasses:
```python
from transformers import AutoConfig, AutoModel, AutoTokenizer
config = AutoConfig.from_pretrained("your model name", revision=revision)
model = AutoModel.from_pretrained("your model name", revision=revision)
tokenizer = AutoTokenizer.from_pretrained("your model name", revision=revision)
```
Note:
- If this step fails, debug your model before submitting.
- Ensure your model is public.
- Models requiring `use_remote_code=True` are not currently supported.
### 2. Convert Weights to Safetensors
[Safetensors](https://huggingface.co/docs/safetensors/index) is a new format for storing weights which is safer and faster to load and use. It will also allow us to add the number of parameters of your model to the `Extended Viewer`!
### 3. Verify Your Model Open License
This is a leaderboard for Open LLMs, and we'd love for as many people as possible to know they can use your model 🤗
### 4. Complete Your Model Card
When we add extra information about models to the leaderboard, it will be automatically taken from the model card
### 5. Select Appropriate Precision
The "auto" option supports fp16, fp32, and bf16 precisions. If your model uses any other precision format, please select the appropriate option.
If auto is specified, precision in config.json is automatically selected.
### Note about large models
Currently, we support models up to 70B parameters. However, we are working on infrastructure improvements to accommodate larger models (70B+) in the near future. Stay tuned for updates!

"""
EVALUATION_QUEUE_TEXT_JA = """
## モデル提出前の最初のステップ
### 1. AutoClasses でモデルが読み込めることを確認
AutoClasses を使用してモデルとトークナイザーを読み込めることを確認してください:
```python
from transformers import AutoConfig, AutoModel, AutoTokenizer
config = AutoConfig.from_pretrained("your model name", revision=revision)
model = AutoModel.from_pretrained("your model name", revision=revision)
tokenizer = AutoTokenizer.from_pretrained("your model name", revision=revision)
```
注意:
- この手順が失敗する場合は、提出前にモデルをデバッグしてください。
- モデルが公開されていることを確認してください。
- `use_remote_code=True` を必要とするモデルは現時点ではサポートされていません。

### 2. 重みを Safetensors に変換
[Safetensors](https://huggingface.co/docs/safetensors/index) は、より安全で高速に読み込めるウェイトの新しい保存形式です。これにより、`Extended Viewer` にモデルのパラメータ数を追加することも可能になります！

### 3. モデルのオープンライセンスを確認
これはオープン LLM のリーダーボードです。できるだけ多くの人があなたのモデルを使用できることを知ってもらえると嬉しいです🤗

### 4. モデルカードを完成させる
リーダーボードにモデルの追加情報を掲載する際は、モデルカードから自動的に情報が取得されます

### 5. 適切なPrecisionの選択
"auto"オプションはfp16、fp32、bf16のprecisionに対応しています。これら以外のprecisionを使用している場合は、適切なオプションを選択してください。
また、autoを指定した場合、config.jsonのprecisionが自動的に選択されます。

### 大規模モデルに関する注意
現在、70Bパラメータまでのモデルをサポートしています。より大規模なモデル（70Bよりも大きいもの）については、インフラストラクチャの改善を進めており、近い将来対応予定です。続報をお待ちください！

"""

BOTTOM_LOGO = """
<div style="display: flex; flex-direction: row; justify-content: center; align-items: center;">
  <a href="https://llm-jp.nii.ac.jp/en/" style="margin: 0 10px;">
    <img src="https://raw.githubusercontent.com/AkimfromParis/akimfromparis/refs/heads/main/images/LLM-jp-Logo-Oct-2024.png" alt="LLM-jp" style="max-height: 100px;">
  </a>
  <a href="https://mdx.jp/" style="margin: 0 10px;">
    <img src="https://raw.githubusercontent.com/AkimfromParis/akimfromparis/refs/heads/main/images/MDX-Logo-Oct-2024.jpg" alt="MDX" style="max-height: 100px;">
  </a>
  <a href="https://huggingface.co/" style="margin: 0 10px;">
    <img src="https://raw.githubusercontent.com/AkimfromParis/akimfromparis/refs/heads/main/images/HuggingFace-Logo-Oct-2024.png" alt="HuggingFace" style="max-height: 100px;">
  </a>
</div>
"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_LABEL_JA = "引用の際は、次のスニペットをコピーしてご利用ください"

CITATION_BUTTON_TEXT = r"""@misc{OJLL,
  author = {Miyao, Yusuke and Ishida, Shigeki and Okamoto, Takumi and Han, Namgi and Mousterou, Akim and Fourrier, Clémentine and Hayashi, Toshihiro and Tachibana, Yuichiro},
  title = {Open Japanese LLM Leaderboard},
  year = {2024},
  publisher = {OJLL},
  howpublished = "\url{https://huggingface.co/spaces/llm-jp/open-japanese-llm-leaderboard}"
}
@misc{llmjp2024llmjpcrossorganizationalprojectresearch,
      title={LLM-jp: A Cross-organizational Project for the Research and Development of Fully Open Japanese LLMs},
      author={LLM-jp and : and Akiko Aizawa and Eiji Aramaki and Bowen Chen and Fei Cheng and Hiroyuki Deguchi and Rintaro Enomoto and Kazuki Fujii and Kensuke Fukumoto and Takuya Fukushima and Namgi Han and Yuto Harada and Chikara Hashimoto and Tatsuya Hiraoka and Shohei Hisada and Sosuke Hosokawa and Lu Jie and Keisuke Kamata and Teruhito Kanazawa and Hiroki Kanezashi and Hiroshi Kataoka and Satoru Katsumata and Daisuke Kawahara and Seiya Kawano and Atsushi Keyaki and Keisuke Kiryu and Hirokazu Kiyomaru and Takashi Kodama and Takahiro Kubo and Yohei Kuga and Ryoma Kumon and Shuhei Kurita and Sadao Kurohashi and Conglong Li and Taiki Maekawa and Hiroshi Matsuda and Yusuke Miyao and Kentaro Mizuki and Sakae Mizuki and Yugo Murawaki and Ryo Nakamura and Taishi Nakamura and Kouta Nakayama and Tomoka Nakazato and Takuro Niitsuma and Jiro Nishitoba and Yusuke Oda and Hayato Ogawa and Takumi Okamoto and Naoaki Okazaki and Yohei Oseki and Shintaro Ozaki and Koki Ryu and Rafal Rzepka and Keisuke Sakaguchi and Shota Sasaki and Satoshi Sekine and Kohei Suda and Saku Sugawara and Issa Sugiura and Hiroaki Sugiyama and Hisami Suzuki and Jun Suzuki and Toyotaro Suzumura and Kensuke Tachibana and Yu Takagi and Kyosuke Takami and Koichi Takeda and Masashi Takeshita and Masahiro Tanaka and Kenjiro Taura and Arseny Tolmachev and Nobuhiro Ueda and Zhen Wan and Shuntaro Yada and Sakiko Yahata and Yuya Yamamoto and Yusuke Yamauchi and Hitomi Yanaka and Rio Yokota and Koichiro Yoshino},
      year={2024},
      eprint={2407.03963},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.03963},
}
"""
