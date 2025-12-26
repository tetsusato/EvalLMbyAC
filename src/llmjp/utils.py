from dataclasses import dataclass, make_dataclass
from enum import Enum

import pandas as pd

from src.llmjp.about import Tasks, TaskType


def fields(raw_class):
    return [v for k, v in raw_class.__dict__.items() if k[:2] != "__" and k[-2:] != "__"]


# These classes are for user facing column names,
# to avoid having to change them all around the code
# when a modif is needed
@dataclass(frozen=True)
class ColumnContent:
    name: str
    type: str
    displayed_by_default: bool
    hidden: bool = False
    never_hidden: bool = False
    dummy: bool = False
    task_type: TaskType = TaskType.NotTask
    average: bool = False


## Leaderboard columns
auto_eval_column_dict = []
# Init
auto_eval_column_dict.append(["model_type_symbol", ColumnContent, ColumnContent("T", "str", True, never_hidden=True)])
auto_eval_column_dict.append(["model", ColumnContent, ColumnContent("Model", "markdown", True, never_hidden=True)])
# Scores
# auto_eval_column_dict.append(["average", ColumnContent, ColumnContent("Average ⬆️", "number", True)])
for task in Tasks:
    auto_eval_column_dict.append(
        [
            task.name,
            ColumnContent,
            ColumnContent(
                task.value.col_name,
                "number",
                displayed_by_default=(task.value.task_type == TaskType.AVG or task.value.average),
                task_type=task.value.task_type,
                average=task.value.average,
            ),
        ]
    )
# Model information
auto_eval_column_dict.append(["model_type", ColumnContent, ColumnContent("Type", "str", False)])
auto_eval_column_dict.append(["architecture", ColumnContent, ColumnContent("Architecture", "str", False)])
auto_eval_column_dict.append(["weight_type", ColumnContent, ColumnContent("Weight type", "str", False, True)])
auto_eval_column_dict.append(["precision", ColumnContent, ColumnContent("Precision", "str", False)])
auto_eval_column_dict.append(["license", ColumnContent, ColumnContent("Hub License", "str", False)])
auto_eval_column_dict.append(["params", ColumnContent, ColumnContent("#Params (B)", "number", False)])
auto_eval_column_dict.append(["likes", ColumnContent, ColumnContent("Hub ❤️", "number", False)])
auto_eval_column_dict.append(["revision", ColumnContent, ColumnContent("Revision", "str", False, False)])
auto_eval_column_dict.append(["num_few_shots", ColumnContent, ColumnContent("Few-shot", "number", False)])
auto_eval_column_dict.append(["add_special_tokens", ColumnContent, ColumnContent("Add Special Tokens", "bool", False)])
auto_eval_column_dict.append(
    ["llm_jp_eval_version", ColumnContent, ColumnContent("llm-jp-eval version", "str", False)]
)
auto_eval_column_dict.append(["vllm_version", ColumnContent, ColumnContent("vllm version", "str", False)])
auto_eval_column_dict.append(["dummy", ColumnContent, ColumnContent("model_name_for_query", "str", False, dummy=True)])
auto_eval_column_dict.append(["row_id", ColumnContent, ColumnContent("ID", "number", False, dummy=True)])

# We use make dataclass to dynamically fill the scores from Tasks
AutoEvalColumn = make_dataclass("AutoEvalColumn", auto_eval_column_dict, frozen=True)


## For the queue columns in the submission tab
@dataclass(frozen=True)
class EvalQueueColumn:  # Queue column
    model = ColumnContent("model", "markdown", True)
    revision = ColumnContent("revision", "str", True)
    model_type = ColumnContent("model_type", "str", True)
    precision = ColumnContent("precision", "str", True)
    add_special_tokens = ColumnContent("add_special_tokens", "str", True)
    llm_jp_eval_version = ColumnContent("llm_jp_eval_version", "str", True)
    vllm_version = ColumnContent("vllm_version", "str", True)
    status = ColumnContent("status", "str", True)


# This class is used to store the model data in the queue
@dataclass(frozen=True)
class EvalQueuedModel:
    model: str
    revision: str
    precision: str
    add_special_tokens: str
    llm_jp_eval_version: str
    vllm_version: str


## All the model information that we might need
@dataclass
class ModelDetails:
    name: str
    display_name: str = ""
    symbol: str = ""  # emoji


class ModelType(Enum):
    PT = ModelDetails(name="pretrained", symbol="🟢")
    FT = ModelDetails(name="fine-tuned", symbol="🔶")
    IFT = ModelDetails(name="instruction-tuned", symbol="⭕")
    RL = ModelDetails(name="RL-tuned (Preference optimization)", symbol="🟦")
    MM = ModelDetails(name="multimodal", symbol="🌸")
    BM = ModelDetails(name="base merges and moerges", symbol="🤝")

    def to_str(self, separator=" "):
        return f"{self.value.symbol}{separator}{self.value.name}"

    @staticmethod
    def from_str(type):
        if "fine-tuned" in type or "🔶" in type:
            return ModelType.FT
        if "pretrained" in type or "🟢" in type:
            return ModelType.PT
        if "RL-tuned" in type or "🟦" in type:
            return ModelType.RL
        if "instruction-tuned" in type or "⭕" in type:
            return ModelType.IFT
        if "multimodal" in type or "🌸" in type:
            return ModelType.MM
        if "base merges and moerges" in type or "🤝" in type:
            return ModelType.BM
        raise ValueError(f"Unsupported model type: {type}")


class WeightType(Enum):
    Adapter = ModelDetails("Adapter")
    Original = ModelDetails("Original")
    Delta = ModelDetails("Delta")


class Precision(Enum):
    float16 = ModelDetails("float16")
    bfloat16 = ModelDetails("bfloat16")
    float32 = ModelDetails("float32")

    @staticmethod
    def from_str(precision: str) -> "Precision":
        if precision == "float16":
            return Precision.float16
        if precision == "bfloat16":
            return Precision.bfloat16
        if precision == "float32":
            return Precision.float32
        raise ValueError(
            f"Unsupported precision type: {precision}. Please use 'auto' (recommended), 'float32', 'float16', or 'bfloat16'"
        )


class AddSpecialTokens(Enum):
    true = ModelDetails("True")
    false = ModelDetails("False")


class NumFewShots(Enum):
    shots_0 = 0
    shots_4 = 4


class LLMJpEvalVersion(Enum):
    current = ModelDetails("v1.4.1")

    @staticmethod
    def from_str(version: str) -> "LLMJpEvalVersion":
        if version == "1.4.1":
            return LLMJpEvalVersion.current
        raise ValueError(f"Unsupported LLMJpEval version: {version}")


class VllmVersion(Enum):
    current = ModelDetails("v0.6.3.post1")

    @staticmethod
    def from_str(version: str) -> "VllmVersion":
        if version == "v0.6.3.post1":
            return VllmVersion.current
        raise ValueError(f"Unsupported VLLM version: {version}")


# Column selection
COLS = [c.name for c in fields(AutoEvalColumn) if not c.hidden]
TYPES = [c.type for c in fields(AutoEvalColumn)]

EVAL_COLS = [c.name for c in fields(EvalQueueColumn)]
EVAL_TYPES = [c.type for c in fields(EvalQueueColumn)]

BENCHMARK_COLS = [t.value.col_name for t in Tasks]

NUMERIC_INTERVALS = {
    "0~3B": pd.Interval(0, 3, closed="right"),
    "3~7B": pd.Interval(3, 7.3, closed="right"),
    "7~13B": pd.Interval(7.3, 13, closed="right"),
    "13~35B": pd.Interval(13, 35, closed="right"),
    "35~60B": pd.Interval(35, 60, closed="right"),
    "60B+": pd.Interval(60, 10000, closed="right"),
    "?": pd.Interval(-1, 0, closed="right"),
}
