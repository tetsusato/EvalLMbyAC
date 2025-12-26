import pytest
from perplexity import Perplexity
from transformers import AutoTokenizer, AutoModelForCausalLM


class TestPerplexity():
    def test_calculate(self):
        poor_model = "HuggingFaceTB/SmolLM2-135M-Instruct"
        model = AutoModelForCausalLM\
                   .from_pretrained(poor_model,
                                    device_map="auto",
                                    )
        tokenizer = AutoTokenizer.from_pretrained(poor_model)

        perplexity = Perplexity(model,
                                tokenizer,
                                )
        score = perplexity.calculate("This is a pen.")
        assert score == pytest.approx(171.1189, 1e-5)
        score = perplexity.calculate("This is a cans and dogs.")
        assert score == pytest.approx(419.3890, 1e-5)
        score = perplexity.calculate("A quick brown fox jumps over the lazy dog.")
        assert score == pytest.approx(19.8726, 1e-5)
        
