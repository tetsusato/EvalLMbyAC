import torch
from torcheval.metrics.functional import perplexity
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

class Perplexity():
    def __init__(self,
                 lm: AutoModelForCausalLM,
                 tokenizer: AutoTokenizer,
                 stride: int=1, # Sliding Windowの移動距離
                 max_length=4096, # 最大入力トークン数
                 ):
        self.model = lm
        self.tokenizer = tokenizer
        self.stride = stride
        self.max_length = max_length

    def calculate(self,
                  msg: str,
                  ) -> float:
        device = "cuda"
        past_key_values = DynamicCache()
        
        encodings = self.tokenizer(msg, return_tensors="pt")
        input_ids = encodings.input_ids

        seq_len = input_ids.size(1) # [batch, len] -> len
        prev_end_loc = 0
        num_loop = 0
        sum_ppl = 0
        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            current_input_ids = input_ids[:, begin_loc:end_loc]
            target_ids = current_input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = self.model(current_input_ids.to(device),
                                     device_map="auto",
                                     use_cache=False,
                                     past_key_values=past_key_values
                                     )
                past_key_values = outputs.past_key_values
                logits = outputs.logits
                ppl = perplexity(logits[:, :-1, :], current_input_ids[:, 1:])
                sum_ppl += ppl.item()
            prev_end_loc = end_loc
            num_loop +=1
            if end_loc == seq_len:
                break

        final_ppl = sum_ppl/num_loop
        return final_ppl
