from typing import Iterator, Optional, Tuple, Union

import argparse
from decimal import Decimal
from diskcache import Cache
import joblib
import numpy as np
import os
import pickle
import torch
import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, cache_utils, HybridCache, StaticCache

from transformers.cache_utils import DynamicCache, SinkCache, StaticCache, SlidingWindowCache, QuantoQuantizedCache, QuantizedCacheConfig
from .utils import bits_to_bytes, bytes_to_bits, normalize_pdf_for_arithmetic_coding
from .helpers import Encoder, Decoder

from logging import config
import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)
progress = logging.getLogger("progress")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ArithmeticCoder:
    # Helpful links:
    #   > https://github.com/google-deepmind/language_modeling_is_compression
    #   > https://www.cs.cmu.edu/~aarti/Class/10704/Intro_Arith_coding.pdf
    #   > https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html
    #   > https://www.cs.ucf.edu/courses/cap5015/Arithmetic%20Coding%20note%202004.pdf
    # Base 2 means that the coder writes bits.
    ARITHMETIC_CODER_BASE = 2
    # Precision 32 implies 32 bit arithmetic.
    ARITHMETIC_CODER_PRECISION = 32
    #ARITHMETIC_CODER_PRECISION = 56

    def __init__(self, lm, tokenizer, use_cache = True):
        #lm.forward = torch.compile(lm.forward, mode="reduce-overhead", fullgraph=True)
        self.lm = lm
        self.tokenizer = tokenizer
        self.use_cache = use_cache # cache past_key_values
        self.cache_size = 0 # cache size for past_key_values
        #self.cache = Cache("cache")
        self.cache = None
    @property
    def _lm_device(self) -> torch.device:
        return next(self.lm.parameters()).device
    
    #def _next_token_probs(self, input_ids: torch.Tensor, past_key_values: Tuple) -> torch.Tensor:
    def _next_token_probs(self, input_ids: torch.Tensor, past_key_values: DynamicCache) -> torch.Tensor:
        """
          LLMに流してoutputのlogitsを取りnumpyにして返す
        """
        if self.cache is not None:
            # handle cache
            key = input_ids.tolist()
            key = [str(x) for row in key for x in row] # 2d to 1d
            key = ",".join(key)
            val = self.cache.get(key)
            logger.debug(f"Cache key={key}, len(Cache key)={len(key)}")
            if val is not None:
                logger.debug(f"Hit! Cache key={key}")
        logger.debug(f"past_key_values={past_key_values}")
        if past_key_values is not None:
            logger.debug(f"len(past_key_values.keys)={len(past_key_values.key_cache)}")
            logger.debug(f"past_key_values.keys[0].shape={past_key_values.key_cache[0].shape}")
            logger.debug(f"past_key_values.keys[0].shape[2]={past_key_values.key_cache[0].shape[2]}")
            kv_cache_len = past_key_values.key_cache[0].shape[2]
            logger.debug(f"kv_cache_len will be: {kv_cache_len}")
        else:
            logger.debug(f"init past_key_values by DynamicCache()")
            past_key_values = DynamicCache()

        #if val is None:
        #if True:
        if self.cache is None:
            if self.cache is not None:
                logger.debug(f"Cache missed! key={key}")
            #if isinstance(past_key_values, DynamicCache):
            #if isinstance(past_key_values, StaticCache):
            if isinstance(past_key_values, DynamicCache):
                #logger.debug(f"past_key_values.vals.shape={past_key_values.value_cache.shape}")
                #max_cache_length = past_key_values.get_max_length()
                #max_cache_length = 1024
                cache_length = past_key_values.get_seq_length()
                #logger.debug(f"max_cache_length={max_cache_length}")
                #logger.debug(f"current cache length={self.cache_size}")
                #input_ids = input_ids[:, self.cache_size:]
                #logger.debug(f"Change input_ids from {input_ids}")
                input_ids = input_ids[:, cache_length:]
                #logger.debug(f"Change input_ids to {input_ids}")
            elif (past_key_values is not None):
                # HuggingFace doesn't want us to provide input ids for anything that's in the kv cache.
                # We have to trim this part.
                #kv_cache_seq_length = past_key_values[0][0].shape[2]
                kv_cache_seq_length = 1024
                input_ids = input_ids[:, kv_cache_seq_length:]

            assert len(input_ids.shape) == 2, f"can't get probs for input_ids shape {input_ids.shape}"
            logger.debug(f"lm_device={self._lm_device}")
            logger.debug(f"input past_key_values={past_key_values}({past_key_values.__class__})")
            logger.debug(f"input past_key_values.seq_len={past_key_values.get_seq_length()}")
            logger.debug(f"use_cache={self.use_cache}")
            #logger.debug(f"input_ids={input_ids}")
            logger.debug(f"input_ids.shape={input_ids.shape}")
            with torch.no_grad():
                output = self.lm(
                    #input_ids=input_ids.to(self._lm_device),
                    #input_ids=input_ids.to("cuda"),
                    input_ids=input_ids,

                    past_key_values=past_key_values,
                    #use_cache=False,
                    use_cache=self.use_cache,
                    #cache_implementation="dynamic"
                )
            logger.debug(f"output past_key_values={output.past_key_values}({output.past_key_values.__class__})")
            logger.debug(f"logits.shape={output.logits.shape}")
            #probs = output.logits.to(torch.float32).softmax(dim=-1)
            probs = output.logits.to(torch.float64).softmax(dim=-1)
            #return (probs.cpu().numpy(), output.past_key_values)
            logger.debug(f"returned past_key_values [0][0].shape[2]={output.past_key_values[0][0].shape[2]}")
            """
            logger.debug(f"len(returned past_key_values)={len(output.past_key_values)}")
            logger.debug(f"returned past_key_values[0]={output.past_key_values[0]}")
            logger.debug(f"returned past_key_values[0][0]={output.past_key_values[0][0]}")
            logger.debug(f"returned past_key_values[0][0].shape={output.past_key_values[0][0].shape}")
            #logger.debug(f"returned past_key_values[0].shape={output.past_key_values[0].shape}")
            logger.debug(f"returned past_key_values[1]={output.past_key_values[1]}")
            logger.debug(f"returned past_key_values[1][0]={output.past_key_values[1][0]}")
            logger.debug(f"returned past_key_values[1][0].shape={output.past_key_values[1][0].shape}")
            logger.debug(f"returned past_key_values[1][1]={output.past_key_values[1][1]}")
            logger.debug(f"returned past_key_values[1][1].shape={output.past_key_values[1][1].shape}")
            logger.debug(f"returned past_key_values[2]={output.past_key_values[2]}")
            logger.debug(f"returned past_key_values[2][0]={output.past_key_values[2][0]}")
            logger.debug(f"returned past_key_values[2][0].shape={output.past_key_values[2][0].shape}")
            logger.debug(f"returned past_key_values[3].shape={output.past_key_values[3].shape}")
            """
            probs = probs.cpu().numpy()
            #past_key_values = DynamicCache.from_legacy_cache(output.past_key_values)
            past_key_values = output.past_key_values
            #kv_cache_pkl = pickle.dumps(kv_cache)
            #logger.debug(f"kv cache pickle object size={len(kv_cache_pkl)}")
            #val = (probs.cpu().numpy(), kv_cache_pkl)
            val = (probs, past_key_values)
            #logger.debug(f"Cache key={key} Cache val={val}")
            if self.cache is not None:
                logger.debug(f"Set by Cache key={key}")
                self.cache.set(key, val)
        else:
            logger.debug(f"Cache hit!key={key}") # valはでかいので表示しない
            probs = val[0]
            #past_key_values = pickle.loads(val[1]) # DynamicCache
            past_key_values = val[1] # DynamicCache
        return (probs, past_key_values)
        #probs = np.vectorize(Decimal.from_float)(probs.cpu())
        #return (probs, output.past_key_values)
    def _next_token_probs_gemma(self,
                                input_ids: torch.Tensor,
                                past_key_values: HybridCache
                                ) -> torch.Tensor:
        """
          LLMに流してoutputのlogitsを取りnumpyにして返す
        """
        device_map="auto"
        if (past_key_values is not None):
            # HuggingFace doesn't want us to provide input ids for anything that's in the kv cache.
            # We have to trim this part.
            #kv_cache_seq_length = past_key_values[0][0].shape[2]
            #kv_cache_seq_length = past_key_values.max_cache_len
            kv_cache_seq_length = 1024
            input_ids = input_ids[:, kv_cache_seq_length:]
            cache_position = torch.tensor([[0]])

        assert len(input_ids.shape) == 2, f"can't get probs for input_ids shape {input_ids.shape}"

        with torch.no_grad():
            output = self.lm(
                input_ids=input_ids.to(self._lm_device),
                past_key_values=past_key_values,
                use_cache=self.use_cache,
                #use_cache=False,
                cache_position=cache_utils,
                device_map=device_map,
                cache_implementation="dynamic"
            )
        
        probs = output.logits.to(torch.float32).softmax(dim=-1)
        return (probs.cpu().numpy(), output.past_key_values)

    def encode(
        self,
        data: str,
        return_num_padded_bits: bool = False,
    ) -> Union[bytes, tuple[bytes, int]]:
        """Compresses the `data` using arithmetic coding and a pretrained model.

        Args:
            data: The data to be compressed.
            return_num_padded_bits: Whether to return the number of zeros added to the
                encoded bitstream in order to make it byte-decodeable (i.e., divisible by
                8). Usually, this is used when the encoded data has to be decoded again.

Returns:
        

    The compressed data.
        """
        logger.info(f"encode start")
        logger.debug(f"encode start")
        input_ids_tensor = self.tokenizer(data, return_tensors='pt').input_ids
        #print(f"seq array.shape={input_ids_tensor.shape}") # (1, token数)
        #print(f"tokens={input_ids_tensor[0].tolist()}")
        #print(f"tokens={self.tokenizer.convert_ids_to_tokens(input_ids_tensor[0].tolist())}")
        if "qwen" in str(self.lm.__class__) :
            input_ids_tensor = torch.cat(
                [
                    torch.tensor([151643]), # <|endoftext|>
                    input_ids_tensor.flatten(),
                ]
            )
        else:
            input_ids_tensor = torch.cat(
                [
                    torch.tensor([self.tokenizer.bos_token_id]),
                    input_ids_tensor.flatten(),
                ]
            )
        # 先頭にBOSを追加するだけ？
        #print(f"new seq array={input_ids_tensor}")
        log_probs = [] 
        past_key_values = None
        max_generated_length = len(data)+10
        #print(f"max_generated_length={max_generated_length}")
        #past_key_values = HybridCache(config=self.lm.model.config,
        #                              batch_size=1,
        #                              max_cache_len=max_generated_length)
        #past_key_values = DynamicCache()
        past_key_values = None
        #past_key_values = QuantoQuantizedCache(
        #                      cache_config=QuantizedCacheConfig()
        #                  )
        #max_cache_length = past_key_values.get_max_length()

        logger.debug("Writing probs...")
        with open("log_probs.npy", "wb") as fp:
            for subsequence_length in tqdm.trange(len(input_ids_tensor), leave=False):
                #logger.debug(f"input_ids={input_ids_tensor[None, : subsequence_length + 1]}")
                subsequence_probs, past_key_values = self._next_token_probs(
                #subsequence_probs, past_key_values = self._next_token_probs_gemma(
                    input_ids=input_ids_tensor[None, : subsequence_length + 1],
                    past_key_values=past_key_values
                )
                #print(f"subsequence_probs={subsequence_probs}")
                #print(f"past_key_values={past_key_values}")
                #log_probs.append(subsequence_probs[0, -1])
                np.save(fp, subsequence_probs[0, -1])
                #print(f"log_probs={log_probs}, len={len(log_probs)}")
                #print(f"probs={probs}, shape={probs.shape}")
                logger.debug(f"past_key_values={past_key_values}")

        
        #probs = np.vstack(log_probs)

        logger.debug("Reading probs...")
        probs = []
        with open("log_probs.npy", "rb") as fp:
            for i in tqdm.trange(len(input_ids_tensor)):
                subsequence_probs = np.load(fp)
                probs.append(subsequence_probs)
        probs = np.vstack(probs)
        output = list()
        encoder = Encoder(
            base=ArithmeticCoder.ARITHMETIC_CODER_BASE,
            precision=ArithmeticCoder.ARITHMETIC_CODER_PRECISION,
            output_fn=output.append,
        )
        #print(f"probs.shape={probs.shape}") # [token数, vocab数](numpy)
        #print(f"input_ids_tensor.shape={input_ids_tensor.shape}") # [token数]
        normalize_pdf_array =  joblib.Parallel(n_jobs=-1, prefer="threads")(
                                   joblib.delayed(normalize_pdf_for_arithmetic_coding)
                                   (probs[i,])
                                   for i in range(len(probs[:,]))
                               )
        normalize_pdf_array = np.array(normalize_pdf_array)
        print(f"normalize_pdf_array={normalize_pdf_array}")
        #for pdf, symbol in zip(probs[:,], input_ids_tensor[1:]):
        for pdf, symbol in tqdm.tqdm(zip(normalize_pdf_array[:,], input_ids_tensor[1:])):
            #print(f"pdf={pdf}, pdf.shape={pdf.shape}")
            #print(f"symbol={symbol}")
            encoder.encode(pdf, symbol.item())
            #logger.debug(f"current output={output}")
        encoder.terminate()

        logger.debug(f"output={output}, len={len(output)}")
        compressed_bits = ''.join(map(str, output))
        logger.debug(f"output bits={compressed_bits}")
        compressed_bytes, num_padded_bits = bits_to_bytes(compressed_bits)

        if return_num_padded_bits:
            return compressed_bytes, num_padded_bits
        else:
            return compressed_bytes


    def decode(
            self,
            data: bytes,
            num_padded_bits: int = 0,
            skip_special_tokens: bool = True,
        ) -> bytes:
        """Decompresses the `data` using arithmetic coding and a pretrained model.

        See https://en.wikipedia.org/wiki/Arithmetic_coding for details.

        Args:
            data: The data to be decompressed.
            num_padded_bits: The number of zeros added to the encoded bitstream in order
            to make it byte-decodeable (i.e., divisble by 8).
            skip_special_tokens: Whether to filter out e.g. <eos> in tokens-to-string
                conversion.

        Returns:
            The decompressed data.
        """
        logger.info(f"decode start")
        logger.debug(f"decode start")
        logger.debug(f"num padding bits={num_padded_bits}")
        data_iter = iter(bytes_to_bits(data, num_padded_bits=num_padded_bits))

        # The decoder requires a function that reads digits from {0, 1, ..., base - 1}
        # from the compressed input and returns `None` when the input is exhausted.
        def _input_fn(bit_sequence: Iterator[str] = data_iter) -> Optional[int]:
            try:
                return int(next(bit_sequence))
            except StopIteration:
                return None

        decoder = Decoder(
            base=ArithmeticCoder.ARITHMETIC_CODER_BASE,
            precision=ArithmeticCoder.ARITHMETIC_CODER_PRECISION,
            input_fn=_input_fn,
        )
        # We need a dummy token because the language model right-shifts the sequence
        # by onde when computing the conditional probabilities. Concretely, at every
        # step, we need the `pdf` of the next token given all currently decompressed
        # tokens, but without a dummy token, the last `pdf` would be that of the last
        # already decompressed token. The value of the dummy token is irrelevant.
        if "qwen" in str(self.lm.__class__):
            sequence_array = torch.tensor([151643], dtype=torch.int32)
        else:
            sequence_array = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.int32)            
        # print("3 >> sequence_array.shape", sequence_array.shape)
        probs, past_key_values = self._next_token_probs(
            input_ids=sequence_array[None], 
            past_key_values=None
        )
        probs = probs[0, 0]

        idx = 0
        is_success=True
        while True:
            #print(f"probs={probs}")
            #logger.debug(f"probs={probs}, shape={probs.shape}")
            #probs = np.vectorize(Decimal.from_float)(probs)
            #logger.debug(f"new probs={probs}, shape={probs.shape}")
            # print("idx", idx, "probs.shape", probs.shape, "/ argmax", probs.argmax().item(), "sequence_arr", sequence_array)

            try:
                token = decoder.decode(
                    normalize_pdf_for_arithmetic_coding(probs)
                )
            except StopIteration:
                break
            except AssertionError as e:
                logger.info(f"AssertionError: {e}")
                progress.info(f"AssertionError: {e}")
                logger.debug(f"AssertionError: {e}"
                             +f"sequence_array={sequence_array}")
                is_success=False
                break
            # print("\t token:", token)
            logger.debug(f"token id={token}, token={self.tokenizer.convert_ids_to_tokens([token])}")
            sequence_array = torch.tensor(
                np.append(sequence_array, token)
                , dtype=torch.int32
            )
            probs, past_key_values = self._next_token_probs(sequence_array[None], past_key_values=past_key_values)
            probs = probs[0, -1]
            idx += 1

        # Remove the dummy token and convert to bytes.
        print(f"Decoded {len(sequence_array)} tokens:", sequence_array)
        decoded_string = self.tokenizer.decode(sequence_array,
                                               skip_special_tokens=skip_special_tokens)
        return decoded_string, is_success

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("message", type=str, help="The message to print")
    args = parser.parse_args()

    #model = "gpt2"
    model = "nvidia--Llama-3.1-Nemotron-70B-Instruct-HF"
    lm = AutoModelForCausalLM.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    string = args.message
    msg = """
    Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?”

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, “Oh dear! Oh dear! I shall be late!” (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.
    """
    
    coder = ArithmeticCoder(lm=lm, tokenizer=tokenizer)
    print(f"[0] Encoding... `{msg}`")
    code, num_padded_bits = coder.encode(
        msg, 
        return_num_padded_bits=True, 
    )
    print(f"[1] Code... `{code}` ({len(code)} bytes, num_padded_bits={num_padded_bits})")
    print("\n" * 5)
    decoded_string = coder.decode(code, num_padded_bits=num_padded_bits)
    print(f"[2] Decoded: {decoded_string}")
