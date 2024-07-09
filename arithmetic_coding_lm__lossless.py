from typing import Iterator

import argparse
import numpy as np
import torch


from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import bits_to_bytes, bytes_to_bits, normalize_pdf_for_arithmetic_coding
from helpers import Encoder, Decoder

CHUNK_SIZE_BYTES = 1024

# Base 2 means that the coder writes bits.
ARITHMETIC_CODER_BASE = 2
# Precision 32 implies 32 bit arithmetic.
ARITHMETIC_CODER_PRECISION = 32

class ArithmeticCoder:
    # Helpful links:
    #   > https://www.cs.cmu.edu/~aarti/Class/10704/Intro_Arith_coding.pdf
    #   > https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html
    #   > https://www.cs.ucf.edu/courses/cap5015/Arithmetic%20Coding%20note%202004.pdf

    def __init__(self, lm, tokenizer):
        self.lm = lm
        self.tokenizer = tokenizer
    
    def _next_token_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert len(input_ids.shape) == 2, f"can't get probs for input_ids shape {input_ids.shape}"
        with torch.no_grad():
            output = self.lm(input_ids)
        probs = output.logits.to(torch.float32).log_softmax(dim=-1).exp()
        print("PROBS SUM:", probs.sum())
        return probs.cpu().numpy()


    def encode(
        self,
        data: str,
        return_num_padded_bits: bool = False,
        use_slow_lossless_compression: bool = False,
    ) -> bytes | tuple[bytes, int]:
        """Compresses the `data` using arithmetic coding and a pretrained model.

        Args:
            data: The data to be compressed.
            return_num_padded_bits: Whether to return the number of zeros added to the
            encoded bitstream in order to make it byte-decodeable (i.e., divisible by
            8). Usually, this is used when the encoded data has to be decoded again.
            use_slow_lossless_compression: Whether to compute the `pdf`s for all tokens
            in the data stream in one go or separately for every proper subsequence.
            When only compressing data (i.e., without decompression) use the first
            approach (i.e., `True`) since it has an O(n) runtime complexity, while the
            latter is O(n^2). However, the goal is to losslessly decompress the
            compressed output, use the second option (i.e., `False`) since this is
            what happens in the decoder (which iteratively reconstructs the sequence).

        Returns:
            The compressed data.
        """

        # Convert the `data` into an array of integers (representing the bytes).
        sequence_array = self.tokenizer(data, return_tensors='pt').input_ids
        sequence_array = torch.cat(
            [
                torch.tensor([self.tokenizer.bos_token_id]),
                sequence_array.flatten(),
            ]
        )
        print("Tokens:", data, "//", sequence_array)

        if use_slow_lossless_compression:
            log_probs = list()
            for subsequence_length in range(len(sequence_array)):
                subsequence_probs = self._next_token_probs(
                    sequence_array[None, : subsequence_length + 1]
                )
                log_probs.append(subsequence_probs[0, -1])
                probs = np.vstack(log_probs)
        else:
            probs = self._next_token_probs(sequence_array)
        print("probs.shape:", probs.shape, "sequence_array.shape", sequence_array.shape)

        output = list()
        encoder = Encoder(
            base=ARITHMETIC_CODER_BASE,
            precision=ARITHMETIC_CODER_PRECISION,
            output_fn=output.append,
        )
        print("iterating", probs.shape, "?", sequence_array.shape)
        for pdf, symbol in zip(probs[:, :], sequence_array[:]):
            print("symbol:", symbol.item(), "pdf argmax:", pdf.argmax())
            encoder.encode(normalize_pdf_for_arithmetic_coding(pdf), symbol.item())
        encoder.terminate()

        compressed_bits = ''.join(map(str, output))
        compressed_bytes, num_padded_bits = bits_to_bytes(compressed_bits)

        if return_num_padded_bits:
            return compressed_bytes, num_padded_bits

        return compressed_bytes


    def decode(
            self,
            data: bytes,
            num_padded_bits: int = 0,
            uncompressed_length: int = CHUNK_SIZE_BYTES,
        ) -> bytes:
        """Decompresses the `data` using arithmetic coding and a pretrained model.

        See https://en.wikipedia.org/wiki/Arithmetic_coding for details.

        Args:
            data: The data to be decompressed.
            num_padded_bits: The number of zeros added to the encoded bitstream in order
            to make it byte-decodeable (i.e., divisble by 8).
            uncompressed_length: The length of the original data stream (in bytes).

        Returns:
            The decompressed data.
        """
        data_iter = iter(bytes_to_bits(data, num_padded_bits=num_padded_bits))

        # The decoder requires a function that reads digits from {0, 1, ..., base - 1}
        # from the compressed input and returns `None` when the input is exhausted.
        def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
            try:
                return int(next(bit_sequence))
            except StopIteration:
                return None

        decoder = Decoder(
            base=ARITHMETIC_CODER_BASE,
            precision=ARITHMETIC_CODER_PRECISION,
            input_fn=_input_fn,
        )
        # We need a dummy token because the language model right-shifts the sequence
        # by onde when computing the conditional probabilities. Concretely, at every
        # step, we need the `pdf` of the next token given all currently decompressed
        # tokens, but without a dummy token, the last `pdf` would be that of the last
        # already decompressed token. The value of the dummy token is irrelevant.
        sequence_array = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.int32)
        print("3 >> sequence_array.shape", sequence_array.shape)
        probs = self._next_token_probs(sequence_array[None])[0]

        idx = 0
        while True:
            print("idx", idx, "probs.shape", probs.shape, "/ argmax", probs.argmax().item(), "sequence_arr", sequence_array)
            print("pre-decode probs.sum:", probs.sum(), probs.dtype)
            try:
                token = decoder.decode(
                    normalize_pdf_for_arithmetic_coding(probs)
                )
                print("decoder.decode() returned token:", token, "w/ argmax", probs[-1].argmax())
                # if token == self.tokenizer.eos_token_id:
                #     raise StopIteration
            except StopIteration:
                break
            print("\t token:", token)
            sequence_array = torch.tensor(
                np.append(sequence_array, token)
                , dtype=torch.int32
            )
            probs = self._next_token_probs(sequence_array[None])[0][-1]
            idx += 1

        # Remove the dummy token and convert to bytes.
        print(f"Decoded {len(sequence_array)} tokens:", sequence_array)
        return self.tokenizer.decode(sequence_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("message", type=str, help="The message to print")
    args = parser.parse_args()

    model = "gpt2"
    lm = AutoModelForCausalLM.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    string = args.message
    coder = ArithmeticCoder(lm=lm, tokenizer=tokenizer)
    print(f"[0] Encoding... `{string}`")
    code, num_padded_bits = coder.encode(
        string, 
        return_num_padded_bits=True, 
        use_slow_lossless_compression=True
    )
    print(f"[1] Code... `{code}` ({len(code)} bytes, num_padded_bits={num_padded_bits})")
    decoded_string = coder.decode(code, num_padded_bits=num_padded_bits)
    print(f"[2] Decoded: {decoded_string}")