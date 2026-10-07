import os
import regex
from typing import BinaryIO
from collections import Counter

PRETOKEN_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def find_chunk_boundaries(
    file : BinaryIO,
    desired_num_chunks : int,
    split_special_token : bytes,
) -> list[int]:

    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position) 
        while True:
            mini_chunk = file.read(mini_chunk_size)  

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def train_bpe (
    input_path : str,
    vocab_size : int,
    special_tokens : list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # 1. 初始化词表
    vocab = {i : bytes([i]) for i in range(256)}

    special_tokens = list(dict.fromkeys(special_tokens))
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # 2. 分割特殊字符并统计
    special_pattern = regex.compile("|".join(regex.escape(token) for token in special_tokens))

    sequences = Counter()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks = 8,
            split_special_token = special_tokens[0].encode("utf-8")
        )

        for start, end in zip(boundaries[: -1], boundaries[1 :]):
            f.seek(start)
            text = f.read(end - start).decode("utf-8")
            parts = special_pattern.split(text)

            for part in parts:
                for match in PRETOKEN_PATTERN.finditer(part):
                    token_bytes = match.group().encode("utf-8")
                    tokens = tuple(bytes([value]) for value in token_bytes)
                    sequences[tokens] += 1

    # 3.BPE合并
    merges = []
    while len(vocab) < vocab_size:
        counts = Counter()
        for tokens, count in sequences.items():
            for pair in zip(tokens[: -1], tokens[1 :]):
                counts[pair] += count

        if not counts:
            break

        choose = max(counts, key = lambda pair : (counts[pair], pair))
        merges.append(choose)

        left, right = choose
        merged = left + right
        vocab[len(vocab)] = merged

        next_sequences = Counter()
        for tokens, count in sequences.items():
            next_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == left and tokens[i + 1] == right:
                    next_tokens.append(merged)
                    i += 2
                else:
                    next_tokens.append(tokens[i])
                    i += 1
            next_sequences[tuple(next_tokens)] += count
        sequences = next_sequences

    return vocab, merges