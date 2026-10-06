"""Lecture 1 teaching examples. Run with Python 3; no third-party packages.

Simplified byte-level BPE, adapted conceptually from the official lecture.
No pre-tokenization, special tokens, or production optimizations.
"""
from collections import Counter
from math import log


def merge(ids, pair, new_id):
    result = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result


def train_bpe(text, num_merges):
    ids = list(text.encode("utf-8"))
    vocab = {i: bytes([i]) for i in range(256)}
    rules = []
    trace = []
    for _ in range(num_merges):
        counts = Counter(zip(ids, ids[1:]))
        if not counts:
            break
        # Equal counts: choose the first pair encountered, as in the lecture.
        pair = max(counts, key=counts.get)
        new_id = len(vocab)
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        rules.append((pair, new_id))
        ids = merge(ids, pair, new_id)
        trace.append((pair, counts[pair], new_id, list(ids)))
    return vocab, rules, trace


def encode(text, rules):
    ids = list(text.encode("utf-8"))
    for pair, new_id in rules:
        ids = merge(ids, pair, new_id)
    return ids


def decode(ids, vocab):
    # Concatenate bytes BEFORE UTF-8 decoding: one token may be a partial character.
    return b"".join(vocab[i] for i in ids).decode("utf-8")


def bytes_per_token(text, ids):
    if not ids:
        raise ValueError("Bytes per token is undefined for empty input")
    return len(text.encode("utf-8")) / len(ids)


def main():
    text = "A你🌍"
    print("Unicode code points:", [ord(c) for c in text])
    print("UTF-8 bytes:", list(text.encode("utf-8")))
    training_text = "the cat in the hat"
    vocab, rules, trace = train_bpe(training_text, 3)
    for pair, count, new_id, ids in trace:
        print("merge:", pair, "count:", count, "new ID:", new_id,
              "bytes:", vocab[new_id], "tokens:", len(ids))
    for sample in [training_text, "the quick brown fox", "A你🌍", "", "aaa", "hello\nworld"]:
        ids = encode(sample, rules)
        recovered = decode(ids, vocab)
        assert recovered == sample
        print(repr(sample), "->", ids, "->", repr(recovered))
        if ids:
            print("  bytes/token:", round(bytes_per_token(sample, ids), 4))
    # Check the hand-worked example and overlapping-pair behavior.
    assert [vocab[new_id] for _, new_id in rules] == [b"th", b"the", b"the "]
    assert len(encode(training_text, rules)) == 12
    assert len(encode("the quick brown fox", rules)) == 16
    assert merge([97, 97, 97], (97, 97), 256) == [256, 97]
    singleton_vocab, singleton_rules, _ = train_bpe("x", 10)
    assert decode(encode("x", singleton_rules), singleton_vocab) == "x"
    print("Teaching probability example: NLL =", round(-log(0.5 * 0.8 * 0.25), 6))
    print("All example checks passed.")


if __name__ == "__main__":
    main()
