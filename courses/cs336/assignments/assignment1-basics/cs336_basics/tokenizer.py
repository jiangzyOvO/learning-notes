import regex

PRETOKEN_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

class Tokenizer:

    def __init__(
        self,
        vocab : dict[int, bytes],
        merges : list[tuple[bytes, bytes]],
        special_tokens : list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))

        # 如果不存在词表中，就加进去
        existing_tokens = set(self.vocab.values())
        next_id = max(self.vocab, default = -1) + 1

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in existing_tokens:
                self.vocab[next_id] = token_bytes
                existing_tokens.add(token_bytes)
                next_id += 1

        self.token_to_id = {
            token_bytes : token_id
            for token_id, token_bytes in self.vocab.items()
        }

    def encode(self, text : str) -> list[int]:
        special_set = set(self.special_tokens)

        if special_set:
            specials = sorted(special_set, key = len, reverse = True)
            pattern = "(" + "|".join(regex.escape(token) for token in specials) + ")"
            parts = regex.split(pattern, text)
        else:
            parts = [text]

        merges_rank = {
            pair : rank
            for rank, pair in enumerate(self.merges)
        }

        ids = []
        for part in parts:
            if part in special_set:
                ids.append(self.token_to_id[part.encode("utf-8")])
                continue

            for match in PRETOKEN_PATTERN.finditer(part):
                token_bytes = match.group().encode("utf-8")
                tokens = [bytes([value]) for value in token_bytes]

                while len(tokens) > 1:
                    best_pair = None
                    best_rank = float("inf")

                    for pair in zip(tokens[: -1], tokens[1 :]):
                        rank = merges_rank.get(pair)
                        if rank is not None and rank < best_rank:
                            best_pair = pair
                            best_rank = rank

                    if best_pair == None:
                        break

                    next_tokens = []
                    i = 0

                    while i < len(tokens):
                        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair:
                            next_tokens.append(tokens[i] + tokens[i + 1])
                            i += 2
                        else:
                            next_tokens.append(tokens[i])
                            i += 1

                    tokens = next_tokens

                ids.extend(self.token_to_id[token] for token in tokens)

        return ids

    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids : list[int]) -> str:
        token_bytes = [self.vocab[token_id] for token_id in ids]
        complete_bytes = b"".join(token_bytes)
        return complete_bytes.decode("utf-8", errors = "replace")
        