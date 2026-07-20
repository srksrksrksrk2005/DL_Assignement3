from collections import Counter
from typing import Iterable, Optional

import spacy
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN = SPECIAL_TOKENS


class Vocab:
    def __init__(
        self,
        counter: Optional[Counter] = None,
        min_freq: int = 2,
        specials: Optional[list[str]] = None,
    ) -> None:
        specials = specials or SPECIAL_TOKENS
        self.itos = list(specials)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.default_index = self.stoi[UNK_TOKEN]

        if counter is not None:
            for token, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
                if freq >= min_freq and token not in self.stoi:
                    self.stoi[token] = len(self.itos)
                    self.itos.append(token)

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.default_index)

    def lookup_token(self, index: int) -> str:
        if 0 <= int(index) < len(self.itos):
            return self.itos[int(index)]
        return UNK_TOKEN

    def lookup_tokens(self, indices: Iterable[int]) -> list[str]:
        return [self.lookup_token(index) for index in indices]

    def lookup_indices(self, tokens: Iterable[str]) -> list[int]:
        return [self[token] for token in tokens]


class Multi30kDataset(Dataset):
    _raw_dataset = None
    _tokenizers: dict[str, spacy.language.Language] = {}
    _cached_vocabs: dict[tuple[int], tuple[Vocab, Vocab]] = {}

    def __init__(
        self,
        split: str = "train",
        src_vocab: Optional[Vocab] = None,
        tgt_vocab: Optional[Vocab] = None,
        min_freq: int = 2,
        max_len: Optional[int] = None,
    ) -> None:
        """
        Load Multi30k and prepare German-to-English token id pairs.

        Vocabularies are built from the training split only. Validation and
        test datasets should receive the training vocabularies or will reuse
        the cached training vocabularies for the same min_freq.
        """
        self.split = self._normalise_split(split)
        self.min_freq = min_freq
        self.max_len = max_len
        self.dataset = self._load_raw_dataset()[self.split]
        self.src_tokenizer = self._get_tokenizer("de")
        self.tgt_tokenizer = self._get_tokenizer("en")

        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        else:
            self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab

        self.pad_idx = self.src_pad_idx = self.tgt_pad_idx = self.src_vocab[PAD_TOKEN]
        self.sos_idx = self.tgt_sos_idx = self.tgt_vocab[SOS_TOKEN]
        self.eos_idx = self.tgt_eos_idx = self.tgt_vocab[EOS_TOKEN]
        self.examples = self.process_data()

    @staticmethod
    def _normalise_split(split: str) -> str:
        split_aliases = {"val": "validation", "valid": "validation", "dev": "validation"}
        split = split_aliases.get(split, split)
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be one of: train, validation, test")
        return split

    @classmethod
    def _load_raw_dataset(cls):
        if cls._raw_dataset is None:
            cls._raw_dataset = load_dataset("bentrevett/multi30k")
        return cls._raw_dataset

    @classmethod
    def _get_tokenizer(cls, language: str):
        if language not in cls._tokenizers:
            cls._tokenizers[language] = spacy.blank(language)
        return cls._tokenizers[language]

    def _tokenize_src(self, sentence: str) -> list[str]:
        return [token.text.lower() for token in self.src_tokenizer(sentence)]

    def _tokenize_tgt(self, sentence: str) -> list[str]:
        return [token.text.lower() for token in self.tgt_tokenizer(sentence)]

    def build_vocab(self) -> tuple[Vocab, Vocab]:
        """
        Build source (German) and target (English) vocabularies from train only.
        """
        cache_key = (self.min_freq,)
        if cache_key in self._cached_vocabs:
            return self._cached_vocabs[cache_key]

        src_counter: Counter = Counter()
        tgt_counter: Counter = Counter()
        for row in self._load_raw_dataset()["train"]:
            src_counter.update(self._tokenize_src(row["de"]))
            tgt_counter.update(self._tokenize_tgt(row["en"]))

        vocabs = (Vocab(src_counter, self.min_freq), Vocab(tgt_counter, self.min_freq))
        self._cached_vocabs[cache_key] = vocabs
        return vocabs

    def process_data(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Tokenize sentences and convert them to integer tensors.
        """
        examples = []
        for row in self.dataset:
            src_tokens = self._tokenize_src(row["de"])
            tgt_tokens = self._tokenize_tgt(row["en"])

            if self.max_len is not None:
                src_tokens = src_tokens[: self.max_len - 2]
                tgt_tokens = tgt_tokens[: self.max_len - 2]

            src_ids = [self.src_vocab[SOS_TOKEN]]
            src_ids.extend(self.src_vocab.lookup_indices(src_tokens))
            src_ids.append(self.src_vocab[EOS_TOKEN])

            tgt_ids = [self.tgt_vocab[SOS_TOKEN]]
            tgt_ids.extend(self.tgt_vocab.lookup_indices(tgt_tokens))
            tgt_ids.append(self.tgt_vocab[EOS_TOKEN])

            examples.append(
                (
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                )
            )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]

    def collate_fn(
        self, batch: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_batch, tgt_batch = zip(*batch)
        src = pad_sequence(src_batch, batch_first=True, padding_value=self.src_pad_idx)
        tgt = pad_sequence(tgt_batch, batch_first=True, padding_value=self.tgt_pad_idx)
        return src, tgt
