"""Head+Tail truncation to a string (plan §5): head 128 tokens + tail rest.

Shared by training, serving, and TF-IDF so every model sees the same text.
Token positions are mapped back to character offsets, so the output is the
original text with the middle cut out, not a decoded token list.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

MAX_LENGTH = 512
HEAD_TOKENS = 128

# Placed between head and tail when a call is cut. See truncate_head_tail().
CUT_SEPARATOR = " "  # Plain space: a visible cut marker would leak length (vishing is cut ~2x more often).


def text_token_budget(tokenizer: PreTrainedTokenizerBase, max_length: int = MAX_LENGTH) -> int:
    """Tokens available for text after the tokenizer's own special tokens."""
    return max_length - tokenizer.num_special_tokens_to_add(pair=False)


def truncate_head_tail(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_LENGTH,
    head_tokens: int = HEAD_TOKENS,
) -> str:
    """Return text unchanged if it fits, else head + CUT_SEPARATOR + tail.

    head = first `head_tokens` tokens; tail = the last (budget - head_tokens)
    tokens, where budget = max_length - special tokens.
    """
    budget = text_token_budget(tokenizer, max_length)
    tail_tokens = budget - head_tokens
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    if len(offsets) <= budget:
        return text

    head_end_char = offsets[head_tokens - 1][1]
    tail_start_char = offsets[-tail_tokens][0]
    return text[:head_end_char] + CUT_SEPARATOR + text[tail_start_char:]
