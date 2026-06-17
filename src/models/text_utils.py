import torch


def find_text_content_positions(tokenizer, input_ids: torch.Tensor, text: str) -> list[int]:
    """
    Find token positions corresponding to `text` within the full token sequence.
    Tokenizes `text` standalone and searches for the subsequence.
    """
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    ids = input_ids[0].tolist()
    for i in range(len(ids) - len(text_ids) + 1):
        if ids[i : i + len(text_ids)] == text_ids:
            return list(range(i, i + len(text_ids)))
    return []
