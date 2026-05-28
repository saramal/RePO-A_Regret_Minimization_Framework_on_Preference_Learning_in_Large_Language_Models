import torch
import torch.nn.functional as F


def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)

def zero_pad_sequences_for_topk(sequences, side: str = "left", value=0, pad_dim: int =-1):
    """
    sequences: list of tensors. all tensors must have the same rank.
    side: 'left' | 'right'
    value: padding value
    pad_dim: dimension to pad (negative indices allowed). e.g. to pad L in (1, L, K), use pad_dim=-2
    """
    assert side in ("left", "right")
    assert len(sequences) > 0, "sequences must be non-empty"

    ref = sequences[0]
    D = ref.dim()
    pad_dim = pad_dim if pad_dim >= 0 else D + pad_dim
    assert 0 <= pad_dim < D, f"pad_dim out of range for ndim={D}"

    # check if all tensors have the same rank (shape validation is ok for pad_dim)
    for t in sequences:
        assert t.dim() == D, f"rank mismatch: expected {D}, got {t.dim()}"

    max_len = max(t.size(pad_dim) for t in sequences)
    if max_len == 0:
        return torch.stack(sequences, dim=0)

    padded = []
    for t in sequences:
        pad_len = max_len - t.size(pad_dim)
        if pad_len <= 0:
            padded.append(t)
            continue

        left  = pad_len if side == "left" else 0
        right = 0       if side == "left" else pad_len

        # F.pad pads only the last n dimensions.
        # set n to include pad_dim, and pad the other dimensions with (0,0)
        n = D - pad_dim
        pad_tuple = []
        for d in range(D - 1, D - n - 1, -1):  # pad from the last dimension
            if d == pad_dim:
                pad_tuple.extend([left, right])
            else:
                pad_tuple.extend([0, 0])
        pad_tuple = tuple(pad_tuple)

        padded.append(F.pad(t, pad_tuple, value=value))
    return torch.stack(padded, dim=0)

def exist_and_not_none(d, key):
    return key in d and not d[key] is None
