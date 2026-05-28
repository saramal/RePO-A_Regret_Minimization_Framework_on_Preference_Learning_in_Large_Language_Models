from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask

    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # The k2_loss is approximately equivalent to the
    # one-step KL divergence penalty with the k1 estimator
    # used in https://arxiv.org/pdf/2310.10505.
    if kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    if kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    return log_ratio

#TODO: ORM to PRM
def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    action_mask: Optional[torch.Tensor] = None,
    num_actions: Optional[Union[int, list[int]]] = None,
    reward_clip_range: Tuple[float, float] = None,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    if reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    logger.info(f"[DEBUG] LINE 65 : action_mask: {action_mask}")
    logger.info(f"[DEBUG] LINE 79 : r: {r.shape}")


    if action_mask is not None:
        kl_reward = -kl_coef * kl
        # The following code is equivalent to:
        #
        # last_reward = torch.zeros_like(kl)
        # for i in range(last_reward.size(0)):
        #     for t in reversed(range(last_reward.size(1))):
        #         if action_mask[i][t] > 0.5:
        #             last_reward[i][t] = r[i]
        #             break
        #
        ##############################################################################################
        # PRM: token-level reward
        # r: (B, S)
        if r.squeeze().ndim == 2:
            reward = r + kl_reward
            logger.info(f"[DEBUG] PRM Reward Shape: {reward.shape}")
            logger.info(f"[DEBUG] PRM Reward Sample: {reward[0]}")

        # ORM: sequence-level reward
        # r: (B,) 
        else:
            eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
            last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))
            reward = last_reward + kl_reward
            logger.info(f"[DEBUG] ORM Reward Shape: {reward.shape}")
            logger.info(f"[DEBUG] ORM Reward Sample: {reward[0]}")
        ##############################################################################################


        # eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
        # last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

        # reward = last_reward + kl_reward

    else:
        reward = []
        for i, (kl_seg, action_len) in enumerate(zip(kl, num_actions)):
            logger.info(f"\n[Sample {i}]")
            logger.info(f"KL segment (kl_seg): {kl_seg}")
            logger.info(f"Action length (num_actions): {action_len}")
            kl_reward = -kl_coef * kl_seg
            logger.info(f"KL reward before adding r[i]: {kl_reward}")

            kl_reward[action_len - 1] += r[i]
            logger.info(f"Reward r[{i}]: {r[i]}")
            logger.info(f"KL reward after adding r[i] at index {action_len - 1}: {kl_reward}")

            reward.append(kl_reward)
        logger.info(f"\nFinal reward list: {reward}")



    return reward


def _logsumexp_by_chunk(logits: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    seq_len = logits.shape[0]
    logsumexp_values = torch.zeros((seq_len), device=logits.device, dtype=logits.dtype)
    for s_idx in range(0, seq_len, chunk_size):
        end_idx = min(s_idx + chunk_size, seq_len)
        logsumexp_values[s_idx:end_idx] = torch.logsumexp(logits[s_idx:end_idx], dim=-1)

    return logsumexp_values


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature != 1.0:
        logits.div_(temperature)
    # https://github.com/OpenRLHF/OpenRLHF/pull/718#issuecomment-2641081881
    if logits.dtype in [torch.float32, torch.float64]:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        try:
            from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

            output = cross_entropy_loss(logits.reshape(-1, last_dim), labels.reshape(-1))
            log_probs_labels = -output[0].view(*batch_dim)
        except ImportError:
            logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            logsumexp_values = _logsumexp_by_chunk(logits.reshape(-1, last_dim))
            logsumexp_values = logsumexp_values.view(*batch_dim)
            log_probs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        log_probs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_log_probs = F.log_softmax(row_logits, dim=-1)
            row_log_probs_labels = row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            log_probs_labels.append(row_log_probs_labels)
        log_probs_labels = torch.stack(log_probs_labels)
    return log_probs_labels


def _logsumexp_by_chunk_topk(logits: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    seq_len = logits.shape[0]
    logsumexp_values = torch.zeros((seq_len), device=logits.device, dtype=logits.dtype)
    for s_idx in range(0, seq_len, chunk_size):
        end_idx = min(s_idx + chunk_size, seq_len)
        logsumexp_values[s_idx:end_idx] = torch.logsumexp(logits[s_idx:end_idx], dim=-1)
    return logsumexp_values

def log_probs_from_logits_topk(
    logits: torch.Tensor,  # (B, N, V)
    labels: torch.Tensor,  # (B, N, k)
    temperature: float = 1.0
) -> torch.Tensor:
    """
    logits: (B, N, V)
    labels: (B, N, k)
    returns: (B, N, k) log probabilities for each candidate token
    """
    if temperature != 1.0:
        logits = logits / temperature

    B, N, V = logits.shape
    _, _, K = labels.shape

    # Flatten batch/sequence for efficient processing
    logits_flat = logits.reshape(-1, V)  # (B*N, V)
    labels_flat = labels.reshape(-1, K)  # (B*N, K)

    # Compute logsumexp for normalization (same as denominator in softmax)
    logsumexp_values = _logsumexp_by_chunk_topk(logits_flat)
    logsumexp_values = logsumexp_values.view(B, N, 1)  # (B, N, 1)

    # Gather the logits corresponding to each candidate token id
    labels = labels.to(dtype=torch.int64)
    logits_selected = torch.gather(
        logits, dim=-1, index=labels
    )  # (B, N, k)

    # log_softmax(x_i) = x_i - logsumexp(x)
    log_probs = logits_selected - logsumexp_values  # (B, N, k)

    return log_probs




def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


# Reset positions for packed samples
# For example
# Input: attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 0]])
# Output: position_ids  = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 0]])
def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids


def unpacking_samples(values: torch.Tensor, packed_seqlens: list[int]):
    values = values.squeeze(0)
    unpacked_values = []
    offset = 0
    for seqlen in packed_seqlens:
        unpacked_values.append(values[offset : offset + seqlen])
        offset += seqlen
    return unpacked_values



def flush_left_and_truncate(tensor: torch.Tensor, mask: torch.Tensor):
    """
    Vectorized flush + truncate function supporting both 2D and 3D inputs.
    Flushes along dim=1 (sequence dim), keeps relative order of other dims.

    Args:
        tensor: (B, N) or (B, N, K)
        mask: (B, N) or (B, N, K)
    Returns:
        (tensor_shifted, mask_shifted)
            tensor_shifted: flushed + truncated tensor
            mask_shifted: flushed + truncated mask
    """
    assert tensor.dim() in (2, 3), "tensor must be (B, N) or (B, N, K)"
    assert mask.dim() == tensor.dim(), "mask must have the same dim as tensor"
    assert tensor.shape[:2] == mask.shape[:2], "tensor and mask must match on (B, N)"

    B, N = tensor.shape[:2]
    device = tensor.device

    # ---- Compute first non-pad index (for flush left) ----
    # if mask is 3D, check if any of the K dimensions is 1 for valid non-pad index
    valid_mask_2d = mask.any(dim=-1) if mask.dim() == 3 else mask
    nonzero = (valid_mask_2d.cumsum(dim=1) > 0)
    first_idx = (~nonzero).sum(dim=1).clamp(max=N-1)  # (B,)

    # ---- Flush left (vectorized gather) ----
    arange_N = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    rolled_idx = (arange_N + first_idx.unsqueeze(1)) % N  # (B, N)

    if tensor.dim() == 2:
        tensor_shifted = torch.gather(tensor, dim=1, index=rolled_idx)
        mask_shifted = torch.gather(mask, dim=1, index=rolled_idx)
    else:
        B, N, K = tensor.shape
        rolled_idx_expand = rolled_idx.unsqueeze(-1).expand(B, N, K)
        tensor_shifted = torch.gather(tensor, dim=1, index=rolled_idx_expand)
        mask_shifted = torch.gather(mask, dim=1, index=rolled_idx_expand)

    # ---- Truncate (remove tail padding columns) ----
    # if mask is 3D, check if all of the K dimensions are 0 for padding columns
    valid_mask_shifted = mask_shifted.any(dim=-1) if mask_shifted.dim() == 3 else mask_shifted
    empty_cols = torch.sum(valid_mask_shifted, dim=0) == 0
    first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else N

    if tensor.dim() == 2:
        return tensor_shifted[:, :first_empty_col], mask_shifted[:, :first_empty_col]
    else:
        return tensor_shifted[:, :first_empty_col, :], mask_shifted[:, :first_empty_col, :]

import torch

def estimate_kl_divergence_batchwise(
    p_log_probs: torch.Tensor,  # (B, N, K)
    q_log_probs: torch.Tensor,  # (B, N, K)
    valid_k: int,
    vocab_size_p: int,
    vocab_size_q: int,
    eps: float = 1e-9
):
    """
    KL(p‖q) ≈ sum_topk + uniform_tail estimate.
    Input:
        p_log_probs, q_log_probs : (B, N, K)
        valid_k : top-k number
        vocab_size_p, vocab_size_q : total vocab size
    Output:
        kl_est : (B, N)
    """
    # convert to probabilities
    p_probs = torch.exp(p_log_probs)
    q_probs = torch.exp(q_log_probs)

    # top-k section KL calculation → (B, N)
    kl_top_k = torch.sum(p_probs * (p_log_probs - q_log_probs), dim=-1)

    # calculate probability mass sum for each (B, N)
    sum_p = torch.sum(p_probs, dim=-1).clamp(max=1.0)
    sum_q = torch.sum(q_probs, dim=-1).clamp(max=1.0)

    # remaining mass
    p_rem = (1.0 - sum_p).clamp(min=0.0) + eps
    q_rem = (1.0 - sum_q).clamp(min=0.0) + eps

    # uniform probability (scalar)
    denom_p = max(vocab_size_p - valid_k, 1)
    denom_q = max(vocab_size_q - valid_k, 1)
    p_other = p_rem / denom_p
    q_other = q_rem / denom_q

    # uniform area KL contribution
    # p_rem * (log p_other - log q_other) — broadcast to generate (B, N) result
    kl_other = p_rem * (torch.log(p_other) - torch.log(q_other))

    kl_total = kl_top_k + kl_other
    return kl_total  # (B, N)
