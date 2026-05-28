from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean, flush_left_and_truncate, estimate_kl_divergence_batchwise


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self, ring_attn_group=None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

        self.ring_attn_group = ring_attn_group
        if self.ring_attn_group:
            self.ring_attn_rank = dist.get_rank(self.ring_attn_group)
            self.ring_attn_world_size = dist.get_world_size(self.ring_attn_group)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # RingAttention
        if self.ring_attn_group is not None:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.ring_attn_world_size
            start_idx = self.ring_attn_rank * seq_len_per_process
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)
            labels = labels[..., start_idx:end_idx]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # if labels are all IGNORE_INDEX, then nn.CrossEntropyLoss will be nan
            if torch.all(shift_labels == self.IGNORE_INDEX):
                # Use mean of logits multiplied by 0 to maintain gradient flow
                loss = shift_logits.mean() * 0
            else:
                loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.ring_attn_group)
            loss = loss / self.ring_attn_world_size
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return 0.5 * loss


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False, simpo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo
        self.simpo = simpo
        if simpo:
            self.beta = 2.5
            self.gamma = 0.3


    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.simpo:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            gamma_logratios = self.gamma / self.beta
            logits = pi_logratios - gamma_logratios
        else: 
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps
            logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        if self.simpo:
            chosen_rewards = self.beta * policy_chosen_logps.detach()
            rejected_rewards = self.beta * policy_rejected_logps.detach()
        else:
            chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


class TDPOLoss(nn.Module):
    """
    Token-level Direct Preference Optimization (TDPO) loss.

    Implements the objective from https://arxiv.org/abs/2406.08414 with
    the TDPO1 / TDPO2 variants controlled by ``if_tdpo2``.
    """

    def __init__(self, beta: float, alpha: float = 0.5, if_tdpo2: bool = True) -> None:
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.if_tdpo2 = if_tdpo2

    def forward(
        self,
        chosen_logps_margin: torch.Tensor,
        rejected_logps_margin: torch.Tensor,
        chosen_position_kl: torch.Tensor,
        rejected_position_kl: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            chosen_logps_margin: token-level log-prob margin (policy - ref) for chosen replies.
            rejected_logps_margin: token-level log-prob margin (policy - ref) for rejected replies.
            chosen_position_kl: sequential KL between policy and ref for chosen replies.
            rejected_position_kl: sequential KL between policy and ref for rejected replies.
        """

        chosen_values = chosen_logps_margin + chosen_position_kl
        rejected_values = rejected_logps_margin + rejected_position_kl

        logps_margin_diff = chosen_logps_margin - rejected_logps_margin
        if not self.if_tdpo2:
            # TDPO1: use raw KL difference
            logits = logps_margin_diff - (rejected_position_kl - chosen_position_kl)
        else:
            # TDPO2: stop grad on chosen KL to stabilize
            logits = logps_margin_diff - self.alpha * (rejected_position_kl - chosen_position_kl.detach())

        losses = -F.logsigmoid(self.beta * logits)

        chosen_rewards = self.beta * chosen_values.detach()
        rejected_rewards = self.beta * rejected_values.detach()

        return losses.mean(), chosen_rewards, rejected_rewards

class RePO_Loss(nn.Module):
    """

    RePO Loss
    
    """
    def __init__(self, cpl_lambda: float, ref_coef=1, alpha: float = 0.1, normalize_score=True, normalization_type='mean') -> None:
        super().__init__()
        self.cpl_lambda = cpl_lambda
        self.ref_coef = ref_coef
        self.normalize_score = normalize_score
        self.normalization_type='mean'
        self.alpha = alpha

        
        
    def forward(
        self,
        RePO_forward_output : dict,
        chosen_data_mask: bool = True,
        reject_data_mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # import pdb
        # pdb.set_trace()
        
        regret_target_chosen = self.get_regret_score(
            RePO_forward_output["chosen_target_model_logprobs"],
            RePO_forward_output["chosen_label_logprobs"],
            RePO_forward_output["chosen_label_masks"],
            is_ref_regret=False,
            sequence_normalization_type=self.normalization_type
            )
        
        regret_target_rejected = self.get_regret_score(
            RePO_forward_output["rejected_target_model_logprobs"],
            RePO_forward_output["rejected_label_logprobs"],
            RePO_forward_output["rejected_label_masks"],
            is_ref_regret=False,
            sequence_normalization_type=self.normalization_type
            )

        
        if "ref_chosen_target_model_logprobs" in RePO_forward_output:
            regret_ref_chosen = self.get_regret_score(
                RePO_forward_output["ref_chosen_target_model_logprobs"],
                RePO_forward_output["chosen_label_logprobs"],
                RePO_forward_output["chosen_label_masks"],
                is_ref_regret=True, 
                sequence_normalization_type=self.normalization_type
                )

            regret_ref_rejected = self.get_regret_score(
                RePO_forward_output["ref_rejected_target_model_logprobs"],
                RePO_forward_output["rejected_label_logprobs"],
                RePO_forward_output["rejected_label_masks"],
                is_ref_regret=True,
                sequence_normalization_type=self.normalization_type
                )

        else:
            regret_ref_chosen = torch.zeros_like(regret_target_chosen)
            regret_ref_rejected = torch.zeros_like(regret_target_rejected)

        regret_chosen = self.alpha * (regret_target_chosen - self.ref_coef * regret_ref_chosen)
        regret_rejected = self.alpha * (regret_target_rejected - self.ref_coef * regret_ref_rejected)
        

        losses = - F.logsigmoid(regret_chosen - self.cpl_lambda * regret_rejected)
        # losses = - F.logsigmoid(RePO_forward_output["chosen_target_model_logits"].sum(dim=-1).sum(dim=-1))
        loss = losses.mean()
        # dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        
        # pi_logratios = policy_chosen_logps - policy_rejected_logps
        # ref_logratios = reference_chosen_logps - reference_rejected_logps
        # logits = pi_logratios - ref_logratios

        # losses = -F.logsigmoid(-self.beta * logits)

        # loss = losses.mean()
        # chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        # rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        # return loss, 0, 0
        return loss, regret_chosen, regret_rejected, \
                regret_target_chosen.mean().item(), regret_target_rejected.mean().item(), \
                    regret_ref_chosen.mean().item(), regret_ref_rejected.mean().item()
    
    #TODO: remove torch.isnan after debugging
    def get_regret_score(
        self, 
        logprob_target, 
        logprob_label, 
        label_mask_,
        is_ref_regret=False,
        sequence_normalization_type='mean'
         
    ):
    
        # return logprob_target.sum(dim=-1).sum(dim=-1)*0.001
        # logprob_target_masked = torch.multiply(logprob_target, label_mask)
        # chosen_logprob_laebel_masked = torch.multiply(logprob_label, label_mask)
        logprob_target_masked = logprob_target.clone()
        chosen_logprob_laebel_masked = logprob_label
        label_mask = label_mask_
        
        # # Flush left to reduce the memory usage and align
        
        #     # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
        #     #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
        # for i in range(label_mask.size(0)):
        #     first_one_idx = torch.nonzero(label_mask[i])[0].item()
        #     logprob_target_masked[i] = torch.roll(logprob_target_masked[i], shifts=-first_one_idx)
        #     chosen_logprob_laebel_masked[i] = torch.roll(chosen_logprob_laebel_masked[i], shifts=-first_one_idx)
        #     label_mask[i] = torch.roll(label_mask[i], shifts=-first_one_idx)
        
        # # Get the first column idx that is all zeros and remove every column after that
        # empty_cols = torch.sum(label_mask, dim=0) == 0
        # first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else label_mask.size(1)
        # logprob_target_masked = logprob_target_masked[:, :first_empty_col]
        # label_mask = label_mask[:, :first_empty_col]
        # chosen_logprob_laebel_masked = chosen_logprob_laebel_masked[:, :first_empty_col]

        logprob_target_masked, _ = flush_left_and_truncate(logprob_target_masked, label_mask)
        chosen_logprob_laebel_masked, label_mask = flush_left_and_truncate(chosen_logprob_laebel_masked, label_mask)

        assert label_mask.dim()==2 and logprob_target_masked.dim()==2 and chosen_logprob_laebel_masked.dim()==2,\
                "The dim of label_mask, logprob_target_masked and chosen_logprob_laebel_masked should be 2"
        assert label_mask.size()==logprob_target_masked.size()==chosen_logprob_laebel_masked.size(),\
                "The size of label_mask, logprob_target_masked and chosen_logprob_laebel_masked should be equal"
        

        # # Method 1: Using For loop for regret score calculation
        # len_chosen_generated_sequence = label_mask.sum(dim=-1)

        # regret_score = 0
        # for t in range(label_mask.size(-1)):
        #     # logprob term
        #     target_logprob_term = logprob_target_masked[:, t]
        #     if torch.isnan(target_logprob_term).any():
        #         print(f"Nan in target_logprob_term: {target_logprob_term}")
        #         import pdb
        #         pdb.set_trace()
        #     # KL divergence term
        #     rollout_window_size = len_chosen_generated_sequence - t - 1
        #     # rollout_window_size = len_chosen_generated_sequence - t
        #     sequence_max_len = len_chosen_generated_sequence.max().item()
        #     kl_divergence_term = torch.zeros_like(target_logprob_term)
        #     for l in range(1, sequence_max_len - t):
        #         # tail of logprob tensor is masked(=0), so just sum up all.
        #         kl_divergence_term = kl_divergence_term + chosen_logprob_laebel_masked[:, t + l] - logprob_target_masked[:, t + l]
        #         if torch.isnan(kl_divergence_term).any():
        #             print(f"Nan in kl_divergence_term: {kl_divergence_term}")
        #             import pdb
        #             pdb.set_trace()
            
            
        #     assert (kl_divergence_term.dim()==1 or kl_divergence_term == 0) and kl_divergence_term.size(0)==label_mask.size(0), \
        #         "The dim and size of kl_divergence_term should be (batch_size,)"  
        #     # normalized by length
        #     # avoid div-by-zero
        #     div_mask = (kl_divergence_term != 0) & (rollout_window_size != 0)
        #     temp_kl_term = torch.zeros_like(kl_divergence_term)
        #     temp_kl_term[div_mask] = kl_divergence_term[div_mask] / rollout_window_size[div_mask]
        #     kl_divergence_term = temp_kl_term
            
        #     ## Second methods for avoid div-by-zero,
        #     # kl_divergence_term = torch.div(kl_divergence_term, rollout_window_size)
        #     # kl_divergence_term[(kl_divergence_term==0) | (rollout_window_size==0)] = 0
        #     if torch.isnan(kl_divergence_term).any():
        #         print(f"Nan in kl_divergence_term AFTER NORMALIZE: {kl_divergence_term}")
        #         import pdb
        #         pdb.set_trace()


        #     # regret score
        #     regret_score = regret_score + target_logprob_term - kl_divergence_term
            
        # if self.normalize_score:
        #     regret_score = torch.div(regret_score, len_chosen_generated_sequence)
        # return regret_score
        
        # Method 2: Using vectorized operation for regret score calculation
            # sequence len (batch_size,)
        len_chosen_generated_sequence = label_mask.sum(dim=-1)

        batch_size, seq_len = label_mask.size()
        device = label_mask.device

        # mask: (batch, seq_len)
        valid_steps = torch.arange(seq_len, device=device).unsqueeze(0) < len_chosen_generated_sequence.unsqueeze(1)

        # (batch, seq_len): Target logprob term
        target_logprob_term = logprob_target_masked.clone()
        target_logprob_term[~valid_steps] = 0
        
        
        # regret score
        if is_ref_regret:
            regret_score_per_step = target_logprob_term
        else:
            # Use KL divergence term to calculate the regret score            

            # (batch, seq_len): KL divergence term calculation
            # kl_term_raw = chosen_logprob_laebel_masked - logprob_target_masked
            kl_term_raw = chosen_logprob_laebel_masked - logprob_target_masked
            kl_term_raw[~valid_steps] = 0

            # cumsum start from back, accumulated kl after 't'
            # flip for back-cumsum 
            kl_cumsum = torch.flip(torch.cumsum(torch.flip(kl_term_raw, dims=[1]), dim=1), dims=[1])

            # calculate rollout window size at t
            rollout_window_sizes = (len_chosen_generated_sequence.unsqueeze(1) - torch.arange(seq_len, device=device).unsqueeze(0)).clamp(min=1)
            rollout_window_sizes = rollout_window_sizes * valid_steps

            # normalize
            kl_divergence_term = torch.zeros_like(kl_cumsum)
            nonzero_mask = rollout_window_sizes != 0
            kl_divergence_term[nonzero_mask] = kl_cumsum[nonzero_mask] / rollout_window_sizes[nonzero_mask]
            
            
            regret_score_per_step = target_logprob_term - kl_divergence_term

        
        if sequence_normalization_type == 'mean':
            regret_score = regret_score_per_step.sum(dim=-1)
            regret_score = torch.div(regret_score, len_chosen_generated_sequence)
        elif sequence_normalization_type == 'max':
            regret_score = regret_score_per_step.max(dim=-1)
        else:
            raise ValueError("Not supported normalization type, available options:['mean', 'max']")
        
        
        # # normalize final regret score by length
        # if self.normalize_score:
        #     regret_score = torch.div(regret_score, len_chosen_generated_sequence)

        return regret_score




class RePO_Loss_deterministic(nn.Module):
    """

    RePO Loss for deterministic generation
    
    """
    def __init__(self, cpl_lambda: float, ref_coef=1, alpha: float = 0.1, normalize_score=True, normalization_type='mean') -> None:
        super().__init__()
        self.cpl_lambda = cpl_lambda
        self.ref_coef = ref_coef
        self.normalize_score = normalize_score
        self.normalization_type='mean'
        self.alpha = alpha

        
        
    def forward(
        self,
        RePO_forward_output : dict,
        chosen_data_mask: bool = True,
        reject_data_mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # import pdb
        # pdb.set_trace()
        
        regret_target_chosen = self.get_regret_score(
            RePO_forward_output["chosen_target_model_logprobs"],
            # RePO_forward_output["chosen_label_logprobs"],
            # RePO_forward_output["chosen_label_masks"],
            RePO_forward_output["chosen_label_masks"],
            is_ref_regret=False,
            sequence_normalization_type=self.normalization_type
            )
        
        regret_target_rejected = self.get_regret_score(
            RePO_forward_output["rejected_target_model_logprobs"],
            # RePO_forward_output["rejected_label_logprobs"],
            # RePO_forward_output["rejected_label_masks"],
            RePO_forward_output["rejected_label_masks"],
            is_ref_regret=False,
            sequence_normalization_type=self.normalization_type
            )

        
        if "ref_chosen_target_model_logprobs" in RePO_forward_output:
            regret_ref_chosen = self.get_regret_score(
                RePO_forward_output["ref_chosen_target_model_logprobs"],
                # RePO_forward_output["chosen_label_logprobs"],
                # RePO_forward_output["chosen_label_masks"],
                RePO_forward_output["chosen_label_masks"],
                is_ref_regret=True, 
                sequence_normalization_type=self.normalization_type
                )

            regret_ref_rejected = self.get_regret_score(
                RePO_forward_output["ref_rejected_target_model_logprobs"],
                # RePO_forward_output["rejected_label_logprobs"],
                # RePO_forward_output["rejected_label_masks"],
                RePO_forward_output["rejected_label_masks"],
                is_ref_regret=True,
                sequence_normalization_type=self.normalization_type
                )

        else:
            regret_ref_chosen = torch.zeros_like(regret_target_chosen)
            regret_ref_rejected = torch.zeros_like(regret_target_rejected)

        regret_chosen = self.alpha * (regret_target_chosen - self.ref_coef * regret_ref_chosen)
        regret_rejected = self.alpha * (regret_target_rejected - self.ref_coef * regret_ref_rejected)
        

        losses = - F.logsigmoid(regret_chosen - self.cpl_lambda * regret_rejected)
        # losses = - F.logsigmoid(RePO_forward_output["chosen_target_model_logits"].sum(dim=-1).sum(dim=-1))
        loss = losses.mean()
        # dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        
        # pi_logratios = policy_chosen_logps - policy_rejected_logps
        # ref_logratios = reference_chosen_logps - reference_rejected_logps
        # logits = pi_logratios - ref_logratios

        # losses = -F.logsigmoid(-self.beta * logits)

        # loss = losses.mean()
        # chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        # rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        # return loss, 0, 0
        return loss, regret_chosen, regret_rejected, \
                regret_target_chosen.mean().item(), regret_target_rejected.mean().item(), \
                    regret_ref_chosen.mean().item(), regret_ref_rejected.mean().item()
    
    #TODO: remove torch.isnan after debugging
    def get_regret_score(
        self, 
        logprob_target, 
        # logprob_label, 
        label_mask_,
        is_ref_regret=False,
        sequence_normalization_type='mean'
         
    ):
    

        logprob_target_masked = logprob_target.clone()
        # chosen_logprob_laebel_masked = logprob_label
        label_mask = label_mask_
   

        logprob_target_masked, label_mask = flush_left_and_truncate(logprob_target_masked, label_mask)
        # chosen_logprob_laebel_masked, label_mask = flush_left_and_truncate(chosen_logprob_laebel_masked, label_mask)

        assert logprob_target_masked.dim()==2,\
                "The dim of logprob_target_masked should be 2"

        


        len_chosen_generated_sequence = label_mask.sum(dim=-1)

        batch_size, seq_len = label_mask.size()
        device = label_mask.device

        # mask: (batch, seq_len)
        valid_steps = torch.arange(seq_len, device=device).unsqueeze(0) < len_chosen_generated_sequence.unsqueeze(1)

        # (batch, seq_len): Target logprob term
        target_logprob_term = logprob_target_masked.clone()
        target_logprob_term[~valid_steps] = 0
        
        
        # regret score
        if is_ref_regret:
            regret_score_per_step = target_logprob_term
        else:
            # Use KL divergence term to calculate the regret score            

            # (batch, seq_len): KL divergence term calculation
            # kl_term_raw = chosen_logprob_laebel_masked - logprob_target_masked
            kl_term_raw = -1 * logprob_target_masked
            kl_term_raw[~valid_steps] = 0

            # cumsum start from back, accumulated kl after 't'
            # flip for back-cumsum 
            kl_cumsum = torch.flip(torch.cumsum(torch.flip(kl_term_raw, dims=[1]), dim=1), dims=[1])

            # calculate rollout window size at t
            rollout_window_sizes = (len_chosen_generated_sequence.unsqueeze(1) - torch.arange(seq_len, device=device).unsqueeze(0)).clamp(min=1)
            rollout_window_sizes = rollout_window_sizes * valid_steps

            # normalize
            kl_divergence_term = torch.zeros_like(kl_cumsum)
            nonzero_mask = rollout_window_sizes != 0
            kl_divergence_term[nonzero_mask] = kl_cumsum[nonzero_mask] / rollout_window_sizes[nonzero_mask]
            
            
            regret_score_per_step = target_logprob_term - kl_divergence_term

        
        if sequence_normalization_type == 'mean':
            regret_score = regret_score_per_step.sum(dim=-1)
            regret_score = torch.div(regret_score, len_chosen_generated_sequence)
        elif sequence_normalization_type == 'max':
            regret_score = regret_score_per_step.max(dim=-1)
        else:
            raise ValueError("Not supported normalization type, available options:['mean', 'max']")
        
        
        # # normalize final regret score by length
        # if self.normalize_score:
        #     regret_score = torch.div(regret_score, len_chosen_generated_sequence)

        return regret_score




    

class RePO_Loss_topk(nn.Module):
    """

    RePO Loss
    
    """
    def __init__(self, cpl_lambda: float, ref_coef=1, alpha: float = 0.1, normalize_score=True, normalization_type='mean') -> None:
        super().__init__()
        self.cpl_lambda = cpl_lambda
        self.ref_coef = ref_coef
        self.normalize_score = normalize_score
        self.normalization_type='mean'
        self.alpha = alpha

        
        
    def forward(
        self,
        RePO_forward_output : dict,
        chosen_data_mask: bool = True,
        reject_data_mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # import pdb
        # pdb.set_trace()
        
        regret_target_chosen = self.get_regret_score_topk(
            RePO_forward_output["chosen_target_model_logprobs"],
            RePO_forward_output["chosen_label_logprobs"],
            RePO_forward_output["chosen_label_masks"],
            RePO_forward_output["topk_chosen_target_model_logprobs"],
            RePO_forward_output["topk_chosen_label_logprobs"],
            RePO_forward_output["topk_chosen_label_masks"],
            RePO_forward_output["valid_k"],
            RePO_forward_output["vocab_size_p"],
            RePO_forward_output["vocab_size_q"],
            is_ref_regret=False,
            sequence_normalization_type=self.normalization_type
            )
        
        regret_target_rejected = self.get_regret_score_topk(
            RePO_forward_output["rejected_target_model_logprobs"],
            RePO_forward_output["rejected_label_logprobs"],
            RePO_forward_output["rejected_label_masks"],
            RePO_forward_output["topk_rejected_target_model_logprobs"],
            RePO_forward_output["topk_rejected_label_logprobs"],
            RePO_forward_output["topk_rejected_label_masks"],
            RePO_forward_output["valid_k"],
            RePO_forward_output["vocab_size_p"],
            RePO_forward_output["vocab_size_q"],
            is_ref_regret=False,
            sequence_normalization_type=self.normalization_type
            )

        
        if "ref_chosen_target_model_logprobs" in RePO_forward_output:
            regret_ref_chosen = self.get_regret_score(
                RePO_forward_output["ref_chosen_target_model_logprobs"],
                RePO_forward_output["chosen_label_logprobs"],
                RePO_forward_output["chosen_label_masks"],
                is_ref_regret=True, 
                sequence_normalization_type=self.normalization_type
                )

            regret_ref_rejected = self.get_regret_score(
                RePO_forward_output["ref_rejected_target_model_logprobs"],
                RePO_forward_output["rejected_label_logprobs"],
                RePO_forward_output["rejected_label_masks"],
                is_ref_regret=True,
                sequence_normalization_type=self.normalization_type
                )

        else:
            regret_ref_chosen = torch.zeros_like(regret_target_chosen)
            regret_ref_rejected = torch.zeros_like(regret_target_rejected)

        regret_chosen = self.alpha * (regret_target_chosen - self.ref_coef * regret_ref_chosen)
        regret_rejected = self.alpha * (regret_target_rejected - self.ref_coef * regret_ref_rejected)
        

        losses = - F.logsigmoid(regret_chosen - self.cpl_lambda * regret_rejected)
        # losses = - F.logsigmoid(RePO_forward_output["chosen_target_model_logits"].sum(dim=-1).sum(dim=-1))
        loss = losses.mean()
        # dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        
        # pi_logratios = policy_chosen_logps - policy_rejected_logps
        # ref_logratios = reference_chosen_logps - reference_rejected_logps
        # logits = pi_logratios - ref_logratios

        # losses = -F.logsigmoid(-self.beta * logits)

        # loss = losses.mean()
        # chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        # rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        # return loss, 0, 0
        return loss, regret_chosen, regret_rejected, \
                regret_target_chosen.mean().item(), regret_target_rejected.mean().item(), \
                    regret_ref_chosen.mean().item(), regret_ref_rejected.mean().item()
    
    #TODO: remove torch.isnan after debugging
    def get_regret_score(
        self, 
        logprob_target, 
        logprob_label, 
        label_mask_,
        is_ref_regret=False,
        sequence_normalization_type='mean'
         
    ):
    
        # return logprob_target.sum(dim=-1).sum(dim=-1)*0.001
        # logprob_target_masked = torch.multiply(logprob_target, label_mask)
        # chosen_logprob_laebel_masked = torch.multiply(logprob_label, label_mask)
        logprob_target_masked = logprob_target.clone()
        chosen_logprob_laebel_masked = logprob_label
        label_mask = label_mask_
        
        # # Flush left to reduce the memory usage and align
        
        #     # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
        #     #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
        # for i in range(label_mask.size(0)):
        #     first_one_idx = torch.nonzero(label_mask[i])[0].item()
        #     logprob_target_masked[i] = torch.roll(logprob_target_masked[i], shifts=-first_one_idx)
        #     chosen_logprob_laebel_masked[i] = torch.roll(chosen_logprob_laebel_masked[i], shifts=-first_one_idx)
        #     label_mask[i] = torch.roll(label_mask[i], shifts=-first_one_idx)
        
        # # Get the first column idx that is all zeros and remove every column after that
        # empty_cols = torch.sum(label_mask, dim=0) == 0
        # first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else label_mask.size(1)
        # logprob_target_masked = logprob_target_masked[:, :first_empty_col]
        # label_mask = label_mask[:, :first_empty_col]
        # chosen_logprob_laebel_masked = chosen_logprob_laebel_masked[:, :first_empty_col]

        logprob_target_masked, _ = flush_left_and_truncate(logprob_target_masked, label_mask)
        chosen_logprob_laebel_masked, label_mask = flush_left_and_truncate(chosen_logprob_laebel_masked, label_mask)


        assert label_mask.dim()==2 and logprob_target_masked.dim()==2 and chosen_logprob_laebel_masked.dim()==2,\
                "The dim of label_mask, logprob_target_masked and chosen_logprob_laebel_masked should be 2"
        assert label_mask.size()==logprob_target_masked.size()==chosen_logprob_laebel_masked.size(),\
                "The size of label_mask, logprob_target_masked and chosen_logprob_laebel_masked should be equal"
        

        # # Method 1: Using For loop for regret score calculation
        # len_chosen_generated_sequence = label_mask.sum(dim=-1)

        # regret_score = 0
        # for t in range(label_mask.size(-1)):
        #     # logprob term
        #     target_logprob_term = logprob_target_masked[:, t]
        #     if torch.isnan(target_logprob_term).any():
        #         print(f"Nan in target_logprob_term: {target_logprob_term}")
        #         import pdb
        #         pdb.set_trace()
        #     # KL divergence term
        #     rollout_window_size = len_chosen_generated_sequence - t - 1
        #     # rollout_window_size = len_chosen_generated_sequence - t
        #     sequence_max_len = len_chosen_generated_sequence.max().item()
        #     kl_divergence_term = torch.zeros_like(target_logprob_term)
        #     for l in range(1, sequence_max_len - t):
        #         # tail of logprob tensor is masked(=0), so just sum up all.
        #         kl_divergence_term = kl_divergence_term + chosen_logprob_laebel_masked[:, t + l] - logprob_target_masked[:, t + l]
        #         if torch.isnan(kl_divergence_term).any():
        #             print(f"Nan in kl_divergence_term: {kl_divergence_term}")
        #             import pdb
        #             pdb.set_trace()
            
            
        #     assert (kl_divergence_term.dim()==1 or kl_divergence_term == 0) and kl_divergence_term.size(0)==label_mask.size(0), \
        #         "The dim and size of kl_divergence_term should be (batch_size,)"  
        #     # normalized by length
        #     # avoid div-by-zero
        #     div_mask = (kl_divergence_term != 0) & (rollout_window_size != 0)
        #     temp_kl_term = torch.zeros_like(kl_divergence_term)
        #     temp_kl_term[div_mask] = kl_divergence_term[div_mask] / rollout_window_size[div_mask]
        #     kl_divergence_term = temp_kl_term
            
        #     ## Second methods for avoid div-by-zero,
        #     # kl_divergence_term = torch.div(kl_divergence_term, rollout_window_size)
        #     # kl_divergence_term[(kl_divergence_term==0) | (rollout_window_size==0)] = 0
        #     if torch.isnan(kl_divergence_term).any():
        #         print(f"Nan in kl_divergence_term AFTER NORMALIZE: {kl_divergence_term}")
        #         import pdb
        #         pdb.set_trace()


        #     # regret score
        #     regret_score = regret_score + target_logprob_term - kl_divergence_term
            
        # if self.normalize_score:
        #     regret_score = torch.div(regret_score, len_chosen_generated_sequence)
        # return regret_score
        
        # Method 2: Using vectorized operation for regret score calculation
            # sequence len (batch_size,)
        len_chosen_generated_sequence = label_mask.sum(dim=-1)

        batch_size, seq_len = label_mask.size()
        device = label_mask.device

        # mask: (batch, seq_len)
        valid_steps = torch.arange(seq_len, device=device).unsqueeze(0) < len_chosen_generated_sequence.unsqueeze(1)

        # (batch, seq_len): Target logprob term
        target_logprob_term = logprob_target_masked.clone()
        target_logprob_term[~valid_steps] = 0
        
        
        # regret score
        if is_ref_regret:
            regret_score_per_step = target_logprob_term
        else:
            # Use KL divergence term to calculate the regret score            

            # (batch, seq_len): KL divergence term calculation
            # kl_term_raw = chosen_logprob_laebel_masked - logprob_target_masked
            kl_term_raw = chosen_logprob_laebel_masked - logprob_target_masked
            kl_term_raw[~valid_steps] = 0

            # cumsum start from back, accumulated kl after 't'
            # flip for back-cumsum 
            kl_cumsum = torch.flip(torch.cumsum(torch.flip(kl_term_raw, dims=[1]), dim=1), dims=[1])

            # calculate rollout window size at t
            rollout_window_sizes = (len_chosen_generated_sequence.unsqueeze(1) - torch.arange(seq_len, device=device).unsqueeze(0)).clamp(min=1)
            rollout_window_sizes = rollout_window_sizes * valid_steps

            # normalize
            kl_divergence_term = torch.zeros_like(kl_cumsum)
            nonzero_mask = rollout_window_sizes != 0
            kl_divergence_term[nonzero_mask] = kl_cumsum[nonzero_mask] / rollout_window_sizes[nonzero_mask]
            
            
            regret_score_per_step = target_logprob_term - kl_divergence_term

        
        if sequence_normalization_type == 'mean':
            regret_score = regret_score_per_step.sum(dim=-1)
            regret_score = torch.div(regret_score, len_chosen_generated_sequence)
        elif sequence_normalization_type == 'max':
            regret_score = regret_score_per_step.max(dim=-1)
        else:
            raise ValueError("Not supported normalization type, available options:['mean', 'max']")
        
        
        # # normalize final regret score by length
        # if self.normalize_score:
        #     regret_score = torch.div(regret_score, len_chosen_generated_sequence)

        return regret_score
    def get_regret_score_topk(
        self, 
        logprob_target, 
        logprob_label, 
        label_mask_,
        logprob_topk_target,
        logprob_topk_label,
        label_topk_mask_,
        valid_k,
        vocab_size_p,
        vocab_size_q,
        is_ref_regret=False,
        sequence_normalization_type='mean'
         
    ):
    
        # return logprob_target.sum(dim=-1).sum(dim=-1)*0.001
        # logprob_target_masked = torch.multiply(logprob_target, label_mask)
        # chosen_logprob_laebel_masked = torch.multiply(logprob_label, label_mask)
        logprob_target_masked = logprob_target.clone()
        chosen_logprob_laebel_masked = logprob_label
        label_mask = label_mask_
        

        # # Flush left to reduce the memory usage and align
        
        #     # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
        #     #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
        # for i in range(label_mask.size(0)):
        #     first_one_idx = torch.nonzero(label_mask[i])[0].item()
        #     logprob_target_masked[i] = torch.roll(logprob_target_masked[i], shifts=-first_one_idx)
        #     chosen_logprob_laebel_masked[i] = torch.roll(chosen_logprob_laebel_masked[i], shifts=-first_one_idx)
        #     label_mask[i] = torch.roll(label_mask[i], shifts=-first_one_idx)
        
        # # Get the first column idx that is all zeros and remove every column after that
        # empty_cols = torch.sum(label_mask, dim=0) == 0
        # first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else label_mask.size(1)
        # logprob_target_masked = logprob_target_masked[:, :first_empty_col]
        # label_mask = label_mask[:, :first_empty_col]
        # chosen_logprob_laebel_masked = chosen_logprob_laebel_masked[:, :first_empty_col]

        logprob_target_masked, _ = flush_left_and_truncate(logprob_target_masked, label_mask)
        chosen_logprob_laebel_masked, label_mask = flush_left_and_truncate(chosen_logprob_laebel_masked, label_mask)


        
        logprob_topk_target_masked = logprob_topk_target.clone()
        chosen_logprob_topk_laebel_masked = logprob_topk_label
        label_topk_mask = label_topk_mask_

        # Flush left to reduce the memory usage and align, with topk logprobs
        logprob_topk_target_masked, _ = flush_left_and_truncate(logprob_topk_target_masked, label_topk_mask)
        chosen_logprob_topk_laebel_masked, label_topk_mask = flush_left_and_truncate(chosen_logprob_topk_laebel_masked, label_topk_mask)



    

        assert label_mask.dim()==2 and logprob_target_masked.dim()==2 and chosen_logprob_laebel_masked.dim()==2,\
                "The dim of label_mask, logprob_target_masked and chosen_logprob_laebel_masked should be 2"
        assert label_mask.size()==logprob_target_masked.size()==chosen_logprob_laebel_masked.size(),\
                "The size of label_mask, logprob_target_masked and chosen_logprob_laebel_masked should be equal"
        
        assert label_topk_mask.dim()==3 and logprob_topk_target_masked.dim()==3 and chosen_logprob_topk_laebel_masked.dim()==3,\
                "The dim of label_topk_mask, logprob_topk_target_masked and chosen_logprob_topk_laebel_masked should be 3"
        assert label_topk_mask.size()==logprob_topk_target_masked.size()==chosen_logprob_topk_laebel_masked.size(),\
                "The size of label_topk_mask, logprob_topk_target_masked and chosen_logprob_topk_laebel_masked should be equal"

        len_chosen_generated_sequence = label_mask.sum(dim=-1)

        batch_size, seq_len = label_mask.size()
        device = label_mask.device

        # mask: (batch, seq_len)
        valid_steps = torch.arange(seq_len, device=device).unsqueeze(0) < len_chosen_generated_sequence.unsqueeze(1)

        # (batch, seq_len): Target logprob term
        target_logprob_term = logprob_target_masked.clone()
        target_logprob_term[~valid_steps] = 0
        
        
        # regret score
        if is_ref_regret:
            regret_score_per_step = target_logprob_term
        else:
            # Use KL divergence term to calculate the regret score            

            # (batch, seq_len): KL divergence term calculation

            # kl_term_raw = chosen_logprob_laebel_masked - logprob_target_masked
            # kl_term_raw[~valid_steps] = 0

            kl_topk_term_raw = estimate_kl_divergence_batchwise(chosen_logprob_topk_laebel_masked, logprob_topk_target_masked, valid_k, vocab_size_p, vocab_size_q)
            kl_topk_term_raw[~valid_steps] = 0

            # cumsum start from back, accumulated kl after 't'
            # flip for back-cumsum 
            kl_cumsum = torch.flip(torch.cumsum(torch.flip(kl_topk_term_raw, dims=[1]), dim=1), dims=[1])

            # calculate rollout window size at t
            rollout_window_sizes = (len_chosen_generated_sequence.unsqueeze(1) - torch.arange(seq_len, device=device).unsqueeze(0)).clamp(min=1)
            rollout_window_sizes = rollout_window_sizes * valid_steps

            # normalize
            kl_divergence_term = torch.zeros_like(kl_cumsum)
            nonzero_mask = rollout_window_sizes != 0
            kl_divergence_term[nonzero_mask] = kl_cumsum[nonzero_mask] / rollout_window_sizes[nonzero_mask]
            
            
            regret_score_per_step = target_logprob_term - kl_divergence_term

        
        if sequence_normalization_type == 'mean':
            regret_score = regret_score_per_step.sum(dim=-1)
            regret_score = torch.div(regret_score, len_chosen_generated_sequence)
        elif sequence_normalization_type == 'max':
            regret_score = regret_score_per_step.max(dim=-1)
        else:
            raise ValueError("Not supported normalization type, available options:['mean', 'max']")
        
        
        # # normalize final regret score by length
        # if self.normalize_score:
        #     regret_score = torch.div(regret_score, len_chosen_generated_sequence)

        return regret_score

    


    
class RePO_Unvalanced_Loss(nn.Module):
    """

    Unpaired RePO Loss
    For paired dataset, just using portion of dataset
    to calculate the regret score.

    """
    def __init__(self, beta: float, ref_coef=1, normalize_score=True, normalization_type='mean') -> None:
        super().__init__()
        self.beta = beta
        self.ref_coef = ref_coef
        self.normalize_score = normalize_score
        self.normalization_type = normalization_type

        
        
    def forward(
        self,
        RePO_forward_output : dict,
        chosen_data_mask: bool = True,
        reject_data_mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # import pdb
        # pdb.set_trace()
        
        regret_target_chosen = self.get_regret_score(
            RePO_forward_output["chosen_target_model_logprobs"],
            RePO_forward_output["chosen_label_logprobs"],
            RePO_forward_output["chosen_label_masks"],
            is_ref_regret=False,
            sequence_normalization_type=self.normalization_type
            )
        
        regret_target_rejected = self.get_regret_score(
            RePO_forward_output["rejected_target_model_logprobs"],
            RePO_forward_output["rejected_label_logprobs"],
            RePO_forward_output["rejected_label_masks"],
            is_ref_regret=False,
            sequence_normalization_type=self.normalization_type
            )

        
        if "ref_chosen_target_model_logprobs" in RePO_forward_output:
            regret_ref_chosen = self.get_regret_score(
                RePO_forward_output["ref_chosen_target_model_logprobs"],
                RePO_forward_output["chosen_label_logprobs"],
                RePO_forward_output["chosen_label_masks"],
                is_ref_regret=True, 
                sequence_normalization_type=self.normalization_type
                )

            regret_ref_rejected = self.get_regret_score(
                RePO_forward_output["ref_rejected_target_model_logprobs"],
                RePO_forward_output["rejected_label_logprobs"],
                RePO_forward_output["rejected_label_masks"],
                is_ref_regret=True,
                sequence_normalization_type=self.normalization_type
                )

        else:
            regret_ref_chosen = torch.zeros_like(regret_target_chosen)
            regret_ref_rejected = torch.zeros_like(regret_target_rejected)

        regret_chosen = regret_target_chosen - self.ref_coef * regret_ref_chosen
        regret_rejected = regret_target_rejected - self.ref_coef * regret_ref_rejected
        
        
        chosen_losses = 1 - F.logsigmoid(self.beta * regret_chosen)
        rejected_losses = 1 - F.logsigmoid(-(self.beta * regret_rejected))

        losses_list = []
        if chosen_data_mask:
            losses_list.append(chosen_losses)
        if reject_data_mask:
            losses_list.append(rejected_losses)
        
        losses = torch.cat(losses_list, dim=0) if losses_list else torch.zeros_like(chosen_losses)
        # losses = torch.cat(
        #     (
        #         chosen_losses,
        #         rejected_losses,
        #     ),
        #     0,
        # )
        # losses = - F.logsigmoid(regret_chosen - self.cpl_lambda * regret_rejected)

        loss = losses.mean()

        return loss, regret_chosen, regret_rejected, \
                regret_target_chosen.mean().item(), regret_target_rejected.mean().item(), \
                    regret_ref_chosen.mean().item(), regret_ref_rejected.mean().item()
    
    #TODO: remove torch.isnan after debugging
    def get_regret_score(
        self, 
        logprob_target, 
        logprob_label, 
        label_mask_, 
        is_ref_regret=False,
        sequence_normalization_type='mean'
    ):
    
        # return logprob_target.sum(dim=-1).sum(dim=-1)*0.001
        # logprob_target_masked = torch.multiply(logprob_target, label_mask)
        # chosen_logprob_laebel_masked = torch.multiply(logprob_label, label_mask)
        logprob_target_masked = logprob_target.clone()
        chosen_logprob_laebel_masked = logprob_label
        label_mask = label_mask_
        
        # Flush left to reduce the memory usage and align
        
            # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
            #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
        for i in range(label_mask.size(0)):
            first_one_idx = torch.nonzero(label_mask[i])[0].item()
            logprob_target_masked[i] = torch.roll(logprob_target_masked[i], shifts=-first_one_idx)
            chosen_logprob_laebel_masked[i] = torch.roll(chosen_logprob_laebel_masked[i], shifts=-first_one_idx)
            label_mask[i] = torch.roll(label_mask[i], shifts=-first_one_idx)
        
        # Get the first column idx that is all zeros and remove every column after that
        empty_cols = torch.sum(label_mask, dim=0) == 0
        first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else label_mask.size(1)
        logprob_target_masked = logprob_target_masked[:, :first_empty_col]
        label_mask = label_mask[:, :first_empty_col]
        chosen_logprob_laebel_masked = chosen_logprob_laebel_masked[:, :first_empty_col]


        assert label_mask.dim()==2 and logprob_target_masked.dim()==2 and chosen_logprob_laebel_masked.dim()==2,\
                "The dim of label_mask, logprob_target_masked and chosen_logprob_laebel_masked should be 2"
        assert label_mask.size()==logprob_target_masked.size()==chosen_logprob_laebel_masked.size(),\
                "The size of label_mask, logprob_target_masked and chosen_logprob_laebel_masked should be equal"
        

        # # Method 1: Using For loop for regret score calculation
        # len_chosen_generated_sequence = label_mask.sum(dim=-1)

        # regret_score = 0
        # for t in range(label_mask.size(-1)):
        #     # logprob term
        #     target_logprob_term = logprob_target_masked[:, t]
        #     if torch.isnan(target_logprob_term).any():
        #         print(f"Nan in target_logprob_term: {target_logprob_term}")
        #         import pdb
        #         pdb.set_trace()
        #     # KL divergence term
        #     rollout_window_size = len_chosen_generated_sequence - t - 1
        #     # rollout_window_size = len_chosen_generated_sequence - t
        #     sequence_max_len = len_chosen_generated_sequence.max().item()
        #     kl_divergence_term = torch.zeros_like(target_logprob_term)
        #     for l in range(1, sequence_max_len - t):
        #         # tail of logprob tensor is masked(=0), so just sum up all.
        #         kl_divergence_term = kl_divergence_term + chosen_logprob_laebel_masked[:, t + l] - logprob_target_masked[:, t + l]
        #         if torch.isnan(kl_divergence_term).any():
        #             print(f"Nan in kl_divergence_term: {kl_divergence_term}")
        #             import pdb
        #             pdb.set_trace()
            
            
        #     assert (kl_divergence_term.dim()==1 or kl_divergence_term == 0) and kl_divergence_term.size(0)==label_mask.size(0), \
        #         "The dim and size of kl_divergence_term should be (batch_size,)"  
        #     # normalized by length
        #     # avoid div-by-zero
        #     div_mask = (kl_divergence_term != 0) & (rollout_window_size != 0)
        #     temp_kl_term = torch.zeros_like(kl_divergence_term)
        #     temp_kl_term[div_mask] = kl_divergence_term[div_mask] / rollout_window_size[div_mask]
        #     kl_divergence_term = temp_kl_term
            
        #     ## Second methods for avoid div-by-zero,
        #     # kl_divergence_term = torch.div(kl_divergence_term, rollout_window_size)
        #     # kl_divergence_term[(kl_divergence_term==0) | (rollout_window_size==0)] = 0
        #     if torch.isnan(kl_divergence_term).any():
        #         print(f"Nan in kl_divergence_term AFTER NORMALIZE: {kl_divergence_term}")
        #         import pdb
        #         pdb.set_trace()


        #     # regret score
        #     regret_score = regret_score + target_logprob_term - kl_divergence_term
            
        # if self.normalize_score:
        #     regret_score = torch.div(regret_score, len_chosen_generated_sequence)
        # return regret_score
        
        # Method 2: Using vectorized operation for regret score calculation
            # sequence len (batch_size,)
        len_chosen_generated_sequence = label_mask.sum(dim=-1)

        batch_size, seq_len = label_mask.size()
        device = label_mask.device

        # mask: (batch, seq_len)
        valid_steps = torch.arange(seq_len, device=device).unsqueeze(0) < len_chosen_generated_sequence.unsqueeze(1)

        # (batch, seq_len): Target logprob term
        target_logprob_term = logprob_target_masked.clone()
        target_logprob_term[~valid_steps] = 0

        # (batch, seq_len): KL divergence term calculation
        # kl_term_raw = chosen_logprob_laebel_masked - logprob_target_masked
        kl_term_raw = chosen_logprob_laebel_masked - logprob_target_masked
        kl_term_raw[~valid_steps] = 0

        # cumsum start from back, accumulated kl after 't'
        # flip for back-cumsum 
        kl_cumsum = torch.flip(torch.cumsum(torch.flip(kl_term_raw, dims=[1]), dim=1), dims=[1])

        # calculate rollout window size at t
        rollout_window_sizes = (len_chosen_generated_sequence.unsqueeze(1) - torch.arange(seq_len, device=device).unsqueeze(0)).clamp(min=1)
        rollout_window_sizes = rollout_window_sizes * valid_steps

        # normalize
        kl_divergence_term = torch.zeros_like(kl_cumsum)
        nonzero_mask = rollout_window_sizes != 0
        kl_divergence_term[nonzero_mask] = kl_cumsum[nonzero_mask] / rollout_window_sizes[nonzero_mask]

        # regret score
        if is_ref_regret:
            regret_score_per_step = target_logprob_term
        else:
            regret_score_per_step = target_logprob_term - kl_divergence_term
        
        if sequence_normalization_type == 'mean':
            regret_score = regret_score_per_step.sum(dim=-1)
            regret_score = torch.div(regret_score, len_chosen_generated_sequence)
        elif sequence_normalization_type == 'max':
            regret_score = regret_score_per_step.max(dim=-1)
        else:
            raise ValueError("Not supported normalization type, available options:['mean', 'max']")
        
        
        # # normalize final regret score by length
        # if self.normalize_score:
        #     regret_score = torch.div(regret_score, len_chosen_generated_sequence)

        return regret_score


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L742
class VanillaKTOLoss(nn.Module):
    """
    KTO loss for even sampling
    """

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        ).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class KTOLoss(nn.Module):
    """
    KTO loss for uneven sampling
    """

    def __init__(
        self, beta: float, desirable_weight: float, undesirable_weight: float, world_size: int, device: torch.device
    ) -> None:
        super().__init__()
        self.beta = beta
        self.world_size = world_size
        self.device = device
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0
        ).mean()
        return losses, chosen_rewards, rejected_rewards, KL


# Adapted from https://github.com/microsoft/LMOps/blob/main/minillm/finetune.py#L166
class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != self.IGNORE_INDEX).int()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss


# import logging

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("prm_loss_mistral_1.log"),  # Logs will be saved to this file
#         logging.StreamHandler()  # Logs will also be printed to the console
#     ]
# )

class PRMLoss(nn.Module):
    """
    Process Reward Model Loss
    """

    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.placeholder_token_id = placeholder_token_id
        #WARNING : QWEN tokenizer has space as placeholder token id.
        # QWEN series uses both "¿" and " ¿", so it must be specified with the version that includes a space.
        # self.placeholder_token_id = " " + placeholder_token_id

        self.reward_token_ids = reward_token_ids

    def forward(self, inputs: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, *, return_acc: bool = False):


        placeholder_mask = inputs == self.placeholder_token_id
        logits = logits[placeholder_mask]
        labels = labels[placeholder_mask]


        if labels.dtype == torch.float:
            # soft label
            assert len(self.reward_token_ids) == 2, "reward_token_ids should have 2 tokens for soft labels"
            logits = logits[..., self.reward_token_ids]
            positive_labels = labels.to(logits.dtype)
            negative_labels = 1 - positive_labels
            negative_labels[positive_labels != -100] = 1 - positive_labels[positive_labels != -100]
            labels = torch.stack([positive_labels, negative_labels], dim=-1)
        elif self.reward_token_ids is not None:
            # hard label with reward_token_ids set. (otherwise the whole vocab will be trained together.)
            logits = logits[..., self.reward_token_ids]
            # this is slow....
            for i, token in enumerate(self.reward_token_ids):
                labels = torch.where(labels == token, i, labels)
            # logging.info(f"Hard labels: {labels}")

        loss = self.loss(logits, labels)
        # logging.info(f"Loss: {loss.item()}")

        if not return_acc:
            return loss

        if labels.dtype == logits.dtype:
            labels = labels.argmax(dim=-1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc

