# import os
# import time
# from abc import ABC
# from datetime import timedelta

# import ray
# import torch
# from tqdm import tqdm

# from openrlhf.datasets import PromptDataset
# from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
# from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker
# from openrlhf.trainer.ray.launcher import PPORayActorGroup
# from openrlhf.utils import blending_datasets, get_tokenizer
# from openrlhf.utils.deepspeed import DeepspeedStrategy
# from openrlhf.utils.logging_utils import init_logger
# from openrlhf.utils.remote_rm_utils import remote_rm_fn_ray

# logger = init_logger(__name__)


# @ray.remote
# class PPOTrainer(ABC):
#     """
#     Trainer for Proximal Policy Optimization (PPO) / REINFORCE++ / GRPO / RLOO and their variants.
#     Single Controller with Multiple ActorGroups
#     """

#     def __init__(
#         self,
#         pretrain: str,
#         strategy: DeepspeedStrategy,
#         actor_model_group: PPORayActorGroup,
#         critic_model_group: PPORayActorGroup,
#         reward_model_group: PPORayActorGroup,
#         reference_model_group: PPORayActorGroup,
#         vllm_engines=None,
#         prompt_max_len: int = 120,
#         dataloader_pin_memory: bool = True,
#         prompt_split: str = "train",
#         eval_split: str = "test",
#         **generate_kwargs,
#     ) -> None:
#         super().__init__()

#         self.strategy = strategy
#         self.args = strategy.args

#         self.tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not self.args.disable_fast_tokenizer)
#         self.actor_model_group = actor_model_group
#         self.critic_model_group = critic_model_group
#         self.reward_model_group = reward_model_group
#         self.reference_model_group = reference_model_group
#         self.dataloader_pin_memory = dataloader_pin_memory
#         self.vllm_engines = vllm_engines

#         self.prompt_split = prompt_split
#         self.eval_split = eval_split

#         self.prompt_max_len = prompt_max_len
#         self.generate_kwargs = generate_kwargs

#         self.max_epochs = self.args.max_epochs
#         self.remote_rm_url = self.args.remote_rm_url
#         self.init_kl_coef = self.args.init_kl_coef
#         self.kl_target = self.args.kl_target
#         self.kl_horizon = self.args.kl_horizon

#         self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

#         if self.kl_target:
#             self.kl_ctl = AdaptiveKLController(self.init_kl_coef, self.kl_target, self.kl_horizon)
#         else:
#             self.kl_ctl = FixedKLController(self.init_kl_coef)

#         self.experience_maker = RemoteExperienceMaker(
#             self.actor_model_group,
#             self.critic_model_group,
#             self.reward_model_group,
#             self.reference_model_group,
#             self.tokenizer,
#             self.prompt_max_len,
#             self.kl_ctl,
#             self.strategy,
#             self.remote_rm_url,
#             vllm_engines=self.vllm_engines,
#             packing_samples=self.strategy.args.packing_samples,
#         )

#         self.prepare_datasets()

#         # wandb/tensorboard setting
#         self._wandb = None
#         self._tensorboard = None
#         self.generated_samples_table = None
#         if self.strategy.args.use_wandb:
#             import wandb

#             self._wandb = wandb
#             if not wandb.api.api_key:
#                 wandb.login(key=self.strategy.args.use_wandb)
#             wandb.init(
#                 entity=self.strategy.args.wandb_org,
#                 project=self.strategy.args.wandb_project,
#                 group=self.strategy.args.wandb_group,
#                 name=self.strategy.args.wandb_run_name,
#                 config=self.strategy.args.__dict__,
#                 reinit=True,
#             )

#             wandb.define_metric("train/global_step")
#             wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
#             wandb.define_metric("eval/epoch")
#             wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)
#             self.generated_samples_table = wandb.Table(columns=["global_step", "text", "reward"])

#         # Initialize TensorBoard writer if wandb is not available
#         if self.strategy.args.use_tensorboard and self._wandb is None:
#             from torch.utils.tensorboard import SummaryWriter

#             os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
#             log_dir = os.path.join(self.strategy.args.use_tensorboard, self.strategy.args.wandb_run_name)
#             self._tensorboard = SummaryWriter(log_dir=log_dir)

#     def fit(
#         self,
#     ) -> None:
#         args = self.args

#         # Load datasets
#         num_rollouts_per_episodes = len(self.prompts_dataloader)

#         # get eval and save steps
#         if args.eval_steps == -1:
#             args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
#         if args.save_steps == -1:
#             args.save_steps = float("inf")  # do not save ckpt

#         # broadcast init checkpoint to vllm
#         ckpt_path = os.path.join(args.ckpt_path, "_actor")
#         if args.load_checkpoint and os.path.exists(ckpt_path) and not self.vllm_engines is None:
#             # vLLM wakeup when vllm_enable_sleep
#             if self.strategy.args.vllm_enable_sleep:
#                 from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

#                 batch_vllm_engine_call(self.vllm_engines, "wake_up")

#             ref = self.actor_model_group.async_run_method(method_name="broadcast_to_vllm")
#             ray.get(ref)

#             # vLLM offload when vllm_enable_sleep
#             if self.strategy.args.vllm_enable_sleep:
#                 batch_vllm_engine_call(self.vllm_engines, "sleep")

#         # Restore step and start_epoch
#         consumed_samples = ray.get(self.actor_model_group.async_run_method(method_name="get_consumed_samples"))[0]
#         steps = consumed_samples // args.rollout_batch_size + 1
#         start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
#         consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

#         for episode in range(start_episode, args.num_episodes):
#             self.prompts_dataloader.sampler.set_epoch(
#                 episode, consumed_samples=0 if episode > start_episode else consumed_samples
#             )
#             pbar = tqdm(
#                 range(self.prompts_dataloader.__len__()),
#                 desc=f"Episode [{episode + 1}/{args.num_episodes}]",
#                 disable=False,
#             )

#             for _, rand_prompts, labels in self.prompts_dataloader:
#                 experiences = self.experience_maker.make_experience_list(rand_prompts, labels, **self.generate_kwargs)
#                 sample0 = self.tokenizer.batch_decode(
#                     experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True
#                 )
#                 print(sample0)
#                 refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
#                 if self.critic_model_group is not None:
#                     refs.extend(
#                         self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences)
#                     )
#                 ray.get(refs)

#                 status = self.ppo_train(steps)

#                 if "kl" in status:
#                     self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
#                 pbar.set_postfix(status)

#                 # Add generated samples to status dictionary
#                 status["generated_samples"] = [sample0[0], experiences[0].info["reward"][0]]
#                 # logs/checkpoints
#                 client_states = {"consumed_samples": steps * args.rollout_batch_size}
#                 self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

#                 pbar.update()
#                 steps = steps + 1

#         if self._wandb is not None:
#             self._wandb.finish()
#         if self._tensorboard is not None:
#             self._tensorboard.close()

#     def ppo_train(self, global_steps):
#         status = {}

#         # triger remote critic model training
#         if self.critic_model_group is not None:
#             # sync for deepspeed_enable_sleep
#             if self.strategy.args.deepspeed_enable_sleep:
#                 ray.get(self.critic_model_group.async_run_method(method_name="reload_states"))

#             critic_status_ref = self.critic_model_group.async_run_method(method_name="fit")

#             if self.strategy.args.colocate_all_models or self.strategy.args.deepspeed_enable_sleep:
#                 status.update(ray.get(critic_status_ref)[0])
#             if self.strategy.args.deepspeed_enable_sleep:
#                 ray.get(self.critic_model_group.async_run_method(method_name="offload_states"))

#         # actor model training
#         if global_steps > self.freezing_actor_steps:
#             if self.strategy.args.deepspeed_enable_sleep:
#                 self.actor_model_group.async_run_method(method_name="reload_states")

#             actor_status_ref = self.actor_model_group.async_run_method(method_name="fit", kl_ctl=self.kl_ctl.value)
#             status.update(ray.get(actor_status_ref)[0])

#             if self.strategy.args.deepspeed_enable_sleep:
#                 self.actor_model_group.async_run_method(method_name="offload_states")

#             # 4. broadcast weights to vllm engines
#             if self.vllm_engines is not None:
#                 if self.strategy.args.vllm_enable_sleep:
#                     from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

#                     batch_vllm_engine_call(self.vllm_engines, "wake_up")

#                 ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

#                 if self.strategy.args.vllm_enable_sleep:
#                     batch_vllm_engine_call(self.vllm_engines, "sleep")

#         # 5. wait remote critic model training done
#         if self.critic_model_group and not self.strategy.args.colocate_all_models:
#             status.update(ray.get(critic_status_ref)[0])

#         return status

#     def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
#         if global_step % args.logging_steps == 0:
#             # wandb
#             if self._wandb is not None:
#                 # Add generated samples to wandb using Table
#                 if "generated_samples" in logs_dict:
#                     # https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
#                     new_table = self._wandb.Table(
#                         columns=self.generated_samples_table.columns, data=self.generated_samples_table.data
#                     )
#                     new_table.add_data(global_step, *logs_dict.pop("generated_samples"))
#                     self.generated_samples_table = new_table
#                     self._wandb.log({"train/generated_samples": new_table})
#                 logs = {
#                     "train/%s" % k: v
#                     for k, v in {
#                         **logs_dict,
#                         "global_step": global_step,
#                     }.items()
#                 }
#                 self._wandb.log(logs)
#             # TensorBoard
#             elif self._tensorboard is not None:
#                 for k, v in logs_dict.items():
#                     if k == "generated_samples":
#                         # Record generated samples in TensorBoard using simple text format
#                         text, reward = v
#                         formatted_text = f"Sample:\n{text}\n\nReward: {reward:.4f}"
#                         self._tensorboard.add_text("train/generated_samples", formatted_text, global_step)
#                     else:
#                         self._tensorboard.add_scalar(f"train/{k}", v, global_step)

#         # TODO: Add evaluation mechanism for PPO
#         if global_step % args.eval_steps == 0 and self.eval_dataloader and len(self.eval_dataloader) > 0:
#             logger.info(f"Evaluating model at step {global_step}...")
#             self.evaluate(self.eval_dataloader, global_step, args.eval_temperature, args.eval_n_samples_per_prompt)
#         # save ckpt
#         # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
#         if global_step % args.save_steps == 0:
#             tag = f"global_step{global_step}"
#             ref = self.actor_model_group.async_run_method(
#                 method_name="save_checkpoint", tag=tag, client_states=client_states
#             )
#             if self.critic_model_group is not None:
#                 ref.extend(self.critic_model_group.async_run_method(method_name="save_checkpoint", tag=tag))
#             ray.get(ref)

#     def evaluate(self, eval_dataloader, global_step, temperature=0.6, n_samples_per_prompt=1):
#         """Evaluate model performance on eval dataset.

#         Args:
#             eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
#             global_step: Current training step for logging
#             n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
#         """
#         start_time = time.time()
#         logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

#         # vLLM wakeup when vllm_enable_sleep
#         if self.strategy.args.vllm_enable_sleep:
#             from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

#             batch_vllm_engine_call(self.vllm_engines, "wake_up")

#         with torch.no_grad():
#             # First collect all prompts and labels
#             all_prompts = []
#             all_labels = []
#             all_datasources = []

#             for datasources, prompts, labels in eval_dataloader:
#                 all_prompts.extend(prompts)
#                 all_labels.extend(labels)
#                 all_datasources.extend(datasources)

#             # Generate samples and calculate rewards
#             generate_kwargs = self.generate_kwargs.copy()
#             generate_kwargs["temperature"] = temperature
#             generate_kwargs["n_samples_per_prompt"] = n_samples_per_prompt
#             samples = self.experience_maker.generate_samples(all_prompts, all_labels, **generate_kwargs)
#             queries_list = self.tokenizer.batch_decode(samples.sequences, skip_special_tokens=False)

#             # duplicate prompts and labels for each sample
#             all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in all_prompts], [])
#             all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])

#             # Calculate rewards
#             if self.experience_maker.custom_reward_func:
#                 # Let Ray automatically distribute the workload across available resources
#                 batch_size = self.strategy.args.micro_rollout_batch_size
#                 num_chunks = (len(queries_list) + batch_size - 1) // batch_size
#                 r_refs = []
#                 for i in range(num_chunks):
#                     start_idx = i * batch_size
#                     end_idx = min((i + 1) * batch_size, len(queries_list))
#                     r = self.experience_maker.custom_reward_func.remote(
#                         queries_list[start_idx:end_idx],
#                         all_prompts[start_idx:end_idx],
#                         all_labels[start_idx:end_idx],
#                     )
#                     r_refs.append(r)
#             else:
#                 # Distribute data across different remote reward function servers
#                 num_servers = len(self.remote_rm_url)
#                 batch_size = (len(queries_list) + num_servers - 1) // num_servers
#                 r_refs = []
#                 for i in range(num_servers):
#                     start_idx = i * batch_size
#                     end_idx = min((i + 1) * batch_size, len(queries_list))
#                     rm = self.remote_rm_url[i]
#                     r = remote_rm_fn_ray.remote(
#                         rm,
#                         queries=queries_list[start_idx:end_idx],
#                         prompts=all_prompts[start_idx:end_idx],
#                         labels=all_labels[start_idx:end_idx],
#                     )
#                     r_refs.append(r)

#             # Reshape rewards to (num_prompts, n_samples_per_prompt)
#             rewards = ray.get(r_refs)
#             rewards = torch.cat(rewards, dim=0).reshape(-1, n_samples_per_prompt)

#             # Collect local statistics for each data source
#             global_metrics = {}  # {datasource: {"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}}

#             for i, datasource in enumerate(all_datasources):
#                 if datasource not in global_metrics:
#                     global_metrics[datasource] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}

#                 # Calculate pass@k and pass@1
#                 prompt_rewards = rewards[i]
#                 if n_samples_per_prompt > 1:
#                     global_metrics[datasource][f"pass{n_samples_per_prompt}"] += prompt_rewards.max().float().item()
#                 global_metrics[datasource]["pass1"] += prompt_rewards.mean().float().item()
#                 global_metrics[datasource]["count"] += 1

#             # Calculate global averages
#             logs = {}
#             for datasource, metrics in global_metrics.items():
#                 logs[f"eval_{datasource}_pass{n_samples_per_prompt}"] = (
#                     metrics[f"pass{n_samples_per_prompt}"] / metrics["count"]
#                 )
#                 logs[f"eval_{datasource}_pass1"] = metrics["pass1"] / metrics["count"]

#             # Log to wandb/tensorboard
#             if self._wandb is not None:
#                 logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
#                 self._wandb.log(logs)
#             elif self._tensorboard is not None:
#                 for k, v in logs.items():
#                     self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

#         if self.strategy.args.vllm_enable_sleep:
#             batch_vllm_engine_call(self.vllm_engines, "sleep")

#         end_time = time.time()
#         duration = end_time - start_time
#         time_str = str(timedelta(seconds=duration)).split(".")[0]
#         logger.info(f"✨ Evaluation completed in {time_str}")

#     def prepare_datasets(self):
#         args = self.args
#         strategy = self.strategy

#         # prepare datasets
#         train_data = blending_datasets(
#             args.prompt_data,
#             args.prompt_data_probs,
#             strategy,
#             args.seed,
#             max_count=args.max_samples,
#             dataset_split=self.prompt_split,
#         )

#         # Create train dataset
#         train_data = train_data.select(range(min(args.max_samples, len(train_data))))
#         prompts_dataset = PromptDataset(train_data, self.tokenizer, strategy, input_template=args.input_template)
#         prompts_dataloader = strategy.setup_dataloader(
#             prompts_dataset,
#             args.rollout_batch_size,
#             True,
#             True,
#         )

#         # Create eval dataset if eval data exists
#         if getattr(args, "eval_dataset", None):
#             eval_data = blending_datasets(
#                 args.eval_dataset,
#                 None,  # No probability sampling for eval datasets
#                 strategy,
#                 dataset_split=self.eval_split,
#             )
#             eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
#             eval_dataset = PromptDataset(eval_data, self.tokenizer, strategy, input_template=args.input_template)
#             eval_dataloader = strategy.setup_dataloader(eval_dataset, 1, True, False)
#         else:
#             eval_dataloader = None

#         self.prompts_dataloader = prompts_dataloader
#         self.eval_dataloader = eval_dataloader
#         self.max_steps = (
#             len(prompts_dataset)
#             * args.n_samples_per_prompt
#             // args.train_batch_size
#             * args.num_episodes
#             * args.max_epochs
#         )

#     def get_max_steps(self):
#         return self.max_steps



from abc import ABC
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, ValueLoss

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveReplayBuffer


class BasePPOTrainer(ABC):
    """
    Base Trainer for Proximal Policy Optimization (PPO) algorithm.

    Args:
        strategy (Strategy): The training strategy to use.
        actor (Actor): The actor model in the PPO algorithm.
        critic (nn.Module): The critic model in the PPO algorithm.
        reward_model (nn.Module): The reward model for calculating rewards in the RLHF setup.
        initial_model (Actor): The initial model for reference logits to limit actor updates in RLHF.
        ema_model (Actor): The exponential moving average model for stable training.
        actor_optim (Optimizer): The optimizer for the actor model.
        critic_optim (Optimizer): The optimizer for the critic model.
        actor_scheduler (Scheduler): The learning rate scheduler for the actor.
        critic_scheduler (Scheduler): The learning rate scheduler for the critic.
        ema_beta (float, defaults to 0.992): EMA decay rate for model stability.
        init_kl_coef (float, defaults to 0.001): Initial coefficient for KL divergence.
        kl_target (float, optional): Target value for KL divergence.
        kl_horizon (int, defaults to 10000): Horizon for KL annealing.
        ptx_coef (float, defaults to 0): Coefficient for supervised loss from pre-trained data.
        micro_train_batch_size (int, defaults to 8): Micro-batch size for actor training.
        buffer_limit (int, defaults to 0): Maximum size of the replay buffer.
        buffer_cpu_offload (bool, defaults to True): If True, offloads replay buffer to CPU.
        eps_clip (float, defaults to 0.2): Clipping coefficient for policy loss.
        value_clip (float, defaults to 0.2): Clipping coefficient for value function loss.
        micro_rollout_batch_size (int, defaults to 8): Micro-batch size for generating rollouts.
        gradient_checkpointing (bool, defaults to False): If True, enables gradient checkpointing.
        max_epochs (int, defaults to 1): Number of epochs to train.
        max_norm (float, defaults to 1.0): Maximum gradient norm for gradient clipping.
        tokenizer (Callable, optional): Tokenizer for input data.
        prompt_max_len (int, defaults to 128): Maximum length for prompts.
        dataloader_pin_memory (bool, defaults to True): If True, pins memory in the data loader.
        remote_rm_url (str, optional): URL for remote reward model API.
        reward_fn (Callable, optional): Custom reward function for computing rewards.
        save_hf_ckpt (bool): Whether to save huggingface-format model weight.
        disable_ds_ckpt (bool): Whether not to save deepspeed-format model weight. (Deepspeed model weight is used for training recovery)
        **generate_kwargs: Additional arguments for model generation.
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
        **generate_kwargs,
    ) -> None:
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn

        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size, buffer_limit, buffer_cpu_offload, getattr(self.args, "packing_samples", False)
        )

    def ppo_train(self, global_steps=0):
        raise NotImplementedError("This method should be implemented by the subclass.")

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        raise NotImplementedError("This method should be implemented by the subclass.")
