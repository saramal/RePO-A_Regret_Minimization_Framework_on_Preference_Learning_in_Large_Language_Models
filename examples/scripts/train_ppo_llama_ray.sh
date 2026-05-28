set -x 

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "working_dir": "/openrlhf",
     "excludes": ["wandb", "custom_examples/wandb"]
   }' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
   --reward_pretrain /openrlhf/custom_examples/checkpoint/deepseek-r1-qwen-1.5b-prm \
   --save_path /openrlhf/examples/checkpoint/deepseek-r1-1.5b_debug \
   --micro_train_batch_size 2 \
   --train_batch_size 8 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 32 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data zhuzilin/Math-Shepherd \
   --input_key input \
   --aRePOy_chat_template \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   --use_wandb fd4b71d27293c0b73bfdaf64096853089c9533d2 \
   --wandb_run_name 73_r1_1.5b_ppo_debug \
   --vllm_gpu_memory_utilization 0.5 \


# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
# --remote_rm_url http://localhost:5000/get_reward

