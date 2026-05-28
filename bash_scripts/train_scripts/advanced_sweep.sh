#!/bin/bash

 

 
check_gpu_memory() {
    local threshold=80   
    local gpu_usage=$(nvidia-smi --query-gpu=memory.percent --format=csv,noheader,nounits | head -1)
    
    while [ "$gpu_usage" -gt "$threshold" ]; do
        echo "GPU memory usage is ${gpu_usage}%. Waiting..."
        sleep 30
        gpu_usage=$(nvidia-smi --query-gpu=memory.percent --format=csv,noheader,nounits | head -1)
    done
}

 
log_experiment() {
    local config="$1"
    local status="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $config - $status" >> experiment_log.txt
}

 
declare -A experiments

 
experiments["lr_sweep_1"]="--learning_rate 1e-6 --wandb_run_name lr_1e6"
experiments["lr_sweep_2"]="--learning_rate 5e-6 --wandb_run_name lr_5e6"
experiments["lr_sweep_3"]="--learning_rate 1e-5 --wandb_run_name lr_1e5"

 
experiments["lora_1"]="--lora_rank 64 --lora_dropout 0.1 --wandb_run_name lora_64_01"
experiments["lora_2"]="--lora_rank 128 --lora_dropout 0.05 --wandb_run_name lora_128_005"
experiments["lora_3"]="--lora_rank 256 --lora_dropout 0.02 --wandb_run_name lora_256_002"

 
experiments["model_qwen15"]="--pretrain Qwen/Qwen2.5-1.5B-Instruct --wandb_run_name qwen15b"
experiments["model_qwen3"]="--pretrain Qwen/Qwen2.5-3B-Instruct --wandb_run_name qwen3b"

echo "Starting advanced parameter sweep..."
echo "Total experiments: ${#experiments[@]}"

 
for exp_name in "${!experiments[@]}"; do
    echo ""
    echo "=== Running experiment: $exp_name ==="
    
     
    check_gpu_memory
    
     
    log_experiment "$exp_name" "STARTED"
    
     
    echo "Command: bash train_sft_sweep.sh ${experiments[$exp_name]}"
    
    if bash train_sft_sweep.sh ${experiments[$exp_name]}; then
        log_experiment "$exp_name" "SUCCESS"
        echo "✅ Experiment $exp_name completed successfully"
    else
        log_experiment "$exp_name" "FAILED"
        echo "❌ Experiment $exp_name failed"
        
         
        # read -p "Continue with next experiment? (y/n): " continue_choice
        # if [[ $continue_choice != "y" ]]; then
        #     break
        # fi
    fi
    
     
    echo "Waiting 10 seconds before next experiment..."
    sleep 10
done

echo ""
echo "🎉 All experiments completed!"
echo "Check experiment_log.txt for detailed results."
