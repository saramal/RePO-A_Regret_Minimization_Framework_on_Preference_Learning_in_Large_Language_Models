#!/bin/bash

 
echo "=== Single Parameter Sweep Example ==="

# Learning rate sweep
learning_rates=(1e-6 5e-6 1e-5 2e-5 5e-5)

for lr in "${learning_rates[@]}"; do
    echo "Running experiment with learning_rate=$lr"
    bash train_sft_sweep.sh \
        --learning_rate $lr \
        --wandb_run_name "sft_qwen3.4B_lr_${lr}"
done

echo ""
echo "=== Multiple Parameter Sweep Example ==="

 
learning_rates=(1e-6 5e-6 1e-5)
lora_ranks=(64 128 256)
batch_sizes=(16 32 64)

for lr in "${learning_rates[@]}"; do
    for rank in "${lora_ranks[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            echo "Running experiment: lr=$lr, lora_rank=$rank, batch_size=$batch_size"
            bash train_sft_sweep.sh \
                --learning_rate $lr \
                --lora_rank $rank \
                --train_batch_size $batch_size \
                --wandb_run_name "sft_lr${lr}_rank${rank}_bs${batch_size}"
        done
    done
done

echo ""
echo "=== Model Sweep Example ==="

 
models=(
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct" 
    "Qwen/Qwen2.5-7B-Instruct"
)

datasets=(
    "./RePO_datasets/MetamathQA/RePO_train.jsonl"
    "./RePO_datasets/stepDPO/RePO_train.jsonl"
)

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
         
        model_name=$(echo $model | sed 's/.*\///g' | sed 's/[^a-zA-Z0-9]/_/g')
        dataset_name=$(echo $dataset | sed 's/.*\///g' | sed 's/\.jsonl//g')
        
        echo "Running experiment: model=$model, dataset=$dataset"
        bash train_sft_sweep.sh \
            --pretrain "$model" \
            --dataset "$dataset" \
            --wandb_run_name "sft_${model_name}_${dataset_name}"
    done
done

echo ""
echo "=== Configuration-based Sweep Example ==="

 
configs=(
    "1e-6 64 16"
    "5e-6 128 32" 
    "1e-5 256 64"
    "2e-5 128 32"
)

for config in "${configs[@]}"; do
     
    config_array=($config)
    lr=${config_array[0]}
    lora_rank=${config_array[1]}
    batch_size=${config_array[2]}
    
    echo "Running predefined config: lr=$lr, lora_rank=$lora_rank, batch_size=$batch_size"
    bash train_sft_sweep.sh \
        --learning_rate $lr \
        --lora_rank $lora_rank \
        --train_batch_size $batch_size \
        --wandb_run_name "sft_config_${lr}_${lora_rank}_${batch_size}"
done

echo "Parameter sweep completed!"
