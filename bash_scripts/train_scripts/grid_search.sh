#!/bin/bash

 
 

 
learning_rates=(1e-6 5e-6 1e-5)
lora_ranks=(64 128 256)
batch_sizes=(16 32)
max_epochs=(3 5)

 
total_experiments=$((${#learning_rates[@]} * ${#lora_ranks[@]} * ${#batch_sizes[@]} * ${#max_epochs[@]}))
current_experiment=0

echo "🔬 Starting Grid Search Parameter Sweep"
echo "Parameters:"
echo "  - Learning rates: ${learning_rates[@]}"
echo "  - LoRA ranks: ${lora_ranks[@]}"
echo "  - Batch sizes: ${batch_sizes[@]}"
echo "  - Max epochs: ${max_epochs[@]}"
echo "  - Total experiments: $total_experiments"
echo ""

 
results_file="grid_search_results.csv"
echo "experiment_id,learning_rate,lora_rank,batch_size,max_epochs,wandb_run_name,status,start_time,end_time" > $results_file

 
for lr in "${learning_rates[@]}"; do
    for rank in "${lora_ranks[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for epochs in "${max_epochs[@]}"; do
                current_experiment=$((current_experiment + 1))
                
                 
                exp_id=$(printf "exp_%03d" $current_experiment)
                run_name="grid_lr${lr}_r${rank}_bs${batch_size}_ep${epochs}"
                
                echo "[$current_experiment/$total_experiments] Running $exp_id"
                echo "  Config: lr=$lr, lora_rank=$rank, batch_size=$batch_size, epochs=$epochs"
                
                 
                start_time=$(date '+%Y-%m-%d %H:%M:%S')
                
                 
                if bash train_sft_sweep.sh \
                    --learning_rate $lr \
                    --lora_rank $rank \
                    --train_batch_size $batch_size \
                    --max_epochs $epochs \
                    --wandb_run_name "$run_name"; then
                    
                    status="SUCCESS"
                    echo "  ✅ $exp_id completed successfully"
                else
                    status="FAILED"
                    echo "  ❌ $exp_id failed"
                fi
                
                 
                end_time=$(date '+%Y-%m-%d %H:%M:%S')
                
                 
                echo "$exp_id,$lr,$rank,$batch_size,$epochs,$run_name,$status,$start_time,$end_time" >> $results_file
                
                echo "  Status: $status"
                echo "  Progress: $current_experiment/$total_experiments ($(( current_experiment * 100 / total_experiments ))%)"
                echo ""
                
                 
                if [ $current_experiment -lt $total_experiments ]; then
                    echo "  Waiting 30 seconds before next experiment..."
                    sleep 30
                fi
            done
        done
    done
done

echo "🎉 Grid Search completed!"
echo "Results saved to: $results_file"
echo ""
echo "Summary:"
echo "  Total experiments: $total_experiments"
echo "  Successful: $(grep -c SUCCESS $results_file)"
echo "  Failed: $(grep -c FAILED $results_file)"
