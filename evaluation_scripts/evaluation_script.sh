


#TODO: change the model name, save_path, and prompt
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m evaluation_scripts.eval_math \
    --model Qwen/Qwen2.5-Math-1.5B \
    --data_file ./data/test/GSM8K_test_data.jsonl \
    --save_path 'eval_results/gsm8k/qwen2-72b-instruct-step-dpo.json' \
    --prompt 'qwen2-boxed' \
    --tensor_parallel_size 4