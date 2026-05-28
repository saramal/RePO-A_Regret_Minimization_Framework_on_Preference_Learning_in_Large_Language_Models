import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import os
from tqdm import tqdm

# ---------------------------------------------------------
 
# ---------------------------------------------------------
 
# MODEL_B_PATH = "./checkpoint/dpo_Qwen3-4B-non_sft_masked_pair_lora_32_metamathqa_filtered_1209T09:41_merged"

 
# MODEL_B_PATH = "checkpoint/dpo_Qwen3-4B-non_sft_base_pair_sft0.0_lora_32_metamathqa_filtered_1203T00:43_merged"

 
# MODEL_B_PATH = "./checkpoint/dpo_Qwen3-4B-non_sft_masked_pair_lora_32_metamathqa_filtered_1209T09:41_merged"

# MODEL_A_PATH = "checkpoint/RePO_Qwen/RePO_Qwen3-1.7B-non_sft_base_pair_sft0.0_cpl0.5_lora_32_alpha_1_lr_5e-7_metamathqa_1205T12:22_merged"
# MODEL_B_PATH = "checkpoint/dpo_Qwen3-1.7B-non_sft_base_pair_lora_32_metamathqa_filtered_1125T02:51_merged"

MODEL_A_PATH = "checkpoint/RePO_Qwen/RePO_Qwen3-1.7B-non_sft_base_pair_sft0.0_cpl0.5_lora_32_alpha_1_lr_5e-7_metamathqa_1205T12:22_merged"
MODEL_B_PATH = "checkpoint/dpo_Qwen3-1.7B-non_sft_masked_pair_lora_32_metamathqa_filtered_1130T00:02_merged"


USE_REJECTED = True
USE_CHOSEN = True
 
DATA_PATH = "RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct_dist_batch_new/RePO_test_rank_0.jsonl"

 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def load_model_and_tokenizer(path):
    print(f"Loading model from {path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, 
            device_map=device, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        exit(1)

def get_response_log_probs(model, tokenizer, prompt, response):
    """
    given prompt + response, return log probabilities of each token in the response.
    """
    full_text = prompt + response
    full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    full_ids = full_inputs.input_ids[0]
    
    prompt_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    prompt_len = len(prompt_inputs.input_ids[0])

    with torch.no_grad():
        outputs = model(**full_inputs)
        logits = outputs.logits

    # Shift logits for next token prediction
    shift_logits = logits[0, :-1, :]
    shift_labels = full_ids[1:]

     
    log_probs = F.log_softmax(shift_logits, dim=-1)
    target_log_probs = torch.gather(log_probs, 1, shift_labels.unsqueeze(-1)).squeeze(-1)

     
    start_index = max(0, prompt_len - 1)
    response_log_probs = target_log_probs[start_index:]
    response_ids = shift_labels[start_index:]

    tokens = [tokenizer.decode([tid]) for tid in response_ids]
    scores = response_log_probs.cpu().float().numpy()

    return tokens, scores

def main():
    print(">>> Initializing Models...")
    mod_a, tok_a = load_model_and_tokenizer(MODEL_A_PATH)
    mod_b, tok_b = load_model_and_tokenizer(MODEL_B_PATH)
    print(">>> Models loaded.\n")

    if not os.path.exists(DATA_PATH):
        print(f"Data file {DATA_PATH} not found.")
        return

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    
     
    total_squared_error = 0.0
    total_token_count = 0
    skipped_samples = 0

    total_calculated_samples = 0
    print(f">>> Start calculating MSE over {total_lines} samples...")

    for idx, line in tqdm(enumerate(lines), total=total_lines):
        line = line.strip()
        if not line: continue

        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        prompt = entry.get('prompt', '')
        responses=[]
        if USE_REJECTED:
            responses.append(entry.get('rejected', ''))  
        if USE_CHOSEN:
            responses.append(entry.get('chosen', ''))  
         
        if (idx + 1) % 10 == 0:
            # print(f"[{idx+1}/{total_lines}] Processing...")
            pass
        for response in responses:
             
            tokens_a, scores_a = get_response_log_probs(mod_a, tok_a, prompt, response)
            tokens_b, scores_b = get_response_log_probs(mod_b, tok_b, prompt, response)

             
            if len(scores_a) != len(scores_b):
                 
                # print(f"   [Warning] Index {idx}: Token length mismatch (A:{len(scores_a)}, B:{len(scores_b)}). Skipping.")
                skipped_samples += 1
                continue
            
             
            diff = scores_a - scores_b
            squared_diff_sum = np.sum(diff ** 2)
            
             
            total_squared_error += squared_diff_sum
            total_token_count += len(scores_a)
            total_calculated_samples += 1
    print("\n" + "="*50)
    print(">>> CALCULATION FINISHED")
    print("="*50)

    if total_token_count > 0:
        mse = total_squared_error / total_token_count
        rmse = np.sqrt(mse)
        
        print(f"Total Samples Processed : {total_calculated_samples}")
        print(f"Skipped Samples         : {skipped_samples} (Due to token length mismatch)")
        print(f"Total Tokens Evaluated  : {total_token_count}")
        print(f"-"*30)
        print(f"MSE (Mean Squared Error): {mse:.6f}")
        print(f"RMSE (Root Mean Sq Err) : {rmse:.6f}")
    else:
        print("No valid tokens were processed (Check data path or tokenizers).")

if __name__ == "__main__":
    main()