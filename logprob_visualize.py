import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import math
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import math
import os
import textwrap

from matplotlib.colors import TwoSlopeNorm
# ---------------------------------------------------------
 
# ---------------------------------------------------------
MODEL_A_PATH = "Qwen/Qwen3-4B-Base"   
 
# MODEL_A_PATH = "./checkpoint/dpo_Qwen3-4B-non_sft_masked_pair_lora_32_metamathqa_filtered_1209T09:41_merged"



# MODEL_B_PATH = "./checkpoint/dpo_Qwen3-4B-non_sft_masked_pair_lora_32_metamathqa_filtered_1209T09:41_merged"
 
MODEL_B_PATH = "checkpoint/RePO_Qwen/RePO_Qwen3-4B-non_sft_base_pair_sft0.0_cpl0.5_lora_32_alpha_1_lr_5e-7_metamathqa_1205T00:10_merged"   

# INPUT_TEXT = "The quick brown fox jumps over the lazy dog."
DATA_PATH = "RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct_dist_batch_new/RePO_test_rank_0.jsonl"
# OUTPUT_IMAGE_PATH = "model_comparison_logprobs.png"

PROMPT_TEMPLATE = problem_prompt = (
                "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
                "<|im_start|>user\n{input}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
OUTPUT_DIR = "visualization/comparison_results_absolute_ref_RePO"
TOKENS_PER_ROW = 40
OUTLIER_CUTOFF = 0
TEXT_WRAP_WIDTH = 120
START_IDX = 0
USE_CHOSEN = True
USE_REJECTED = False
 
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
    full_text = prompt + response
    full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    full_ids = full_inputs.input_ids[0]
    
    prompt_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    prompt_len = len(prompt_inputs.input_ids[0])

    with torch.no_grad():
        outputs = model(**full_inputs)
        logits = outputs.logits

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

def plot_wrapped_heatmap(tokens, scores, ax, title, vmin, vmax, cmap="RdYlGn"):
    num_tokens = len(tokens)
    num_rows = math.ceil(num_tokens / TOKENS_PER_ROW)
    
    grid_scores = np.full((num_rows, TOKENS_PER_ROW), np.nan)
    grid_labels = np.full((num_rows, TOKENS_PER_ROW), "", dtype=object)

    for i in range(num_tokens):
        row = i // TOKENS_PER_ROW
        col = i % TOKENS_PER_ROW
        grid_scores[row, col] = scores[i]
        grid_labels[row, col] = tokens[i]

    sns.heatmap(
        grid_scores, annot=grid_labels, fmt="", cmap=cmap,
        ax=ax, cbar=False, 
        vmin=vmin, vmax=vmax,
        xticklabels=False, yticklabels=False,
        linewidths=0.5, linecolor='white'
    )
    ax.set_title(title)

def plot_wrapped_heatmap_v2(tokens, scores, ax, title, vmin, vmax, cmap="RdYlGn", tokens_per_row=10):
    """
    tokens_per_row: number of tokens to display per row (default 10)
    """
    num_tokens = len(tokens)
    num_rows = math.ceil(num_tokens / tokens_per_row)
    
     
    grid_scores = np.full((num_rows, tokens_per_row), np.nan)
    grid_labels = np.full((num_rows, tokens_per_row), "", dtype=object)

    for i in range(num_tokens):
        row = i // tokens_per_row
        col = i % tokens_per_row
        grid_scores[row, col] = scores[i]
        grid_labels[row, col] = tokens[i]

     
     
     
     
    current_width, _ = ax.figure.get_size_inches()
    new_height = num_rows * 0.6   
     
    new_height = max(new_height, 3) 
    ax.figure.set_size_inches(current_width, new_height)

    sns.heatmap(
        grid_scores, 
        annot=grid_labels, 
        fmt="", 
        cmap=cmap,
        ax=ax, 
        cbar=False, 
        vmin=vmin, 
        vmax=vmax,
        xticklabels=False, 
        yticklabels=False,
        linewidths=0.5, 
        linecolor='white',
         
        annot_kws={"size": 10, "va": "center", "ha": "center"} 
    )
    
    ax.set_title(title, pad=20)

def save_diff_heatmap_only(tokens, scores, prompt, output_path, outlier_cutoff=5):
     
    diff_p_low = np.percentile(scores, outlier_cutoff)
    diff_p_high = np.percentile(scores, 100 - outlier_cutoff)
    limit = max(abs(diff_p_low), abs(diff_p_high), 0.5)
    vmin, vmax = -limit, limit

     
     
    FIG_WIDTH_INCHES = 16    
    TOKENS_PER_ROW = 12      
    
     
     
     
    ROW_HEIGHT_INCHES = (FIG_WIDTH_INCHES / TOKENS_PER_ROW) * 0.35 

    num_tokens = len(tokens)
    num_rows = math.ceil(num_tokens / TOKENS_PER_ROW)
    
     
    grid_scores = np.full((num_rows, TOKENS_PER_ROW), np.nan)
    grid_labels = np.full((num_rows, TOKENS_PER_ROW), "", dtype=object)

    for i in range(num_tokens):
        r, c = divmod(i, TOKENS_PER_ROW)
        grid_scores[r, c] = scores[i]
        grid_labels[r, c] = tokens[i]

     
     
    wrapped_prompt = textwrap.fill(prompt, width=100)  
    prompt_lines = wrapped_prompt.count('\n') + 1
    prompt_height_inches = prompt_lines * 0.3 + 1.0  

     
    heatmap_height = num_rows * ROW_HEIGHT_INCHES
    total_height = heatmap_height + prompt_height_inches + 2.0  

     
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, total_height))

     
     
    title_y_pos = 1.0 - (prompt_height_inches / total_height)
    fig.suptitle(
        f"{wrapped_prompt}", 
        x=0.01, y=0.99, 
        ha='left', va='top', 
        fontsize=14, fontfamily='monospace', fontweight='bold'
    )

     
     
     
    plt.subplots_adjust(top=title_y_pos - 0.02, bottom=0.05)

    sns.heatmap(
        grid_scores, annot=grid_labels, fmt="", cmap="RdYlGn",
        ax=ax, cbar=False, vmin=vmin, vmax=vmax,
        xticklabels=False, yticklabels=False,
        linewidths=0.5, linecolor='white',
        annot_kws={"size": 11, "va": "center", "ha": "center"}
    )
    
    ax.set_title("Diff (Model B - Model A) [Green: B better, Red: A better]", pad=10)

     
     
     
    cbar_height = min(0.3, heatmap_height * 0.5)  
    cbar_bottom = (title_y_pos / 2) - (cbar_height / 2)
    
    cbar_ax = fig.add_axes([0.91, cbar_bottom, 0.015, cbar_height]) 
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Diff Score (B - A)')

     
    plt.savefig(output_path, dpi=150, bbox_inches='tight')  
    plt.close(fig)
    print(f"   Saved Diff plot to: {output_path}")


def save_diff_heatmap_only_unnormalized(tokens, scores, prompt, output_path, outlier_cutoff=OUTLIER_CUTOFF):
     
     
    p_low = np.percentile(scores, outlier_cutoff)
    p_high = np.percentile(scores, 100 - outlier_cutoff)
    
     
     
    scores_clipped = np.clip(scores, p_low, p_high)

     
     
     
     
    
     
    actual_min = min(scores_clipped.min(), -0.00000001)
    
     
    actual_max = max(scores_clipped.max(), 0.000000001)

     
    norm = TwoSlopeNorm(vmin=actual_min, vcenter=0, vmax=actual_max)
     

     
    FIG_WIDTH_INCHES = 16 
    TOKENS_PER_ROW = 12 
    ROW_HEIGHT_INCHES = (FIG_WIDTH_INCHES / TOKENS_PER_ROW) * 0.35 

    num_tokens = len(tokens)
    num_rows = math.ceil(num_tokens / TOKENS_PER_ROW)
    
    grid_scores = np.full((num_rows, TOKENS_PER_ROW), np.nan)
    grid_labels = np.full((num_rows, TOKENS_PER_ROW), "", dtype=object)

    for i in range(num_tokens):
        r, c = divmod(i, TOKENS_PER_ROW)
        grid_scores[r, c] = scores_clipped[i]  
        grid_labels[r, c] = tokens[i]

     
    wrapped_prompt = textwrap.fill(prompt, width=100)
    prompt_lines = wrapped_prompt.count('\n') + 1
    prompt_height_inches = prompt_lines * 0.3 + 1.0

    heatmap_height = num_rows * ROW_HEIGHT_INCHES
    total_height = heatmap_height + prompt_height_inches + 2.0

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, total_height))

    title_y_pos = 1.0 - (prompt_height_inches / total_height)
    fig.suptitle(
        f"{wrapped_prompt}", 
        x=0.01, y=0.99, 
        ha='left', va='top', 
        fontsize=14, fontfamily='monospace', fontweight='bold'
    )

    plt.subplots_adjust(top=title_y_pos - 0.02, bottom=0.05)

     
    sns.heatmap(
        grid_scores, annot=grid_labels, fmt="", cmap="RdYlGn",
        ax=ax, cbar=False, 
        norm=norm,  
        xticklabels=False, yticklabels=False,
        linewidths=0.5, linecolor='white',
        annot_kws={"size": 11, "va": "center", "ha": "center"}
    )
    
    ax.set_title("Diff (Model B - Model A) [Green: B better, Red: A better]", pad=10)

     
    cbar_height = min(0.3, heatmap_height * 0.5)
    cbar_bottom = (title_y_pos / 2) - (cbar_height / 2)
    cbar_ax = fig.add_axes([0.91, cbar_bottom, 0.015, cbar_height]) 
    
     
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Diff Score (B - A)')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved Diff plot to: {output_path}")


def save_diff_heatmap_absolute(tokens, scores, prompt, output_path, abs_limit=2.0):
    """
    abs_limit: absolute score that determines the maximum intensity of the color.
               regardless of whether the data is 100 or 10,
               scores above abs_limit (e.g. 2.0) are always displayed in the most intense color.
               this value needs to be fixed to enable absolute color comparison between multiple images.
    """
    
     
     
     
    target_vmin = -abs_limit
    target_vmax = abs_limit

     
     
    norm = TwoSlopeNorm(vmin=target_vmin, vcenter=0, vmax=target_vmax)

     
    FIG_WIDTH_INCHES = 16 
    TOKENS_PER_ROW = 12 
    ROW_HEIGHT_INCHES = (FIG_WIDTH_INCHES / TOKENS_PER_ROW) * 0.35 

    num_tokens = len(tokens)
    num_rows = math.ceil(num_tokens / TOKENS_PER_ROW)
    
    grid_scores = np.full((num_rows, TOKENS_PER_ROW), np.nan)
    grid_labels = np.full((num_rows, TOKENS_PER_ROW), "", dtype=object)

    for i in range(num_tokens):
        r, c = divmod(i, TOKENS_PER_ROW)
        grid_scores[r, c] = scores[i]  
        grid_labels[r, c] = tokens[i]

     
    wrapped_prompt = textwrap.fill(prompt, width=100)
    prompt_lines = wrapped_prompt.count('\n') + 1
    prompt_height_inches = prompt_lines * 0.3 + 1.0

    heatmap_height = num_rows * ROW_HEIGHT_INCHES
    total_height = heatmap_height + prompt_height_inches + 2.0

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, total_height))

    title_y_pos = 1.0 - (prompt_height_inches / total_height)
    fig.suptitle(
        f"{wrapped_prompt}", 
        x=0.01, y=0.99, ha='left', va='top', 
        fontsize=14, fontfamily='monospace', fontweight='bold'
    )

    plt.subplots_adjust(top=title_y_pos - 0.02, bottom=0.05)

     
    sns.heatmap(
        grid_scores, annot=grid_labels, fmt="", cmap="RdYlGn",
        ax=ax, cbar=False, 
        norm=norm,  
        xticklabels=False, yticklabels=False,
        linewidths=0.5, linecolor='white',
        annot_kws={"size": 11, "va": "center", "ha": "center"}
    )
    
     
    ax.set_title(f"Diff Score (Fixed Scale: ±{abs_limit}) [Green: B better, Red: A better]", pad=10)

     
    cbar_height = min(0.3, heatmap_height * 0.5)
    cbar_bottom = (title_y_pos / 2) - (cbar_height / 2)
    cbar_ax = fig.add_axes([0.91, cbar_bottom, 0.015, cbar_height]) 
    
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])
     
    fig.colorbar(sm, cax=cbar_ax, label='Diff Score', ticks=[target_vmin, 0, target_vmax])

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved Fixed-Scale Diff plot to: {output_path}")



def main():
    print(">>> Initializing Models...")
    mod_a, tok_a = load_model_and_tokenizer(MODEL_A_PATH)
    mod_b, tok_b = load_model_and_tokenizer(MODEL_B_PATH)
    print(">>> Models loaded.\n")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(DATA_PATH):
        print(f"Data file {DATA_PATH} not found.")
        return
    # import pdb; pdb.set_trace()
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    
    for idx, line in enumerate(lines):
        if START_IDX > idx:
            continue
        line = line.strip()
        if not line: continue

        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        prompt = entry.get('prompt', '')
        if USE_REJECTED:
            response = entry.get('rejected', '')
        elif USE_CHOSEN:
            response = entry.get('chosen', '')
        else:
            print("No response found")
            exit(1)

        # prompt = PROMPT_TEMPLATE.format(input=prompt)
        print(f"prompt: {prompt}")
        print(f"response: {response}")

        print(f"[{idx+1}/{total_lines}] Processing...")

         
        tokens_a, scores_a = get_response_log_probs(mod_a, tok_a, prompt, response)
        tokens_b, scores_b = get_response_log_probs(mod_b, tok_b, prompt, response)

         
        can_compute_diff = False
        if len(tokens_a) == len(tokens_b):
             
            can_compute_diff = True
            diff_scores = scores_b - scores_a
        else:
            print("   [Warning] Token lists have different lengths. Skipping Diff plot.")

         
        combined_scores = np.concatenate([scores_a, scores_b])
        robust_min = np.percentile(combined_scores, OUTLIER_CUTOFF)
        abs_vmin = min(robust_min, -2.0)
        abs_vmax = 0

         
        wrapped_prompt = textwrap.fill(prompt, width=TEXT_WRAP_WIDTH)
        prompt_lines = wrapped_prompt.count('\n') + 1
        
         
        text_height_buffer = prompt_lines * 0.3 

         
        num_plots = 3 if can_compute_diff else 2
        max_tokens = max(len(tokens_a), len(tokens_b))
        num_rows = math.ceil(max_tokens / TOKENS_PER_ROW)
        
         
        base_plot_height = max(4, num_rows * 2.5) * (num_plots / 2)
        total_fig_height = base_plot_height + text_height_buffer + 1.0  

         
        fig, axes = plt.subplots(num_plots, 1, figsize=(20, total_fig_height))
        
         
         
        plt.subplots_adjust(top=1.0 - (text_height_buffer / total_fig_height), hspace=0.3)

        # Plot 1: Model A
        plot_wrapped_heatmap_v2(tokens_a, scores_a, axes[0], f"Model A: {MODEL_A_PATH}", abs_vmin, abs_vmax)
        
        # Plot 2: Model B
        plot_wrapped_heatmap_v2(tokens_b, scores_b, axes[1], f"Model B: {MODEL_B_PATH}", abs_vmin, abs_vmax)

        # Plot 3: Diff
        diff_vmin, diff_vmax = -0.5, 0.5 # Default
        if can_compute_diff:
            diff_p_low = np.percentile(diff_scores, OUTLIER_CUTOFF)
            diff_p_high = np.percentile(diff_scores, 100 - OUTLIER_CUTOFF)
            limit = max(abs(diff_p_low), abs(diff_p_high), 0.5)
            diff_vmin, diff_vmax = -limit, limit

            plot_wrapped_heatmap_v2(
                tokens_b, diff_scores, axes[2], 
                "Diff (Model B - Model A) [Green: B better, Red: A better]", 
                diff_vmin, diff_vmax, cmap="RdYlGn"
            )
            if USE_REJECTED:
                response_type = "rejected"
            elif USE_CHOSEN:
                response_type = "chosen"
            else:
                print("No response type found")
                exit(1)
            # save_diff_heatmap_only(tokens_b, diff_scores, prompt, os.path.join(OUTPUT_DIR, f"result_{idx:03d}_{response_type}_diff.png"))
            # save_diff_heatmap_only_unnormalized(tokens_b, diff_scores, "", os.path.join(OUTPUT_DIR, f"result_{idx:03d}_{response_type}_diff.png"))
            save_diff_heatmap_absolute(tokens_b, diff_scores, "", os.path.join(OUTPUT_DIR, f"result_{idx:03d}_{response_type}_diff.png"))
         
        fig.suptitle(
            f"PROMPT:\n{wrapped_prompt}", 
            x=0.01, y=0.99,  
            ha='left', va='top', 
            fontsize=14, fontfamily='monospace', fontweight='bold'
        )

         
         
        cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.3]) 
        norm1 = plt.Normalize(abs_vmin, abs_vmax)
        sm1 = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm1)
        sm1.set_array([])
        fig.colorbar(sm1, cax=cbar_ax1, label='Log Probability (Abs)')

        if can_compute_diff:
            cbar_ax2 = fig.add_axes([0.92, 0.1, 0.02, 0.25]) 
            norm2 = plt.Normalize(diff_vmin, diff_vmax)
            sm2 = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm2)
            sm2.set_array([])
            fig.colorbar(sm2, cax=cbar_ax2, label='Diff Score (B - A)')
        if USE_REJECTED:
            response_type = "rejected"
        elif USE_CHOSEN:
            response_type = "chosen"
        else:
            print("No response type found")
            exit(1)
        save_filename = os.path.join(OUTPUT_DIR, f"result_{idx:03d}_{response_type}.png")
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   Saved to: {save_filename}")

        if idx < total_lines - 1:
            user_input = input(">>> Press [Enter] to next (or 'q' to quit): ")
            if user_input.lower().strip() == 'q':
                break

if __name__ == "__main__":
    main()
