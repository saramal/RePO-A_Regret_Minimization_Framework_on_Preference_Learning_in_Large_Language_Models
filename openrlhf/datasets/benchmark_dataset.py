from torch.utils.data import Dataset
from tqdm import tqdm


def split_query(data, split_key="Step 1"):
    return data.split(split_key)[0]

# def split_answer(data, split_key="The answer is: "):
#     return int(data.split(split_key)[1].split(" ")[0])

def preprocess_data(data, input_template=None, input_key="input", label_key=None, data_id_key=None, aRePOy_chat_template=None, few_shot_example=None) -> str:
    if aRePOy_chat_template:
        chat = data[input_key]
        chat = split_query(chat)
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = aRePOy_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(input=prompt)

    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    data_id = data[data_id_key] if data_id_key else None
    # WARINING

    # TODO: if you want use numerical label, please modify the code
    # split 'chat' with Answer key, like "The answer is: "
    # maybe you can just aRePOy the split_answer function

    # add customized prompt
    # deepseek-R1-QWEN-distill-1.5B system prompt
    assert isinstance(prompt, str)
    if few_shot_example:
        prompt = few_shot_example + prompt

    return prompt, label, data_id


class BenchmarkDataset(Dataset):
    """
    Dataset for benchmark model

    Args:
        dataset: dataset for benchmark model
        tokenizer: tokenizer for benchmark model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
        prompt_type=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        if prompt_type:
            self.input_template = self.get_prompt_template(prompt_type)
        else:
            self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "answer_key", None)
        data_id_key = getattr(self.strategy.args, "data_id_key", None)
        aRePOy_chat_template = getattr(self.strategy.args, "aRePOy_chat_template", False)
        few_shot_example = getattr(self.strategy.args, "few_shot_example", None)
        
        if aRePOy_chat_template:
            aRePOy_chat_template = self.tokenizer.aRePOy_chat_template

        self.prompts = []
        self.labels = []
        self.data_ids = []
        self.datasources = [] #TODO:AFTER

        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, label, data_id = preprocess_data(data, self.input_template, input_key, label_key, data_id_key, aRePOy_chat_template, few_shot_example)
            self.prompts.append(prompt)
            self.labels.append(label)
            self.data_ids.append(data_id)
            self.datasources.append(data.get("datasource", "default")) #TODO:AFTER


    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.data_ids[idx], self.prompts[idx], self.labels[idx] #AFTER
        # return self.prompts[idx], self.labels[idx] #BEFORE

    def get_prompt_template(self, prompt_type):
        if prompt_type == 'alpaca':
            problem_prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{input}\n\n### Response: Let's think step by step."
                )
        elif prompt_type == 'alpaca-cot-step':
            problem_prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{input}\n\n### Response:\nLet's think step by step.\nStep 1: "
            )
        elif prompt_type == 'alpaca-cot-prefix':
            problem_prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{input}\n\n### Response:\nLet's think step by step.\n{prefix}"
            )
        elif prompt_type == 'deepseek-math':
            problem_prompt = (
                "User: {instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant:"
            )
        elif prompt_type == 'deepseek-math-step':
            problem_prompt = (
                "User: {instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant: Let's think step by step.\nStep 1: "
            )
        elif prompt_type == 'qwen2-boxed':
            problem_prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        elif prompt_type == 'qwen2-boxed-cot':
            problem_prompt = (
                "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
                "<|im_start|>user\n{input}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        elif prompt_type == 'qwen2-boxed-step':
            problem_prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
                "<|im_start|>assistant\nLet's think step by step.\nStep 1: "
            )
        elif prompt_type == 'qwen2-boxed-prefix':
            problem_prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
                "<|im_start|>assistant\nLet's think step by step.\n{prefix}"
            )
            
        elif prompt_type == 'qwen-stepdpo':
            problem_prompt = (
                "<|user|>:\n{input}\nPlease reason step by step, and put your final answer with 'The answer is: '.\n<|assistant|>:\n"
            )
        elif prompt_type == 'llama3':
            problem_prompt = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{{answer}}$.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: {input}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        elif prompt_type == 'phi3':
            problem_prompt = (
                "<|system|>\nYou are a helpful AI assistant.<|end|>\n"
                "<|user|>\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n"
                "<|assistant|>\n"
            )
        elif prompt_type == 'qwen-instruct-basic-prompt':
            problem_prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n{input}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        elif prompt_type == 'cdpo-basic-prompt':
            problem_prompt = (
                "Question: {input}\nAnswer: "
            )
        elif prompt_type == 'cdpo-gsm8k-fewshot-prompt':
            problem_prompt = GSM8K_PROMPT + "\n" + "Question: {input}\nAnswer: "
            
        elif prompt_type == 'cdpo-math-fewshot-prompt':
            problem_prompt = MATH_PROMPT + "\n" + "Question: {input}\nAnswer: "
            
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
        return problem_prompt
    






GSM8K_PROMPT = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? 
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? 
Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? 
Answer: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? 
Answer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? 
Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

"""


MATH_PROMPT = """Problem: Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}
Solution: The expressions inside each square root must be non-negative.
Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$.
Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$.
Therefore, the domain of the expression is $\\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem: If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$
Solution: We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$
Final Answer: The final answer is $24$. I hope it is correct.

Problem: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?
Solution: If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$: \\begin{align*}
30n&=480\\\\
\\Rightarrow\\qquad n&=480/30=\\boxed{16}
\\end{align*}
Final Answer: The final answer is $16$. I hope it is correct.

Problem: If the system of equations

\\begin{align*}
6x-4y&=a,\\\\
6y-9x &=b.
\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.
Solution: If we multiply the first equation by $-\\frac{3}{2}$, we obtain

$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have

$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$
Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.

"""