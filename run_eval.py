import json
import copy

from transformers import AutoTokenizer
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
import datasets
import torch
import numpy as np

from tasks import TASKS, ISSUE_TASKS
from utils import generate_prompts
from evaluation import EXACT_MATCH_BALANCED_ACC_TASKS, evaluate

OTHER_TASKS = [
    'sara_numeric',
    'successor_liability',
    'citation_prediction_open',
    'definition_extraction',
    'ssla_company_defendants',
    'ssla_individual_defendants',
    'ssla_plaintiff',
]


def main():
    available_exact_match_tasks = copy.deepcopy(EXACT_MATCH_BALANCED_ACC_TASKS)
    # It needs a separtate metric
    index = available_exact_match_tasks.index("successor_liability")
    del available_exact_match_tasks[index]
    
    # This task doesn't even exist
    index = available_exact_match_tasks.index("intra_rule_distinguishing")
    del available_exact_match_tasks[index]

    hf_org = "Equall"
    model_name = "Saul-7B-Base"
    model_dtype = "float32"
    seed = 10
    batch_size = 3
    max_tokens = 16
    
    sampling_params = SamplingParams(
        temperature=0.0, # Greedy
        top_p=1.0,
        max_tokens=max_tokens,
        seed=seed,
        stop=["\n"],
    )
    model = LLM(
        model=f"{hf_org}/{model_name}",
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        seed=seed,
        dtype=model_dtype,
        max_num_seqs=batch_size,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    all_tasks = OTHER_TASKS + available_exact_match_tasks
    
    for task in tqdm(all_tasks):
        with open(f"tasks/{task}/base_prompt.txt") as in_file:
            prompt_template = in_file.read()
    
        dataset = datasets.load_dataset("nguha/legalbench", task)
        test_df = dataset["test"].to_pandas()
        
        prompts = generate_prompts(prompt_template=prompt_template, data_df=test_df)
        outputs = model.generate(prompts, sampling_params)
        generations = [o.outputs[0].text.strip() for o in outputs]
        answers = test_df["answer"].tolist()

        prompt_tokens = [len(tokenizer.encode(text)) for text in prompts]
        full_tokens = [prompt_token_count + len(tokenizer.encode(answer)) for prompt_token_count, answer in zip(prompt_tokens, answers)]

        metric_name = "exact_match_balanced_accuracy" if task in available_exact_match_tasks else "task_specific"
        metric = evaluate(task, generations, answers)
        
        data = {
            "metric_name": metric_name,
            "score": metric,
            "task": task,
            "pred": generations,
            "label": answers,
            "prompt_token_counts": prompt_tokens,
            "full_token_counts": full_tokens,
        }
        
        with open(f'{model_name}_evaluations.jsonl', 'a', encoding='utf-8') as f:
            json_line = json.dumps(data, ensure_ascii=False)
            f.write(json_line + '\n')

if __name__ == "__main__":
    main()
