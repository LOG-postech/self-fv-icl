"""
Data processing and formatting utilities for few-shot learning.
"""

import random


def get_examples(data, args, is_ood=False, is_cie=False, datum_id=0):
    """Get few-shot examples (either in-domain or out-of-domain)
    
    Args:
        data: Dataset to sample from
        args: Arguments containing shot configuration
        is_ood: Whether to use out-of-domain settings
        is_cie: Whether to use CIE settings
        datum_id: ID of the target example
    Returns:
        list: Selected examples or None if num_shots is 0
    """
    num_shots = args.num_ood_shots if is_ood else args.num_shots
    num_noisy = args.num_noisy_ood_shots if is_ood else args.num_noisy_shots
    
    if num_shots == 0:
        return None
        
    # Sample examples
    local_random = random.Random(args.seed + datum_id) if not is_cie else random.Random(args.cie_seed + datum_id)
    local_random.shuffle(data)
    classified = {}
    for item in data:
        if isinstance(item[args.answer], list):
            answer = str(item[args.answer])
            if answer not in classified:
                classified[answer] = []
            if len(classified[answer]) < num_shots:
                classified[answer].append(item)
        else:
            if item[args.answer] not in classified:
                classified[item[args.answer]] = []
            if len(classified[item[args.answer]]) < num_shots:
                classified[item[args.answer]].append(item)
    examples = []
    class_num = len(classified.keys())
    quotient, remainder = divmod(num_shots, class_num)
    for v in classified.values():
        examples.extend(v[:min(quotient, len(v))])
    if remainder > 0:
        class_keys = list(classified.keys())
        local_random.shuffle(class_keys)
        for k in class_keys[:remainder]:
            v = classified[k]
            if len(v) > quotient:
                examples.append(v[quotient])
            else:
                examples.append(v[-1])
    
    # Add noise to examples
    if num_noisy > 0:
        local_random = random.Random(args.seed + datum_id + 1000) if not is_cie else random.Random(args.cie_seed + datum_id + 1000)
        noisy_example_idx = local_random.sample(range(len(examples)), num_noisy)
            
        for idx in noisy_example_idx:
            options = [item[args.answer] for item in data]
            if all(isinstance(elem, list) for elem in options):
                unique_options = sorted([list(t) for t in set(tuple(pair) for pair in options)])
                remove_option = examples[idx][args.answer]
                remove_option_rev = remove_option[::-1]
                to_remove = {tuple(remove_option), tuple(remove_option_rev)}
                options = [pair for pair in unique_options if tuple(pair) not in to_remove]
                options = [p for p in options if len(set(p)) > 1]
            else:
                options = sorted(list(set(options)))
                options.remove(examples[idx][args.answer])
            
            examples[idx][args.answer] = local_random.choice(options)
    return examples


def format_example(example, args):
    """Format a single example into input-output prompt format
    
    Args:
        example: Data example to format
        args: Arguments containing formatting configuration
        
    Returns:
        dict: Formatted example with 'input' and 'output' keys
    """
    # QA
    prompt = ""
    if "source" not in example.keys():
        example["source"] = args.data
    if example["source"] in ["MMLU"]:
        prompt += "Question: " + example["question"]
    elif example["source"] in [
        "gsm8k", "emotion", "wordnet", "wordnet_multi", "wordnet_triple", "wordnet_quad", "moons",
        "wordnet_test_ood1", "wordnet_test_ood2", "wordnet_test_ood3", "wordnet_test_ood4", "wordnet_test_ood5"
    ]:
        prompt += "Question: " + example["input"]
    elif example["source"] in ["ag_news"]:
        prompt += "Question: " + example["input"]
    # Reading Comprehension
    elif example["source"] in ["CosmosQA", "Halu-CNN/DailyMail", "hellaswag", "Halu-OpenDialKG"]:
        prompt += "Context: " + example["context"] + "\n" + "Question: " + example["input"]
    else:
        raise NotImplementedError("Not supported dataset.")
    if args.answer_type in ["choice", "number"]:
        prompt += "\nChoices:\n"
        for k, v in example["choices"].items():
            if args.answer_type == "number":
                k = str(ord(k.upper()) - ord('A'))
            prompt += k + ". " + str(v) + "\n"
    else:
        prompt += "\n"
    prompt+= "Answer:"
    if ("Qwen" in args.model) or ("gpt-j-6b" in args.model):
        answer_prompt = " "
    else:
        answer_prompt = ""
    if args.answer_type in ["choice", "number"]:
        if isinstance(example[args.answer], list):
            # Convert a list like ["A", "B"] to ["0", "1"]
            ans_list = [
                str(ord(ans.upper()) - ord('A')) if args.answer_type == "number" and isinstance(ans, str) else str(ans)
                for ans in example[args.answer]
            ]
            for idx, ans in enumerate(ans_list):
                if idx==0:
                    answer_prompt+= str(ans)
                else:
                    answer_prompt+= ", " + str(ans)
        else:
            if args.answer_type == "number" and isinstance(example[args.answer], str):
                answer = str(ord(example[args.answer].upper()) - ord('A'))
            else:
                answer = str(example[args.answer])
            answer_prompt+= answer
    else:
        answer_prompt+= example["output"]
    return {"input":prompt, "output" : answer_prompt}


def format_prompt(example, args, fewshot_exps=None):
    """Format a complete prompt with few-shot examples
    
    Args:
        example: Target example to format
        args: Arguments containing formatting configuration
        fewshot_exps: Optional list of few-shot examples
        
    Returns:
        list: Formatted samples including few-shot examples and target
    """
    samples = []
    if fewshot_exps is not None:
        for fs_exp in fewshot_exps:
            samples.append(format_example(fs_exp, args))
    samples.append(format_example(example, args))
    samples[-1]["id"] = example["id"]
    return samples
