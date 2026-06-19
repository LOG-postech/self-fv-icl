"""
Intervention utilities for function vectors and attention head manipulation.
"""

import json
import os
import random
import numpy as np
import torch
from itertools import combinations
from tqdm import tqdm
from baukit import TraceDict, get_module
from einops import rearrange
from utils.data_utils import get_examples, format_prompt
from utils.prompt_utils import verbalize
import copy


def split_act_by_heads(activation, model):
    """Split activation tensor by attention heads.
    
    Args:
        activation: Input activation tensor
        model: Model containing config with num_attention_heads
        
    Returns:
        torch.Tensor: Reshaped activation with separated heads
    """
    n_heads = model.config.num_attention_heads
    return rearrange(activation, "... (n h) -> ... n h", n=n_heads)


def extract_attn_act(token, ans_mask, model, args):
    """Extract attention activations from specified layers.
    
    Args:
        token: Input tokens
        ans_mask: Answer mask
        model: Language model
        args: Arguments containing target_layers
        
    Returns:
        torch.Tensor: Stacked attention activations from all target layers
    """
    shift_ans_mask = ans_mask[:, 1:]
    attn_layers = []
    with TraceDict(model, layers=args.target_layers, retain_input=True, retain_output=False) as t:
        model(**token)
    for k in args.target_layers:
        shift_input = t[k].input[:, :-1]
        attn = shift_input[shift_ans_mask.to(shift_input.device)]
        split_attn = split_act_by_heads(attn[0], model)
        attn_layers.append(split_attn.cpu())
    return torch.stack(attn_layers)


def o_project(layer_name: str, inp: torch.Tensor, model, args):
    """Apply output projection for a given layer.
    
    Args:
        layer_name: Name of the layer for projection
        inp: Input tensor
        model: Language model
        args: Arguments containing model name
        
    Returns:
        torch.Tensor: Projected output
    """
    proj_module = get_module(model, layer_name)
    o_proj_w = proj_module.weight
    # Move inp to the same device as the projection weights (for model parallelism)
    inp = inp.to(o_proj_w.device)
    if "llama" in args.model or "gpt-j" in args.model or "Qwen" in args.model or "Mistral" in args.model or "Mixtral" in args.model:
        out = torch.matmul(inp, o_proj_w.T)
    elif "gpt-neox" in args.model:
        out_proj_bias = proj_module.bias
        out = torch.addmm(out_proj_bias, inp.squeeze(), o_proj_w.T)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")
    return out


def get_mean_head_act_adder(mean_head_acts, layer_name: str, layer: int, head: int, model, args):
    """Create an editor function that adds mean head activations.
    
    Args:
        mean_head_acts: Mean head activations tensor
        layer_name: Name of the target layer
        layer: Layer index
        head: Head index
        model: Language model
        args: Arguments containing model configuration
        
    Returns:
        function: Editor function for intervention
    """
    def editor(output, current_layer, inp):
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = split_act_by_heads(inp, model)  # [batch, seq_len, heads, head_dim]
        inp[:, :, head] = mean_head_acts[layer, head].to(inp.device)
        inp = rearrange(inp, "... n h -> ... (n h)")
        output = o_project(layer_name, inp, model, args)
        return output
    return editor


def last_only_io_mask(io_mask):
    """Create a mask that only keeps the last True position.
    
    Args:
        io_mask: Input/output mask tensor
        
    Returns:
        torch.Tensor: Mask with only the last True position
    """
    ones_indices = io_mask.squeeze().nonzero()
    last_one_index = ones_indices[-1].item()
    last_only_mask = torch.zeros_like(io_mask)
    last_only_mask[..., last_one_index] = True
    return last_only_mask


def get_function_vector_editor(function_vector, io_mask):
    """Create an editor function that applies function vector intervention.
    
    Args:
        function_vector: Function vector to add
        io_mask: Input/output mask for intervention position
        
    Returns:
        function: Editor function for function vector intervention
    """
    last_only_mask = last_only_io_mask(io_mask)
    applied = False  # Flag to track if intervention was already applied
    
    def editor(output):
        nonlocal applied
        
        # Only apply intervention once, during the first forward pass
        if not applied and output.shape[1] == last_only_mask.shape[1]:
            mask_on_device = last_only_mask.to(output.device)
            output[mask_on_device] += function_vector.to(output.device)
            applied = True
        # Skip intervention for subsequent generation steps
        
        return output
    return editor


def get_function_vector(model, args):
    """Construct function vector from top heads and mean activations.
    
    Args:
        model: Language model
        args: Arguments containing top_heads_path, num_top_heads, etc.
        
    Returns:
        torch.Tensor: Constructed function vector
    """
    with open(args.top_heads_path, 'r') as f:
        top_heads = json.load(f)
    top_heads = [tuple(t) for t in top_heads[:args.num_top_heads]]
    
    if args.mean_head_act_path is not None:
        mean_head_act = torch.load(args.mean_head_act_path)
    else:
        mean_head_act = torch.load(os.path.join(os.path.dirname(args.top_heads_path), "mean_head_act.pt"))
    function_vector = torch.zeros(model.config.hidden_size, device=model.device)
    
    for layer, head, _ in top_heads:
        head_size = model.config.hidden_size // model.config.num_attention_heads
        x = torch.zeros(model.config.hidden_size, device=model.device, dtype=model.dtype)
        x[head * head_size : (head + 1) * head_size] = mean_head_act[layer, head].to(model.device)
        function_vector += o_project(args.target_layers[layer], x, model, args).to(model.device)
    function_vector = function_vector.to(model.device, model.dtype).unsqueeze(0)
    
    return function_vector


def _prepare_cie_experiment(args, train_data, tokenizer):
    """Handle setup, paths, and data splitting for CIE experiment.
    
    Returns:
        dict: Setup data containing paths, data splits, and token mappings
    """
    # Setup paths
    if args.cie_save_path is None:
        args.cie_save_path = args.result_path
    mean_head_act_dir_path = os.path.join(args.cie_save_path, f"cie_seed{args.cie_seed}")
    os.makedirs(mean_head_act_dir_path, exist_ok=True)
    mean_head_act_path = os.path.join(mean_head_act_dir_path, f"mean_head_act.pt")
    
    # Prepare data
    random.seed(args.cie_seed)
    train_val_data = copy.deepcopy(train_data)
    
    if not hasattr(args, 'options_letters'):
        args.options_letters = list(train_val_data[0]['choices'].values())
    if ('gpt-j-6b' in args.model) or ('Qwen' in args.model) or ('Mistral' in args.model) or ('Mixtral' in args.model):
        first_tokens = {i: l for i, l in enumerate(args.options_tokens)}
    else:
        first_tokens = {i: tokenizer.encode(l, add_special_tokens=False)[0] for i, l in enumerate(args.options_letters)}
    
    # Split train data for CIE
    random.shuffle(train_val_data)
    split_idx = int(len(train_val_data) * 0.9)
    train_cie, val_cie = train_val_data[:split_idx], train_val_data[split_idx:]
    
    return {
        'mean_head_act_dir_path': mean_head_act_dir_path,
        'mean_head_act_path': mean_head_act_path,
        'first_tokens': first_tokens,
        'train_cie': train_cie,
        'val_cie': val_cie
    }


def _get_model_prediction(model, tokenizer, sq_token, sq_ans_mask, datum, first_tokens, args, 
                         compute_probs=False, editor=None, layer_name=None):
    """Centralize forward vs generate decision with optional intervention and probability computation.
    
    Args:
        compute_probs: If True, also compute and return probability scores
        editor: Optional editor function for intervention
        layer_name: Layer name for intervention (required if editor is provided)
        
    Returns:
        If compute_probs=False: (is_correct: bool, q_token_id)
        If compute_probs=True: (probabilities, q_token_id, is_correct: bool)
    """
    # Create appropriate context manager
    if editor is not None and layer_name is not None:
        context_manager = TraceDict(model, layers=[layer_name], edit_output=editor)
    else:
        context_manager = torch.no_grad()
    
    if isinstance(datum["label"], list):
        q_token_id = [first_tokens[l] for l in sorted(datum["label"])]
        if "Qwen" in args.model:
            q_token_id = [first_tokens[sorted(datum["label"])[0]]] + [args.second_tokens[l] for l in sorted(datum["label"])[1:]]
        new_sq_token = {
            "input_ids": sq_token["input_ids"][:, ~sq_ans_mask[0]], 
            "attention_mask": sq_token["attention_mask"][:, ~sq_ans_mask[0]]
        }
        with context_manager:
            outputs = model.generate(
                **new_sq_token,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings='\n\n',
                tokenizer=tokenizer,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        generated_tokens = outputs[0].squeeze(0)[new_sq_token['input_ids'].shape[1]:]
        suffix_ids = tokenizer.encode("\n\n", add_special_tokens=False)
        suffix = torch.tensor(suffix_ids, device=generated_tokens.device, dtype=generated_tokens.dtype)

        if "Qwen" in args.model:
            if len(suffix) > 0 and len(generated_tokens) >= len(suffix) and torch.equal(generated_tokens[-len(suffix):], suffix):
                generated_tokens = generated_tokens[:-len(suffix)]
            elif tokenizer.eos_token_id is not None and len(generated_tokens) > 0 and generated_tokens[-1].item() == tokenizer.eos_token_id:
                generated_tokens = generated_tokens[:-1]
        else:
            if len(suffix) > 0 and len(generated_tokens) >= len(suffix) and torch.equal(generated_tokens[-len(suffix):], suffix):
                generated_tokens = generated_tokens[:-len(suffix)]
        match = generated_tokens[:, None] == torch.tensor(args.options_tokens, device=generated_tokens.device)[None, :] 
        match_idx, _ = torch.where(match)
        
        # Check correctness
        is_correct = False
        if len(match_idx) > 0:
            clean_pred = generated_tokens[match_idx].tolist()
            is_correct = set(clean_pred) == set(q_token_id)
        
        if compute_probs:
            # Always compute probabilities for target tokens, regardless of correctness
            target_probs = 0.0
            num_target_tokens = len(q_token_id)
            
            # Get probabilities for each target token from the generation scores
            for i, target_token in enumerate(q_token_id):
                if i < len(outputs.scores):
                    score_probs = torch.softmax(outputs.scores[i], dim=-1)
                    target_probs += score_probs[0, target_token].item()
            
            # Average over target tokens
            avg_target_prob = target_probs / num_target_tokens if num_target_tokens > 0 else 0.0
            return avg_target_prob, q_token_id, is_correct
        else:
            return is_correct, q_token_id
    else:
        q_token_id = first_tokens[datum["label"]]
        
        with context_manager:
            clean_logits = model(**sq_token).logits
        clean_ans = clean_logits[:, :-1][sq_ans_mask[:, 1:]][0]
        clean_probs = torch.softmax(clean_ans, dim=-1)
        clean_pred = clean_probs.argmax().item()
        
        # Check correctness
        is_correct = clean_pred == q_token_id
        
        if compute_probs:
            # Always return probability of target token, regardless of correctness
            return clean_probs, q_token_id, is_correct
        else:
            return is_correct, q_token_id


def _add_noise_to_examples(examples, args):
    """Add noise to few-shot examples for CIE experiment."""
    if not examples:
        return
        
    example_labels = [example[args.answer] for example in examples]
    repl_options = [[o for o in args.options_letters if o != lbl] if isinstance(lbl, str) else [list(c) for c in combinations(args.options_letters, len(lbl)) if set(c) != set(lbl)] for lbl in example_labels]
    
    if args.cie_multi_single_noise:
        for idx, item in enumerate(repl_options):
            repl_options[idx].extend([[o] for o in args.options_letters])
    
    if not args.data in ["moons"]:
        for idx, example in enumerate(examples):
            if repl_options[idx]:
                example[args.answer] = random.choice(repl_options[idx])
    
    else:
        repl_ids = random.sample(range(len(examples)), k=len(examples)//2)
        for idx, example in enumerate(examples):
            if idx in repl_ids and repl_options[idx]:
                example[args.answer] = random.choice(repl_options[idx])
    
    if any(not repl_opt for repl_opt in repl_options):
        random.shuffle(examples)            


def _collect_activations(model, tokenizer, vb, setup_data, ood_data, args, device):
    """Collect clean activations from validation data.
    
    Returns:
        torch.Tensor: Mean head activations
    """
    mean_head_act_path = setup_data['mean_head_act_path']
    first_tokens = setup_data['first_tokens']
    val_cie = setup_data['val_cie']
    train_cie = setup_data['train_cie']
    
    # Collect activations
    activations = []
    random.seed(args.cie_seed)
    for datum in tqdm(val_cie, desc="Collecting activations"):
        train_cie_copy = copy.deepcopy(train_cie)
        examples = get_examples(train_cie_copy, args, is_cie=True, datum_id=datum["id"])
        if ood_data is not None:
            ood_examples = get_examples(ood_data, args, is_ood=True, is_cie=True, datum_id=datum["id"])
            if ood_examples is not None:
                examples = (examples or []) + ood_examples
        if examples is not None:
            local_random = random.Random(args.cie_seed + datum["id"] + 2000)
            local_random.shuffle(examples)
        
        if os.path.exists(mean_head_act_path):
            continue
            
        sq_token, sq_ans_mask, _ = verbalize(vb, format_prompt(datum, args, examples), device)
        
        # Get clean prediction (no probabilities needed)
        is_valid, q_token_id = _get_model_prediction(model, tokenizer, sq_token, sq_ans_mask, datum, first_tokens, args, compute_probs=False)
        if not is_valid:
            continue
            
        attn_acts = extract_attn_act(sq_token, sq_ans_mask, model, args)
        activations.append(attn_acts)
    
    if os.path.exists(mean_head_act_path):
        mean_head_act = torch.load(mean_head_act_path)
    else:
        mean_head_act = torch.stack(activations).mean(0).to(device, model.dtype)
        torch.save(mean_head_act, mean_head_act_path)
    
    return mean_head_act


def _compute_indirect_effects(model, tokenizer, vb, setup_data, mean_head_act, ood_data, args, device):
    """Apply interventions and compute indirect effects."""
    mean_head_act_dir_path = setup_data['mean_head_act_dir_path']
    first_tokens = setup_data['first_tokens']
    val_cie = setup_data['val_cie']
    train_cie = setup_data['train_cie']
    mean_head_act_path = setup_data['mean_head_act_path']
    
    # Compute indirect effects
    ep = 0
    correct_ep = 0
    indirect_effects = torch.zeros(args.ie_episodes, len(args.target_layers), model.config.num_attention_heads)
    correct_indirect_effects = torch.zeros(len(val_cie), len(args.target_layers), model.config.num_attention_heads)
    random.seed(args.cie_seed)
    correct_ranking = []
    
    for datum in tqdm(val_cie, desc="Computing indirect effects"):
        # Similar logic as above but with interventions
        train_cie_copy = copy.deepcopy(train_cie)
        examples = get_examples(train_cie_copy, args, is_cie=True, datum_id=datum["id"])
        if ood_data is not None:
            ood_examples = get_examples(ood_data, args, is_ood=True, is_cie=True, datum_id=datum["id"])
            if ood_examples is not None:
                examples = (examples or []) + ood_examples
        if examples is not None:
            local_random = random.Random(args.cie_seed + datum["id"] + 2000)
            local_random.shuffle(examples)
            
        sq_token, sq_ans_mask, _ = verbalize(vb, format_prompt(datum, args, examples), device)
        
        # Get clean prediction with probabilities
        clean_probs, q_token_id, is_clean_correct = _get_model_prediction(model, tokenizer, sq_token, sq_ans_mask, datum, first_tokens, args, compute_probs=True)
        if args.cie_only_correct_pred and not is_clean_correct:
            continue
        
        # Add noise to examples
        _add_noise_to_examples(examples, args)
                
        sq_token, sq_ans_mask, _ = verbalize(vb, format_prompt(datum, args, examples), device)
        layer = args.cie_layer
        layer_name = args.target_layers[layer]
        
        correct_pred = 0
        for head in range(model.config.num_attention_heads):
            editor = get_mean_head_act_adder(mean_head_act, layer_name, layer, head, model, args)
            
            # Get intervened prediction using the unified function
            subs_probs, subs_q_token_id, is_subs_correct = _get_model_prediction(
                model, tokenizer, sq_token, sq_ans_mask, datum, first_tokens, args, 
                compute_probs=True, editor=editor, layer_name=layer_name
            )
            
            # Compute indirect effect
            if isinstance(datum["label"], list):
                # List case: both subs_probs and clean_probs are floats
                e = subs_probs - clean_probs
                if ep < args.ie_episodes:
                    indirect_effects[ep, layer, head] = e
                if subs_q_token_id and set(subs_q_token_id) == set(q_token_id):
                    correct_indirect_effects[correct_ep, layer, head] = e
                    correct_pred += 1
            else:
                # Single label case: both subs_probs and clean_probs are tensors
                e = (subs_probs - clean_probs).squeeze()
                if ep < args.ie_episodes:
                    indirect_effects[ep, layer, head] = e[q_token_id]
                if is_subs_correct:
                    correct_indirect_effects[correct_ep, layer, head] = e[q_token_id]
                    correct_pred += 1
        correct_ranking.append(correct_pred)
        ep += 1
        correct_ep += 1

    ep = min(ep, args.ie_episodes)
    correct_ep = min(correct_ep, args.ie_episodes)
    
    scores = torch.as_tensor(correct_ranking, device=correct_indirect_effects.device, dtype=torch.float32)

    k = int(min(max(correct_ep, 0), scores.numel()))
    if k == 0:
        correct_indirect_effects = correct_indirect_effects[:0]
    else:
        idx = torch.argsort(scores, descending=True, stable=True)[:k]
        correct_indirect_effects = correct_indirect_effects.index_select(0, idx)
    
    # Save results
    mean_effect = indirect_effects[:ep].mean(0)
    correct_mean_effect = correct_indirect_effects[:correct_ep].mean(0)
    
    mean_effect_path = os.path.join(mean_head_act_dir_path, f"{args.cie_layer}_mean_effect.pt" if not args.cie_multi_single_noise else f"{args.cie_layer}_mean_effect_single_noise.pt")
    correct_mean_effect_path = os.path.join(mean_head_act_dir_path, f"{args.cie_layer}_correct_mean_effect.pt" if not args.cie_multi_single_noise else f"{args.cie_layer}_correct_mean_effect_single_noise.pt")
    
    torch.save(mean_effect, mean_effect_path)
    torch.save(correct_mean_effect, correct_mean_effect_path)
    torch.save(mean_head_act, mean_head_act_path)
    print(f"ep: {ep}, correct_ep: {correct_ep} out of {len(val_cie)}, args.ie_episodes: {args.ie_episodes}")
    print(f"CIE results saved to {mean_effect_path} and {correct_mean_effect_path}")


def run_cie_experiment(model, tokenizer, vb, train_data, ood_data, args, device):
    """Run Causal Intervention Experiment (CIE) - refactored for clarity.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer
        vb: Verbalizer for prompt formatting
        train_data: Training dataset
        ood_data: Out-of-domain dataset (optional)
        args: Arguments containing CIE configuration
        device: Device for computation
    """   
    print("Running CIE experiment...")
    
    # Phase 1: Preparation
    setup_data = _prepare_cie_experiment(args, train_data, tokenizer)
    
    # Phase 2: Activation Collection  
    mean_head_act = _collect_activations(model, tokenizer, vb, setup_data, ood_data, args, device)
    
    # Phase 3: Indirect Effects Computation
    _compute_indirect_effects(model, tokenizer, vb, setup_data, mean_head_act, ood_data, args, device)
