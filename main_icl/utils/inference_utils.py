"""
Model inference utilities for generation and entropy computation.
"""

import torch
import torch.nn.functional as F
from baukit import TraceDict

def get_model_output(model, tokenizer, inputs, ans_mask, args, example_id, editor=None, target_layer=None):
    """Generate model output and compute entropy metrics
    
    Args:
        model: The language model
        tokenizer: Model tokenizer
        inputs: Input tokens
        ans_mask: Answer mask
        args: Arguments containing options and configuration
        example_id: ID of the example
        editor: Optional editor function for intervention (default: None)
        target_layer: Optional target layer for intervention (default: None)
        
    Returns:
        dict: Dictionary containing predicted answers and entropy metrics
    """
    option_ids = args.options_tokens if (args.data != "gsm8k" or args.answer_type != "generation") else sorted(list(tokenizer.get_vocab().values()))
    stop_strings = args.stop_strings if hasattr(args, 'stop_strings') else '\n\n'
    
    # Create context manager for intervention if needed
    if editor is not None and target_layer is not None:
        context_manager = TraceDict(model, layers=target_layer, edit_output=editor)
    else:
        context_manager = torch.no_grad()
    
    if args.data in ["wordnet_multi", "wordnet_triple", "wordnet_quad"]:
        max_new_tokens = 10
        with context_manager:
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=tokenizer.eos_token_id, 
                stop_strings=stop_strings, 
                tokenizer=tokenizer, 
                return_dict_in_generate=True, 
                output_scores=True
            )
        generated_tokens = outputs[0].squeeze(0)[inputs['input_ids'].shape[1]:]
        if len(generated_tokens) != max_new_tokens:
            if args.answer_type == "generation" and args.data not in ["gsm8k"]:
                generated_tokens = generated_tokens[:1]  # 'Sci'
            else:
                generated_tokens = generated_tokens[:-3]  # remove ' \n\n'
        match = generated_tokens[:, None] == torch.tensor(option_ids, device=generated_tokens.device)[None, :]
        match_idx, option_idx = torch.where(match)
        if len(match_idx) == 0:
            match = generated_tokens[:, None] == torch.tensor(sorted(list(tokenizer.get_vocab().values())), device=generated_tokens.device)[None, :]
            match_idx, option_idx = torch.where(match)
        step_logits = torch.stack(outputs.scores, dim=1)
        picked = step_logits.index_select(1, match_idx)
    else:
        with context_manager:
            outputs = model(**inputs)
            logits = outputs.logits
            picked = logits[:, :-1][ans_mask[:, 1:]][0]
            generated_tokens = picked.unsqueeze(0).argmax(dim=-1)
        

    # Compute entropy metrics
    
    logp = F.log_softmax(picked, dim=-1)
    entropy = -(logp.exp() * logp).sum(-1)
    first_token_entropy = 0
    first_token_option_entropy = 0
    cols = torch.tensor(option_ids, device=picked.device)
    picked_option_sliced = picked.index_select(-1, cols)
    option_logp = F.log_softmax(picked_option_sliced, dim=-1)
    option_entropy = -(option_logp.exp() * option_logp).sum(-1)
    
    # Compute separator entropy
    if args.data in ["wordnet_multi", "wordnet_triple", "wordnet_quad"]:
        first_token_entropy = entropy[0, 0].cpu().item()
        first_token_option_entropy = option_entropy[0, 0].cpu().item()
        separator_logits = outputs.scores[match_idx[0]+1]
        cols = torch.tensor([29892, 13], device=separator_logits.device)
        separator_logits = separator_logits.index_select(-1, cols)
        separator_logp = F.log_softmax(separator_logits, dim=-1)
        separator_entropy = -(separator_logp.exp() * separator_logp).sum(-1)
        separator_entropy = separator_entropy.mean().cpu().item()
    else:
        separator_entropy = 0.0
    
    # Format output
    if len(generated_tokens)>1:
        gen_id =[p.argmax().item() for p in picked[0]]
    elif len(generated_tokens)==1:
        gen_id = generated_tokens[0].item()
    else:
        gen_id = None
        
    if gen_id is not None:
        if gen_id in args.options_tokens:
            predicted_answers = [args.options_letters[args.options_tokens.index(gen_id)]]
        elif isinstance(gen_id, list):
            if all(g in args.options_tokens for g in gen_id):
                predicted_answers = [args.options_letters[args.options_tokens.index(g)] for g in gen_id]
            else:
                predicted_answers = [tokenizer.decode([g]) for g in gen_id]
        else:
            predicted_answers = [tokenizer.decode([gen_id])]
    else:
        predicted_answers = []
        
    return {
        "id": example_id, 
        "predicted_answers": predicted_answers, 
        "entropy": entropy.mean().cpu().item(), 
        "option_entropy": option_entropy.mean().cpu().item(), 
        "separator_entropy": separator_entropy,
        "first_token_entropy": first_token_entropy,
        "first_token_option_entropy": first_token_option_entropy
    }
