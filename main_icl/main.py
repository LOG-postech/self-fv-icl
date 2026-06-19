import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import instruction as pt
import pandas as pd
import copy
import re

from utils.prompt_utils import Verbalizer, verbalize
from utils.metrics import compute_accuracy, compute_auroc_aupr_from_entropy, compute_auroc_aupr_from_option_entropy, compute_cross_auroc_aupr, compute_cross_auroc_aupr_from_option_entropy, compute_macro_f1
from utils.model_utils import set_seed, load_model
from utils.data_utils import get_examples, format_prompt
from utils.intervention_utils import split_act_by_heads, o_project, get_function_vector_editor, get_function_vector, run_cie_experiment
from utils.inference_utils import get_model_output

from baukit import TraceDict


def setup_args():
    """Parse and configure arguments"""
    parser = argparse.ArgumentParser(description="In-Context Learning with Uncertainty Estimation")
    
    # Model and data arguments
    parser.add_argument('--model', type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--data_path', type=str, default="dataset_files/icl")
    parser.add_argument('--data', type=str, help="Dataset to use")
    parser.add_argument('--ood_data', type=str, default=None, help="Out-of-domain dataset")
    parser.add_argument('--output_dir', type=str, default='results')
    
    # Few-shot learning arguments
    parser.add_argument('--uniform_example', type=int, default=0)
    parser.add_argument('--num_shots', type=int, default=0)
    parser.add_argument('--num_ood_shots', type=int, default=0)
    parser.add_argument('--num_noisy_shots', type=int, default=0)
    parser.add_argument('--num_noisy_ood_shots', type=int, default=0)
    parser.add_argument('--is_fixed_total_shots', type=int, default=0)
    
    # Model configuration
    parser.add_argument('--cot', type=int, default=0, help="Use chain-of-thought")
    parser.add_argument('--deterministic', type=int, default=1, help="Use deterministic sampling")
    parser.add_argument('--answer_type', type=str, default='choice', 
                       choices=['choice', 'generation', 'number'])
    parser.add_argument('--num_choices', type=int, default=4)
    parser.add_argument('--num_multiple_choices', type=int, default=1)
    parser.add_argument('--prompt_template', type=str, default="base")
    parser.add_argument('--instruction', type=str, default=None)
    
    # Experimental settings
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--cie_seed', type=int, default=24, help="CIE random seed")
    parser.add_argument('--use_wandb', type=int, default=1, help="Use wandb logging")
    parser.add_argument('--cie', type=int, default=0, help="Run CIE experiment")
    parser.add_argument('--cie_only_correct_pred', type=int, default=0, help="CIE only correct prediction")
    parser.add_argument('--ie_episodes', type=int, default=200, help="CIE episodes")
    parser.add_argument("--cie_layer", type=int, default=10, help="CIE target layer")
    parser.add_argument("--layer_info", type=str, default="model.layers.{}.self_attn.o_proj", help="Layer name template")
    parser.add_argument("--layer_start", type=int, default=0, help="CIE target layer start")
    parser.add_argument("--layer_end", type=int, default=32, help="CIE target layer end")
    parser.add_argument("--cie_save_path", type=str, default=None)
    parser.add_argument("--cie_multi_single_noise", type=int, default=0, help="CIE multi-single noise")
    
    parser.add_argument('--use_fv', type=int, default=0, help="Use FV")
    parser.add_argument('--mean_head_act_path', type=str, default=None)
    parser.add_argument('--use_self_fv', type=int, default=0, help="Use self FV")
    parser.add_argument('--top_heads_path', type=str, default=None, help="Path to top heads")
    parser.add_argument('--num_top_heads', type=int, default=10, help="Number of top heads")
    parser.add_argument('--interv_layer', type=int, default=None, help="Intervention layer")
    
    parser.add_argument('--debug', type=int, default=0, help="Debug mode")
    parser.add_argument('--log_prompt', type=int, default=0, help="Log prompts to JSON file")
    args = parser.parse_args()
    
    # Configure answer field and options
    args.answer = 'choices_label' if args.answer_type == 'choice' else 'output' if args.answer_type == 'generation' else 'label'
    
    # Dataset-specific configurations
    if args.data == "emotion":
        args.num_choices = 6
    if args.data in ["wordnet_multi"] and args.ood_data is None:
        args.num_multiple_choices = 2
    elif args.data in ["wordnet_triple"] and args.ood_data is None:
        args.num_multiple_choices = 3
    elif args.data in ["wordnet_quad"] and args.ood_data is None:
        args.num_multiple_choices = 4
    
    # Setup answer options
    if args.answer_type == 'choice':
        if 'Mistral' in args.model:
            args.options = ["Answer:" + chr(ord('A') + i) for i in range(args.num_choices)]
        else:
            args.options = ["Answer: " + chr(ord('A') + i) for i in range(args.num_choices)]
        args.options_letters = [chr(ord('A') + i) for i in range(args.num_choices)]
    elif args.answer_type == 'generation':
        if 'Mistral' in args.model:
            args.options = ["Answer:"]
        else:
            args.options = ["Answer: "]
    elif args.answer_type == 'number':
        args.options = ["Answer: " + str(i) for i in range(args.num_choices)]
        args.options_letters = [str(i) for i in range(args.num_choices)]
    else:
        raise NotImplementedError("Not supported answer type.")
    
    # Setup target layers for CIE
    layer_name, start, end = args.layer_info, args.layer_start, args.layer_end
    args.target_layers = [layer_name.format(i) for i in range(start, end)]
    
    # Configure instruction prompt
    if args.instruction is not None:
        if args.cot:
            args.instruction = args.instruction + "\n" + pt.base_cot_prompt
    else:
        args.instruction = pt.base_cot_prompt if args.cot else ""
    
    return args


if __name__ == "__main__":
    args = setup_args()

    set_seed(args.seed)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        import wandb
        run_name = f"interv_layer{args.interv_layer}_num_top_heads{args.num_top_heads}_cie_only_correct_pred{args.cie_only_correct_pred}"
        wandb.init(project="icl", name=run_name, config=args)
        print(f"wandb run ID: {wandb.run.id}")
    
    # Load model and tokenizer
    tokenizer, model = load_model(args)
    device = model.device
    vb = Verbalizer(tokenizer, instruction=args.instruction)
    
    # Configure model options and generation
    if args.answer_type in ["choice", "number"]:
        args.options_tokens = [tokenizer.encode(opt)[-1] for opt in args.options]
        if args.data == "wordnet_multi":
            args.second_tokens = {i: opt_token for i, opt_token in enumerate(args.options_tokens)}
            args.options_tokens.extend([tokenizer.encode(opt)[-1] for opt in args.options_letters])
            
    torch.set_grad_enabled(False)
    if args.deterministic:
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
    model.eval()
    print(f"Model device map: {model.hf_device_map}")
    if 'cpu' in model.hf_device_map.values():
        if args.use_wandb:
            wandb.finish(exit_code=1)
        import sys
        sys.exit(1)
    
    # Load datasets
    train_data = json.load(open(os.path.join(args.data_path, args.data) + "_train.json", "r"))
    if args.answer_type == 'generation' and args.data != 'gsm8k' :
        args.options_letters = list(train_data[0]['choices'].values())
        args.options = [args.options[0] + l for l in list(train_data[0]['choices'].values())]
        tokens = [tokenizer.encode(l, add_special_tokens=False) for l in args.options]
        filtered_tokens = [[tok for tok in t if tok not in set.intersection(*map(set, tokens))] for t in tokens]
        args.options_tokens = [l[0] for l in filtered_tokens]
    
    test_data = json.load(open(os.path.join(args.data_path, args.data) + "_test.json", "r"))
    ood_data = None
    if args.ood_data is not None:
        ood_data = json.load(open(os.path.join(args.data_path, args.ood_data) + "_train.json", "r"))
    
    dir_name = f"{args.num_shots}_uniform_shot" if args.uniform_example else f"{args.num_shots}_shot"
    dir_name = dir_name + "_cie_only_correct_pred" if args.cie_only_correct_pred else dir_name
    args.result_path = os.path.join(args.output_dir, args.data, args.answer_type, dir_name, f"seed{args.seed}")
    os.makedirs(args.result_path, exist_ok=True)
    # Run CIE experiment if enabled
    if args.cie:
        run_cie_experiment(model, tokenizer, vb, train_data, ood_data, args, device)
        exit()

    # Run main evaluation
    print("Running main evaluation...")
    model_outputs = []
    if args.use_fv:
        fv_model_outputs = []
    if args.use_self_fv:
        self_fv_model_outputs = []
    # Initialize prompt logging
    prompt_logs = []
    
    if args.debug:
        test_data = test_data[:10]
    for datum in tqdm(test_data, desc="Evaluating"):
        # Get few-shot examples
        train_data_copy = copy.deepcopy(train_data)
        examples = get_examples(train_data_copy, args, datum_id=datum["id"])
        
        # Add OOD examples if specified
        if ood_data is not None:
            train_ood_data_copy = copy.deepcopy(ood_data)
            ood_examples = get_examples(train_ood_data_copy, args, is_ood=True, datum_id=datum["id"])
            if ood_examples is not None:
                examples = (examples or []) + ood_examples
        
        if examples is not None:
            local_random = random.Random(args.seed + datum["id"] + 2000)
            local_random.shuffle(examples)
        
        # Generate prompt and get model output
        sq_token, sq_ans_mask, sq_io_mask = verbalize(vb, format_prompt(datum, args, examples), device)
        
        # Log prompt if requested
        if args.log_prompt == 1:
            decoded_prompt = tokenizer.decode(sq_token["input_ids"][0], skip_special_tokens=True)
            prompt_logs.append({
                "example_id": datum["id"],
                "decoded_prompt": decoded_prompt
            })
            # Break after logging 5 examples
            if len(prompt_logs) >= 5:
                break
        
        if args.data in ["wordnet_multi", "wordnet_triple", "wordnet_quad"]:         
            prompt_only_token = {
                "input_ids": sq_token["input_ids"][:, ~sq_ans_mask[0]], 
                "attention_mask": sq_token["attention_mask"][:, ~sq_ans_mask[0]]
            }
            input_token = prompt_only_token
        else:
            input_token = sq_token
        
        output = get_model_output(model, tokenizer, input_token, sq_ans_mask, args, datum["id"])
        model_outputs.append(output)
        
        if args.use_fv or args.use_self_fv:
            if args.interv_layer is None:
                layer = model.config.num_hidden_layers // 3  # add at |L|/3
            else:
                layer = args.interv_layer
            target_layer = args.target_layers[layer - 1 : layer]
            if len(vb.io_sep_ids) == 0 :
                mask = sq_ans_mask.clone()
                idx = mask[0].nonzero(as_tuple=True)[0]

                if len(idx) > 0 and idx[0] > 0:
                    sq_io_mask = torch.zeros_like(mask, dtype=torch.bool)
                    sq_io_mask[0, idx[0] - 1] = True
                else:
                    sq_io_mask = torch.zeros_like(mask, dtype=torch.bool)      
            if args.data in ["wordnet_multi", "wordnet_triple", "wordnet_quad"]:
                sq_io_mask = sq_io_mask[:, ~sq_ans_mask[0]]
                
        if args.use_self_fv:
            with open(args.top_heads_path, 'r') as f:
                top_heads = json.load(f)
            top_heads = [tuple(t) for t in top_heads[:args.num_top_heads]]
            shift_ans_mask = sq_ans_mask[:, 1:]
            attn_layers = []
            with TraceDict(model, layers=args.target_layers, retain_input=True, retain_output=False) as td:
                model(**sq_token)
            for k in args.target_layers:
                shift_input = td[k].input[:, :-1]
                attn = shift_input[shift_ans_mask.to(shift_input.device)]
                split_attn = split_act_by_heads(attn[0], model)
                attn_layers.append(split_attn.cpu())
            activations = torch.stack(attn_layers)  # [layer, heads, head_dim] on CPU
            self_function_vector = torch.zeros(model.config.hidden_size, device=device)
            for layer, head, _ in top_heads:
                head_size = model.config.hidden_size // model.config.num_attention_heads
                x = torch.zeros(model.config.hidden_size, device=device, dtype=model.dtype)
                x[head * head_size : (head + 1) * head_size] = activations[layer, head].to(device)
                self_function_vector += o_project(args.target_layers[layer], x, model, args).to(device)
            self_function_vector = self_function_vector.to(device, model.dtype).unsqueeze(0)      
            
            editor = get_function_vector_editor(self_function_vector, sq_io_mask) # sq_io_mask[:, :-sq_ans_mask.sum().item()]
            output_self_fv = get_model_output(model, tokenizer, input_token, sq_ans_mask, args, datum["id"], editor, target_layer)
            
            self_fv_model_outputs.append(output_self_fv)
    
        if args.use_fv:
            function_vector = get_function_vector(model, args)
            editor = get_function_vector_editor(function_vector, sq_io_mask)
            output_fv = get_model_output(model, tokenizer, input_token, sq_ans_mask,args, datum["id"], editor, target_layer)
            fv_model_outputs.append(output_fv)

    # Compute metrics
    model_outputs, accuracy = compute_accuracy(model_outputs, test_data, args)
    print(f"accuracy: {accuracy}")
    
    # Compute macro-F1
    macro_f1 = compute_macro_f1(model_outputs, test_data, args)
    print(f"macro-F1: {macro_f1}")
    
    if args.use_self_fv:
        self_fv_outputs, self_fv_accuracy = compute_accuracy(self_fv_model_outputs, test_data, args)
        print(f"self_fv_accuracy: {self_fv_accuracy}")
        
        self_fv_macro_f1 = compute_macro_f1(self_fv_model_outputs, test_data, args)
        print(f"self_fv macro-F1: {self_fv_macro_f1}")
        
    if args.use_fv:
        fv_outputs, fv_accuracy = compute_accuracy(fv_model_outputs, test_data, args)
        print(f"fv_accuracy: {fv_accuracy}")
        
        fv_macro_f1 = compute_macro_f1(fv_model_outputs, test_data, args)
        print(f"fv macro-F1: {fv_macro_f1}")
    
    if accuracy == 0.0:
        auroc, aupr = 0.0, 0.0
        option_auroc, option_aupr = 0.0, 0.0
    else:
        try:
            auroc, aupr = compute_auroc_aupr_from_entropy(model_outputs)
        except Exception:
            auroc, aupr = 0.0, 0.0
        try:
            option_auroc, option_aupr = compute_auroc_aupr_from_option_entropy(model_outputs)
        except Exception:
            option_auroc, option_aupr = 0.0, 0.0
    
    if args.use_self_fv:
        if self_fv_accuracy == 0.0:
            self_fv_auroc, self_fv_aupr = 0.0, 0.0
            self_fv_option_auroc, self_fv_option_aupr = 0.0, 0.0
        else:
            try:
                self_fv_auroc, self_fv_aupr = compute_auroc_aupr_from_entropy(self_fv_outputs)
            except Exception:
                self_fv_auroc, self_fv_aupr = 0.0, 0.0
            try:
                self_fv_option_auroc, self_fv_option_aupr = compute_auroc_aupr_from_option_entropy(self_fv_outputs)
            except Exception:
                self_fv_option_auroc, self_fv_option_aupr = 0.0, 0.0
    
    # Compute cross AUROC/AUPR: model_outputs correctness vs self_fv_model_outputs entropy
        if accuracy == 0.0:
            cross_auroc, cross_aupr = 0.0, 0.0
            cross_option_auroc, cross_option_aupr = 0.0, 0.0
        else:
            try:
                cross_auroc, cross_aupr = compute_cross_auroc_aupr(model_outputs, self_fv_model_outputs)
            except Exception:
                cross_auroc, cross_aupr = 0.0, 0.0
            try:
                cross_option_auroc, cross_option_aupr = compute_cross_auroc_aupr_from_option_entropy(model_outputs, self_fv_model_outputs)
            except Exception:
                cross_option_auroc, cross_option_aupr = 0.0, 0.0
    
    if args.use_fv:
        if fv_accuracy == 0.0:
            fv_auroc, fv_aupr = 0.0, 0.0
            fv_option_auroc, fv_option_aupr = 0.0, 0.0
        else:
            try:
                fv_auroc, fv_aupr = compute_auroc_aupr_from_entropy(fv_outputs)
            except Exception:
                fv_auroc, fv_aupr = 0.0, 0.0
            try:
                fv_option_auroc, fv_option_aupr = compute_auroc_aupr_from_option_entropy(fv_outputs)
            except Exception:
                fv_option_auroc, fv_option_aupr = 0.0, 0.0
                
        if accuracy == 0.0:
            fv_cross_auroc, fv_cross_aupr = 0.0, 0.0
            fv_cross_option_auroc, fv_cross_option_aupr = 0.0, 0.0
        else:
            try:
                fv_cross_auroc, fv_cross_aupr = compute_cross_auroc_aupr(model_outputs, fv_model_outputs)
            except Exception:
                fv_cross_auroc, fv_cross_aupr = 0.0, 0.0
            try:
                fv_cross_option_auroc, fv_cross_option_aupr = compute_cross_auroc_aupr_from_option_entropy(model_outputs, fv_model_outputs)
            except Exception:
                fv_cross_option_auroc, fv_cross_option_aupr = 0.0, 0.0
    
    # Log to wandb
    if args.use_wandb:
        result = {
            "clean/accuracy": accuracy,
            "clean/macro_f1": macro_f1,
            "clean/auroc": auroc, 
            "clean/aupr": aupr,
            "clean/option_auroc": option_auroc,
            "clean/option_aupr": option_aupr,
            'clean/avg/entropy': np.mean([output['entropy'] for output in model_outputs]), 
            'clean/avg/option_entropy': np.mean([output['option_entropy'] for output in model_outputs]), 
            'clean/avg/separator_entropy': np.mean([output['separator_entropy'] for output in model_outputs]),
            'clean/avg/first_token_entropy': np.mean([output['first_token_entropy'] for output in model_outputs]),
            'clean/avg/first_token_option_entropy': np.mean([output['first_token_option_entropy'] for output in model_outputs]),
            'clean/avg/correct': np.mean([output['correct'] for output in model_outputs]),
            'clean/model_outputs': wandb.Table(dataframe=pd.DataFrame(model_outputs))
        }
        if args.use_self_fv:
            result['self/accuracy'] = self_fv_accuracy
            result['self/macro_f1'] = self_fv_macro_f1
            result['self/auroc'] = self_fv_auroc
            result['self/aupr'] = self_fv_aupr
            result['self/option_auroc'] = self_fv_option_auroc
            result['self/option_aupr'] = self_fv_option_aupr
            result['self/avg/entropy'] = np.mean([output['entropy'] for output in self_fv_outputs])
            result['self/avg/option_entropy'] = np.mean([output['option_entropy'] for output in self_fv_outputs])
            result['self/avg/separator_entropy'] = np.mean([output['separator_entropy'] for output in self_fv_outputs])
            result['self/avg/first_token_entropy'] = np.mean([output['first_token_entropy'] for output in self_fv_outputs])
            result['self/avg/first_token_option_entropy'] = np.mean([output['first_token_option_entropy'] for output in self_fv_outputs])
            result['self/avg/correct'] = np.mean([output['correct'] for output in self_fv_outputs])
            result['self/cross/auroc'] = cross_auroc
            result['self/cross/aupr'] = cross_aupr
            result['self/cross/option_auroc'] = cross_option_auroc
            result['self/cross/option_aupr'] = cross_option_aupr
            result['self/model_outputs'] = wandb.Table(dataframe=pd.DataFrame(self_fv_outputs))
        if args.use_fv:
            result['fv/accuracy'] = fv_accuracy
            result['fv/macro_f1'] = fv_macro_f1
            result['fv/auroc'] = fv_auroc
            result['fv/aupr'] = fv_aupr
            result['fv/option_auroc'] = fv_option_auroc
            result['fv/option_aupr'] = fv_option_aupr
            result['fv/avg/entropy'] = np.mean([output['entropy'] for output in fv_outputs])
            result['fv/avg/option_entropy'] = np.mean([output['option_entropy'] for output in fv_outputs])
            result['fv/avg/separator_entropy'] = np.mean([output['separator_entropy'] for output in fv_outputs])
            result['fv/avg/first_token_entropy'] = np.mean([output['first_token_entropy'] for output in fv_outputs])
            result['fv/avg/first_token_option_entropy'] = np.mean([output['first_token_option_entropy'] for output in fv_outputs])
            result['fv/avg/correct'] = np.mean([output['correct'] for output in fv_outputs])
            result['fv/cross/auroc'] = fv_cross_auroc
            result['fv/cross/aupr'] = fv_cross_aupr
            result['fv/cross/option_auroc'] = fv_cross_option_auroc
            result['fv/cross/option_aupr'] = fv_cross_option_aupr
            result['fv/model_outputs'] = wandb.Table(dataframe=pd.DataFrame(fv_outputs))
        
        wandb.log(result)
        wandb.finish()
    
    # Save prompt logs to JSON file if requested
    if args.log_prompt == 1 and prompt_logs:
        log_filename = f"prompt_logs_{args.data}_seed{args.seed}_num_noisy_shots{args.num_noisy_shots}_num_ood_shots{args.num_ood_shots}.json"
        log_path = os.path.join(args.result_path, log_filename)
        with open(log_path, 'w') as f:
            json.dump(prompt_logs, f, indent=2)
        print(f"Saved {len(prompt_logs)} prompt logs to: {log_path}")
